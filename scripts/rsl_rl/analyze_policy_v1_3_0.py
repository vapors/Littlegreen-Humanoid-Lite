"""Offline rollout diagnostics for Berkeley Humanoid Lite v1.3.0 ST3215 policies.

This script is intentionally analysis-only. It loads a checkpoint, rolls out the selected
v1.3.0 ST3215 task, and writes JSON/CSV summaries for raw policy output, bounded action use,
physical targets, measured joints, velocity saturation, torque use, standing quality,
and fall/timeout counts.

Example:
    python scripts/rsl_rl/analyze_policy_v1_3_0.py \
        --task Velocity-Lilgreen-Stand-ST3215-v0 \
        --load_run 2026-07-10_12-00-00_stand_st3215_seed42 \
        --checkpoint model_5000.pt \
        --num_envs 256 \
        --steps 1500 \
        --headless
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Analyze a v1.3.0 ST3215-aware Berkeley Humanoid Lite policy rollout.")
parser.add_argument("--task", type=str, required=True, help="Registered Isaac Lab task name.")
parser.add_argument("--num_envs", type=int, default=256, help="Parallel rollout environments.")
parser.add_argument("--steps", type=int, default=1500, help="Policy steps to collect.")
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Output directory. Default: <run>/analysis/<timestamp>.",
)
parser.add_argument("--disable_fabric", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

import littlegreen_humanoid_lite.tasks  # noqa: F401,E402


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float32, copy=False)


def _percentiles(values: np.ndarray, q: tuple[float, ...]) -> list[float]:
    return [float(v) for v in np.percentile(values, q)]


def _safe_scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    selected = values[mask]
    return float(np.mean(selected)) if selected.size else float("nan")


def _extract_timeouts(extras: Any, dones: torch.Tensor) -> torch.Tensor:
    """Best-effort extraction of timeout flags across RSL-RL wrapper versions."""
    if isinstance(extras, dict):
        for key in ("time_outs", "timeouts", "truncated"):
            value = extras.get(key)
            if isinstance(value, torch.Tensor) and value.shape[0] == dones.shape[0]:
                return value.to(device=dones.device, dtype=torch.bool)
    return torch.zeros_like(dones, dtype=torch.bool)


def _joint_stats(
    joint_names: list[str],
    raw: np.ndarray,
    bounded: np.ndarray,
    targets: np.ndarray,
    q: np.ndarray,
    qdot: np.ndarray,
    torque: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    default: np.ndarray,
    standing_mask: np.ndarray,
    velocity_limit: float,
    torque_limit: float,
    torque_soft_ratio: float,
) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    for j, name in enumerate(joint_names):
        raw_j = raw[:, j]
        bounded_j = bounded[:, j]
        target_j = targets[:, j]
        q_j = q[:, j]
        qdot_j = qdot[:, j]
        torque_j = torque[:, j]
        standing_raw_j = raw_j[standing_mask]
        standing_bounded_j = bounded_j[standing_mask]
        standing_target_j = target_j[standing_mask]
        standing_q_j = q_j[standing_mask]
        standing_torque_j = torque_j[standing_mask]
        p95, p99 = _percentiles(raw_j, (95.0, 99.0))
        abs_p95, abs_p99 = _percentiles(np.abs(raw_j), (95.0, 99.0))
        rows.append(
            {
                "index": j,
                "joint": name,
                "raw_mean": float(np.mean(raw_j)),
                "raw_std": float(np.std(raw_j)),
                "raw_min": float(np.min(raw_j)),
                "raw_max": float(np.max(raw_j)),
                "raw_p95": p95,
                "raw_p99": p99,
                "raw_abs_p95": abs_p95,
                "raw_abs_p99": abs_p99,
                "raw_excess_fraction": float(np.mean(np.abs(raw_j) > 1.0)),
                "bounded_saturation_fraction": float(np.mean(np.abs(bounded_j) >= 0.999)),
                "target_lower_limit_fraction": float(np.mean(target_j <= lower[j] + 1.0e-3)),
                "target_upper_limit_fraction": float(np.mean(target_j >= upper[j] - 1.0e-3)),
                "target_min_rad": float(np.min(target_j)),
                "target_max_rad": float(np.max(target_j)),
                "q_mean_rad": float(np.mean(q_j)),
                "q_min_rad": float(np.min(q_j)),
                "q_max_rad": float(np.max(q_j)),
                "qdot_abs_p95_rad_s": float(np.percentile(np.abs(qdot_j), 95.0)),
                "qdot_abs_p99_rad_s": float(np.percentile(np.abs(qdot_j), 99.0)),
                "velocity_limit_fraction": float(np.mean(np.abs(qdot_j) >= 0.95 * velocity_limit)),
                "torque_abs_mean_nm": float(np.mean(np.abs(torque_j))),
                "torque_abs_p95_nm": float(np.percentile(np.abs(torque_j), 95.0)),
                "torque_abs_p99_nm": float(np.percentile(np.abs(torque_j), 99.0)),
                "torque_over_soft_fraction": float(
                    np.mean(np.abs(torque_j) >= torque_soft_ratio * torque_limit)
                ),
                "torque_limit_fraction": float(np.mean(np.abs(torque_j) >= 0.95 * torque_limit)),
                "target_residual_abs_mean_rad": float(np.mean(np.abs(target_j - default[j]))),
                "standing_raw_abs_mean": float(np.mean(np.abs(standing_raw_j))) if standing_raw_j.size else float("nan"),
                "standing_bounded_abs_mean": float(np.mean(np.abs(standing_bounded_j))) if standing_bounded_j.size else float("nan"),
                "standing_raw_excess_fraction": float(np.mean(np.abs(standing_raw_j) > 1.0)) if standing_raw_j.size else float("nan"),
                "standing_target_residual_abs_mean_rad": float(np.mean(np.abs(standing_target_j - default[j]))) if standing_target_j.size else float("nan"),
                "standing_target_limit_fraction": float(
                    np.mean((standing_target_j <= lower[j] + 1.0e-3) | (standing_target_j >= upper[j] - 1.0e-3))
                ) if standing_target_j.size else float("nan"),
                "standing_joint_pos_error_abs_mean_rad": float(np.mean(np.abs(standing_q_j - default[j]))) if standing_q_j.size else float("nan"),
                "standing_torque_abs_mean_nm": float(np.mean(np.abs(standing_torque_j))) if standing_torque_j.size else float("nan"),
                "standing_torque_over_soft_fraction": float(
                    np.mean(np.abs(standing_torque_j) >= torque_soft_ratio * torque_limit)
                ) if standing_torque_j.size else float("nan"),
                "standing_torque_limit_fraction": float(
                    np.mean(np.abs(standing_torque_j) >= 0.95 * torque_limit)
                ) if standing_torque_j.size else float("nan"),
            }
        )
    return rows


def main() -> None:
    if args_cli.steps <= 0:
        raise ValueError("--steps must be positive")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    checkpoint = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    run_dir = Path(checkpoint).parent
    output_dir = Path(args_cli.output_dir) if args_cli.output_dir else (
        run_dir / "analysis" / datetime.now().strftime("%Y%m%dT%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ANALYZE] Task:       {args_cli.task}")
    print(f"[ANALYZE] Checkpoint: {checkpoint}")
    print(f"[ANALYZE] Output:     {output_dir}")

    base_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(base_env.unwrapped, DirectMARLEnv):
        base_env = multi_agent_to_single_agent(base_env)
    vec_env = RslRlVecEnvWrapper(base_env)

    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=vec_env.unwrapped.device)

    env = vec_env.unwrapped
    action_term = env.action_manager.get_term("joint_pos")
    if not all(hasattr(action_term, name) for name in ("bounded_actions", "processed_actions", "lower_limits")):
        raise RuntimeError(
            "This analyzer expects the v1.2.3 bounded action contract. "
            "Use Velocity-Lilgreen-Stand-v0 or Velocity-Lilgreen-Hardware-v0."
        )

    robot = env.scene["robot"]
    joint_ids, joint_names = robot.find_joints(action_term.joint_names, preserve_order=True)
    contact_sensor = env.scene.sensors["contact_forces"]
    contact_body_ids, contact_body_names = contact_sensor.find_bodies(".*_ankle_roll", preserve_order=True)
    foot_body_ids, foot_body_names = robot.find_bodies(contact_body_names, preserve_order=True)
    if foot_body_names != contact_body_names:
        raise RuntimeError(
            f"Foot body ordering mismatch: robot={foot_body_names}, contact_sensor={contact_body_names}"
        )

    velocity_limit = float(getattr(env_cfg.scene.robot.actuators["legs"], "velocity_limit_sim", 4.72))
    torque_limit = float(getattr(env_cfg.scene.robot.actuators["legs"], "effort_limit_sim", 2.94))
    torque_soft_ratio = 0.70
    desired_base_com_height_m = float(env_cfg.rewards.stand_base_height.params["desired_height"])
    action_contract_version = 3 if hasattr(action_term, "residual_scale_rad") else 2
    action_transform = (
        "bounded_default_centered_symmetric_residual"
        if action_contract_version == 3
        else "bounded_default_centered_asymmetric_full_range"
    )
    policy_dt = float(env.step_dt)

    raw_chunks: list[np.ndarray] = []
    bounded_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    q_chunks: list[np.ndarray] = []
    qdot_chunks: list[np.ndarray] = []
    torque_chunks: list[np.ndarray] = []
    command_chunks: list[np.ndarray] = []
    lin_tracking_error_chunks: list[np.ndarray] = []
    yaw_tracking_error_chunks: list[np.ndarray] = []
    standing_mask_chunks: list[np.ndarray] = []
    base_com_height_chunks: list[np.ndarray] = []
    actuator_drive_target_chunks: list[np.ndarray] = []
    actuator_delayed_target_chunks: list[np.ndarray] = []

    stable_standing_age = torch.zeros(env.num_envs, device=env.device)
    posture_conforming_age = torch.zeros(env.num_envs, device=env.device)
    reached_stable_10s = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    reached_stable_20s = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    reached_posture_10s = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    reached_posture_20s = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    zero_cmd_drift_sum = 0.0
    zero_cmd_slip_sum = 0.0
    standing_samples = 0
    standing_gate_counts = {
        "upright": 0.0,
        "quiet_xy": 0.0,
        "quiet_yaw": 0.0,
        "near_default": 0.0,
        "both_feet": 0.0,
        "stable_conditions": 0.0,
        "posture_conditions": 0.0,
    }
    standing_base_com_height_sum = 0.0
    standing_base_com_height_error_sum = 0.0
    max_continuous_stable_standing_s = 0.0
    max_continuous_posture_conforming_s = 0.0
    done_count = 0
    timeout_count = 0
    fall_count = 0

    obs, _ = vec_env.get_observations()

    for step in range(args_cli.steps):
        with torch.inference_mode():
            raw_action = policy(obs)
            step_out = vec_env.step(raw_action)

        if not isinstance(step_out, tuple) or len(step_out) < 4:
            raise RuntimeError(f"Unexpected RSL-RL step return: {type(step_out)}")
        obs, _, dones, extras = step_out[:4]
        dones = dones.to(dtype=torch.bool)
        timeouts = _extract_timeouts(extras, dones)
        falls = dones & ~timeouts

        command = env.command_manager.get_command("base_velocity")
        vel_yaw = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w[:, :3])
        lin_tracking_error = command[:, :2] - vel_yaw[:, :2]
        yaw_tracking_error = command[:, 2] - robot.data.root_ang_vel_w[:, 2]
        standing = torch.linalg.vector_norm(command[:, :3], dim=1) < 0.05
        upright = robot.data.projected_gravity_b[:, 2] < -0.97
        quiet_xy = torch.linalg.vector_norm(robot.data.root_lin_vel_b[:, :2], dim=1) < 0.15
        quiet_yaw = torch.abs(robot.data.root_ang_vel_b[:, 2]) < 0.20
        joint_error = torch.abs(
            robot.data.joint_pos[:, joint_ids] - robot.data.default_joint_pos[:, joint_ids]
        )
        near_default = torch.amax(joint_error, dim=1) < 0.20
        forces = contact_sensor.data.net_forces_w_history[:, :, contact_body_ids, :]
        foot_contact = forces.norm(dim=-1).max(dim=1)[0] > 1.0
        both_feet = torch.all(foot_contact, dim=1)
        stable_standing = standing & upright & quiet_xy & quiet_yaw & both_feet
        posture_conforming = stable_standing & near_default
        stable_standing_age = torch.where(
            stable_standing,
            stable_standing_age + policy_dt,
            torch.zeros_like(stable_standing_age),
        )
        posture_conforming_age = torch.where(
            posture_conforming,
            posture_conforming_age + policy_dt,
            torch.zeros_like(posture_conforming_age),
        )
        reached_stable_10s |= stable_standing_age >= 10.0
        reached_stable_20s |= stable_standing_age >= 20.0
        reached_posture_10s |= posture_conforming_age >= 10.0
        reached_posture_20s |= posture_conforming_age >= 20.0
        max_continuous_stable_standing_s = max(
            max_continuous_stable_standing_s,
            float(torch.max(stable_standing_age).item()),
        )
        max_continuous_posture_conforming_s = max(
            max_continuous_posture_conforming_s,
            float(torch.max(posture_conforming_age).item()),
        )

        stand_n = int(torch.sum(standing).item())
        if stand_n:
            drift = torch.linalg.vector_norm(robot.data.root_lin_vel_b[:, :2], dim=1)
            body_vel = robot.data.body_lin_vel_w[:, foot_body_ids, :2]
            slip = torch.sum(torch.linalg.vector_norm(body_vel, dim=-1) * foot_contact, dim=1)
            zero_cmd_drift_sum += float(torch.sum(drift * standing).item())
            zero_cmd_slip_sum += float(torch.sum(slip * standing).item())
            standing_gate_counts["upright"] += float(torch.sum((upright & standing).float()).item())
            standing_gate_counts["quiet_xy"] += float(torch.sum((quiet_xy & standing).float()).item())
            standing_gate_counts["quiet_yaw"] += float(torch.sum((quiet_yaw & standing).float()).item())
            standing_gate_counts["near_default"] += float(torch.sum((near_default & standing).float()).item())
            standing_gate_counts["both_feet"] += float(torch.sum((both_feet & standing).float()).item())
            standing_gate_counts["stable_conditions"] += float(
                torch.sum(stable_standing.float()).item()
            )
            standing_gate_counts["posture_conditions"] += float(
                torch.sum(posture_conforming.float()).item()
            )
            base_com_height = robot.data.root_com_pos_w[:, 2]
            standing_base_com_height_sum += float(torch.sum(base_com_height * standing).item())
            standing_base_com_height_error_sum += float(
                torch.sum(torch.abs(base_com_height - desired_base_com_height_m) * standing).item()
            )
            standing_samples += stand_n

        raw_chunks.append(_to_numpy(action_term.raw_actions))
        bounded_chunks.append(_to_numpy(action_term.bounded_actions))
        target_chunks.append(_to_numpy(action_term.processed_actions))
        q_chunks.append(_to_numpy(robot.data.joint_pos[:, joint_ids]))
        qdot_chunks.append(_to_numpy(robot.data.joint_vel[:, joint_ids]))
        torque_chunks.append(_to_numpy(robot.data.applied_torque[:, joint_ids]))
        command_chunks.append(_to_numpy(command[:, :3]))
        lin_tracking_error_chunks.append(_to_numpy(lin_tracking_error))
        yaw_tracking_error_chunks.append(_to_numpy(yaw_tracking_error.unsqueeze(-1)))
        standing_mask_chunks.append(_to_numpy(standing).astype(bool, copy=False))
        base_com_height_chunks.append(_to_numpy(robot.data.root_com_pos_w[:, 2]))
        if hasattr(action_term, "drive_targets"):
            actuator_drive_target_chunks.append(_to_numpy(action_term.drive_targets))
            actuator_delayed_target_chunks.append(_to_numpy(action_term.delayed_policy_targets))

        done_count += int(torch.sum(dones).item())
        timeout_count += int(torch.sum(timeouts).item())
        fall_count += int(torch.sum(falls).item())
        stable_standing_age[dones] = 0.0
        posture_conforming_age[dones] = 0.0

        if (step + 1) % max(1, args_cli.steps // 10) == 0:
            print(f"[ANALYZE] collected {step + 1}/{args_cli.steps} policy steps")

    raw = np.concatenate(raw_chunks, axis=0)
    bounded = np.concatenate(bounded_chunks, axis=0)
    targets = np.concatenate(target_chunks, axis=0)
    q = np.concatenate(q_chunks, axis=0)
    qdot = np.concatenate(qdot_chunks, axis=0)
    torque = np.concatenate(torque_chunks, axis=0)
    commands = np.concatenate(command_chunks, axis=0)
    lin_tracking_error = np.concatenate(lin_tracking_error_chunks, axis=0)
    yaw_tracking_error = np.concatenate(yaw_tracking_error_chunks, axis=0).reshape(-1)
    standing_mask = np.concatenate(standing_mask_chunks, axis=0).astype(bool, copy=False)
    base_com_height_all = np.concatenate(base_com_height_chunks, axis=0)
    actuator_drive_targets = np.concatenate(actuator_drive_target_chunks, axis=0) if actuator_drive_target_chunks else None
    actuator_delayed_targets = np.concatenate(actuator_delayed_target_chunks, axis=0) if actuator_delayed_target_chunks else None
    lower = _to_numpy(action_term.lower_limits[0])
    upper = _to_numpy(action_term.upper_limits[0])
    default = _to_numpy(action_term.default_positions[0])

    rows = _joint_stats(
        joint_names,
        raw,
        bounded,
        targets,
        q,
        qdot,
        torque,
        lower,
        upper,
        default,
        standing_mask,
        velocity_limit,
        torque_limit,
        torque_soft_ratio,
    )

    total_env_steps = args_cli.steps * env.num_envs
    summary = {
        "metadata": {
            "task": args_cli.task,
            "checkpoint": os.path.abspath(checkpoint),
            "num_envs": int(env.num_envs),
            "policy_steps": int(args_cli.steps),
            "policy_dt_s": policy_dt,
            "policy_rate_hz": 1.0 / policy_dt,
            "samples": int(raw.shape[0]),
            "action_contract_version": action_contract_version,
            "action_transform": action_transform,
            "desired_base_com_height_m": desired_base_com_height_m,
            "actuator_model_name": getattr(action_term, "actuator_model_name", None),
        },
        "global": {
            "raw_action_mean_abs": float(np.mean(np.abs(raw))),
            "raw_action_std": float(np.std(raw)),
            "raw_action_min": float(np.min(raw)),
            "raw_action_max": float(np.max(raw)),
            "raw_action_excess_fraction": float(np.mean(np.abs(raw) > 1.0)),
            "bounded_saturation_fraction": float(np.mean(np.abs(bounded) >= 0.999)),
            "target_limit_fraction": float(
                np.mean((targets <= lower[None, :] + 1.0e-3) | (targets >= upper[None, :] - 1.0e-3))
            ),
            "joint_velocity_limit_fraction": float(np.mean(np.abs(qdot) >= 0.95 * velocity_limit)),
            "torque_over_soft_fraction": float(
                np.mean(np.abs(torque) >= torque_soft_ratio * torque_limit)
            ),
            "torque_limit_fraction": float(np.mean(np.abs(torque) >= 0.95 * torque_limit)),
            "standing_command_fraction": float(np.mean(np.linalg.norm(commands, axis=1) < 0.05)),
            "linear_velocity_tracking_rmse_mps": float(np.sqrt(np.mean(np.square(lin_tracking_error)))),
            "linear_velocity_tracking_vector_error_mean_mps": float(np.mean(np.linalg.norm(lin_tracking_error, axis=1))),
            "yaw_rate_tracking_rmse_rad_s": float(np.sqrt(np.mean(np.square(yaw_tracking_error)))),
            # Legacy aliases retain posture-conforming meaning for continuity with v1.2.2 reports.
            "standing_success_10s_env_fraction": _safe_scalar(torch.mean(reached_posture_10s.float())),
            "standing_success_20s_env_fraction": _safe_scalar(torch.mean(reached_posture_20s.float())),
            "max_continuous_standing_success_s": max_continuous_posture_conforming_s,
            "standing_stable_success_10s_env_fraction": _safe_scalar(torch.mean(reached_stable_10s.float())),
            "standing_stable_success_20s_env_fraction": _safe_scalar(torch.mean(reached_stable_20s.float())),
            "max_continuous_stable_standing_s": max_continuous_stable_standing_s,
            "standing_posture_success_10s_env_fraction": _safe_scalar(torch.mean(reached_posture_10s.float())),
            "standing_posture_success_20s_env_fraction": _safe_scalar(torch.mean(reached_posture_20s.float())),
            "max_continuous_posture_conforming_s": max_continuous_posture_conforming_s,
            "standing_upright_fraction": standing_gate_counts["upright"] / max(1, standing_samples),
            "standing_quiet_xy_fraction": standing_gate_counts["quiet_xy"] / max(1, standing_samples),
            "standing_quiet_yaw_fraction": standing_gate_counts["quiet_yaw"] / max(1, standing_samples),
            "standing_near_default_fraction": standing_gate_counts["near_default"] / max(1, standing_samples),
            "standing_both_feet_fraction": standing_gate_counts["both_feet"] / max(1, standing_samples),
            "standing_stable_all_conditions_fraction": standing_gate_counts["stable_conditions"] / max(1, standing_samples),
            "standing_posture_all_conditions_fraction": standing_gate_counts["posture_conditions"] / max(1, standing_samples),
            "standing_all_conditions_fraction": standing_gate_counts["posture_conditions"] / max(1, standing_samples),
            "standing_base_com_height_mean_m": standing_base_com_height_sum / max(1, standing_samples),
            "standing_base_com_height_p05_m": float(np.percentile(base_com_height_all[standing_mask], 5.0)) if np.any(standing_mask) else float("nan"),
            "standing_base_com_height_median_m": float(np.percentile(base_com_height_all[standing_mask], 50.0)) if np.any(standing_mask) else float("nan"),
            "standing_base_com_height_p95_m": float(np.percentile(base_com_height_all[standing_mask], 95.0)) if np.any(standing_mask) else float("nan"),
            "standing_base_com_height_error_mean_m": standing_base_com_height_error_sum / max(1, standing_samples),
            "zero_command_xy_drift_mean_mps": zero_cmd_drift_sum / max(1, standing_samples),
            "standing_foot_slip_mean_mps": zero_cmd_slip_sum / max(1, standing_samples),
            "actuator_policy_to_drive_lag_abs_mean_rad": (
                float(np.mean(np.abs(targets - actuator_drive_targets)))
                if actuator_drive_targets is not None else None
            ),
            "actuator_policy_to_delayed_abs_mean_rad": (
                float(np.mean(np.abs(targets - actuator_delayed_targets)))
                if actuator_delayed_targets is not None else None
            ),
            "actuator_delay_mean_ms": (
                float(torch.mean(action_term.sampled_total_delay_s).item() * 1000.0)
                if hasattr(action_term, "sampled_total_delay_s") else None
            ),
            "actuator_tau_mean_ms": (
                float(torch.mean(action_term.sampled_tau_s).item() * 1000.0)
                if hasattr(action_term, "sampled_tau_s") else None
            ),
            "actuator_velocity_scale_mean": (
                float(torch.mean(action_term.sampled_velocity_scale).item())
                if hasattr(action_term, "sampled_velocity_scale") else None
            ),
            "done_count": done_count,
            "timeout_count": timeout_count,
            "fall_count": fall_count,
            "fall_fraction_per_env_step": fall_count / max(1, total_env_steps),
        },
        "joint_stats": rows,
    }

    with open(output_dir / "policy_analysis.json", "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)

    with open(output_dir / "joint_action_stats.csv", "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n[ANALYZE] Global summary")
    for key, value in summary["global"].items():
        print(f"  {key}: {value}")
    print(f"\n[ANALYZE] JSON: {output_dir / 'policy_analysis.json'}")
    print(f"[ANALYZE] CSV:  {output_dir / 'joint_action_stats.csv'}")

    vec_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
