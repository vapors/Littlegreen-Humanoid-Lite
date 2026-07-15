"""Measure q_default base-COM height under several distinct conditions in Isaac Lab.

The script separates three quantities that should not be conflated:

1. Exact reset-state geometry at q_default, sampled immediately after env.reset().
2. Unfiltered zero-action dynamic equilibrium while continuously commanding q_default.
3. Supported q_default equilibrium, using only samples that are upright, quiet,
   close to q_default, and supported by both feet.

For the v1.2.3 bounded residual action contract, zero normalized action maps to
q_default. Reset/randomization terms that would contaminate the measurement are
disabled, zero velocity commands are forced, and fall-triggered auto-reset is disabled
so rejected/fallen states are counted rather than silently replaced by fresh resets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure nominal q_default base COM height.")
parser.add_argument("--task", type=str, default="Velocity-Lilgreen-Stand-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--settle_seconds", type=float, default=5.0)
parser.add_argument("--sample_seconds", type=float, default=2.0)
parser.add_argument("--output", type=str, default="")
parser.add_argument("--disable_fabric", action="store_true", default=False)

# Acceptance-gate thresholds for the supported-q_default measurement.
parser.add_argument("--upright_gravity_z_max", type=float, default=-0.97)
parser.add_argument("--quiet_xy_speed_threshold", type=float, default=0.15)
parser.add_argument("--quiet_yaw_rate_threshold", type=float, default=0.20)
parser.add_argument("--near_default_threshold_rad", type=float, default=0.20)
parser.add_argument("--contact_force_threshold_n", type=float, default=1.0)
parser.add_argument("--feet_body_pattern", type=str, default=".*_ankle_roll")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.utils import parse_env_cfg

import littlegreen_humanoid_lite.tasks  # noqa: F401,E402


def _configure_deterministic_measurement(env_cfg, measurement_duration_s: float) -> None:
    """Remove randomization and command variation from the measurement environment."""
    command_cfg = env_cfg.commands.base_velocity
    command_cfg.rel_standing_envs = 1.0
    command_cfg.rel_heading_envs = 0.0
    command_cfg.ranges.lin_vel_x = (0.0, 0.0)
    command_cfg.ranges.lin_vel_y = (0.0, 0.0)
    command_cfg.ranges.ang_vel_z = (0.0, 0.0)
    command_cfg.ranges.heading = (0.0, 0.0)

    events = env_cfg.events
    for name in ("physics_material", "base_mass", "actuator_gains", "push_robot"):
        if hasattr(events, name):
            setattr(events, name, None)

    if getattr(events, "reset_base", None) is not None:
        events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

    if getattr(events, "reset_robot_joints", None) is not None:
        events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
        events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

    # A fall during this diagnostic should remain a rejected sample. Do not let the
    # orientation termination silently reset that environment into a clean pose.
    terminations = getattr(env_cfg, "terminations", None)
    if terminations is not None and hasattr(terminations, "base_orientation"):
        terminations.base_orientation = None

    # Keep timeout from occurring during a deliberately long settle/sample request.
    env_cfg.episode_length_s = max(
        float(env_cfg.episode_length_s),
        float(measurement_duration_s) + 10.0,
    )


def _stats(samples: torch.Tensor) -> dict[str, int | float | None]:
    """Return robust scalar statistics, including quantiles and explicit sample count."""
    flat = samples.flatten()
    count = int(flat.numel())
    if count == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p05": None,
            "median": None,
            "p95": None,
            "max": None,
        }

    quantiles = torch.quantile(
        flat,
        torch.tensor([0.05, 0.50, 0.95], device=flat.device, dtype=flat.dtype),
    )
    return {
        "count": count,
        "mean": float(torch.mean(flat).item()),
        "std": float(torch.std(flat, unbiased=False).item()),
        "min": float(torch.min(flat).item()),
        "p05": float(quantiles[0].item()),
        "median": float(quantiles[1].item()),
        "p95": float(quantiles[2].item()),
        "max": float(torch.max(flat).item()),
    }


def _fraction(mask: torch.Tensor) -> float:
    return float(mask.float().mean().item())


def _posture_error(robot, joint_ids) -> tuple[torch.Tensor, torch.Tensor]:
    error = robot.data.joint_pos[:, joint_ids] - robot.data.default_joint_pos[:, joint_ids]
    rms = torch.sqrt(torch.mean(torch.square(error), dim=1))
    max_abs = torch.amax(torch.abs(error), dim=1)
    return rms, max_abs


def main() -> None:
    if args_cli.settle_seconds <= 0.0 or args_cli.sample_seconds <= 0.0:
        raise ValueError("settle_seconds and sample_seconds must be positive")
    if args_cli.num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if args_cli.near_default_threshold_rad <= 0.0:
        raise ValueError("near_default_threshold_rad must be positive")
    if args_cli.contact_force_threshold_n < 0.0:
        raise ValueError("contact_force_threshold_n must be non-negative")

    measurement_duration_s = args_cli.settle_seconds + args_cli.sample_seconds
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    _configure_deterministic_measurement(env_cfg, measurement_duration_s)

    env = None
    try:
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        env.reset()

        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        action_term = unwrapped.action_manager.get_term("joint_pos")
        joint_ids, joint_names = robot.find_joints(action_term.joint_names, preserve_order=True)

        feet_sensor_cfg = SceneEntityCfg(
            "contact_forces",
            body_names=args_cli.feet_body_pattern,
            preserve_order=True,
        )
        feet_sensor_cfg.resolve(unwrapped.scene)
        contact_sensor = unwrapped.scene.sensors[feet_sensor_cfg.name]

        if not feet_sensor_cfg.body_ids:
            raise RuntimeError(
                f"No contact-sensor bodies matched feet pattern: {args_cli.feet_body_pattern!r}"
            )

        policy_dt = float(unwrapped.step_dt)
        settle_steps = max(1, round(args_cli.settle_seconds / policy_dt))
        sample_steps = max(1, round(args_cli.sample_seconds / policy_dt))
        num_envs = int(unwrapped.num_envs)
        action_dim = int(env.action_space.shape[-1])
        actions = torch.zeros(
            (num_envs, action_dim),
            device=unwrapped.device,
            dtype=torch.float32,
        )

        # ---------------------------------------------------------------------
        # 1. Exact reset-state geometry at deterministic q_default.
        # ---------------------------------------------------------------------
        reset_posture_rms, reset_posture_max = _posture_error(robot, joint_ids)
        reset_snapshot = {
            "root_frame_height_m": _stats(robot.data.root_pos_w[:, 2].detach().clone()),
            "base_com_height_m": _stats(robot.data.root_com_pos_w[:, 2].detach().clone()),
            "joint_posture_rms_rad": _stats(reset_posture_rms.detach().clone()),
            "joint_posture_max_abs_rad": _stats(reset_posture_max.detach().clone()),
        }

        # ---------------------------------------------------------------------
        # Settle under zero normalized action => q_target = q_default.
        # ---------------------------------------------------------------------
        for _ in range(settle_steps):
            env.step(actions)

        settled_posture_rms, settled_posture_max = _posture_error(robot, joint_ids)
        settled_snapshot = {
            "root_frame_height_m": _stats(robot.data.root_pos_w[:, 2].detach().clone()),
            "base_com_height_m": _stats(robot.data.root_com_pos_w[:, 2].detach().clone()),
            "joint_posture_rms_rad": _stats(settled_posture_rms.detach().clone()),
            "joint_posture_max_abs_rad": _stats(settled_posture_max.detach().clone()),
        }

        # ---------------------------------------------------------------------
        # 2. Unfiltered zero-action equilibrium and
        # 3. supported-q_default filtered equilibrium.
        # ---------------------------------------------------------------------
        root_frame_samples: list[torch.Tensor] = []
        base_com_samples: list[torch.Tensor] = []
        posture_rms_samples: list[torch.Tensor] = []
        posture_max_samples: list[torch.Tensor] = []

        upright_masks: list[torch.Tensor] = []
        quiet_xy_masks: list[torch.Tensor] = []
        quiet_yaw_masks: list[torch.Tensor] = []
        near_default_masks: list[torch.Tensor] = []
        both_feet_masks: list[torch.Tensor] = []
        stable_supported_masks: list[torch.Tensor] = []
        accepted_masks: list[torch.Tensor] = []

        for _ in range(sample_steps):
            env.step(actions)

            root_frame_height = robot.data.root_pos_w[:, 2]
            base_com_height = robot.data.root_com_pos_w[:, 2]
            posture_rms, posture_max = _posture_error(robot, joint_ids)

            upright = robot.data.projected_gravity_b[:, 2] < args_cli.upright_gravity_z_max
            quiet_xy = (
                torch.linalg.vector_norm(robot.data.root_lin_vel_b[:, :2], dim=1)
                < args_cli.quiet_xy_speed_threshold
            )
            quiet_yaw = (
                torch.abs(robot.data.root_ang_vel_b[:, 2])
                < args_cli.quiet_yaw_rate_threshold
            )
            near_default = posture_max < args_cli.near_default_threshold_rad

            forces = contact_sensor.data.net_forces_w_history[
                :, :, feet_sensor_cfg.body_ids, :
            ]
            foot_contact = (
                forces.norm(dim=-1).amax(dim=1)
                > args_cli.contact_force_threshold_n
            )
            both_feet = torch.all(foot_contact, dim=1)

            # Stable support deliberately excludes posture conformity. This preserves
            # the distinction between "standing successfully" and "near q_default".
            stable_supported = upright & quiet_xy & quiet_yaw & both_feet
            accepted = stable_supported & near_default

            root_frame_samples.append(root_frame_height.detach().clone())
            base_com_samples.append(base_com_height.detach().clone())
            posture_rms_samples.append(posture_rms.detach().clone())
            posture_max_samples.append(posture_max.detach().clone())

            upright_masks.append(upright.detach().clone())
            quiet_xy_masks.append(quiet_xy.detach().clone())
            quiet_yaw_masks.append(quiet_yaw.detach().clone())
            near_default_masks.append(near_default.detach().clone())
            both_feet_masks.append(both_feet.detach().clone())
            stable_supported_masks.append(stable_supported.detach().clone())
            accepted_masks.append(accepted.detach().clone())

        root_frame_stack = torch.stack(root_frame_samples)
        base_com_stack = torch.stack(base_com_samples)
        posture_rms_stack = torch.stack(posture_rms_samples)
        posture_max_stack = torch.stack(posture_max_samples)

        upright_stack = torch.stack(upright_masks)
        quiet_xy_stack = torch.stack(quiet_xy_masks)
        quiet_yaw_stack = torch.stack(quiet_yaw_masks)
        near_default_stack = torch.stack(near_default_masks)
        both_feet_stack = torch.stack(both_feet_masks)
        stable_supported_stack = torch.stack(stable_supported_masks)
        accepted_stack = torch.stack(accepted_masks)

        # Filtered one-dimensional sample arrays. _stats() handles an empty result.
        supported_qdefault_root_height = root_frame_stack[accepted_stack]
        supported_qdefault_com_height = base_com_stack[accepted_stack]
        supported_qdefault_posture_rms = posture_rms_stack[accepted_stack]
        supported_qdefault_posture_max = posture_max_stack[accepted_stack]

        stable_supported_com_height = base_com_stack[stable_supported_stack]
        stable_supported_posture_max = posture_max_stack[stable_supported_stack]

        total_env_samples = int(accepted_stack.numel())
        accepted_env_samples = int(accepted_stack.sum().item())

        thresholds: dict[str, Any] = {
            "upright_gravity_z_max": args_cli.upright_gravity_z_max,
            "quiet_xy_speed_threshold_mps": args_cli.quiet_xy_speed_threshold,
            "quiet_yaw_rate_threshold_rad_s": args_cli.quiet_yaw_rate_threshold,
            "near_default_max_abs_threshold_rad": args_cli.near_default_threshold_rad,
            "contact_force_threshold_n": args_cli.contact_force_threshold_n,
            "feet_body_pattern": args_cli.feet_body_pattern,
        }

        report = {
            "task": args_cli.task,
            "num_envs": num_envs,
            "policy_dt_s": policy_dt,
            "settle_seconds": args_cli.settle_seconds,
            "sample_seconds": args_cli.sample_seconds,
            "settle_steps": settle_steps,
            "sample_steps": sample_steps,
            "action_dim": action_dim,
            "action_joint_names": list(joint_names),
            "measurement_interpretation": {
                "reset_qdefault_geometry": (
                    "Immediate deterministic reset state before dynamic settling."
                ),
                "zero_action_dynamic_equilibrium": (
                    "All environment-time samples while zero normalized action continuously commands q_default."
                ),
                "stable_supported_equilibrium": (
                    "Samples that are upright, quiet in XY and yaw, and supported by both feet; posture conformity is not required."
                ),
                "supported_qdefault_equilibrium": (
                    "Stable-supported samples that also satisfy the max joint-error near-q_default gate."
                ),
            },
            "acceptance_thresholds": thresholds,
            "reset_qdefault_geometry": reset_snapshot,
            "settled_zero_action_snapshot": settled_snapshot,
            "zero_action_dynamic_equilibrium": {
                "root_frame_height_m": _stats(root_frame_stack),
                "base_com_height_m": _stats(base_com_stack),
                "joint_posture_rms_rad": _stats(posture_rms_stack),
                "joint_posture_max_abs_rad": _stats(posture_max_stack),
            },
            "gate_fractions": {
                "upright_fraction": _fraction(upright_stack),
                "quiet_xy_fraction": _fraction(quiet_xy_stack),
                "quiet_yaw_fraction": _fraction(quiet_yaw_stack),
                "near_default_fraction": _fraction(near_default_stack),
                "both_feet_fraction": _fraction(both_feet_stack),
                "stable_supported_fraction": _fraction(stable_supported_stack),
                "supported_qdefault_accepted_fraction": _fraction(accepted_stack),
                "accepted_env_samples": accepted_env_samples,
                "total_env_samples": total_env_samples,
            },
            "stable_supported_equilibrium": {
                "base_com_height_m": _stats(stable_supported_com_height),
                "joint_posture_max_abs_rad": _stats(stable_supported_posture_max),
            },
            "supported_qdefault_equilibrium": {
                "root_frame_height_m": _stats(supported_qdefault_root_height),
                "base_com_height_m": _stats(supported_qdefault_com_height),
                "joint_posture_rms_rad": _stats(supported_qdefault_posture_rms),
                "joint_posture_max_abs_rad": _stats(supported_qdefault_posture_max),
            },
        }

        serialized = json.dumps(report, indent=2)
        print(serialized)

        if args_cli.output:
            output = Path(args_cli.output).expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(serialized + "\n")
            print(f"[INFO] Wrote {output}")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
