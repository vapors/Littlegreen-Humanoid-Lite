"""Export an RSL-RL checkpoint to ONNX/JIT plus a deployment YAML.

Example:
    python scripts/rsl_rl/export_policy_littlegreen.py \
        --task Velocity-Lilgreen-Humanoid-v0 \
        --load_run 2026-07-01_12-45-00_fresh_seed42 \
        --checkpoint model_15000.pt \
        --num_envs 1 \
        --headless

Outputs by default:
    logs/rsl_rl/<experiment>/<run>/exported/policy.onnx
    logs/rsl_rl/<experiment>/<run>/exported/policy.pt
    logs/rsl_rl/<experiment>/<run>/exported/policy.yaml
    configs/policy_latest.yaml
    ~/littlegreen_ros2_ws/src/littlegreen_biped_pkg/src/configs/policy_latest.yaml
    ~/littlegreen_ros2_ws/src/littlegreen_biped_pkg/src/configs/policy.onnx
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Export an RSL-RL policy checkpoint to ONNX/JIT and deployment YAML.")
parser.add_argument("--task", type=str, required=True, help="Name of the Isaac Lab task/env.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs to instantiate for export metadata.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Optional export directory. Default: <checkpoint_run_dir>/exported.",
)
parser.add_argument(
    "--onnx_filename", type=str, default="policy.onnx", help="Filename for exported ONNX policy."
)
parser.add_argument("--jit_filename", type=str, default="policy.pt", help="Filename for exported TorchScript/JIT policy.")
parser.add_argument(
    "--no_jit", action="store_true", default=False, help="Only export ONNX and YAML; skip TorchScript/JIT export."
)
parser.add_argument(
    "--latest_config_path",
    type=str,
    default="configs/policy_latest.yaml",
    help="Convenience copy of deployment YAML. Use empty string to disable.",
)
parser.add_argument(
    "--ros2_deploy_dir",
    type=str,
    default=str(Path.home() / "littlegreen_ros2_ws" / "src" / "littlegreen_biped_pkg" / "src" / "configs"),
    help=(
        "Directory that receives an atomic deployment pair: policy_latest.yaml + policy ONNX. "
        "Use an empty string to disable. Default: "
        "~/littlegreen_ros2_ws/src/littlegreen_biped_pkg/src/configs"
    ),
)

parser.add_argument(
    "--allow_action_contract_v2_ros2_deploy",
    action="store_true",
    default=False,
    help=(
        "Allow an action-contract-v2 policy to be copied into the ROS 2 deployment directory. "
        "By default this is blocked because the legacy deployment node must be updated to perform "
        "the bounded asymmetric action transform."
    ),
)

parser.add_argument(
    "--allow_nonlegacy_action_contract_ros2_deploy",
    action="store_true",
    default=False,
    help=(
        "Allow action contract v2 or newer to be copied into the ROS 2 deployment directory. "
        "Use only when the deployment node explicitly supports the exported action metadata."
    ),
)

# RSL-RL args include --experiment_name, --load_run, --checkpoint, etc.
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Export does not need cameras/rendering.
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


PROJECT_NAME = "Littlegreen-Humanoid-Lite"
PROJECT_VERSION = "2.0.0"
TRAINING_PACKAGE = "littlegreen_humanoid_lite"
ASSET_PACKAGE = "littlegreen_humanoid_lite_assets"


import gymnasium as gym
import torch
from omegaconf import OmegaConf
from rsl_rl.runners import OnPolicyRunner

import isaaclab.utils.string as string_utils
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# Import extensions to register project tasks.
import littlegreen_humanoid_lite.tasks  # noqa: F401,E402
from littlegreen_humanoid_lite.tasks.locomotion.velocity.mdp.st3215_actuator_model import (
    DATASET_NAME as ST3215_DATASET_NAME,
    SOURCE_DATASET_NAME as ST3215_SOURCE_DATASET_NAME,
)


def _safe_to_list(value: Any):
    """Convert common Isaac/Torch/OmegaConf objects to YAML-safe Python values."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    try:
        return OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    return value


def _assign_group_values(target: torch.Tensor, indices, value: Any):
    """Assign actuator stiffness/damping/effort values to selected indices robustly."""
    if not indices:
        return
    if isinstance(value, dict):
        # This script resolves the matching joints before assignment; if the actuator config
        # stores per-regex values as a dict, use the first scalar/list value as a fallback.
        value = next(iter(value.values()))
    if isinstance(value, torch.Tensor):
        value = value.detach().to(device=target.device, dtype=target.dtype)
        if value.numel() == 1:
            target[indices] = value.item()
        else:
            target[indices] = value.flatten()[: len(indices)]
    elif isinstance(value, (list, tuple)):
        if len(value) == 1:
            target[indices] = float(value[0])
        else:
            target[indices] = torch.tensor(value[: len(indices)], device=target.device, dtype=target.dtype)
    else:
        target[indices] = float(value)


def _get_observation_dim(env) -> int:
    """Return the policy observation dimension across common Isaac/RSL-RL wrappers."""
    obs_space = getattr(env, "observation_space", None)
    if isinstance(obs_space, dict) and "policy" in obs_space:
        return int(obs_space["policy"].shape[-1])
    if hasattr(obs_space, "shape") and obs_space.shape is not None:
        return int(obs_space.shape[-1])

    obs_out = env.reset()
    obs = obs_out[0] if isinstance(obs_out, tuple) else obs_out
    if isinstance(obs, dict) and "policy" in obs:
        return int(obs["policy"].shape[-1])
    return int(obs.shape[-1])


def _get_history_length(env_cfg) -> int:
    try:
        return int(env_cfg.observations.policy.actions.history_length)
    except Exception:
        return 1


def _get_command_velocity(env, env_cfg):
    """Try to sample the same default command vector used by the current play.py deployment YAML."""
    try:
        term = env_cfg.observations.policy.velocity_commands
        cmd_name = term.params["command_name"]
        cmd = term.func(env.unwrapped, cmd_name)[0]
        return _safe_to_list(cmd)
    except Exception:
        return [0.0, 0.0, 0.0]


def _sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_copy(source: str | Path, destination: str | Path) -> None:
    """Copy a file into place using a same-directory temporary file + atomic replace."""
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_save_yaml(config: dict, destination: str | Path) -> None:
    """Write YAML into place using a same-directory temporary file + atomic replace."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        OmegaConf.save(config=config, f=str(temporary))
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _build_deploy_config(
    env,
    env_cfg,
    checkpoint_path: str,
    export_dir: str,
    onnx_filename: str,
    onnx_sha256: str,
):
    joint_pos_cfg = env_cfg.scene.robot.init_state.joint_pos
    joint_names = list(joint_pos_cfg.keys())
    init_joint_pos = [_safe_to_list(v) for v in joint_pos_cfg.values()]
    num_joints = len(joint_names)

    device = env.unwrapped.device
    joint_kp = torch.zeros(num_joints, device=device)
    joint_kd = torch.zeros(num_joints, device=device)
    effort_limits = torch.zeros(num_joints, device=device)

    for group in env_cfg.scene.robot.actuators.values():
        match_expr_dict = {expr: None for expr in group.joint_names_expr}
        indices, _, _ = string_utils.resolve_matching_names_values(
            match_expr_dict, joint_names, preserve_order=True
        )
        _assign_group_values(joint_kp, indices, group.stiffness)
        _assign_group_values(joint_kd, indices, group.damping)
        _assign_group_values(effort_limits, indices, group.effort_limit)

    action_expr_dict = {expr: None for expr in env_cfg.actions.joint_pos.joint_names}
    action_indices, _, _ = string_utils.resolve_matching_names_values(
        action_expr_dict, joint_names, preserve_order=True
    )

    physics_dt = float(env_cfg.sim.dt)
    decimation = int(env_cfg.decimation)
    policy_dt = float(physics_dt * decimation)
    onnx_path = os.path.abspath(os.path.join(export_dir, onnx_filename))

    # policy.yaml is saved inside export_dir, so this is portable when
    # policy.yaml and policy.onnx are copied together.
    onnx_relative_path = onnx_filename

    action_cfg = env_cfg.actions.joint_pos
    action_defaults = [float(init_joint_pos[index]) for index in action_indices]

    if hasattr(action_cfg, "residual_scale_rad"):
        scale_value = _safe_to_list(action_cfg.residual_scale_rad)
        if isinstance(scale_value, (int, float)):
            residual_scale = [float(scale_value)] * len(action_indices)
        else:
            residual_scale = [float(value) for value in scale_value]
        if len(residual_scale) != len(action_indices):
            raise ValueError(
                "Residual action scale metadata length does not match action dimension: "
                f"scale={len(residual_scale)}, actions={len(action_indices)}"
            )
        nonuniform_residual_scale = (max(residual_scale) - min(residual_scale)) > 1.0e-9
        action_contract_version = 4 if nonuniform_residual_scale else 3
        action_contract_name = (
            "bounded_default_centered_vector_residual_athletic"
            if action_contract_version == 4
            else "bounded_default_centered_symmetric_residual"
        )
        action_transform = (
            "bounded_default_centered_vector_residual"
            if action_contract_version == 4
            else "bounded_default_centered_symmetric_residual"
        )

        physical_lower = [float(value) for value in _safe_to_list(action_cfg.lower_limits)]
        physical_upper = [float(value) for value in _safe_to_list(action_cfg.upper_limits)]
        nominal_lower = [
            max(lower, default - scale)
            for default, lower, scale in zip(action_defaults, physical_lower, residual_scale)
        ]
        nominal_upper = [
            min(upper, default + scale)
            for default, upper, scale in zip(action_defaults, physical_upper, residual_scale)
        ]

        training_stage_for_profile = str(getattr(action_cfg, "actuator_model_stage", ""))
        if action_contract_version == 4:
            if "stabilized" in training_stage_for_profile:
                deployment_contract_profile = "v1_4_5_stabilized_vector_residual"
            else:
                deployment_contract_profile = "v1_4_5_athletic_vector_residual"
        else:
            deployment_contract_profile = "v1_2_3_scalar_residual"

        action_metadata = {
            "action_contract_version": action_contract_version,
            "action_contract_name": action_contract_name,
            "action_transform": action_transform,
            "action_equation": "q_target = clip(q_default + clip(a_raw,-1,1) * residual_scale_rad, physical_lower, physical_upper)",
            "action_limit_lower": -1.0,
            "action_limit_upper": 1.0,
            "action_residual_scale_rad": residual_scale,
            "action_target_lower_rad": physical_lower,
            "action_target_upper_rad": physical_upper,
            "action_nominal_residual_lower_rad": nominal_lower,
            "action_nominal_residual_upper_rad": nominal_upper,
            "action_default_rad": action_defaults,
            "previous_action_observation": "bounded_normalized_action",
            "deployment_requires_action_contract_transform": True,
            "deployment_requires_action_contract_v3_transform": action_contract_version == 3,
            "deployment_requires_action_contract_v4_transform": action_contract_version == 4,
            "deployment_contract_profile": deployment_contract_profile,
        }
        if hasattr(action_cfg, "actuator_model_name"):
            action_metadata.update({
                "training_actuator_model_name": str(action_cfg.actuator_model_name),
                "training_actuator_model_source_name": str(action_cfg.actuator_model_name).replace(
                    ST3215_DATASET_NAME, ST3215_SOURCE_DATASET_NAME
                ),
                "training_actuator_model_stage": str(
                    getattr(action_cfg, "actuator_model_stage", "stage_a")
                ),
                "training_actuator_bus_phase_delay_s_range": _safe_to_list(action_cfg.bus_phase_delay_s_range),
                "training_actuator_response_delay_s_range": _safe_to_list(action_cfg.response_delay_s_range),
                "training_actuator_response_delay_scale": float(action_cfg.response_delay_scale),
                "training_actuator_tau_median_s": _safe_to_list(action_cfg.tau_median_s),
                "training_actuator_tau_p10_s": _safe_to_list(action_cfg.tau_p10_s),
                "training_actuator_tau_p90_s": _safe_to_list(action_cfg.tau_p90_s),
                "training_actuator_velocity_scale_range": _safe_to_list(action_cfg.velocity_scale_range),
                "training_actuator_special_transients_enabled": False,
                "deployment_actuator_model_is_training_only": True,
            })

            if bool(getattr(action_cfg, "loaded_envelope_enabled", False)):
                action_metadata.update({
                    "training_loaded_actuator_dataset_name": str(action_cfg.loaded_dataset_name),
                    "training_loaded_envelope_enabled": True,
                    "training_loaded_envelope_combination_rule": str(
                        action_cfg.loaded_envelope_combination_rule
                    ),
                    "training_loaded_crouch_direction_sign": _safe_to_list(
                        action_cfg.loaded_crouch_direction_sign
                    ),
                    "training_loaded_crouch_low_demand_gain": _safe_to_list(
                        action_cfg.loaded_crouch_low_demand_gain
                    ),
                    "training_loaded_crouch_vmax_rad_s": _safe_to_list(
                        action_cfg.loaded_crouch_vmax_rad_s
                    ),
                    "training_loaded_return_low_demand_gain": _safe_to_list(
                        action_cfg.loaded_return_low_demand_gain
                    ),
                    "training_loaded_return_vmax_rad_s": _safe_to_list(
                        action_cfg.loaded_return_vmax_rad_s
                    ),
                    "training_loaded_direction_conditioning_weight": _safe_to_list(
                        action_cfg.loaded_direction_conditioning_weight
                    ),
                    "training_loaded_return_tau_scale": _safe_to_list(
                        action_cfg.loaded_return_tau_scale
                    ),
                    "training_loaded_velocity_scale_range": _safe_to_list(
                        action_cfg.loaded_velocity_scale_range
                    ),
                    "training_loaded_envelope_interpretation": (
                        "empirical loaded whole-body trajectory envelope; not an intrinsic "
                        "motor speed, torque-speed, stiffness, or damping model"
                    ),
                })

    elif hasattr(action_cfg, "lower_limits") and hasattr(action_cfg, "upper_limits"):
        action_metadata = {
            "action_contract_version": 2,
            "action_contract_name": "bounded_default_centered_asymmetric_full_range",
            "action_transform": "bounded_default_centered_asymmetric",
            "action_limit_lower": -1.0,
            "action_limit_upper": 1.0,
            "action_scale": 1.0,
            "action_target_lower_rad": _safe_to_list(action_cfg.lower_limits),
            "action_target_upper_rad": _safe_to_list(action_cfg.upper_limits),
            "action_default_rad": action_defaults,
            "previous_action_observation": "bounded_normalized_action",
            "deployment_requires_action_contract_transform": True,
            "deployment_requires_action_contract_v2_transform": True,
        }
    else:
        action_metadata = {
            "action_contract_version": 1,
            "action_transform": "legacy_scaled_default_offset",
            "action_scale": _safe_to_list(action_cfg.scale),
            "action_limit_lower": -10000,
            "action_limit_upper": 10000,
            "previous_action_observation": "raw_policy_action",
            "deployment_requires_action_contract_transform": False,
            "deployment_requires_action_contract_v2_transform": False,
        }

    phase_metadata = {}
    if args_cli.task == "Velocity-Lilgreen-Hardware-ST3215-Loaded-v10":
        phase_metadata = {
            "observation_contract_name": "47d_command_synchronized_phase_v10",
            "observation_layout": (
                "cmd3,base_ang_vel3,projected_gravity3,joint_pos_error12,"
                "joint_vel12,previous_bounded_action12,phase_sin1,phase_cos1"
            ),
            "gait_phase_semantics": "command_synchronized_alternating_first_swing",
            "gait_phase_standing_behavior": "frozen_double_support_phase_zero",
            "gait_phase_movement_onset_behavior": "reset_to_alternating_swing_boundary",
            "gait_phase_first_swing_balancing": "environment_parity_then_alternate_per_onset",
            "gait_phase_period_s_initial": 0.90,
            "gait_phase_period_s_curriculum": [0.90, 0.86, 0.82, 0.78],
            "gait_phase_transition_fraction_per_boundary": [0.08, 0.07, 0.06, 0.05],
            "deployment_requires_command_synchronized_phase_v10": True,
        }

    return {
        "metadata": {
            "project_name": PROJECT_NAME,
            "project_version": PROJECT_VERSION,
            "training_package": TRAINING_PACKAGE,
            "asset_package": ASSET_PACKAGE,
            "source_project_name": "Berkeley-Humanoid-Lite",
            "source_project_version": "1.4.8",
            "exported_at": datetime.now().isoformat(timespec="seconds"),
            "task": args_cli.task,
            "checkpoint_path": os.path.abspath(checkpoint_path),
            "export_dir": os.path.abspath(export_dir),
            "policy_filename": onnx_filename,
            "policy_format": "onnx",
            "policy_sha256": onnx_sha256,
        },

        # Policy configuration
        # Legacy/host-machine path:
        "policy_checkpoint_path": onnx_path,

        # Preferred portable deployment path:
        # Resolve relative to the directory containing this YAML file.
        "policy_checkpoint_relative_path": onnx_relative_path,
        "policy_checkpoint_filename": onnx_filename,
        "policy_sha256": onnx_sha256,

        # Networking configuration used by current ROS2 deployment bridge
        "ip_robot_addr": "127.0.0.1",
        "ip_policy_obs_port": 10000,
        "ip_host_addr": "127.0.0.1",
        "ip_policy_acs_port": 10001,

        # Timing / physics configuration extracted from the resolved Isaac environment.
        # control_dt is retained as a compatibility alias for the policy/action update period.
        "control_dt": policy_dt,
        "policy_dt": policy_dt,
        "physics_dt": physics_dt,
        "decimation": decimation,

        # Articulation configuration
        "num_joints": num_joints,
        "joints": joint_names,
        "joint_kp": joint_kp.detach().cpu().tolist(),
        "joint_kd": joint_kd.detach().cpu().tolist(),
        "effort_limits": effort_limits.detach().cpu().tolist(),
        "default_base_position": _safe_to_list(env_cfg.scene.robot.init_state.pos),
        "default_joint_positions": init_joint_pos,

        # Observation configuration
        "num_observations": _get_observation_dim(env),
        "history_length": _get_history_length(env_cfg),
        **phase_metadata,

        # Command configuration
        "command_velocity": _get_command_velocity(env, env_cfg),

        # Action configuration
        "num_actions": int(env.action_space.shape[-1]),
        "action_indices": list(action_indices),
        **action_metadata,
    }


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    checkpoint_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    run_dir = os.path.dirname(checkpoint_path)
    export_dir = args_cli.output_dir or os.path.join(run_dir, "exported")
    os.makedirs(export_dir, exist_ok=True)

    print(f"[INFO] Experiment root: {log_root_path}")
    print(f"[INFO] Checkpoint:      {checkpoint_path}")
    print(f"[INFO] Export dir:      {os.path.abspath(export_dir)}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path)

    print(f"[EXPORT] ONNX: {os.path.join(export_dir, args_cli.onnx_filename)}")
    export_policy_as_onnx(
        runner.alg.policy,
        normalizer=runner.obs_normalizer,
        path=export_dir,
        filename=args_cli.onnx_filename,
    )

    if not args_cli.no_jit:
        print(f"[EXPORT] JIT:  {os.path.join(export_dir, args_cli.jit_filename)}")
        export_policy_as_jit(
            runner.alg.policy,
            runner.obs_normalizer,
            path=export_dir,
            filename=args_cli.jit_filename,
        )

    onnx_export_path = Path(export_dir) / args_cli.onnx_filename
    onnx_sha256 = _sha256_file(onnx_export_path)
    print(f"[EXPORT] ONNX SHA-256: {onnx_sha256}")

    deploy_config = _build_deploy_config(
        env,
        env_cfg,
        checkpoint_path,
        export_dir,
        args_cli.onnx_filename,
        onnx_sha256,
    )

    run_yaml_path = Path(export_dir) / "policy.yaml"
    _atomic_save_yaml(deploy_config, run_yaml_path)
    print(f"[EXPORT] YAML: {run_yaml_path}")

    if args_cli.latest_config_path:
        latest_path = Path(args_cli.latest_config_path).expanduser()
        _atomic_save_yaml(deploy_config, latest_path)
        print(f"[EXPORT] Latest YAML: {latest_path}")

    contract_version = int(deploy_config.get("action_contract_version", 1))
    is_nonlegacy_contract = contract_version >= 2
    allow_ros2_deploy = (
        not is_nonlegacy_contract
        or args_cli.allow_nonlegacy_action_contract_ros2_deploy
        or (contract_version == 2 and args_cli.allow_action_contract_v2_ros2_deploy)
    )
    if args_cli.ros2_deploy_dir and is_nonlegacy_contract and not allow_ros2_deploy:
        transform_name = deploy_config.get("action_transform", "unknown")
        print(
            f"[DEPLOY][SAFETY] ROS 2 copy skipped: this policy uses action contract "
            f"v{contract_version} ({transform_name}). The deployment node must explicitly "
            "implement the exported transform before this policy/model pair is published. "
            "Re-run with --allow_nonlegacy_action_contract_ros2_deploy only after the "
            "deployment path is compatible."
        )

    if args_cli.ros2_deploy_dir and allow_ros2_deploy:
        deploy_dir = Path(args_cli.ros2_deploy_dir).expanduser()
        deploy_dir.mkdir(parents=True, exist_ok=True)

        # Copy model first, then publish the YAML pointer. The policy node resolves the
        # relative model beside this YAML and no longer silently falls back to an old model.
        deployed_onnx = deploy_dir / args_cli.onnx_filename
        deployed_yaml = deploy_dir / "policy_latest.yaml"
        _atomic_copy(onnx_export_path, deployed_onnx)
        _atomic_save_yaml(deploy_config, deployed_yaml)

        deployed_sha256 = _sha256_file(deployed_onnx)
        if deployed_sha256 != onnx_sha256:
            raise RuntimeError(
                f"Deployment ONNX checksum mismatch: expected {onnx_sha256}, got {deployed_sha256}"
            )

        print(f"[DEPLOY] ONNX: {deployed_onnx}")
        print(f"[DEPLOY] YAML: {deployed_yaml}")
        print(f"[DEPLOY] SHA-256 verified: {deployed_sha256}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
