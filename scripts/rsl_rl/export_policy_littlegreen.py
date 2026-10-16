"""Export supported LittleGreen checkpoints for deployment.

The maintained exporter supports the canonical v2.5 foundation policies plus the
current v11 baseline locomotion policy (v10.3.1 frozen recipe):

* Velocity-Lilgreen-Stand-ST3215-Loaded-v24
* Velocity-Lilgreen-Transfer-ST3215-Loaded-v24
* Velocity-Lilgreen-Walk-ST3215-Loaded-v25
* Velocity-Lilgreen-Locomotion-ST3215-Loaded-v102

All supported policies use the shared 47-D actor observation contract and 12-D
action contract v4.  v11 exports the training-exact ``neutral_static`` phase
tail (sin=0, cos=1) and its joystick command envelope so Track 2 can reconstruct
the observation without inventing a gait clock or accepting out-of-distribution
commands.  The exporter writes an atomic ONNX/YAML bundle and validates the
actual ONNX tensor interface before publishing convenience or ROS 2 copies.

Example — Stand::

    python scripts/rsl_rl/export_policy.py \
      --task Velocity-Lilgreen-Stand-ST3215-Loaded-v24 \
      --load_run <STAND_RUN_DIRECTORY> \
      --checkpoint model_2500.pt \
      --num_envs 1 \
      --headless

Example — Walk::

    python scripts/rsl_rl/export_policy.py \
      --task Velocity-Lilgreen-Walk-ST3215-Loaded-v25 \
      --load_run <WALK_RUN_DIRECTORY> \
      --checkpoint model_5000.pt \
      --num_envs 1 \
      --headless

Default run outputs::

    <run>/exported/policy.onnx
    <run>/exported/policy.pt
    <run>/exported/policy.yaml
    <run>/exported/deployment_contract.yaml
    <run>/exported/policy.sha256
    <run>/exported/bundle_manifest.yaml

The default project-level convenience copy is an atomic pair, not a YAML-only
copy::

    configs/policy.onnx
    configs/policy_latest.yaml
    configs/policy.sha256
    configs/policy_bundle_manifest.yaml

ROS 2 publication is disabled unless ``--deploy_ros2`` is supplied explicitly.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


STAND_TASK = "Velocity-Lilgreen-Stand-ST3215-Loaded-v24"
TRANSFER_TASK = "Velocity-Lilgreen-Transfer-ST3215-Loaded-v24"
WALK_TASK = "Velocity-Lilgreen-Walk-ST3215-Loaded-v25"
LOCOMOTION_TASK = "Velocity-Lilgreen-Locomotion-ST3215-Loaded-v110"
SUPPORTED_TASKS = (LOCOMOTION_TASK,)

PROJECT_NAME = "Littlegreen-Humanoid-Lite"
PROJECT_VERSION = "11.0.0"
EXPORT_SCHEMA_VERSION = 2
TRAINING_PACKAGE = "littlegreen_humanoid_lite"
ASSET_PACKAGE = "littlegreen_humanoid_lite_assets"
EXPECTED_POLICY_OBSERVATIONS = 47
EXPECTED_CRITIC_OBSERVATIONS = 50
EXPECTED_ACTIONS = 12
EXPECTED_PHYSICS_DT = 0.005
EXPECTED_POLICY_DT = 0.02


parser = argparse.ArgumentParser(
    description="Export a supported LittleGreen checkpoint to ONNX/JIT and deployment metadata."
)
parser.add_argument("--task", type=str, required=True, choices=SUPPORTED_TASKS, help="Supported LittleGreen deployment task ID.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments instantiated during export.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable Fabric and use USD I/O operations.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Export directory. Default: <checkpoint_run_dir>/exported.",
)
parser.add_argument("--onnx_filename", type=str, default="policy.onnx", help="Exported ONNX filename.")
parser.add_argument("--jit_filename", type=str, default="policy.pt", help="Exported TorchScript filename.")
parser.add_argument("--no_jit", action="store_true", default=False, help="Skip TorchScript export.")
parser.add_argument(
    "--deployment_stage",
    type=int,
    choices=(0, 1, 2, 3),
    default=None,
    help=(
        "Transfer/Walk curriculum stage whose phase period should be written into deployment metadata. "
        "Stand and v11 neutral-static locomotion ignore this option. Use the stage associated with the selected checkpoint."
    ),
)
parser.add_argument(
    "--latest_config_path",
    type=str,
    default="configs/policy_latest.yaml",
    help=(
        "Project-level convenience YAML. The matching ONNX and checksum are copied beside it. "
        "Use an empty string to disable."
    ),
)
parser.add_argument(
    "--deploy_ros2",
    action="store_true",
    default=False,
    help=(
        "Explicitly publish the validated policy bundle into --ros2_deploy_dir. "
        "Use only after the ROS 2 runtime supports the exported observation/phase contract."
    ),
)
parser.add_argument(
    "--ros2_deploy_dir",
    type=str,
    default=str(
        Path.home()
        / "littlegreen_ros2_ws"
        / "src"
        / "littlegreen_biped_pkg"
        / "src"
        / "configs"
    ),
    help="ROS 2 destination directory used only with --deploy_ros2.",
)

# RSL-RL arguments include --load_run, --checkpoint, --experiment_name, etc.
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if not args_cli.load_run:
    parser.error("--load_run is required for policy export")
if not args_cli.checkpoint:
    parser.error("--checkpoint is required for policy export")
if args_cli.num_envs < 1:
    parser.error("--num_envs must be at least 1")

# Export does not need cameras/rendering.
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


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
)
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# Import extensions to register project tasks.
import littlegreen_humanoid_lite.tasks  # noqa: F401,E402
from littlegreen_humanoid_lite.tasks.locomotion.velocity.config.littlegreen_humanoid.observations_cfg import (
    OBSERVATION_LAYOUT_V24,
    POLICY_OBSERVATION_COUNT_V24,
)
from littlegreen_humanoid_lite.tasks.locomotion.velocity.mdp.hardware_contract import (
    ACTIONABLE_JOINTS_V1_2_3,
)
from littlegreen_humanoid_lite.tasks.locomotion.velocity.mdp.st3215_actuator_model import (
    DATASET_NAME as ST3215_DATASET_NAME,
    SOURCE_DATASET_NAME as ST3215_SOURCE_DATASET_NAME,
)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_to_list(value: Any):
    """Convert common Isaac/Torch/OmegaConf values to YAML-safe Python values."""
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
        return value


def _float_list(value: Any, expected_length: int, field_name: str) -> list[float]:
    converted = _safe_to_list(value)
    if isinstance(converted, (int, float)):
        result = [float(converted)] * expected_length
    else:
        result = [float(item) for item in converted]
    if len(result) != expected_length:
        raise ValueError(f"{field_name} has {len(result)} values; expected {expected_length}")
    return result


def _assign_group_values(target: torch.Tensor, indices: list[int], value: Any) -> None:
    """Assign actuator stiffness/damping/effort values to selected joint indices."""
    if not indices:
        return
    if isinstance(value, dict):
        value = next(iter(value.values()))
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device=target.device, dtype=target.dtype).flatten()
        if tensor.numel() == 1:
            target[indices] = tensor.item()
        else:
            if tensor.numel() < len(indices):
                raise ValueError("Actuator tensor is shorter than the resolved joint group")
            target[indices] = tensor[: len(indices)]
    elif isinstance(value, (list, tuple)):
        if len(value) == 1:
            target[indices] = float(value[0])
        else:
            if len(value) < len(indices):
                raise ValueError("Actuator value list is shorter than the resolved joint group")
            target[indices] = torch.tensor(value[: len(indices)], device=target.device, dtype=target.dtype)
    else:
        target[indices] = float(value)


def _get_policy_observation_dim(env) -> int:
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


def _get_group_observation_dim(raw_env, group_name: str, fallback: int) -> int:
    """Read an Isaac observation-group dimension without changing the policy tensor."""
    manager = getattr(raw_env.unwrapped, "observation_manager", None)
    group_dims = getattr(manager, "group_obs_dim", None)
    if isinstance(group_dims, dict) and group_name in group_dims:
        value = group_dims[group_name]
        if isinstance(value, int):
            return int(value)
        if isinstance(value, (tuple, list)):
            result = 1
            for item in value:
                result *= int(item)
            return result
    return int(fallback)


def _history_length() -> int:
    """No temporal observation stacking is used; previous action is one explicit term."""
    return 0


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_copy(source: str | Path, destination: str | Path) -> None:
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
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        OmegaConf.save(config=config, f=str(temporary))
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_text(text: str, destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _normalize_onnx_shape(shape: list[Any]) -> list[int | str | None]:
    """Convert ONNX Runtime shape values to YAML-safe dimensions."""
    normalized: list[int | str | None] = []
    for dim in shape:
        if isinstance(dim, int):
            normalized.append(int(dim))
        elif isinstance(dim, str):
            normalized.append(dim)
        else:
            normalized.append(None)
    return normalized


class _FeedForwardPolicyExportWrapper(torch.nn.Module):
    """CPU-only actor + observation-normalizer wrapper for deterministic export."""

    def __init__(self, policy: object, normalizer: object | None):
        super().__init__()
        if not hasattr(policy, "actor"):
            raise ValueError("The active LittleGreen v2.5.0 policy must expose an actor module.")
        if bool(getattr(policy, "is_recurrent", False)):
            raise ValueError("The v2.4.3 exporter currently supports the feed-forward actor only.")

        self.actor = copy.deepcopy(policy.actor).to(device="cpu").eval()
        if normalizer is None:
            self.normalizer = torch.nn.Identity()
        else:
            self.normalizer = copy.deepcopy(normalizer).to(device="cpu").eval()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor(self.normalizer(observations))


def _export_policy_as_onnx_legacy(
    policy: object,
    normalizer: object | None,
    destination: Path,
    observation_dim: int,
) -> None:
    """Export with PyTorch's legacy TorchScript ONNX path and atomically publish it.

    PyTorch 2.7 changed ``torch.onnx.export`` to prefer the torch.export/Dynamo
    exporter.  In the Isaac Sim 5.0 training environment that path can finish
    writing the ONNX file but fail to return control to the Python script.  The
    legacy exporter is fully sufficient for this small feed-forward MLP and has
    been the stable deployment path for LittleGreen.
    """

    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp{destination.suffix}")
    if temporary.exists():
        temporary.unlink()

    wrapper = _FeedForwardPolicyExportWrapper(policy, normalizer)
    dummy_observation = torch.zeros(1, observation_dim, dtype=torch.float32, device="cpu")

    print(
        "[EXPORT] ONNX backend: PyTorch legacy exporter (dynamo=False, opset=18)",
        flush=True,
    )
    try:
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                dummy_observation,
                str(temporary),
                export_params=True,
                opset_version=18,
                do_constant_folding=True,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={},
                dynamo=False,
            )
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise RuntimeError(f"ONNX exporter did not create a valid file: {temporary}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(
        f"[EXPORT] ONNX write complete: {destination} ({destination.stat().st_size} bytes)",
        flush=True,
    )


def _inspect_onnx_contract(onnx_path: Path) -> dict[str, Any]:
    """Inspect and enforce the exported ONNX 47-to-12 tensor contract."""
    try:
        import onnxruntime as ort
    except ImportError:
        ort = None

    if ort is not None:
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(
                "LittleGreen policy ONNX must expose exactly one runtime input and one output; "
                f"got inputs={len(inputs)}, outputs={len(outputs)}"
            )

        input_shape = _normalize_onnx_shape(list(inputs[0].shape))
        output_shape = _normalize_onnx_shape(list(outputs[0].shape))
        input_name = inputs[0].name
        output_name = outputs[0].name
        input_type = inputs[0].type
        output_type = outputs[0].type
        validator = "onnxruntime"
        validator_version = str(ort.__version__)
    else:
        try:
            import onnx
        except ImportError as exc:
            raise RuntimeError(
                "Install either 'onnxruntime' or 'onnx' to validate the exported tensor contract."
            ) from exc

        model = onnx.load(str(onnx_path), load_external_data=True)
        onnx.checker.check_model(model)
        graph_inputs = [value for value in model.graph.input]
        graph_outputs = [value for value in model.graph.output]
        if len(graph_inputs) != 1 or len(graph_outputs) != 1:
            raise RuntimeError(
                "LittleGreen policy ONNX must expose exactly one runtime input and one output; "
                f"got inputs={len(graph_inputs)}, outputs={len(graph_outputs)}"
            )

        def _value_shape(value_info) -> list[int | str | None]:
            dims: list[int | str | None] = []
            for dim in value_info.type.tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    dims.append(int(dim.dim_value))
                elif dim.HasField("dim_param"):
                    dims.append(str(dim.dim_param))
                else:
                    dims.append(None)
            return dims

        input_shape = _value_shape(graph_inputs[0])
        output_shape = _value_shape(graph_outputs[0])
        input_name = graph_inputs[0].name
        output_name = graph_outputs[0].name
        input_type = "tensor(float)"
        output_type = "tensor(float)"
        validator = "onnx"
        validator_version = str(onnx.__version__)

    if not input_shape or input_shape[-1] != EXPECTED_POLICY_OBSERVATIONS:
        raise RuntimeError(f"ONNX input shape must end in 47, got {input_shape}")
    if not output_shape or output_shape[-1] != EXPECTED_ACTIONS:
        raise RuntimeError(f"ONNX output shape must end in 12, got {output_shape}")

    return {
        "onnx_input_name": input_name,
        "onnx_input_shape": input_shape,
        "onnx_input_type": input_type,
        "onnx_output_name": output_name,
        "onnx_output_shape": output_shape,
        "onnx_output_type": output_type,
        "onnx_validator": validator,
        "onnx_validator_version": validator_version,
    }


def _resolved_phase_metadata(env_cfg, task: str) -> dict[str, Any]:
    phase_term = env_cfg.observations.policy.gait_phase
    params = dict(phase_term.params)
    configured_mode = str(params.get("phase_mode", ""))
    expected_modes = {
        STAND_TASK: "random_static",
        TRANSFER_TASK: "continuous_transfer",
        WALK_TASK: "command_synchronized",
        LOCOMOTION_TASK: "neutral_static",
    }
    expected_mode = expected_modes[task]
    if configured_mode != expected_mode:
        raise RuntimeError(
            f"Resolved phase mode for {task} is {configured_mode!r}; expected {expected_mode!r}"
        )

    if task == STAND_TASK:
        role = "stand"
    elif task == TRANSFER_TASK:
        role = "transfer"
    elif task == WALK_TASK:
        role = "walk"
    else:
        role = "locomotion"

    exported_modes = {
        "stand": "randomized_static_per_episode",
        "transfer": "continuous_alternating_transfer_clock",
        "walk": "command_synchronized_continuous_nonblocking",
        "locomotion": "neutral_static",
    }
    reset_semantics = {
        "stand": "sample_uniform_once_for_each_reset_environment_and_hold",
        "transfer": "balanced_left_right_initial_offset_then_continuous_advance",
        "walk": "zero_while_standing_alternating_onset_then_continuous_nonblocking_advance",
        "locomotion": "constant_neutral_phase_sin0_cos1_for_all_environments_and_ticks",
    }
    stage_periods = {
        "transfer": (1.20, 1.10, 1.00, 0.92),
        "walk": (0.96, 0.92, 0.88, 0.84),
    }
    deployment_stage = None
    phase_period_source = "resolved_task_default"
    phase_period = float(params.get("period_s", 0.90))
    if role in stage_periods and args_cli.deployment_stage is not None:
        deployment_stage = int(args_cli.deployment_stage)
        phase_period = float(stage_periods[role][deployment_stage])
        phase_period_source = "explicit_curriculum_stage"
    if role == "locomotion":
        phase_period_source = "inactive_neutral_static_metadata_only"

    return {
        "observation_contract_version": 1,
        "observation_contract_name": "littlegreen_velocity_47d_phase_v1",
        "observation_layout": OBSERVATION_LAYOUT_V24,
        "observation_layout_ranges": {
            "command_velocity": [0, 2],
            "base_angular_velocity": [3, 5],
            "projected_gravity": [6, 8],
            "joint_position_relative_default": [9, 20],
            "joint_velocity": [21, 32],
            "previous_bounded_action": [33, 44],
            "phase_sin_cos": [45, 46],
        },
        "observation_count": EXPECTED_POLICY_OBSERVATIONS,
        "critic_observation_count": EXPECTED_CRITIC_OBSERVATIONS,
        "phase_indices": [45, 46],
        "phase_encoding": "sin_cos_2pi",
        "phase_mode": exported_modes[role],
        "phase_source_function": "littlegreen_phase_sin_cos_v24",
        "phase_period_s": phase_period,
        "phase_period_active": role != "locomotion",
        "phase_constant_sin_cos": [0.0, 1.0] if role == "locomotion" else None,
        "deployment_curriculum_stage": deployment_stage,
        "phase_period_source": phase_period_source,
        "phase_transition_fraction": float(params.get("transition_fraction", 0.10)),
        "phase_linear_command_threshold": float(params.get("linear_command_threshold", 0.08)),
        "phase_yaw_command_threshold": float(params.get("yaw_command_threshold", 0.08)),
        "phase_reset_semantics": reset_semantics[role],
        # Keep the historical key for Track-2 backward compatibility while naming
        # the broader contract accurately for modern locomotion exports.
        "shared_47d_stand_walk_contract": True,
        "shared_47d_foundation_contract": True,
        "stand_transfer_walk_observation_difference": "phase_evolution_only",
        "legacy_45d_checkpoint_support": False,
        "deployment_requires_shared_47d_observation_builder": True,
        "deployment_requires_random_static_phase_for_stand": role == "stand",
        "deployment_requires_continuous_transfer_phase": role == "transfer",
        "deployment_requires_command_synchronized_phase_for_walk": role == "walk",
        "deployment_requires_neutral_static_phase_for_locomotion": role == "locomotion",
    }


def _resolved_command_metadata(env_cfg, task: str) -> dict[str, Any]:
    command_cfg = env_cfg.commands.base_velocity
    ranges = command_cfg.ranges

    def _pair(value: Any, name: str) -> list[float]:
        result = _float_list(value, 2, name)
        if not result[0] <= result[1]:
            raise RuntimeError(f"{name} lower bound exceeds upper bound: {result}")
        return result

    metadata: dict[str, Any] = {
        "command_contract_version": 1,
        "command_velocity_source": "runtime_command_input",
        "command_limit_lin_vel_x": _pair(ranges.lin_vel_x, "lin_vel_x"),
        "command_limit_lin_vel_y": _pair(ranges.lin_vel_y, "lin_vel_y"),
        "command_limit_ang_vel_z": _pair(ranges.ang_vel_z, "ang_vel_z"),
        "deployment_requires_command_clamp": task == LOCOMOTION_TASK,
        "command_limit_behavior": "clamp_each_axis_to_training_range",
    }
    if task == LOCOMOTION_TASK:
        metadata.update(
            {
                "command_training_profile": "v10_3_1_forward_bilateral_acquisition",
                "command_resampling_time_range_s": _float_list(
                    command_cfg.resampling_time_range, 2, "resampling_time_range"
                ),
                "command_rel_standing_envs": float(command_cfg.rel_standing_envs),
            }
        )
    return metadata


def _build_deploy_config(
    env,
    raw_env,
    env_cfg,
    agent_cfg: RslRlOnPolicyRunnerCfg,
    checkpoint_path: str,
    export_dir: Path,
    onnx_filename: str,
    onnx_sha256: str,
    onnx_contract: dict[str, Any],
) -> dict[str, Any]:
    """Build the v2.4.3 deployment YAML from the fully resolved environment."""
    if POLICY_OBSERVATION_COUNT_V24 != EXPECTED_POLICY_OBSERVATIONS:
        raise RuntimeError(
            f"Source constant POLICY_OBSERVATION_COUNT_V24={POLICY_OBSERVATION_COUNT_V24}; expected 47"
        )

    joint_pos_cfg = env_cfg.scene.robot.init_state.joint_pos
    sim_joint_names = list(joint_pos_cfg.keys())
    sim_default_joint_positions = [float(value) for value in joint_pos_cfg.values()]
    sim_num_joints = len(sim_joint_names)

    device = env.unwrapped.device
    joint_kp = torch.zeros(sim_num_joints, device=device)
    joint_kd = torch.zeros(sim_num_joints, device=device)
    effort_limits = torch.zeros(sim_num_joints, device=device)

    for group in env_cfg.scene.robot.actuators.values():
        match_expr_dict = {expr: None for expr in group.joint_names_expr}
        indices, _, _ = string_utils.resolve_matching_names_values(
            match_expr_dict, sim_joint_names, preserve_order=True
        )
        _assign_group_values(joint_kp, indices, group.stiffness)
        _assign_group_values(joint_kd, indices, group.damping)
        _assign_group_values(effort_limits, indices, group.effort_limit)

    action_cfg = env_cfg.actions.joint_pos
    action_expr_dict = {expr: None for expr in action_cfg.joint_names}
    action_indices, _, _ = string_utils.resolve_matching_names_values(
        action_expr_dict, sim_joint_names, preserve_order=True
    )
    action_indices = [int(index) for index in action_indices]
    action_joint_names = [sim_joint_names[index] for index in action_indices]

    if action_joint_names != list(ACTIONABLE_JOINTS_V1_2_3):
        raise RuntimeError(
            "Resolved action joint order does not match the canonical 12-joint hardware contract:\n"
            f"resolved={action_joint_names}\nexpected={list(ACTIONABLE_JOINTS_V1_2_3)}"
        )
    if len(action_indices) != EXPECTED_ACTIONS:
        raise RuntimeError(f"Resolved {len(action_indices)} action joints; expected 12")

    action_defaults = [sim_default_joint_positions[index] for index in action_indices]
    residual_scale = _float_list(action_cfg.residual_scale_rad, EXPECTED_ACTIONS, "residual_scale_rad")
    physical_lower = _float_list(action_cfg.lower_limits, EXPECTED_ACTIONS, "lower_limits")
    physical_upper = _float_list(action_cfg.upper_limits, EXPECTED_ACTIONS, "upper_limits")

    if max(residual_scale) - min(residual_scale) <= 1.0e-9:
        raise RuntimeError("LittleGreen v2.5.0 requires the non-uniform action-contract-v4 residual vector")

    nominal_lower = [
        max(lower, default - scale)
        for default, lower, scale in zip(action_defaults, physical_lower, residual_scale)
    ]
    nominal_upper = [
        min(upper, default + scale)
        for default, upper, scale in zip(action_defaults, physical_upper, residual_scale)
    ]

    for index, (lower, default, upper) in enumerate(zip(physical_lower, action_defaults, physical_upper)):
        if not lower <= default <= upper:
            raise RuntimeError(
                f"Default joint position lies outside the physical limit at action index {index}: "
                f"{lower} <= {default} <= {upper} is false"
            )

    physics_dt = float(env_cfg.sim.dt)
    decimation = int(env_cfg.decimation)
    policy_dt = physics_dt * decimation
    if not math.isclose(physics_dt, EXPECTED_PHYSICS_DT, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError(f"v2.5.0 export requires physics_dt=0.005, got {physics_dt}")
    if not math.isclose(policy_dt, EXPECTED_POLICY_DT, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError(f"v2.5.0 export requires policy_dt=0.02, got {policy_dt}")

    policy_observation_dim = _get_policy_observation_dim(env)
    critic_observation_dim = _get_group_observation_dim(
        raw_env, "critic", EXPECTED_CRITIC_OBSERVATIONS
    )
    if policy_observation_dim != EXPECTED_POLICY_OBSERVATIONS:
        raise RuntimeError(
            f"LittleGreen v2.5.0 requires a 47-D actor observation, got {policy_observation_dim}"
        )
    if critic_observation_dim != EXPECTED_CRITIC_OBSERVATIONS:
        raise RuntimeError(
            f"LittleGreen v2.4.3 expects a 50-D critic observation, got {critic_observation_dim}"
        )

    phase_metadata = _resolved_phase_metadata(env_cfg, args_cli.task)
    command_metadata = _resolved_command_metadata(env_cfg, args_cli.task)
    onnx_path = (export_dir / onnx_filename).resolve()

    action_metadata: dict[str, Any] = {
        "action_contract_version": 4,
        "action_contract_name": "bounded_default_centered_vector_residual_athletic",
        "action_transform": "bounded_default_centered_vector_residual",
        "action_equation": (
            "q_target = clip(q_default + clip(a_raw,-1,1) * residual_scale_rad, "
            "physical_lower, physical_upper)"
        ),
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
        "deployment_requires_action_contract_v4_transform": True,
        # The transform and physical vectors remain the stabilized v1.4.5 contract,
        # while the actor observation contract is now the shared v2.4.3 47-D contract.
        "deployment_contract_profile": "v1_4_5_stabilized_vector_residual",
        "training_release_profile": (
            "v10_3_1_forward_bilateral_acquisition"
            if args_cli.task == LOCOMOTION_TASK
            else "v2_4_3_whole_foot_transfer_walk_foundation"
        ),
    }

    if hasattr(action_cfg, "actuator_model_name"):
        action_metadata.update(
            {
                "training_actuator_model_name": str(action_cfg.actuator_model_name),
                "training_actuator_model_source_name": str(action_cfg.actuator_model_name).replace(
                    ST3215_DATASET_NAME, ST3215_SOURCE_DATASET_NAME
                ),
                "training_actuator_model_stage": str(
                    getattr(action_cfg, "actuator_model_stage", "stage_a")
                ),
                "training_actuator_bus_phase_delay_s_range": _safe_to_list(
                    action_cfg.bus_phase_delay_s_range
                ),
                "training_actuator_response_delay_s_range": _safe_to_list(
                    action_cfg.response_delay_s_range
                ),
                "training_actuator_response_delay_scale": float(action_cfg.response_delay_scale),
                "training_actuator_tau_median_s": _safe_to_list(action_cfg.tau_median_s),
                "training_actuator_tau_p10_s": _safe_to_list(action_cfg.tau_p10_s),
                "training_actuator_tau_p90_s": _safe_to_list(action_cfg.tau_p90_s),
                "training_actuator_velocity_scale_range": _safe_to_list(
                    action_cfg.velocity_scale_range
                ),
                "training_actuator_special_transients_enabled": False,
                "deployment_actuator_model_is_training_only": True,
            }
        )

        if bool(getattr(action_cfg, "loaded_envelope_enabled", False)):
            action_metadata.update(
                {
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
                }
            )

    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "metadata": {
            "project_name": PROJECT_NAME,
            "project_version": PROJECT_VERSION,
            "training_package": TRAINING_PACKAGE,
            "asset_package": ASSET_PACKAGE,
            "source_lineage": "Berkeley-Humanoid-Lite v1.4.8 renamed and evolved",
            "exported_at_utc": _utc_timestamp(),
            "task": args_cli.task,
            "task_role": (
                "stand"
                if args_cli.task == STAND_TASK
                else "transfer"
                if args_cli.task == TRANSFER_TASK
                else "walk"
                if args_cli.task == WALK_TASK
                else "locomotion"
            ),
            "training_release": PROJECT_VERSION,
            "experiment_name": str(agent_cfg.experiment_name),
            "load_run": str(args_cli.load_run),
            "checkpoint": str(args_cli.checkpoint),
            "checkpoint_path": str(Path(checkpoint_path).resolve()),
            "export_dir": str(export_dir.resolve()),
            "policy_filename": onnx_filename,
            "policy_format": "onnx",
            "policy_sha256": onnx_sha256,
        },
        "policy_checkpoint_path": str(onnx_path),
        "policy_checkpoint_relative_path": onnx_filename,
        "policy_checkpoint_filename": onnx_filename,
        "policy_sha256": onnx_sha256,
        **onnx_contract,
        "ip_robot_addr": "127.0.0.1",
        "ip_policy_obs_port": 10000,
        "ip_host_addr": "127.0.0.1",
        "ip_policy_acs_port": 10001,
        "control_dt": policy_dt,
        "policy_dt": policy_dt,
        "physics_dt": physics_dt,
        "decimation": decimation,
        "physics_hz": int(round(1.0 / physics_dt)),
        "policy_hz": int(round(1.0 / policy_dt)),
        "num_joints": sim_num_joints,
        "joints": sim_joint_names,
        "joint_kp": joint_kp.detach().cpu().tolist(),
        "joint_kd": joint_kd.detach().cpu().tolist(),
        "effort_limits": effort_limits.detach().cpu().tolist(),
        "default_base_position": _safe_to_list(env_cfg.scene.robot.init_state.pos),
        "default_joint_positions": sim_default_joint_positions,
        "num_actions": EXPECTED_ACTIONS,
        "action_indices": action_indices,
        "action_joint_names": action_joint_names,
        "num_observations": policy_observation_dim,
        "critic_num_observations": critic_observation_dim,
        "history_length": _history_length(),
        **phase_metadata,
        # Runtime supplies the actual command. Avoid embedding a random sampled export command.
        "command_velocity": [0.0, 0.0, 0.0],
        **command_metadata,
        **action_metadata,
    }


def _deployment_contract(config: dict[str, Any]) -> dict[str, Any]:
    """Create a concise contract document paired with the full policy YAML."""
    return {
        "release": f"{PROJECT_NAME} v{config['metadata'].get('training_release', PROJECT_VERSION)}",
        "schema_version": EXPORT_SCHEMA_VERSION,
        "task": config["metadata"]["task"],
        "task_role": config["metadata"]["task_role"],
        "artifact": {
            "policy_filename": config["policy_checkpoint_filename"],
            "policy_sha256": config["policy_sha256"],
            "onnx_input_name": config["onnx_input_name"],
            "onnx_input_shape": config["onnx_input_shape"],
            "onnx_output_name": config["onnx_output_name"],
            "onnx_output_shape": config["onnx_output_shape"],
        },
        "observation_contract": {
            "name": config["observation_contract_name"],
            "actor_count": config["num_observations"],
            "critic_count": config["critic_num_observations"],
            "layout": config["observation_layout"],
            "phase_indices": config["phase_indices"],
            "phase_encoding": config["phase_encoding"],
            "phase_mode": config["phase_mode"],
            "phase_period_active": config.get("phase_period_active", True),
            "phase_constant_sin_cos": config.get("phase_constant_sin_cos"),
            "legacy_45d_checkpoint_support": False,
        },
        "command_contract": {
            "version": config["command_contract_version"],
            "source": config["command_velocity_source"],
            "lin_vel_x": config["command_limit_lin_vel_x"],
            "lin_vel_y": config["command_limit_lin_vel_y"],
            "ang_vel_z": config["command_limit_ang_vel_z"],
            "runtime_clamp_required": config["deployment_requires_command_clamp"],
            "clamp_behavior": config["command_limit_behavior"],
        },
        "action_contract": {
            "version": config["action_contract_version"],
            "count": config["num_actions"],
            "joint_names": config["action_joint_names"],
            "transform": config["action_transform"],
            "previous_action_observation": config["previous_action_observation"],
        },
        "timing": {
            "physics_dt": config["physics_dt"],
            "policy_dt": config["policy_dt"],
            "decimation": config["decimation"],
        },
        "deployment_requirements": {
            "shared_47d_observation_builder": True,
            "phase_mode_must_match_training": True,
            "action_contract_v4_transform": True,
            "canonical_joint_order": True,
            "paired_yaml_onnx_sha256": True,
            "runtime_command_range_clamp": config["deployment_requires_command_clamp"],
            "neutral_static_phase_tail": config["phase_mode"] == "neutral_static",
        },
    }


def _config_for_destination(
    base_config: dict[str, Any],
    destination_onnx: Path,
    publication_kind: str,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["policy_checkpoint_path"] = str(destination_onnx.resolve())
    config["policy_checkpoint_relative_path"] = destination_onnx.name
    config["policy_checkpoint_filename"] = destination_onnx.name
    config["metadata"]["publication_kind"] = publication_kind
    config["metadata"]["published_at_utc"] = _utc_timestamp()
    config["metadata"]["published_policy_path"] = str(destination_onnx.resolve())
    return config


def _write_bundle_sidecars(
    directory: Path,
    config: dict[str, Any],
    onnx_path: Path,
    yaml_filename: str,
    manifest_filename: str,
) -> None:
    onnx_sha = _sha256_file(onnx_path)
    if onnx_sha != config["policy_sha256"]:
        raise RuntimeError(
            f"ONNX checksum mismatch while publishing {onnx_path}: "
            f"expected {config['policy_sha256']}, got {onnx_sha}"
        )

    _atomic_write_text(f"{onnx_sha}  {onnx_path.name}\n", directory / "policy.sha256")
    yaml_path = directory / yaml_filename
    yaml_sha = _sha256_file(yaml_path)
    manifest = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "project_name": PROJECT_NAME,
        "project_version": PROJECT_VERSION,
        "task": config["metadata"]["task"],
        "created_at_utc": _utc_timestamp(),
        "files": {
            "onnx": {"filename": onnx_path.name, "sha256": onnx_sha},
            "yaml": {"filename": yaml_path.name, "sha256": yaml_sha},
        },
        "interface": {
            "observations": config["num_observations"],
            "actions": config["num_actions"],
            "action_contract_version": config["action_contract_version"],
            "phase_mode": config["phase_mode"],
        },
    }
    _atomic_save_yaml(manifest, directory / manifest_filename)


def _publish_policy_pair(
    source_onnx: Path,
    base_config: dict[str, Any],
    destination_yaml: Path,
    publication_kind: str,
    manifest_filename: str,
) -> None:
    """Publish model first, then atomically publish its YAML pointer and checksum."""
    destination_yaml = destination_yaml.expanduser().resolve()
    destination_dir = destination_yaml.parent
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_onnx = destination_dir / source_onnx.name

    _atomic_copy(source_onnx, destination_onnx)
    destination_config = _config_for_destination(
        base_config, destination_onnx, publication_kind
    )
    _atomic_save_yaml(destination_config, destination_yaml)
    _write_bundle_sidecars(
        destination_dir,
        destination_config,
        destination_onnx,
        destination_yaml.name,
        manifest_filename,
    )

    print(f"[PUBLISH] ONNX: {destination_onnx}")
    print(f"[PUBLISH] YAML: {destination_yaml}")
    print(f"[PUBLISH] SHA-256 verified: {destination_config['policy_sha256']}")


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root_path = Path("logs") / "rsl_rl" / str(agent_cfg.experiment_name)
    checkpoint_path = get_checkpoint_path(
        str(log_root_path.resolve()), agent_cfg.load_run, agent_cfg.load_checkpoint
    )
    run_dir = Path(checkpoint_path).resolve().parent
    export_dir = (
        Path(args_cli.output_dir).expanduser().resolve()
        if args_cli.output_dir
        else run_dir / "exported"
    )
    export_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Project:         {PROJECT_NAME} v{PROJECT_VERSION}")
    print(f"[INFO] Task:            {args_cli.task}")
    print(f"[INFO] Experiment root: {log_root_path.resolve()}")
    print(f"[INFO] Checkpoint:      {checkpoint_path}")
    print(f"[INFO] Export dir:      {export_dir}")
    if args_cli.task in (TRANSFER_TASK, WALK_TASK) and args_cli.deployment_stage is None:
        print(
            "[EXPORT][NOTICE] Transfer/Walk export is using the resolved task-default phase period. "
            "For a deployment bundle, pass --deployment_stage 0..3 to record the checkpoint's intended period."
        )

    raw_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = raw_env
    try:
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        env = RslRlVecEnvWrapper(env)

        observation_dim = _get_policy_observation_dim(env)
        action_dim = int(env.action_space.shape[-1])
        if observation_dim != EXPECTED_POLICY_OBSERVATIONS:
            raise RuntimeError(
                f"LittleGreen deployment export requires obs[47], got obs[{observation_dim}]"
            )
        if action_dim != EXPECTED_ACTIONS:
            raise RuntimeError(
                f"LittleGreen deployment export requires actions[12], got actions[{action_dim}]"
            )

        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(checkpoint_path)

        onnx_export_path = export_dir / args_cli.onnx_filename
        print(f"[EXPORT] ONNX: {onnx_export_path}", flush=True)
        _export_policy_as_onnx_legacy(
            policy=runner.alg.policy,
            normalizer=runner.obs_normalizer,
            destination=onnx_export_path,
            observation_dim=observation_dim,
        )

        if not args_cli.no_jit:
            jit_export_path = export_dir / args_cli.jit_filename
            print(f"[EXPORT] JIT:  {jit_export_path}", flush=True)
            export_policy_as_jit(
                runner.alg.policy,
                runner.obs_normalizer,
                path=str(export_dir),
                filename=args_cli.jit_filename,
            )

        onnx_sha256 = _sha256_file(onnx_export_path)
        onnx_contract = _inspect_onnx_contract(onnx_export_path)
        print(
            "[EXPORT] ONNX contract: "
            f"{onnx_contract['onnx_input_name']}{onnx_contract['onnx_input_shape']} -> "
            f"{onnx_contract['onnx_output_name']}{onnx_contract['onnx_output_shape']}"
        )
        print(f"[EXPORT] ONNX SHA-256: {onnx_sha256}")

        deploy_config = _build_deploy_config(
            env=env,
            raw_env=raw_env,
            env_cfg=env_cfg,
            agent_cfg=agent_cfg,
            checkpoint_path=checkpoint_path,
            export_dir=export_dir,
            onnx_filename=args_cli.onnx_filename,
            onnx_sha256=onnx_sha256,
            onnx_contract=onnx_contract,
        )

        run_yaml_path = export_dir / "policy.yaml"
        _atomic_save_yaml(deploy_config, run_yaml_path)
        _atomic_save_yaml(_deployment_contract(deploy_config), export_dir / "deployment_contract.yaml")
        _write_bundle_sidecars(
            export_dir,
            deploy_config,
            onnx_export_path,
            "policy.yaml",
            "bundle_manifest.yaml",
        )
        print(f"[EXPORT] YAML:     {run_yaml_path}")
        print(f"[EXPORT] Contract: {export_dir / 'deployment_contract.yaml'}")
        print(f"[EXPORT] Manifest: {export_dir / 'bundle_manifest.yaml'}")

        if args_cli.latest_config_path:
            _publish_policy_pair(
                onnx_export_path,
                deploy_config,
                Path(args_cli.latest_config_path),
                publication_kind="project_latest",
                manifest_filename="policy_bundle_manifest.yaml",
            )

        if args_cli.deploy_ros2:
            if not args_cli.ros2_deploy_dir:
                raise RuntimeError("--deploy_ros2 requires a non-empty --ros2_deploy_dir")
            deploy_dir = Path(args_cli.ros2_deploy_dir).expanduser().resolve()
            print(
                "[DEPLOY][ACK] Publishing requires the ROS 2 runtime to implement "
                f"phase_mode={deploy_config['phase_mode']}, the shared 47-D observation builder, "
                "action contract v4, and the exported command clamp contract when required."
            )
            _publish_policy_pair(
                onnx_export_path,
                deploy_config,
                deploy_dir / "policy_latest.yaml",
                publication_kind="ros2_deployment",
                manifest_filename="policy_bundle_manifest.yaml",
            )
        else:
            print("[DEPLOY] ROS 2 publication skipped; use --deploy_ros2 after runtime-contract validation.")
    finally:
        # RslRlVecEnvWrapper.close() closes the underlying Gym environment.
        try:
            env.close()
        except Exception:
            try:
                raw_env.close()
            except Exception:
                pass


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
