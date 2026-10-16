"""LittleGreen v2.4 phase generators shared by observations and rewards.

The implementation is intentionally small and compatible with the manager-based
Isaac Lab environment used by the existing Isaac Sim 5.0 project.

Three phase modes are supported:

* Stand: one random phase per episode, held static.
* Transfer: a continuous alternating clock independent of velocity commands.
* Walk: a command-synchronized continuous clock.  The phase is zero while the
  command is below threshold, starts on alternating left/right swing boundaries,
  and never waits for a physical contact event.

No reward-success state gates the observation clock.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _env_ids(env: ManagerBasedRLEnv, env_ids) -> torch.Tensor:
    device = getattr(env, "device", None)
    if device is None:
        device = env.scene["robot"].device
    all_ids = torch.arange(int(env.num_envs), device=device, dtype=torch.long)
    if env_ids is None:
        return all_ids
    if isinstance(env_ids, slice):
        return all_ids[env_ids]
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=device, dtype=torch.long)


def _tensor_state(
    env: ManagerBasedRLEnv,
    name: str,
    default: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    value = getattr(env, name, None)
    if not isinstance(value, torch.Tensor) or value.shape != default.shape:
        value = default.clone()
    value = value.to(device=default.device)
    if dtype is not None:
        value = value.to(dtype=dtype)
    setattr(env, name, value)
    return value


def transfer_half_cycle_shape_v24(phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return scheduled stance side, half-cycle progress, and mid-swing shape.

    Phase semantics are intentionally aligned with ``_phase_swing_state_v24``:
    ``phase < 0.5`` means left stance/right swing and ``phase >= 0.5`` means
    right stance/left swing.  The returned shape is zero at double-support
    boundaries and one at the middle of each swing half-cycle.
    """
    half_progress = torch.remainder(2.0 * phase, 1.0)
    left_stance = phase < 0.5
    swing_shape = torch.sin(torch.pi * half_progress)
    return left_stance, half_progress, swing_shape


def reset_random_static_phase_v24(env: ManagerBasedRLEnv, env_ids) -> None:
    """Sample a phase independently for each reset Stand environment."""
    ids = _env_ids(env, env_ids)
    device = ids.device
    phase = _tensor_state(
        env,
        "_littlegreen_v24_static_phase",
        torch.zeros(int(env.num_envs), device=device, dtype=torch.float32),
    )
    if ids.numel() > 0:
        phase[ids] = torch.rand(ids.numel(), device=device, dtype=torch.float32)


def random_static_phase_fraction_v24(env: ManagerBasedRLEnv) -> torch.Tensor:
    phase = getattr(env, "_littlegreen_v24_static_phase", None)
    if not isinstance(phase, torch.Tensor) or phase.shape != (int(env.num_envs),):
        reset_random_static_phase_v24(env, None)
        phase = env._littlegreen_v24_static_phase
    return phase


def reset_continuous_transfer_phase_v24(env: ManagerBasedRLEnv, env_ids) -> None:
    """Reset Transfer phase offsets with balanced first-swing assignment.

    Even environment IDs start with the right foot scheduled to swing
    (phase=0); odd environment IDs start with the left foot scheduled to swing
    (phase=0.5).  A tiny random offset avoids all environments crossing the
    phase boundary on the same simulation frame.
    """
    ids = _env_ids(env, env_ids)
    device = ids.device
    offsets = _tensor_state(
        env,
        "_littlegreen_v24_transfer_phase_offset",
        torch.zeros(int(env.num_envs), device=device, dtype=torch.float32),
    )
    active = _tensor_state(
        env,
        "_littlegreen_v24_transfer_phase_active",
        torch.ones(int(env.num_envs), device=device, dtype=torch.bool),
        dtype=torch.bool,
    )
    if ids.numel() > 0:
        half = (ids % 2).to(torch.float32) * 0.5
        jitter = (torch.rand(ids.numel(), device=device) - 0.5) * 0.02
        offsets[ids] = torch.remainder(half + jitter, 1.0)
        stand_fraction = float(
            getattr(env, "_littlegreen_v24_transfer_phase_standing_fraction", 0.25)
        )
        stand_fraction = min(max(stand_fraction, 0.0), 1.0)
        active[ids] = torch.rand(ids.numel(), device=device) >= stand_fraction


def continuous_transfer_phase_fraction_v24(
    env: ManagerBasedRLEnv,
    period_s: float = 1.10,
) -> torch.Tensor:
    """Return a deterministic continuous phase derived from episode step count."""
    active_period = float(getattr(env, "_littlegreen_v24_phase_period_s", period_s))
    if active_period <= 1.0e-6:
        raise ValueError("v2.4 Transfer phase period must be positive")
    offsets = getattr(env, "_littlegreen_v24_transfer_phase_offset", None)
    if not isinstance(offsets, torch.Tensor) or offsets.shape != (int(env.num_envs),):
        reset_continuous_transfer_phase_v24(env, None)
        offsets = env._littlegreen_v24_transfer_phase_offset
    episode_steps = getattr(env, "episode_length_buf", None)
    if not isinstance(episode_steps, torch.Tensor):
        episode_steps = torch.zeros(int(env.num_envs), device=offsets.device)
    phase = torch.remainder(
        offsets + episode_steps.to(dtype=torch.float32) * float(env.step_dt) / active_period,
        1.0,
    )
    active = getattr(env, "_littlegreen_v24_transfer_phase_active", None)
    if not isinstance(active, torch.Tensor) or active.shape != phase.shape:
        reset_continuous_transfer_phase_v24(env, None)
        active = env._littlegreen_v24_transfer_phase_active
    phase = torch.where(active, phase, torch.zeros_like(phase))
    env._littlegreen_v24_phase_value = phase
    env._littlegreen_v24_phase_moving = active
    return phase


def reset_command_synchronized_phase_v24(env: ManagerBasedRLEnv, env_ids) -> None:
    """Reset command-synchronized Walk phase state for selected environments."""
    ids = _env_ids(env, env_ids)
    device = ids.device
    num_envs = int(env.num_envs)
    phase = _tensor_state(
        env,
        "_littlegreen_v24_walk_phase",
        torch.zeros(num_envs, device=device, dtype=torch.float32),
    )
    was_moving = _tensor_state(
        env,
        "_littlegreen_v24_walk_was_moving",
        torch.zeros(num_envs, device=device, dtype=torch.bool),
        dtype=torch.bool,
    )
    next_left = _tensor_state(
        env,
        "_littlegreen_v24_walk_next_first_swing_left",
        (torch.arange(num_envs, device=device) % 2 == 1),
        dtype=torch.bool,
    )
    active_left = _tensor_state(
        env,
        "_littlegreen_v24_walk_active_first_swing_left",
        torch.zeros(num_envs, device=device, dtype=torch.bool),
        dtype=torch.bool,
    )
    if ids.numel() > 0:
        phase[ids] = 0.0
        was_moving[ids] = False
        next_left[ids] = (ids % 2 == 1)
        active_left[ids] = False
    env._littlegreen_v24_walk_last_common_step = -1


def command_synchronized_phase_fraction_v24(
    env: ManagerBasedRLEnv,
    command_name: str,
    linear_command_threshold: float = 0.08,
    yaw_command_threshold: float = 0.08,
    period_s: float = 0.90,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the v2.4 command-synchronized Walk phase and moving mask.

    This matches the Track 2 policy phase state contract: zero phase while
    standing, alternating first-swing boundary on command onset, then a fixed
    non-blocking advance of ``policy_dt / period``.
    """
    command = env.command_manager.get_command(command_name)
    linear_norm = torch.linalg.vector_norm(command[:, :2], dim=1)
    yaw_abs = torch.abs(command[:, 2]) if command.shape[1] > 2 else torch.zeros_like(linear_norm)
    moving = (linear_norm > linear_command_threshold) | (yaw_abs > yaw_command_threshold)
    device = command.device
    num_envs = int(env.num_envs)

    phase = _tensor_state(
        env,
        "_littlegreen_v24_walk_phase",
        torch.zeros(num_envs, device=device, dtype=torch.float32),
    )
    was_moving = _tensor_state(
        env,
        "_littlegreen_v24_walk_was_moving",
        torch.zeros(num_envs, device=device, dtype=torch.bool),
        dtype=torch.bool,
    )
    next_left = _tensor_state(
        env,
        "_littlegreen_v24_walk_next_first_swing_left",
        (torch.arange(num_envs, device=device) % 2 == 1),
        dtype=torch.bool,
    )
    active_left = _tensor_state(
        env,
        "_littlegreen_v24_walk_active_first_swing_left",
        torch.zeros(num_envs, device=device, dtype=torch.bool),
        dtype=torch.bool,
    )

    active_period = float(getattr(env, "_littlegreen_v24_phase_period_s", period_s))
    if active_period <= 1.0e-6:
        raise ValueError("v2.4 Walk phase period must be positive")

    common_step = int(getattr(env, "common_step_counter", 0))
    last_step = int(getattr(env, "_littlegreen_v24_walk_last_common_step", -1))
    if common_step != last_step:
        onset = moving & (~was_moving)
        active_left = torch.where(onset, next_left, active_left)
        next_left = torch.where(onset, ~next_left, next_left)
        onset_phase = torch.where(
            active_left,
            torch.full_like(phase, 0.5),
            torch.zeros_like(phase),
        )
        phase = torch.where(onset, onset_phase, phase)
        advance = float(env.step_dt) / active_period
        phase = torch.where(
            moving & (~onset),
            torch.remainder(phase + advance, 1.0),
            phase,
        )
        phase = torch.where(moving, phase, torch.zeros_like(phase))
        was_moving = moving.clone()

        env._littlegreen_v24_walk_phase = phase
        env._littlegreen_v24_walk_was_moving = was_moving
        env._littlegreen_v24_walk_next_first_swing_left = next_left
        env._littlegreen_v24_walk_active_first_swing_left = active_left
        env._littlegreen_v24_walk_last_common_step = common_step

    env._littlegreen_v24_phase_value = phase
    env._littlegreen_v24_phase_moving = moving
    return phase, moving


def phase_sin_cos(phase: torch.Tensor) -> torch.Tensor:
    angle = 2.0 * math.pi * phase
    return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
