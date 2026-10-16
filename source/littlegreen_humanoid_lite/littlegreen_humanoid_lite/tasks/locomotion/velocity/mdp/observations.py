"""Canonical 47-D policy observation helpers.

The actor observation is the original 45-D Berkeley/LittleGreen vector plus a
command-synchronized gait phase encoded as ``sin`` and ``cos``.  The phase clock
is intentionally simple: it never reads foot contact state and it never freezes
mid-step.  This keeps observation semantics independent from reward experiments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(
    env: ManagerBasedRLEnv,
    command_name: str,
    period_s: float = 0.84,
    linear_command_threshold: float = 0.10,
    yaw_command_threshold: float = 0.08,
) -> torch.Tensor:
    """Return a deterministic normalized gait phase in ``[0, 1)``.

    The phase is derived only from episode time.  Standing commands receive phase
    zero, corresponding to the double-support boundary ``[sin, cos] = [0, 1]``.
    No contact, COM, touchdown, or curriculum state can alter this clock.
    """
    if period_s <= 0.0:
        raise ValueError("period_s must be positive")
    command = env.command_manager.get_command(command_name)
    moving = (
        torch.linalg.vector_norm(command[:, :2], dim=1) > linear_command_threshold
    ) | (torch.abs(command[:, 2]) > yaw_command_threshold)
    elapsed_s = env.episode_length_buf.to(dtype=torch.float32) * float(env.step_dt)
    phase = torch.remainder(elapsed_s / period_s, 1.0)
    return torch.where(moving, phase, torch.zeros_like(phase))


def command_phase_sin_cos(
    env: ManagerBasedRLEnv,
    command_name: str,
    period_s: float = 0.84,
    linear_command_threshold: float = 0.10,
    yaw_command_threshold: float = 0.08,
) -> torch.Tensor:
    """Append the canonical two-value phase representation to the 45-D vector."""
    phase = gait_phase(
        env,
        command_name=command_name,
        period_s=period_s,
        linear_command_threshold=linear_command_threshold,
        yaw_command_threshold=yaw_command_threshold,
    )
    angle = 2.0 * torch.pi * phase
    return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
