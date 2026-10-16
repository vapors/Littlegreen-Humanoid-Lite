"""LittleGreen v10.1 exploratory turn-biased velocity command generator.

The command dimensionality and observation contract remain unchanged: [vx, vy, yaw_rate].
Most resamples use the configured full uniform joystick-like envelope.  A small,
explicit fraction of non-standing resamples is replaced by a forward-moving,
high-yaw command so turning is represented often enough to challenge the learned
strong-leg/weak-leg cycle without introducing a turning curriculum or new reward.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass


class TurnBiasedUniformVelocityCommandV101(UniformVelocityCommand):
    """Uniform SE(2) commands with a small reserved set of turn-focused draws.

    The parent sampler remains authoritative for the general command distribution.
    After it samples a command, ``rel_turn_focus_envs`` of non-standing draws are
    replaced with a moderate forward command and a deliberately large yaw-rate
    magnitude.  Yaw sign is balanced 50/50.  This is command sampling only; it does
    not add a reward, phase schedule, gait clock, or curriculum.
    """

    cfg: "TurnBiasedUniformVelocityCommandCfgV101"

    def __init__(self, cfg: "TurnBiasedUniformVelocityCommandCfgV101", env):
        if not 0.0 <= float(cfg.rel_turn_focus_envs) <= 1.0:
            raise ValueError("v10.1 rel_turn_focus_envs must be in [0, 1]")
        if cfg.turn_lin_vel_x[0] > cfg.turn_lin_vel_x[1]:
            raise ValueError("v10.1 turn_lin_vel_x range is invalid")
        if cfg.turn_lin_vel_y[0] > cfg.turn_lin_vel_y[1]:
            raise ValueError("v10.1 turn_lin_vel_y range is invalid")
        if cfg.turn_ang_vel_abs[0] < 0.0 or cfg.turn_ang_vel_abs[0] > cfg.turn_ang_vel_abs[1]:
            raise ValueError("v10.1 turn_ang_vel_abs range is invalid")
        super().__init__(cfg, env)
        self.is_turn_focus_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.metrics["turn_focus_fraction"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = super().__str__()
        msg += f"\n\tTurn-focus probability: {self.cfg.rel_turn_focus_envs}"
        msg += f"\n\tTurn-focus vx range: {self.cfg.turn_lin_vel_x}"
        msg += f"\n\tTurn-focus vy range: {self.cfg.turn_lin_vel_y}"
        msg += f"\n\tTurn-focus |yaw| range: {self.cfg.turn_ang_vel_abs}"
        return msg

    def _resample_command(self, env_ids: Sequence[int]):
        # Preserve Isaac Lab's normal uniform sampler, standing-mask semantics,
        # command hold times, debug visualization, and direct yaw-rate behavior.
        super()._resample_command(env_ids)
        if len(env_ids) == 0:
            return

        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.is_turn_focus_env[ids] = False

        r = torch.empty(ids.numel(), device=self.device)
        choose_turn = r.uniform_(0.0, 1.0) <= float(self.cfg.rel_turn_focus_envs)
        # Standing samples remain truly zero-command after the parent's update step.
        choose_turn &= ~self.is_standing_env[ids]
        turn_ids = ids[choose_turn]
        if turn_ids.numel() == 0:
            return

        rt = torch.empty(turn_ids.numel(), device=self.device)
        self.vel_command_b[turn_ids, 0] = rt.uniform_(*self.cfg.turn_lin_vel_x)
        self.vel_command_b[turn_ids, 1] = rt.uniform_(*self.cfg.turn_lin_vel_y)
        yaw_mag = rt.uniform_(*self.cfg.turn_ang_vel_abs)
        yaw_sign = torch.where(rt.uniform_(0.0, 1.0) < 0.5, -torch.ones_like(yaw_mag), torch.ones_like(yaw_mag))
        self.vel_command_b[turn_ids, 2] = yaw_mag * yaw_sign
        self.is_turn_focus_env[turn_ids] = True

    def _update_metrics(self):
        super()._update_metrics()
        # Per-env flag is useful for runtime/analyzer diagnostics.  It is not a reward.
        self.metrics["turn_focus_fraction"][:] = self.is_turn_focus_env.float()


@configclass
class TurnBiasedUniformVelocityCommandCfgV101(UniformVelocityCommandCfg):
    """Configuration for the v10.1 exploratory turn-biased command sampler."""

    class_type: type = TurnBiasedUniformVelocityCommandV101

    # 12.5% of non-standing command resamples are explicitly turn-focused.
    # With 3% standing commands this is ~12.1% of all raw draws, inside the
    # requested 10-15% band.
    rel_turn_focus_envs: float = 0.125

    # Turn-focused commands still move forward so they exercise the successful
    # alternating gait rather than teaching a separate in-place pivot exploit.
    turn_lin_vel_x: tuple[float, float] = (0.10, 0.30)
    turn_lin_vel_y: tuple[float, float] = (-0.08, 0.08)
    turn_ang_vel_abs: tuple[float, float] = (0.30, 0.50)
