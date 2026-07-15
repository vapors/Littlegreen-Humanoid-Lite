"""Hardware-aligned action term for Berkeley Humanoid Lite v1.2.3."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class BoundedDefaultCenteredJointPositionAction(ActionTerm):
    """Clamp policy output to [-1, 1] and map it asymmetrically around q_default.

    Action semantics:

    * ``a = 0`` maps exactly to the training default pose.
    * ``a = -1`` maps to the configured physical lower joint limit.
    * ``a = +1`` maps to the configured physical upper joint limit.
    * raw network output is retained for diagnostics and excess-action penalties.
    * bounded normalized action is exposed for the policy's last-action observation.
    * optional per-environment one-policy-step delay models host/transport latency.
    """

    cfg: "BoundedDefaultCenteredJointPositionActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "BoundedDefaultCenteredJointPositionActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names,
            preserve_order=self.cfg.preserve_order,
        )
        self._num_joints = len(self._joint_ids)

        if self._num_joints != len(self.cfg.lower_limits) or self._num_joints != len(self.cfg.upper_limits):
            raise ValueError(
                "Bounded action limit arrays must match resolved joint count: "
                f"joints={self._num_joints}, lower={len(self.cfg.lower_limits)}, "
                f"upper={len(self.cfg.upper_limits)}"
            )

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._bounded_actions = torch.zeros_like(self._raw_actions)
        self._previous_bounded_actions = torch.zeros_like(self._raw_actions)
        self._applied_bounded_actions = torch.zeros_like(self._raw_actions)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        self._lower = torch.tensor(self.cfg.lower_limits, device=self.device, dtype=torch.float32).unsqueeze(0)
        self._upper = torch.tensor(self.cfg.upper_limits, device=self.device, dtype=torch.float32).unsqueeze(0)
        self._default = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

        if torch.any(self._default < self._lower) or torch.any(self._default > self._upper):
            raise ValueError("Training default joint positions must lie inside the configured hardware limits")

        self._delay_one_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._sample_delay(slice(None))
        self._processed_actions[:] = self._default

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def joint_names(self) -> list[str]:
        return list(self._joint_names)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def bounded_actions(self) -> torch.Tensor:
        """Current clamped normalized policy action."""
        return self._bounded_actions

    @property
    def previous_bounded_actions(self) -> torch.Tensor:
        """Previous clamped normalized policy action, for action-rate penalties."""
        return self._previous_bounded_actions

    @property
    def applied_bounded_actions(self) -> torch.Tensor:
        """Normalized action actually mapped to the position target after optional delay."""
        return self._applied_bounded_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Physical joint-position targets in radians."""
        return self._processed_actions

    @property
    def lower_limits(self) -> torch.Tensor:
        return self._lower

    @property
    def upper_limits(self) -> torch.Tensor:
        return self._upper

    @property
    def default_positions(self) -> torch.Tensor:
        return self._default

    @property
    def delay_one_step(self) -> torch.Tensor:
        return self._delay_one_step

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._previous_bounded_actions[:] = self._bounded_actions
        self._bounded_actions[:] = torch.clamp(actions, -1.0, 1.0)

        # One-step command latency is applied only to the physical target path.
        self._applied_bounded_actions[:] = torch.where(
            self._delay_one_step.unsqueeze(-1),
            self._previous_bounded_actions,
            self._bounded_actions,
        )

        positive_range = self._upper - self._default
        negative_range = self._default - self._lower
        positive_target = self._default + self._applied_bounded_actions * positive_range
        negative_target = self._default + self._applied_bounded_actions * negative_range
        self._processed_actions[:] = torch.where(
            self._applied_bounded_actions >= 0.0,
            positive_target,
            negative_target,
        )
        self._processed_actions[:] = torch.clamp(self._processed_actions, self._lower, self._upper)

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._bounded_actions[env_ids] = 0.0
        self._previous_bounded_actions[env_ids] = 0.0
        self._applied_bounded_actions[env_ids] = 0.0
        # Select action joints first, then select environments.  Using
        # joint_pos[env_ids, joint_ids] with two 1-D index tensors asks PyTorch
        # to broadcast them element-wise; shapes [num_reset_envs] and
        # [num_action_joints] are not broadcastable when they differ (e.g.
        # [64] and [12] during the initial reset).
        current_action_joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        self._processed_actions[env_ids] = current_action_joint_pos[env_ids]
        self._sample_delay(env_ids)

    def _sample_delay(self, env_ids) -> None:
        probability = float(self.cfg.one_step_delay_probability)
        probability = max(0.0, min(1.0, probability))
        if probability <= 0.0:
            self._delay_one_step[env_ids] = False
            return
        if probability >= 1.0:
            self._delay_one_step[env_ids] = True
            return
        sample_shape = self._delay_one_step[env_ids].shape
        self._delay_one_step[env_ids] = torch.rand(sample_shape, device=self.device) < probability


class BoundedDefaultResidualJointPositionAction(BoundedDefaultCenteredJointPositionAction):
    """Map bounded normalized action to a symmetric residual around q_default.

    Action contract v3 semantics:

    * raw network output is retained for diagnostics and raw-excess penalties;
    * network output is clamped to ``[-1, 1]``;
    * optional one-policy-step latency is applied in normalized-action space;
    * ``q_target = q_default + residual_scale_rad * bounded_action``;
    * the resulting target is finally clipped to calibrated physical limits.
    """

    cfg: "BoundedDefaultResidualJointPositionActionCfg"

    def __init__(self, cfg: "BoundedDefaultResidualJointPositionActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)

        scale = cfg.residual_scale_rad
        if isinstance(scale, (int, float)):
            if float(scale) <= 0.0:
                raise ValueError("residual_scale_rad must be positive")
            self._residual_scale = torch.full(
                (1, self._num_joints), float(scale), device=self.device, dtype=torch.float32
            )
        else:
            if len(scale) != self._num_joints:
                raise ValueError(
                    "Residual action scale array must match resolved joint count: "
                    f"joints={self._num_joints}, scale={len(scale)}"
                )
            self._residual_scale = torch.tensor(
                scale, device=self.device, dtype=torch.float32
            ).unsqueeze(0)
            if torch.any(self._residual_scale <= 0.0):
                raise ValueError("All residual_scale_rad values must be positive")

    @property
    def residual_scale_rad(self) -> torch.Tensor:
        """Per-joint residual action scale in radians."""
        return self._residual_scale

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._previous_bounded_actions[:] = self._bounded_actions
        self._bounded_actions[:] = torch.clamp(actions, -1.0, 1.0)

        self._applied_bounded_actions[:] = torch.where(
            self._delay_one_step.unsqueeze(-1),
            self._previous_bounded_actions,
            self._bounded_actions,
        )

        self._processed_actions[:] = (
            self._default + self._applied_bounded_actions * self._residual_scale
        )
        self._processed_actions[:] = torch.clamp(
            self._processed_actions, self._lower, self._upper
        )


@configclass
class BoundedDefaultCenteredJointPositionActionCfg(ActionTermCfg):
    """Configuration for :class:`BoundedDefaultCenteredJointPositionAction`."""

    class_type: type[ActionTerm] = BoundedDefaultCenteredJointPositionAction

    joint_names: list[str] = MISSING
    lower_limits: list[float] = MISSING
    upper_limits: list[float] = MISSING
    preserve_order: bool = True
    one_step_delay_probability: float = 0.0

@configclass
class BoundedDefaultResidualJointPositionActionCfg(
    BoundedDefaultCenteredJointPositionActionCfg
):
    """Configuration for action contract v3 symmetric residual targets."""

    class_type: type[ActionTerm] = BoundedDefaultResidualJointPositionAction
    residual_scale_rad: float | list[float] = 0.20

