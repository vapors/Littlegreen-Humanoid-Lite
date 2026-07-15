from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos


def randomize_actuator_torque_constant(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    torque_constant_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the friction parameters used in joint friction model.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    # sample joint properties from the given ranges and set into the physics simulation
    # -- friction
    if torque_constant_params is not None:
        for actuator in asset.actuators.values():
            actuator_joint_ids = [joint_id in joint_ids for joint_id in actuator.joint_indices]
            if sum(actuator_joint_ids) > 0:
                stiffness = actuator.stiffness.to(asset.device).clone()
                damping = actuator.damping.to(asset.device).clone()
                scale = _randomize_prop_by_op(
                    torch.ones_like(stiffness, device=asset.device),
                    torque_constant_params,
                    env_ids,
                    actuator_joint_ids,
                    operation=operation,
                    distribution=distribution,
                )
                stiffness[env_ids[:, None], actuator_joint_ids] *= scale[env_ids[:, None], actuator_joint_ids]
                damping[env_ids[:, None], actuator_joint_ids] *= scale[env_ids[:, None], actuator_joint_ids]

                asset.write_joint_stiffness_to_sim(stiffness, joint_ids=actuator.joint_indices, env_ids=env_ids)
                asset.write_joint_damping_to_sim(damping, joint_ids=actuator.joint_indices, env_ids=env_ids)


def set_joint_position_limits(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    lower_limits: list[float],
    upper_limits: list[float],
):
    """Set deterministic hardware joint limits in policy/action order.

    This is a startup event used by the v1.2.3 hardware-aligned tasks so the simulated
    articulation cannot exploit the wider legacy USD/URDF joint envelope.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    if asset_cfg.joint_ids == slice(None):
        joint_ids = torch.arange(asset.num_joints, device=asset.device, dtype=torch.long)
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, device=asset.device, dtype=torch.long)

    if len(joint_ids) != len(lower_limits) or len(joint_ids) != len(upper_limits):
        raise ValueError(
            f"Joint-limit length mismatch: joints={len(joint_ids)}, "
            f"lower={len(lower_limits)}, upper={len(upper_limits)}"
        )

    limits = torch.empty((len(env_ids), len(joint_ids), 2), device=asset.device)
    limits[..., 0] = torch.tensor(lower_limits, device=asset.device)
    limits[..., 1] = torch.tensor(upper_limits, device=asset.device)
    asset.write_joint_position_limit_to_sim(
        limits,
        joint_ids=joint_ids,
        env_ids=env_ids,
        warn_limit_violation=False,
    )
