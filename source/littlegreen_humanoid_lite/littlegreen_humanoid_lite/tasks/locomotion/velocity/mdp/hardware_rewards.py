"""Standing-aware and bounded-action reward terms for v1.2.3 tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from littlegreen_humanoid_lite.tasks.locomotion.velocity.mdp.foundation_phase import (
    command_synchronized_phase_fraction_v24,
    continuous_transfer_phase_fraction_v24,
    transfer_half_cycle_shape_v24,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _standing_mask(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    return torch.linalg.vector_norm(command[:, :3], dim=1) < threshold


def _quiet_standing_mask(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
    max_base_xy_speed_mps: float,
) -> torch.Tensor:
    """Gate symmetry shaping to quasi-static standing, not push recovery."""
    asset: Articulation = env.scene[asset_cfg.name]
    quiet = torch.linalg.vector_norm(asset.data.root_lin_vel_b[:, :2], dim=1) < max_base_xy_speed_mps
    return _standing_mask(env, command_name, command_threshold) & quiet


def bounded_last_action(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    """Return the current bounded normalized action for the policy observation."""
    term = env.action_manager.get_term(action_name)
    return term.bounded_actions


def bounded_action_rate_l2(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    """Squared change in bounded normalized action."""
    term = env.action_manager.get_term(action_name)
    delta = term.bounded_actions - term.previous_bounded_actions
    return torch.sum(torch.square(delta), dim=1)


def bounded_action_l2(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    """Squared bounded normalized action magnitude."""
    action = env.action_manager.get_term(action_name).bounded_actions
    return torch.sum(torch.square(action), dim=1)


def raw_action_excess_l2(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    """Penalize only raw policy output magnitude beyond the valid [-1, 1] contract."""
    raw = env.action_manager.get_term(action_name).raw_actions
    excess = torch.relu(torch.abs(raw) - 1.0)
    return torch.sum(torch.square(excess), dim=1)


def standing_base_xy_speed_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize horizontal base speed only for exact/near standing commands."""
    asset = env.scene[asset_cfg.name]
    mask = _standing_mask(env, command_name, command_threshold)
    value = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    return value * mask


def standing_yaw_rate_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize yaw rotation only for standing commands."""
    asset = env.scene[asset_cfg.name]
    mask = _standing_mask(env, command_name, command_threshold)
    return torch.square(asset.data.root_ang_vel_b[:, 2]) * mask


def standing_default_joint_pose_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize deviation from q_default when commanded to stand."""
    asset: Articulation = env.scene[asset_cfg.name]
    mask = _standing_mask(env, command_name, command_threshold)
    error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(error), dim=1) * mask


def standing_base_height_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    desired_height: float,
    std: float = 0.05,
    command_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward nominal root-body COM height under standing commands.

    The Lilgreen asset's root/link frame is near ground level, so ``root_pos_w[:, 2]``
    is not a useful pelvis/base-height measure.  The physical root-body center of mass
    is represented by ``root_com_pos_w`` and is the quantity shaped here.
    """
    asset = env.scene[asset_cfg.name]
    mask = _standing_mask(env, command_name, command_threshold)
    current_height = asset.data.root_com_pos_w[:, 2]
    error = current_height - desired_height
    reward = torch.exp(-0.5 * torch.square(error / std))
    return reward * mask


def soft_torque_utilization_l2(
    env: ManagerBasedRLEnv,
    torque_limit_nm: float,
    soft_ratio: float = 0.70,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize only torque utilization above a soft fraction of peak torque.

    This complements the small global torque L2 term. Moderate control effort remains
    available for balance, while sustained operation near the ST3215 peak/stall limit
    becomes increasingly expensive.
    """
    if torque_limit_nm <= 0.0:
        raise ValueError("torque_limit_nm must be positive")
    if not 0.0 <= soft_ratio < 1.0:
        raise ValueError("soft_ratio must be in [0, 1)")

    asset: Articulation = env.scene[asset_cfg.name]
    torque = torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids])
    utilization = torque / torque_limit_nm
    excess = torch.relu(utilization - soft_ratio)
    return torch.sum(torch.square(excess), dim=1)


def standing_both_feet_contact(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Reward both feet being in contact during standing."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    in_contact = forces.norm(dim=-1).max(dim=1)[0] > force_threshold
    both_contact = torch.all(in_contact, dim=1)
    return both_contact * _standing_mask(env, command_name, command_threshold)


def standing_feet_slide(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize planted-foot horizontal slip during standing."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > force_threshold
    )
    asset = env.scene[asset_cfg.name]
    body_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    slip = torch.sum(torch.linalg.vector_norm(body_vel_xy, dim=-1) * contacts, dim=1)
    return slip * _standing_mask(env, command_name, command_threshold)



def standing_soft_torque_utilization_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    torque_limit_nm: float,
    soft_ratio: float = 0.70,
    command_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Standing-only soft torque utilization penalty.

    This is useful for Stand checkpoints where we want bent knees but do not want
    a static crouch that sits continuously near the ST3215 torque envelope.
    """
    if torque_limit_nm <= 0.0:
        raise ValueError("torque_limit_nm must be positive")
    if not 0.0 <= soft_ratio < 1.0:
        raise ValueError("soft_ratio must be in [0, 1)")
    asset: Articulation = env.scene[asset_cfg.name]
    torque = torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids])
    utilization = torque / torque_limit_nm
    excess = torch.relu(utilization - soft_ratio)
    return torch.sum(torch.square(excess), dim=1) * _standing_mask(env, command_name, command_threshold)


def standing_contact_force_balance_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    command_threshold: float = 0.05,
    force_threshold: float = 1.0,
    eps: float = 1.0e-6,
    vertical_force_only: bool = False,
    imbalance_deadband: float = 0.0,
    imbalance_scale: float = 0.20,
    max_base_xy_speed_mps: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize left/right support-force imbalance under standing commands.

    ``vertical_force_only`` is preferred for the balanced Stand task so
    tangential forces do not masquerade as weight support.  The three physics
    samples in the contact history are averaged so load sharing reflects
    sustained support rather than unrelated per-foot peak impulses.

    v2.3.2 uses a practical ``imbalance_scale`` instead of normalizing by the
    full theoretical 0--100% imbalance range.  A 15--25% static load mismatch
    therefore produces a useful learning signal.  The term is active whenever
    the pair carries meaningful total support, including the case where one
    foot is almost unloaded; the old both-feet-loaded gate made that exact
    bracing escape route unpenalized.
    """
    if not 0.0 <= imbalance_deadband < 1.0:
        raise ValueError("imbalance_deadband must be in [0, 1)")
    if imbalance_scale <= 0.0:
        raise ValueError("imbalance_scale must be positive")
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_history = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    if vertical_force_only:
        forces = torch.clamp(force_history[..., 2], min=0.0).mean(dim=1)
    else:
        forces = force_history.norm(dim=-1).max(dim=1)[0]
    if forces.shape[1] != 2:
        left = torch.sum(forces[:, 0::2], dim=1)
        right = torch.sum(forces[:, 1::2], dim=1)
    else:
        left, right = forces[:, 0], forces[:, 1]
    total = left + right
    support_loaded = total > force_threshold
    normalized_diff = torch.abs(left - right) / (total + eps)
    normalized_excess = torch.relu(normalized_diff - imbalance_deadband) / imbalance_scale
    # Bound the rare one-foot/impact tail so this term remains a shaping signal
    # rather than an accidental second termination penalty.
    normalized_excess = torch.clamp(normalized_excess, max=2.0)
    quiet_standing = _quiet_standing_mask(
        env, command_name, command_threshold, asset_cfg, max_base_xy_speed_mps
    )
    return torch.square(normalized_excess) * support_loaded * quiet_standing


def standing_com_over_feet_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize standing COM displacement from the midpoint of the two feet.

    This keeps the Stand seed from leaning onto one leg without teaching gait-time
    knee symmetry. Use only on standing tasks.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    foot_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    midpoint_xy = torch.mean(foot_xy, dim=1)
    com_xy = asset.data.root_com_pos_w[:, :2]
    return torch.sum(torch.square(com_xy - midpoint_xy), dim=1) * _standing_mask(env, command_name, command_threshold)


def standing_com_forward_over_feet_band_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    command_threshold: float = 0.05,
    target_forward_m: float = 0.055,
    band_half_width_m: float = 0.010,
    lateral_weight: float = 0.25,
) -> torch.Tensor:
    """Penalize standing COM placement outside a forward band over the foot midpoint.

    The existing ``standing_com_over_feet_l2`` centers the COM over the midpoint of
    both feet. For Lilgreen's lower athletic stand, visual checks showed the robot
    was still slightly rear-biased. This term expresses the COM-to-foot-midpoint
    vector in the yaw-aligned robot frame and prefers a small positive forward
    offset while still lightly damping lateral lean.
    """
    if band_half_width_m < 0.0:
        raise ValueError("band_half_width_m must be non-negative")
    asset: Articulation = env.scene[asset_cfg.name]
    foot_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    midpoint_xy = torch.mean(foot_xy, dim=1)
    delta_w = torch.zeros(asset.data.root_com_pos_w.shape[0], 3, device=asset.data.root_com_pos_w.device)
    delta_w[:, :2] = asset.data.root_com_pos_w[:, :2] - midpoint_xy
    delta_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), delta_w)
    forward_error = delta_yaw[:, 0] - target_forward_m
    forward_excess = torch.relu(torch.abs(forward_error) - band_half_width_m)
    lateral_error = delta_yaw[:, 1]
    value = torch.square(forward_excess) + lateral_weight * torch.square(lateral_error)
    return value * _standing_mask(env, command_name, command_threshold)


def standing_forward_lean_projected_gravity_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_projected_gravity_x: float = 0.052,
    std: float = 0.075,
    command_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward a very small stand-only forward lean using projected gravity.

    Upright is approximately projected_gravity_b = [0, 0, -1]. A small positive
    x component corresponds to a slight forward pitch for the current Lilgreen
    convention. This cue is deliberately weak; COM-over-feet remains the primary
    sagittal placement target.
    """
    if std <= 0.0:
        raise ValueError("std must be positive")
    asset: Articulation = env.scene[asset_cfg.name]
    mask = _standing_mask(env, command_name, command_threshold)
    error = asset.data.projected_gravity_b[:, 0] - target_projected_gravity_x
    return torch.exp(-0.5 * torch.square(error / std)) * mask


def standing_com_lateral_over_feet_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    command_threshold: float = 0.05,
    deadband_m: float = 0.004,
    scale_m: float = 0.025,
    max_base_xy_speed_mps: float = 0.10,
) -> torch.Tensor:
    """Penalize lateral COM offset from the two-foot midpoint while standing.

    The offset is expressed in the yaw-aligned base frame and normalized by
    ``scale_m`` so centimeter-level biases produce a meaningful learning signal.
    """
    if deadband_m < 0.0:
        raise ValueError("deadband_m must be non-negative")
    if scale_m <= 0.0:
        raise ValueError("scale_m must be positive")
    asset: Articulation = env.scene[asset_cfg.name]
    foot_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    midpoint_xy = torch.mean(foot_xy, dim=1)
    delta_w = torch.zeros(asset.data.root_com_pos_w.shape[0], 3, device=asset.data.root_com_pos_w.device)
    delta_w[:, :2] = asset.data.root_com_pos_w[:, :2] - midpoint_xy
    delta_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), delta_w)
    lateral_excess = torch.relu(torch.abs(delta_yaw[:, 1]) - deadband_m) / scale_m
    quiet_standing = _quiet_standing_mask(
        env, command_name, command_threshold, asset_cfg, max_base_xy_speed_mps
    )
    return torch.square(lateral_excess) * quiet_standing


def standing_paired_torque_balance_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    torque_limit_nm: float,
    asset_cfg: SceneEntityCfg,
    command_threshold: float = 0.05,
    utilization_deadband: float = 0.06,
    max_base_xy_speed_mps: float = 0.10,
) -> torch.Tensor:
    """Penalize persistent left/right torque-utilization mismatch while standing.

    ``asset_cfg`` must list matching left joints followed by matching right joints.
    Absolute torque utilization is compared, so mirrored sign conventions do not
    affect the balance objective.
    """
    if torque_limit_nm <= 0.0:
        raise ValueError("torque_limit_nm must be positive")
    if not 0.0 <= utilization_deadband < 1.0:
        raise ValueError("utilization_deadband must be in [0, 1)")
    asset: Articulation = env.scene[asset_cfg.name]
    torque = torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids]) / torque_limit_nm
    if torque.shape[1] == 0 or torque.shape[1] % 2 != 0:
        raise ValueError("standing_paired_torque_balance_l2 requires an even, non-zero joint list")
    half = torque.shape[1] // 2
    mismatch = torch.relu(torch.abs(torque[:, :half] - torque[:, half:]) - utilization_deadband)
    quiet_standing = _quiet_standing_mask(
        env, command_name, command_threshold, asset_cfg, max_base_xy_speed_mps
    )
    return torch.mean(torch.square(mismatch), dim=1) * quiet_standing


def standing_paired_target_residual_balance_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    action_name: str = "joint_pos",
    command_threshold: float = 0.05,
    residual_deadband_rad: float = 0.015,
    scale_rad: float = 0.080,
    max_base_xy_speed_mps: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize left/right mismatch in absolute target residual magnitude.

    The active action contract orders all left-leg joints before the matching
    right-leg joints. Comparing magnitudes avoids assuming identical roll/yaw signs.
    """
    if residual_deadband_rad < 0.0:
        raise ValueError("residual_deadband_rad must be non-negative")
    if scale_rad <= 0.0:
        raise ValueError("scale_rad must be positive")
    term = env.action_manager.get_term(action_name)
    residual = torch.abs(term.processed_actions - term.default_positions)
    if residual.shape[1] == 0 or residual.shape[1] % 2 != 0:
        raise ValueError("standing_paired_target_residual_balance_l2 requires an even action dimension")
    half = residual.shape[1] // 2
    mismatch = torch.relu(torch.abs(residual[:, :half] - residual[:, half:]) - residual_deadband_rad) / scale_rad
    quiet_standing = _quiet_standing_mask(
        env, command_name, command_threshold, asset_cfg, max_base_xy_speed_mps
    )
    return torch.mean(torch.square(mismatch), dim=1) * quiet_standing


def standing_target_limit_margin_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    action_name: str = "joint_pos",
    command_threshold: float = 0.05,
    margin_rad: float = 0.020,
) -> torch.Tensor:
    """Penalize standing targets that sit at or very near a physical joint limit."""
    if margin_rad <= 0.0:
        raise ValueError("margin_rad must be positive")
    term = env.action_manager.get_term(action_name)
    distance = torch.minimum(term.processed_actions - term.lower_limits, term.upper_limits - term.processed_actions)
    margin_violation = torch.relu(margin_rad - distance) / margin_rad
    return torch.sum(torch.square(margin_violation), dim=1) * _standing_mask(env, command_name, command_threshold)




def base_com_height_exp_v24(
    env: ManagerBasedRLEnv,
    desired_height: float,
    std: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward base center-of-mass height for Transfer without command gating."""
    if std <= 0.0:
        raise ValueError("std must be positive")
    asset: Articulation = env.scene[asset_cfg.name]
    error = asset.data.root_com_pos_w[:, 2] - desired_height
    return torch.exp(-0.5 * torch.square(error / std))


# -----------------------------------------------------------------------------
# LittleGreen v2.4 foundation rewards
# -----------------------------------------------------------------------------


def normalized_torque_barrier_l2(
    env: ManagerBasedRLEnv,
    torque_limit_nm: float,
    soft_ratio: float = 0.78,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dimensionless torque barrier with bounded, moderate influence.

    Unlike the earlier unnormalized ``relu(utilization-soft_ratio)^2`` term,
    this maps the interval from the soft threshold to the simulated peak into
    approximately ``[0, 1]``.  Averaging across joints keeps the term useful
    without making torque minimization the primary task objective.
    """
    if torque_limit_nm <= 0.0:
        raise ValueError("torque_limit_nm must be positive")
    if not 0.0 <= soft_ratio < 1.0:
        raise ValueError("soft_ratio must be in [0, 1)")
    asset: Articulation = env.scene[asset_cfg.name]
    utilization = torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids]) / torque_limit_nm
    excess = torch.relu((utilization - soft_ratio) / max(1.0 - soft_ratio, 1.0e-6))
    return torch.mean(torch.square(torch.clamp(excess, max=1.5)), dim=1)


def max_torque_barrier_l2(
    env: ManagerBasedRLEnv,
    torque_limit_nm: float,
    soft_ratio: float = 0.88,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize the single most saturated selected actuator with a normalized barrier.

    The interval from ``soft_ratio`` to the physical torque limit maps to roughly
    ``[0, 1]`` and the maximum across selected joints is returned.  Unlike the
    all-joint mean, a policy cannot hide one permanently saturated brace joint among
    quieter actuators.  This does not change the physical torque limit and imposes no
    joint-position or left/right symmetry target.
    """
    if torque_limit_nm <= 0.0:
        raise ValueError("torque_limit_nm must be positive")
    if not 0.0 <= soft_ratio < 1.0:
        raise ValueError("soft_ratio must be in [0, 1)")
    asset: Articulation = env.scene[asset_cfg.name]
    utilization = torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids]) / torque_limit_nm
    excess = torch.relu((utilization - soft_ratio) / max(1.0 - soft_ratio, 1.0e-6))
    return torch.max(torch.square(torch.clamp(excess, max=1.5)), dim=1).values


def pair_max_torque_barrier_l2(
    env: ManagerBasedRLEnv,
    torque_limit_nm: float,
    pair_indices: tuple[int, ...],
    soft_ratio: float = 0.84,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize the more heavily loaded member of selected mirrored joint pairs.

    This is the locomotion-capable counterpart to
    :func:`standing_pair_max_torque_barrier_l2`.  It is intentionally *not*
    command- or posture-gated: a policy cannot dilute a saturated support joint
    by leaving the other nine or ten joints quiet, nor can it reduce the term by
    loading the mirrored mate.  The physical actuator limit is unchanged; this
    only changes reward economics near saturation.

    ``asset_cfg`` must resolve the canonical 12 action joints in left-six then
    right-six order.  ``pair_indices`` selects indices inside each six-joint leg.
    """
    if torque_limit_nm <= 0.0:
        raise ValueError("torque_limit_nm must be positive")
    if not 0.0 <= soft_ratio < 1.0:
        raise ValueError("soft_ratio must be in [0, 1)")
    asset: Articulation = env.scene[asset_cfg.name]
    torque = torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids])
    if torque.shape[1] != 12:
        raise ValueError("pair_max_torque_barrier_l2 requires 12 ordered action joints")
    left, right = torque[:, :6], torque[:, 6:]
    indices = torch.as_tensor(pair_indices, device=torque.device, dtype=torch.long)
    pair_utilization = torch.maximum(left[:, indices], right[:, indices]) / torque_limit_nm
    excess = torch.relu((pair_utilization - soft_ratio) / max(1.0 - soft_ratio, 1.0e-6))
    return torch.mean(torch.square(torch.clamp(excess, max=1.5)), dim=1)


def standing_pair_max_torque_barrier_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    torque_limit_nm: float,
    pair_indices: tuple[int, ...],
    soft_ratio: float = 0.55,
    command_threshold: float = 0.05,
    max_base_xy_speed_mps: float = 0.12,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize the more heavily loaded member of each mirrored joint pair.

    ``asset_cfg`` must resolve the canonical 12 action joints in left-six then
    right-six order.  Taking the pair maximum prevents the policy from reducing
    a symmetry proxy by increasing effort on the quieter side.
    """
    if torque_limit_nm <= 0.0:
        raise ValueError("torque_limit_nm must be positive")
    if not 0.0 <= soft_ratio < 1.0:
        raise ValueError("soft_ratio must be in [0, 1)")
    asset: Articulation = env.scene[asset_cfg.name]
    torque = torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids])
    if torque.shape[1] != 12:
        raise ValueError("standing_pair_max_torque_barrier_l2 requires 12 ordered action joints")
    left, right = torque[:, :6], torque[:, 6:]
    indices = torch.as_tensor(pair_indices, device=torque.device, dtype=torch.long)
    pair_utilization = torch.maximum(left[:, indices], right[:, indices]) / torque_limit_nm
    excess = torch.relu((pair_utilization - soft_ratio) / max(1.0 - soft_ratio, 1.0e-6))
    quiet = _quiet_standing_mask(
        env, command_name, command_threshold, asset_cfg, max_base_xy_speed_mps
    )
    return torch.mean(torch.square(torch.clamp(excess, max=1.5)), dim=1) * quiet


def standing_mirrored_joint_residual_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    mirror_signs: tuple[float, ...] = (-1.0, -1.0, 1.0, 1.0, 1.0, -1.0),
    std_rad: float = 0.12,
    command_threshold: float = 0.05,
    max_base_xy_speed_mps: float = 0.12,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize left/right posture asymmetry relative to ``q_default``.

    The signs encode the robot's mirror geometry:

    * roll/yaw/ankle-roll residuals mirror with opposite sign;
    * sagittal pitch/knee residuals match with the same sign.

    The normalized mean-square form gives a useful signal while still allowing
    asymmetric recovery motion when the base is moving after a push.
    """
    if std_rad <= 0.0:
        raise ValueError("std_rad must be positive")
    asset: Articulation = env.scene[asset_cfg.name]
    q_rel = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    if q_rel.shape[1] != 12:
        raise ValueError("standing_mirrored_joint_residual_l2 requires 12 ordered action joints")
    signs = torch.as_tensor(mirror_signs, device=q_rel.device, dtype=q_rel.dtype)
    if signs.numel() != 6:
        raise ValueError("mirror_signs must contain six left/right pair signs")
    error = (q_rel[:, :6] - signs * q_rel[:, 6:]) / std_rad
    quiet = _quiet_standing_mask(
        env, command_name, command_threshold, asset_cfg, max_base_xy_speed_mps
    )
    return torch.mean(torch.square(error), dim=1) * quiet


def _phase_fraction_and_moving_v24(
    env: ManagerBasedRLEnv,
    phase_mode: str,
    command_name: str,
    linear_command_threshold: float,
    yaw_command_threshold: float,
    period_s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if phase_mode == "continuous_transfer":
        phase = continuous_transfer_phase_fraction_v24(env, period_s=period_s)
        moving = getattr(env, "_littlegreen_v24_phase_moving", None)
        if not isinstance(moving, torch.Tensor) or moving.shape != phase.shape:
            moving = torch.ones_like(phase, dtype=torch.bool)
        return phase, moving
    if phase_mode == "command_synchronized":
        return command_synchronized_phase_fraction_v24(
            env,
            command_name=command_name,
            linear_command_threshold=linear_command_threshold,
            yaw_command_threshold=yaw_command_threshold,
            period_s=period_s,
        )
    raise ValueError(f"Unsupported v2.4 phase reward mode: {phase_mode}")


def _ordered_vertical_foot_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    history = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    vertical = torch.clamp(history[..., 2], min=0.0).mean(dim=1)
    if vertical.shape[1] < 2:
        raise ValueError("v2.4 phase rewards require ordered left/right foot contacts")
    return vertical[:, 0], vertical[:, 1]


def _transfer_static_mask_v24(
    env: ManagerBasedRLEnv,
    period_s: float = 1.10,
) -> torch.Tensor:
    """Return environments reserved for preserving the canonical Stand skill."""
    phase = continuous_transfer_phase_fraction_v24(env, period_s=period_s)
    moving = getattr(env, "_littlegreen_v24_phase_moving", None)
    if not isinstance(moving, torch.Tensor) or moving.shape != phase.shape:
        moving = torch.ones_like(phase, dtype=torch.bool)
    return ~moving


def transfer_static_both_feet_contact_v24(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    period_s: float = 1.10,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward double support only in Transfer environments reserved for standing."""
    left_force, right_force = _ordered_vertical_foot_forces(env, sensor_cfg)
    both = (left_force > force_threshold) & (right_force > force_threshold)
    return both.float() * _transfer_static_mask_v24(env, period_s=period_s)


def transfer_static_default_pose_l2_v24(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    period_s: float = 1.10,
) -> torch.Tensor:
    """Penalize departure from q_default only in Transfer standing-reserve envs."""
    asset: Articulation = env.scene[asset_cfg.name]
    q_rel = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    return torch.mean(torch.square(q_rel), dim=1) * _transfer_static_mask_v24(
        env, period_s=period_s
    )


def phase_support_load_target_exp_v24(
    env: ManagerBasedRLEnv,
    phase_mode: str,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    period_s: float = 0.90,
    linear_command_threshold: float = 0.08,
    yaw_command_threshold: float = 0.08,
    force_threshold: float = 1.0,
    target_amplitude: float = 0.38,
    std_fraction: float = 0.16,
) -> torch.Tensor:
    """Reward a smooth alternating left/right vertical-load target."""
    phase, moving = _phase_fraction_and_moving_v24(
        env,
        phase_mode,
        command_name,
        linear_command_threshold,
        yaw_command_threshold,
        period_s,
    )
    amplitude = float(getattr(env, "_littlegreen_v24_load_amplitude", target_amplitude))
    active_std = float(getattr(env, "_littlegreen_v24_load_std", std_fraction))
    left_force, right_force = _ordered_vertical_foot_forces(env, sensor_cfg)
    total = left_force + right_force
    left_fraction = left_force / torch.clamp(total, min=1.0e-6)
    left_stance, _, swing_shape = transfer_half_cycle_shape_v24(phase)
    # v2.4.2 correction: peak support belongs at MID-SWING, not at the
    # half-cycle boundary.  The previous cosine target returned to 50/50 at
    # phase 0.25/0.75 exactly when the opposite foot needed to unload.
    stance_fraction_target = 0.5 + amplitude * swing_shape
    target_left = torch.where(left_stance, stance_fraction_target, 1.0 - stance_fraction_target)
    score = torch.exp(-0.5 * torch.square((left_fraction - target_left) / max(active_std, 1.0e-6)))
    return score * (total > force_threshold) * moving


def phase_com_lateral_target_exp_v24(
    env: ManagerBasedRLEnv,
    phase_mode: str,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    period_s: float = 0.90,
    linear_command_threshold: float = 0.08,
    yaw_command_threshold: float = 0.08,
    shift_fraction: float = 0.82,
    std_m: float = 0.030,
) -> torch.Tensor:
    """Reward lateral COM motion toward the scheduled support foot."""
    phase, moving = _phase_fraction_and_moving_v24(
        env,
        phase_mode,
        command_name,
        linear_command_threshold,
        yaw_command_threshold,
        period_s,
    )
    active_shift = float(getattr(env, "_littlegreen_v24_com_shift_fraction", shift_fraction))
    active_std = float(getattr(env, "_littlegreen_v24_com_lateral_std_m", std_m))
    asset: Articulation = env.scene[asset_cfg.name]
    feet = asset.data.body_pos_w[:, asset_cfg.body_ids, :3]
    if feet.shape[1] < 2:
        raise ValueError("v2.4 COM phase reward requires ordered left/right feet")
    midpoint = 0.5 * (feet[:, 0, :] + feet[:, 1, :])
    root_yaw = yaw_quat(asset.data.root_quat_w)
    left_rel = quat_apply_inverse(root_yaw, feet[:, 0, :] - midpoint)
    right_rel = quat_apply_inverse(root_yaw, feet[:, 1, :] - midpoint)
    com_rel = quat_apply_inverse(root_yaw, asset.data.root_com_pos_w - midpoint)
    left_stance, _, swing_shape = transfer_half_cycle_shape_v24(phase)
    scheduled_support_y = torch.where(left_stance, left_rel[:, 1], right_rel[:, 1])
    # Zero lateral target at the double-support boundary; move toward the
    # scheduled stance foot and peak at mid-swing when unloading matters most.
    target_y = active_shift * swing_shape * scheduled_support_y
    score = torch.exp(-0.5 * torch.square((com_rel[:, 1] - target_y) / max(active_std, 1.0e-6)))
    return score * moving


def phase_support_progress_v24(
    env: ManagerBasedRLEnv,
    phase_mode: str,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    period_s: float = 1.20,
    linear_command_threshold: float = 0.08,
    yaw_command_threshold: float = 0.08,
    force_threshold: float = 1.0,
    target_amplitude: float = 0.40,
) -> torch.Tensor:
    """Dense reward for progressively unloading the scheduled swing foot."""
    phase, moving = _phase_fraction_and_moving_v24(
        env, phase_mode, command_name, linear_command_threshold, yaw_command_threshold, period_s
    )
    amplitude = float(getattr(env, "_littlegreen_v24_load_amplitude", target_amplitude))
    left_force, right_force = _ordered_vertical_foot_forces(env, sensor_cfg)
    total = left_force + right_force
    left_fraction = left_force / torch.clamp(total, min=1.0e-6)
    left_stance, _, swing_shape = transfer_half_cycle_shape_v24(phase)
    stance_fraction = torch.where(left_stance, left_fraction, 1.0 - left_fraction)
    target = 0.5 + amplitude * swing_shape
    denom = torch.clamp(target - 0.5, min=0.05)
    progress = torch.clamp((stance_fraction - 0.5) / denom, 0.0, 1.0)
    scale = float(getattr(env, "_littlegreen_v24_support_progress_scale", 1.0))
    return scale * progress * swing_shape * (total > force_threshold) * moving


def phase_com_lateral_progress_v24(
    env: ManagerBasedRLEnv,
    phase_mode: str,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    period_s: float = 1.20,
    linear_command_threshold: float = 0.08,
    yaw_command_threshold: float = 0.08,
    shift_fraction: float = 0.90,
) -> torch.Tensor:
    """Dense reward for moving COM from midpoint toward scheduled stance foot."""
    phase, moving = _phase_fraction_and_moving_v24(
        env, phase_mode, command_name, linear_command_threshold, yaw_command_threshold, period_s
    )
    active_shift = float(getattr(env, "_littlegreen_v24_com_shift_fraction", shift_fraction))
    asset: Articulation = env.scene[asset_cfg.name]
    feet = asset.data.body_pos_w[:, asset_cfg.body_ids, :3]
    midpoint = 0.5 * (feet[:, 0, :] + feet[:, 1, :])
    root_yaw = yaw_quat(asset.data.root_quat_w)
    left_rel = quat_apply_inverse(root_yaw, feet[:, 0, :] - midpoint)
    right_rel = quat_apply_inverse(root_yaw, feet[:, 1, :] - midpoint)
    com_rel = quat_apply_inverse(root_yaw, asset.data.root_com_pos_w - midpoint)
    left_stance, _, swing_shape = transfer_half_cycle_shape_v24(phase)
    stance_y = torch.where(left_stance, left_rel[:, 1], right_rel[:, 1])
    direction = torch.sign(stance_y)
    target_distance = torch.clamp(active_shift * swing_shape * torch.abs(stance_y), min=0.005)
    signed_com = com_rel[:, 1] * direction
    progress = torch.clamp(signed_com / target_distance, 0.0, 1.0)
    scale = float(getattr(env, "_littlegreen_v24_com_progress_scale", 1.0))
    return scale * progress * swing_shape * moving


def _phase_swing_state_v24(
    env: ManagerBasedRLEnv,
    phase_mode: str,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    period_s: float,
    linear_command_threshold: float,
    yaw_command_threshold: float,
) -> dict[str, torch.Tensor]:
    phase, moving = _phase_fraction_and_moving_v24(
        env,
        phase_mode,
        command_name,
        linear_command_threshold,
        yaw_command_threshold,
        period_s,
    )
    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :3]
    foot_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :3]
    if foot_pos.shape[1] < 2:
        raise ValueError("v2.4 phase rewards require ordered left/right feet")
    left_stance, half_progress, _ = transfer_half_cycle_shape_v24(phase)
    stance_pos = torch.where(left_stance.unsqueeze(-1), foot_pos[:, 0, :], foot_pos[:, 1, :])
    swing_pos = torch.where(left_stance.unsqueeze(-1), foot_pos[:, 1, :], foot_pos[:, 0, :])
    stance_vel = torch.where(left_stance.unsqueeze(-1), foot_vel[:, 0, :], foot_vel[:, 1, :])
    swing_vel = torch.where(left_stance.unsqueeze(-1), foot_vel[:, 1, :], foot_vel[:, 0, :])
    return {
        "phase": phase,
        "moving": moving,
        "left_stance": left_stance,
        "stance_pos": stance_pos,
        "swing_pos": swing_pos,
        "stance_vel": stance_vel,
        "swing_vel": swing_vel,
        "half_progress": half_progress,
    }


def phase_swing_clearance_trajectory_v24(
    env: ManagerBasedRLEnv,
    phase_mode: str,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    period_s: float = 0.90,
    linear_command_threshold: float = 0.08,
    yaw_command_threshold: float = 0.08,
    target_clearance_m: float = 0.025,
    clearance_std_m: float = 0.012,
) -> torch.Tensor:
    """Reward explicit zero-at-boundary, peak-at-mid-swing foot clearance."""
    data = _phase_swing_state_v24(
        env,
        phase_mode,
        command_name,
        asset_cfg,
        period_s,
        linear_command_threshold,
        yaw_command_threshold,
    )
    active_target = float(getattr(env, "_littlegreen_v24_clearance_m", target_clearance_m))
    active_std = float(getattr(env, "_littlegreen_v24_clearance_std_m", clearance_std_m))
    shape = torch.sin(torch.pi * data["half_progress"])
    reference = active_target * shape
    clearance = data["swing_pos"][:, 2] - data["stance_pos"][:, 2]
    raw = torch.exp(-0.5 * torch.square((clearance - reference) / max(active_std, 1.0e-6)))
    zero = torch.exp(-0.5 * torch.square(reference / max(active_std, 1.0e-6)))
    normalized = torch.clamp((raw - zero) / torch.clamp(1.0 - zero, min=1.0e-5), 0.0, 1.0)
    enabled = float(getattr(env, "_littlegreen_v24_clearance_scale", 1.0))
    return enabled * normalized * shape * data["moving"]


def phase_contact_schedule_reward_v24(
    env: ManagerBasedRLEnv,
    phase_mode: str,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    period_s: float = 0.90,
    linear_command_threshold: float = 0.08,
    yaw_command_threshold: float = 0.08,
    transition_fraction: float = 0.12,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward double support near boundaries and scheduled single support mid-swing."""
    data = _phase_swing_state_v24(
        env,
        phase_mode,
        command_name,
        asset_cfg,
        period_s,
        linear_command_threshold,
        yaw_command_threshold,
    )
    left_force, right_force = _ordered_vertical_foot_forces(env, sensor_cfg)
    left_contact = left_force > force_threshold
    right_contact = right_force > force_threshold
    both = left_contact & right_contact
    none = (~left_contact) & (~right_contact)
    stance_only = torch.where(
        data["left_stance"],
        left_contact & (~right_contact),
        right_contact & (~left_contact),
    )
    wrong_only = torch.where(
        data["left_stance"],
        right_contact & (~left_contact),
        left_contact & (~right_contact),
    )
    edge = torch.minimum(data["half_progress"], 1.0 - data["half_progress"])
    transition = edge < transition_fraction
    single_scale = float(getattr(env, "_littlegreen_v24_single_support_scale", 1.0))
    transition_score = torch.where(
        both,
        torch.ones_like(data["phase"]),
        torch.where(stance_only, torch.full_like(data["phase"], 0.2), torch.full_like(data["phase"], -0.8)),
    )
    swing_score = (
        (1.0 - single_scale) * both.float()
        + single_scale * stance_only.float()
        - wrong_only.float()
        - none.float()
    )
    score = torch.where(transition, transition_score, swing_score)
    return score * data["moving"]


def phase_swing_progress_along_command_v24(
    env: ManagerBasedRLEnv,
    phase_mode: str,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    period_s: float = 0.90,
    linear_command_threshold: float = 0.08,
    yaw_command_threshold: float = 0.08,
) -> torch.Tensor:
    """Reward scheduled swing-foot velocity in the horizontal command direction."""
    data = _phase_swing_state_v24(
        env,
        phase_mode,
        command_name,
        asset_cfg,
        period_s,
        linear_command_threshold,
        yaw_command_threshold,
    )
    command = env.command_manager.get_command(command_name)[:, :2]
    norm = torch.linalg.vector_norm(command, dim=1)
    direction = command / torch.clamp(norm.unsqueeze(-1), min=1.0e-6)
    asset: Articulation = env.scene[asset_cfg.name]
    root_yaw = yaw_quat(asset.data.root_quat_w)
    relative_velocity = data["swing_vel"] - data["stance_vel"]
    relative_velocity_yaw = quat_apply_inverse(root_yaw, relative_velocity)[:, :2]
    progress = torch.relu(torch.sum(relative_velocity_yaw * direction, dim=1))
    shape = torch.sin(torch.pi * data["half_progress"])
    stage_scale = float(getattr(env, "_littlegreen_v24_swing_progress_scale", 1.0))
    return stage_scale * progress * shape * (norm > linear_command_threshold) * data["moving"]


def phase_stance_foot_slide_l2_v24(
    env: ManagerBasedRLEnv,
    phase_mode: str,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    period_s: float = 0.90,
    linear_command_threshold: float = 0.08,
    yaw_command_threshold: float = 0.08,
    force_threshold: float = 1.0,
    deadzone_m_s: float = 0.025,
) -> torch.Tensor:
    """Penalize horizontal slip of the scheduled loaded stance foot."""
    data = _phase_swing_state_v24(
        env,
        phase_mode,
        command_name,
        asset_cfg,
        period_s,
        linear_command_threshold,
        yaw_command_threshold,
    )
    left_force, right_force = _ordered_vertical_foot_forces(env, sensor_cfg)
    stance_loaded = torch.where(data["left_stance"], left_force, right_force) > force_threshold
    stance_speed = torch.linalg.vector_norm(data["stance_vel"][:, :2], dim=1)
    return torch.square(torch.relu(stance_speed - deadzone_m_s)) * stance_loaded * data["moving"]

# -----------------------------------------------------------------------------
# LittleGreen v2.4.3 whole-foot Transfer/March refinements
# -----------------------------------------------------------------------------


def _whole_foot_vertical_forces_v243(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ordered component and per-side vertical forces for main-foot+toe contact.

    ``sensor_cfg`` must resolve exactly in this order:
    ``left ankle_roll, left toe_pitch, right ankle_roll, right toe_pitch``.
    The component tensor is returned as ``[L_main, L_toe, R_main, R_toe]``.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    history = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    vertical = torch.clamp(history[..., 2], min=0.0).mean(dim=1)
    if vertical.shape[1] != 4:
        raise ValueError(
            "v2.4.3 whole-foot rewards require four ordered bodies: "
            "L_main, L_toe, R_main, R_toe"
        )
    left = vertical[:, 0] + vertical[:, 1]
    right = vertical[:, 2] + vertical[:, 3]
    return vertical, left, right


def _whole_foot_kinematics_v243(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> dict[str, torch.Tensor]:
    """Return ordered main/toe positions, velocities, and side centers.

    ``asset_cfg`` must resolve exactly in this order:
    ``left ankle_roll, left toe_pitch, right ankle_roll, right toe_pitch``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :3]
    vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :3]
    if pos.shape[1] != 4:
        raise ValueError(
            "v2.4.3 whole-foot kinematics require four ordered bodies: "
            "L_main, L_toe, R_main, R_toe"
        )
    left_center = 0.5 * (pos[:, 0, :] + pos[:, 1, :])
    right_center = 0.5 * (pos[:, 2, :] + pos[:, 3, :])
    left_center_vel = 0.5 * (vel[:, 0, :] + vel[:, 1, :])
    right_center_vel = 0.5 * (vel[:, 2, :] + vel[:, 3, :])
    return {
        "pos": pos,
        "vel": vel,
        "left_main_pos": pos[:, 0, :],
        "left_toe_pos": pos[:, 1, :],
        "right_main_pos": pos[:, 2, :],
        "right_toe_pos": pos[:, 3, :],
        "left_center": left_center,
        "right_center": right_center,
        "left_center_vel": left_center_vel,
        "right_center_vel": right_center_vel,
    }


def _phase_whole_foot_state_v243(
    env: ManagerBasedRLEnv,
    phase_mode: str,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    period_s: float,
    linear_command_threshold: float,
    yaw_command_threshold: float,
) -> dict[str, torch.Tensor]:
    """Whole-foot version of the v2.4 scheduled swing state."""
    phase, moving = _phase_fraction_and_moving_v24(
        env,
        phase_mode,
        command_name,
        linear_command_threshold,
        yaw_command_threshold,
        period_s,
    )
    kin = _whole_foot_kinematics_v243(env, asset_cfg)
    left_stance, half_progress, swing_shape = transfer_half_cycle_shape_v24(phase)

    def choose(left: torch.Tensor, right: torch.Tensor, stance: bool) -> torch.Tensor:
        selector = left_stance.unsqueeze(-1)
        if stance:
            return torch.where(selector, left, right)
        return torch.where(selector, right, left)

    return {
        "phase": phase,
        "moving": moving,
        "left_stance": left_stance,
        "half_progress": half_progress,
        "swing_shape": swing_shape,
        "left_center": kin["left_center"],
        "right_center": kin["right_center"],
        "stance_center": choose(kin["left_center"], kin["right_center"], True),
        "swing_center": choose(kin["left_center"], kin["right_center"], False),
        "stance_center_vel": choose(kin["left_center_vel"], kin["right_center_vel"], True),
        "swing_center_vel": choose(kin["left_center_vel"], kin["right_center_vel"], False),
        "stance_main_pos": choose(kin["left_main_pos"], kin["right_main_pos"], True),
        "swing_main_pos": choose(kin["left_main_pos"], kin["right_main_pos"], False),
        "stance_toe_pos": choose(kin["left_toe_pos"], kin["right_toe_pos"], True),
        "swing_toe_pos": choose(kin["left_toe_pos"], kin["right_toe_pos"], False),
    }


def transfer_static_both_whole_feet_contact_v243(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    period_s: float = 1.20,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Preserve double support in the reserved Stand subset using whole-foot contact."""
    _, left_force, right_force = _whole_foot_vertical_forces_v243(env, sensor_cfg)
    both = (left_force > force_threshold) & (right_force > force_threshold)
    return both.float() * _transfer_static_mask_v24(env, period_s=period_s)


def phase_support_load_target_exp_v243(
    env: ManagerBasedRLEnv,
    phase_mode: str,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    period_s: float = 1.20,
    linear_command_threshold: float = 0.08,
    yaw_command_threshold: float = 0.08,
    force_threshold: float = 1.0,
    target_amplitude: float = 0.38,
    std_fraction: float = 0.16,
) -> torch.Tensor:
    """v2.4.2 load target using summed main-foot+toe vertical forces."""
    phase, moving = _phase_fraction_and_moving_v24(
        env, phase_mode, command_name, linear_command_threshold, yaw_command_threshold, period_s
    )
    amplitude = float(getattr(env, "_littlegreen_v24_load_amplitude", target_amplitude))
    active_std = float(getattr(env, "_littlegreen_v24_load_std", std_fraction))
    _, left_force, right_force = _whole_foot_vertical_forces_v243(env, sensor_cfg)
    total = left_force + right_force
    left_fraction = left_force / torch.clamp(total, min=1.0e-6)
    left_stance, _, swing_shape = transfer_half_cycle_shape_v24(phase)
    stance_fraction_target = 0.5 + amplitude * swing_shape
    target_left = torch.where(left_stance, stance_fraction_target, 1.0 - stance_fraction_target)
    score = torch.exp(
        -0.5 * torch.square((left_fraction - target_left) / max(active_std, 1.0e-6))
    )
    return score * (total > force_threshold) * moving


def phase_support_progress_v243(
    env: ManagerBasedRLEnv,
    phase_mode: str,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    period_s: float = 1.20,
    linear_command_threshold: float = 0.08,
    yaw_command_threshold: float = 0.08,
    force_threshold: float = 1.0,
    target_amplitude: float = 0.40,
) -> torch.Tensor:
    """Dense whole-foot load-transfer progress toward the scheduled stance side."""
    phase, moving = _phase_fraction_and_moving_v24(
        env, phase_mode, command_name, linear_command_threshold, yaw_command_threshold, period_s
    )
    amplitude = float(getattr(env, "_littlegreen_v24_load_amplitude", target_amplitude))
    _, left_force, right_force = _whole_foot_vertical_forces_v243(env, sensor_cfg)
    total = left_force + right_force
    left_fraction = left_force / torch.clamp(total, min=1.0e-6)
    left_stance, _, swing_shape = transfer_half_cycle_shape_v24(phase)
    stance_fraction = torch.where(left_stance, left_fraction, 1.0 - left_fraction)
    target = 0.5 + amplitude * swing_shape
    denom = torch.clamp(target - 0.5, min=0.05)
    progress = torch.clamp((stance_fraction - 0.5) / denom, 0.0, 1.0)
    scale = float(getattr(env, "_littlegreen_v24_support_progress_scale", 1.0))
    return scale * progress * swing_shape * (total > force_threshold) * moving


def phase_com_lateral_target_exp_v243(
    env: ManagerBasedRLEnv,
    phase_mode: str,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    period_s: float = 1.20,
    linear_command_threshold: float = 0.08,
    yaw_command_threshold: float = 0.08,
    shift_fraction: float = 0.82,
    std_m: float = 0.030,
) -> torch.Tensor:
    """Lateral COM target referenced to whole-foot centers."""
    phase, moving = _phase_fraction_and_moving_v24(
        env, phase_mode, command_name, linear_command_threshold, yaw_command_threshold, period_s
    )
    active_shift = float(getattr(env, "_littlegreen_v24_com_shift_fraction", shift_fraction))
    active_std = float(getattr(env, "_littlegreen_v24_com_lateral_std_m", std_m))
    asset: Articulation = env.scene[asset_cfg.name]
    kin = _whole_foot_kinematics_v243(env, asset_cfg)
    midpoint = 0.5 * (kin["left_center"] + kin["right_center"])
    root_yaw = yaw_quat(asset.data.root_quat_w)
    left_rel = quat_apply_inverse(root_yaw, kin["left_center"] - midpoint)
    right_rel = quat_apply_inverse(root_yaw, kin["right_center"] - midpoint)
    com_rel = quat_apply_inverse(root_yaw, asset.data.root_com_pos_w - midpoint)
    left_stance, _, swing_shape = transfer_half_cycle_shape_v24(phase)
    scheduled_support_y = torch.where(left_stance, left_rel[:, 1], right_rel[:, 1])
    target_y = active_shift * swing_shape * scheduled_support_y
    score = torch.exp(
        -0.5 * torch.square((com_rel[:, 1] - target_y) / max(active_std, 1.0e-6))
    )
    return score * moving


def phase_com_lateral_progress_v243(
    env: ManagerBasedRLEnv,
    phase_mode: str,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    period_s: float = 1.20,
    linear_command_threshold: float = 0.08,
    yaw_command_threshold: float = 0.08,
    shift_fraction: float = 0.90,
) -> torch.Tensor:
    """Dense lateral COM progress referenced to whole-foot centers."""
    phase, moving = _phase_fraction_and_moving_v24(
        env, phase_mode, command_name, linear_command_threshold, yaw_command_threshold, period_s
    )
    active_shift = float(getattr(env, "_littlegreen_v24_com_shift_fraction", shift_fraction))
    asset: Articulation = env.scene[asset_cfg.name]
    kin = _whole_foot_kinematics_v243(env, asset_cfg)
    midpoint = 0.5 * (kin["left_center"] + kin["right_center"])
    root_yaw = yaw_quat(asset.data.root_quat_w)
    left_rel = quat_apply_inverse(root_yaw, kin["left_center"] - midpoint)
    right_rel = quat_apply_inverse(root_yaw, kin["right_center"] - midpoint)
    com_rel = quat_apply_inverse(root_yaw, asset.data.root_com_pos_w - midpoint)
    left_stance, _, swing_shape = transfer_half_cycle_shape_v24(phase)
    stance_y = torch.where(left_stance, left_rel[:, 1], right_rel[:, 1])
    direction = torch.sign(stance_y)
    target_distance = torch.clamp(active_shift * swing_shape * torch.abs(stance_y), min=0.005)
    signed_com = com_rel[:, 1] * direction
    progress = torch.clamp(signed_com / target_distance, 0.0, 1.0)
    scale = float(getattr(env, "_littlegreen_v24_com_progress_scale", 1.0))
    return scale * progress * swing_shape * moving


def phase_whole_foot_contact_schedule_reward_v243(
    env: ManagerBasedRLEnv,
    phase_mode: str,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    period_s: float = 1.20,
    linear_command_threshold: float = 0.08,
    yaw_command_threshold: float = 0.08,
    transition_fraction: float = 0.14,
    force_threshold: float = 1.0,
    heel_only_mid_swing_credit: float = 0.25,
) -> torch.Tensor:
    """Reward true whole-foot single support while giving heel-only lift partial credit.

    Mid-swing counts as true single support only when both the swing main-foot body
    and swing toe body are unloaded.  A main-foot lift with toe contact remaining
    receives limited shaping credit so the policy can discover toe release without
    making the old heel-only solution optimal.
    """
    data = _phase_whole_foot_state_v243(
        env, phase_mode, command_name, asset_cfg, period_s,
        linear_command_threshold, yaw_command_threshold,
    )
    component, left_force, right_force = _whole_foot_vertical_forces_v243(env, sensor_cfg)
    component_contact = component > force_threshold
    left_main, left_toe, right_main, right_toe = [component_contact[:, i] for i in range(4)]
    left_whole = left_main | left_toe
    right_whole = right_main | right_toe
    both_whole = left_whole & right_whole
    none_whole = (~left_whole) & (~right_whole)

    stance_whole = torch.where(data["left_stance"], left_whole, right_whole)
    swing_whole = torch.where(data["left_stance"], right_whole, left_whole)
    swing_main = torch.where(data["left_stance"], right_main, left_main)
    swing_toe = torch.where(data["left_stance"], right_toe, left_toe)
    stance_only = stance_whole & (~swing_whole)
    wrong_only = (~stance_whole) & swing_whole
    heel_only = stance_whole & (~swing_main) & swing_toe

    edge = torch.minimum(data["half_progress"], 1.0 - data["half_progress"])
    transition = edge < transition_fraction
    single_scale = float(getattr(env, "_littlegreen_v24_single_support_scale", 1.0))

    transition_score = torch.where(
        both_whole,
        torch.ones_like(data["phase"]),
        torch.where(
            stance_only,
            torch.full_like(data["phase"], 0.2),
            torch.full_like(data["phase"], -0.8),
        ),
    )
    double_credit = 1.0 - single_scale
    heel_credit = double_credit + single_scale * float(heel_only_mid_swing_credit)
    swing_score = (
        single_scale * stance_only.float()
        + heel_credit * heel_only.float()
        + double_credit * (both_whole & (~heel_only)).float()
        - wrong_only.float()
        - none_whole.float()
    )
    score = torch.where(transition, transition_score, swing_score)
    return score * data["moving"]


def _phase_component_clearance_trajectory_v243(
    env: ManagerBasedRLEnv,
    phase_mode: str,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    component: str,
    period_s: float,
    linear_command_threshold: float,
    yaw_command_threshold: float,
    target_clearance_m: float,
    clearance_std_m: float,
    target_attr: str,
    std_attr: str,
    scale_attr: str,
) -> torch.Tensor:
    data = _phase_whole_foot_state_v243(
        env, phase_mode, command_name, asset_cfg, period_s,
        linear_command_threshold, yaw_command_threshold,
    )
    active_target = float(getattr(env, target_attr, target_clearance_m))
    active_std = float(getattr(env, std_attr, clearance_std_m))
    shape = data["swing_shape"]
    reference = active_target * shape
    if component == "main":
        clearance = data["swing_main_pos"][:, 2] - data["stance_main_pos"][:, 2]
    elif component == "toe":
        clearance = data["swing_toe_pos"][:, 2] - data["stance_toe_pos"][:, 2]
    else:
        raise ValueError(f"Unknown v2.4.3 clearance component: {component}")
    raw = torch.exp(-0.5 * torch.square((clearance - reference) / max(active_std, 1.0e-6)))
    zero = torch.exp(-0.5 * torch.square(reference / max(active_std, 1.0e-6)))
    normalized = torch.clamp((raw - zero) / torch.clamp(1.0 - zero, min=1.0e-5), 0.0, 1.0)
    enabled = float(getattr(env, scale_attr, 1.0))
    return enabled * normalized * shape * data["moving"]


def phase_main_foot_clearance_trajectory_v243(
    env: ManagerBasedRLEnv,
    phase_mode: str,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    period_s: float = 1.20,
    linear_command_threshold: float = 0.08,
    yaw_command_threshold: float = 0.08,
    target_clearance_m: float = 0.035,
    clearance_std_m: float = 0.016,
) -> torch.Tensor:
    """Reward 25-40 mm-class main-foot clearance at mature Transfer stages."""
    return _phase_component_clearance_trajectory_v243(
        env, phase_mode, command_name, asset_cfg, "main", period_s,
        linear_command_threshold, yaw_command_threshold,
        target_clearance_m, clearance_std_m,
        "_littlegreen_v243_main_clearance_m",
        "_littlegreen_v243_main_clearance_std_m",
        "_littlegreen_v243_main_clearance_scale",
    )


def phase_toe_clearance_trajectory_v243(
    env: ManagerBasedRLEnv,
    phase_mode: str,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    period_s: float = 1.20,
    linear_command_threshold: float = 0.08,
    yaw_command_threshold: float = 0.08,
    target_clearance_m: float = 0.025,
    clearance_std_m: float = 0.014,
) -> torch.Tensor:
    """Reward 15-30 mm-class passive-toe clearance at mature Transfer stages."""
    return _phase_component_clearance_trajectory_v243(
        env, phase_mode, command_name, asset_cfg, "toe", period_s,
        linear_command_threshold, yaw_command_threshold,
        target_clearance_m, clearance_std_m,
        "_littlegreen_v243_toe_clearance_m",
        "_littlegreen_v243_toe_clearance_std_m",
        "_littlegreen_v243_toe_clearance_scale",
    )


def transfer_sagittal_com_readiness_band_l2_v243(
    env: ManagerBasedRLEnv,
    phase_mode: str,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    period_s: float = 1.20,
    linear_command_threshold: float = 0.08,
    yaw_command_threshold: float = 0.08,
    min_forward_m: float = -0.035,
    max_forward_m: float = 0.010,
    scale_m: float = 0.025,
) -> torch.Tensor:
    """Weak band penalty that discourages starting Transfer too far behind the feet."""
    _, moving = _phase_fraction_and_moving_v24(
        env, phase_mode, command_name, linear_command_threshold, yaw_command_threshold, period_s
    )
    asset: Articulation = env.scene[asset_cfg.name]
    kin = _whole_foot_kinematics_v243(env, asset_cfg)
    midpoint = 0.5 * (kin["left_center"] + kin["right_center"])
    com_rel = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_com_pos_w - midpoint)
    behind = torch.relu(float(min_forward_m) - com_rel[:, 0])
    ahead = torch.relu(com_rel[:, 0] - float(max_forward_m))
    scale = max(float(scale_m), 1.0e-6)
    return (torch.square(behind / scale) + torch.square(ahead / scale)) * moving


def phase_zero_command_foot_alignment_l2_v243(
    env: ManagerBasedRLEnv,
    phase_mode: str,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    period_s: float = 1.20,
    linear_command_threshold: float = 0.08,
    yaw_command_threshold: float = 0.08,
    zero_translation_threshold_m_s: float = 0.025,
    zero_yaw_threshold_rad_s: float = 0.04,
    transition_fraction: float = 0.18,
    deadband_m: float = 0.018,
    scale_m: float = 0.040,
) -> torch.Tensor:
    """Subtly align feet fore/aft only at in-place double-support transitions.

    This term is intentionally inactive during directional commands and during the
    middle of swing.  It therefore discourages a persistent preferred lead leg in
    zero-command marching without suppressing legitimate fore/aft placement in Walk.
    """
    phase, moving = _phase_fraction_and_moving_v24(
        env, phase_mode, command_name, linear_command_threshold, yaw_command_threshold, period_s
    )
    kin = _whole_foot_kinematics_v243(env, asset_cfg)
    asset: Articulation = env.scene[asset_cfg.name]
    delta = kin["left_center"] - kin["right_center"]
    delta_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), delta)
    fore_aft_sep = torch.abs(delta_yaw[:, 0])
    excess = torch.relu(fore_aft_sep - float(deadband_m)) / max(float(scale_m), 1.0e-6)

    command = env.command_manager.get_command(command_name)
    zero_translation = torch.linalg.vector_norm(command[:, :2], dim=1) <= float(zero_translation_threshold_m_s)
    zero_yaw = torch.abs(command[:, 2]) <= float(zero_yaw_threshold_rad_s)
    _, half_progress, _ = transfer_half_cycle_shape_v24(phase)
    edge = torch.minimum(half_progress, 1.0 - half_progress)
    transition = edge < float(transition_fraction)
    return torch.square(excess) * moving * zero_translation * zero_yaw * transition


def phase_stance_whole_foot_slide_l2_v243(
    env: ManagerBasedRLEnv,
    phase_mode: str,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    period_s: float = 1.20,
    linear_command_threshold: float = 0.08,
    yaw_command_threshold: float = 0.08,
    force_threshold: float = 1.0,
    deadzone_m_s: float = 0.030,
) -> torch.Tensor:
    """Penalize sliding of the scheduled stance whole-foot center."""
    data = _phase_whole_foot_state_v243(
        env, phase_mode, command_name, asset_cfg, period_s,
        linear_command_threshold, yaw_command_threshold,
    )
    _, left_force, right_force = _whole_foot_vertical_forces_v243(env, sensor_cfg)
    stance_loaded = torch.where(data["left_stance"], left_force, right_force) > force_threshold
    stance_speed = torch.linalg.vector_norm(data["stance_center_vel"][:, :2], dim=1)
    return torch.square(torch.relu(stance_speed - deadzone_m_s)) * stance_loaded * data["moving"]

# -----------------------------------------------------------------------------
# LittleGreen v2.5.0 canonical Walk acquisition refinements
# -----------------------------------------------------------------------------


def phase_swing_placement_along_command_exp_v250(
    env: ManagerBasedRLEnv,
    phase_mode: str,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    period_s: float = 1.12,
    linear_command_threshold: float = 0.05,
    yaw_command_threshold: float = 0.06,
    step_length_fraction: float = 0.32,
    min_step_m: float = 0.025,
    max_step_m: float = 0.10,
    placement_std_m: float = 0.045,
) -> torch.Tensor:
    """Reward late-swing whole-foot placement in the planar command direction.

    This is deliberately stateless.  The desired separation is proportional to
    commanded planar speed and the current gait period, then clipped to a
    physically modest range.  The target ramps monotonically through each swing
    half-cycle, so the swing foot is encouraged to finish ahead of the stance
    foot rather than merely move quickly for an instant.
    """
    data = _phase_whole_foot_state_v243(
        env, phase_mode, command_name, asset_cfg, period_s,
        linear_command_threshold, yaw_command_threshold,
    )
    command = env.command_manager.get_command(command_name)[:, :2]
    speed = torch.linalg.vector_norm(command, dim=1)
    direction = command / torch.clamp(speed.unsqueeze(-1), min=1.0e-6)
    asset: Articulation = env.scene[asset_cfg.name]
    root_yaw = yaw_quat(asset.data.root_quat_w)
    relative = quat_apply_inverse(root_yaw, data["swing_center"] - data["stance_center"])[:, :2]
    along = torch.sum(relative * direction, dim=1)

    active_period = float(getattr(env, "_littlegreen_v24_phase_period_s", period_s))
    active_fraction = float(getattr(env, "_littlegreen_v250_step_length_fraction", step_length_fraction))
    active_max = float(getattr(env, "_littlegreen_v250_step_length_max_m", max_step_m))
    active_std = float(getattr(env, "_littlegreen_v250_placement_std_m", placement_std_m))
    desired_step = torch.clamp(
        speed * active_period * active_fraction,
        min=float(min_step_m),
        max=active_max,
    )
    landing_shape = torch.sin(0.5 * torch.pi * data["half_progress"])
    reference = desired_step * landing_shape
    score = torch.exp(-0.5 * torch.square((along - reference) / max(active_std, 1.0e-6)))
    enabled = float(getattr(env, "_littlegreen_v250_placement_scale", 1.0))
    return enabled * score * landing_shape * (speed > linear_command_threshold) * data["moving"]


def walk_sagittal_com_readiness_band_l2_v250(
    env: ManagerBasedRLEnv,
    phase_mode: str,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    period_s: float = 1.12,
    linear_command_threshold: float = 0.05,
    yaw_command_threshold: float = 0.06,
    min_forward_m: float = -0.045,
    max_forward_m: float = 0.025,
    scale_m: float = 0.030,
) -> torch.Tensor:
    """Weak moving COM band; it does not prescribe body pitch or an exact COM target."""
    _, moving = _phase_fraction_and_moving_v24(
        env, phase_mode, command_name, linear_command_threshold, yaw_command_threshold, period_s
    )
    asset: Articulation = env.scene[asset_cfg.name]
    kin = _whole_foot_kinematics_v243(env, asset_cfg)
    midpoint = 0.5 * (kin["left_center"] + kin["right_center"])
    com_rel = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_com_pos_w - midpoint)
    active_min = float(getattr(env, "_littlegreen_v250_sagittal_min_m", min_forward_m))
    active_max = float(getattr(env, "_littlegreen_v250_sagittal_max_m", max_forward_m))
    behind = torch.relu(active_min - com_rel[:, 0])
    ahead = torch.relu(com_rel[:, 0] - active_max)
    scale = max(float(scale_m), 1.0e-6)
    return (torch.square(behind / scale) + torch.square(ahead / scale)) * moving


def phase_lateral_foot_crossing_l2_v250(
    env: ManagerBasedRLEnv,
    phase_mode: str,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    period_s: float = 1.12,
    linear_command_threshold: float = 0.05,
    yaw_command_threshold: float = 0.06,
    min_lateral_separation_m: float = 0.035,
    scale_m: float = 0.030,
) -> torch.Tensor:
    """Discourage left/right foot crossing while still allowing stance width to vary."""
    _, moving = _phase_fraction_and_moving_v24(
        env, phase_mode, command_name, linear_command_threshold, yaw_command_threshold, period_s
    )
    asset: Articulation = env.scene[asset_cfg.name]
    kin = _whole_foot_kinematics_v243(env, asset_cfg)
    root_yaw = yaw_quat(asset.data.root_quat_w)
    delta = quat_apply_inverse(root_yaw, kin["left_center"] - kin["right_center"])
    lateral_ordered_sep = delta[:, 1]
    excess = torch.relu(float(min_lateral_separation_m) - lateral_ordered_sep)
    return torch.square(excess / max(float(scale_m), 1.0e-6)) * moving


def standing_both_whole_feet_contact_v250(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Standing contact preservation where either main-foot or toe contact counts per side."""
    _, left_force, right_force = _whole_foot_vertical_forces_v243(env, sensor_cfg)
    command = env.command_manager.get_command(command_name)
    standing = torch.linalg.vector_norm(command, dim=1) <= float(command_threshold)
    return ((left_force > force_threshold) & (right_force > force_threshold)).float() * standing


# -----------------------------------------------------------------------------
# LittleGreen v2.5.3 open-gait floor rewards
# -----------------------------------------------------------------------------

def _phase_component_clearance_floor_v253(
    env: ManagerBasedRLEnv,
    phase_mode: str,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    component: str,
    period_s: float,
    linear_command_threshold: float,
    yaw_command_threshold: float,
    floor_m: float,
    active_half_width: float,
) -> torch.Tensor:
    """Saturating swing-clearance floor, not a trajectory target.

    During the central part of the scheduled swing half-cycle, reward rises linearly
    from 0 at zero clearance to 1 at ``floor_m`` and stays at 1 above the floor.
    Therefore exaggerated high stepping receives no extra reward.
    """
    data = _phase_whole_foot_state_v243(
        env, phase_mode, command_name, asset_cfg, period_s,
        linear_command_threshold, yaw_command_threshold,
    )
    if component == "main":
        clearance = data["swing_main_pos"][:, 2] - data["stance_main_pos"][:, 2]
    elif component == "toe":
        clearance = data["swing_toe_pos"][:, 2] - data["stance_toe_pos"][:, 2]
    else:
        raise ValueError(f"Unknown v2.5.3 clearance component: {component}")
    floor = max(float(floor_m), 1.0e-6)
    floor_score = torch.clamp(clearance / floor, 0.0, 1.0)
    # swing_shape peaks at mid-swing. Thresholding it creates a broad central window
    # without prescribing a height trajectory within that window.
    active = data["swing_shape"] >= float(active_half_width)
    return floor_score * active * data["moving"]


def phase_main_foot_clearance_floor_v253(
    env: ManagerBasedRLEnv,
    phase_mode: str,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    period_s: float = 1.12,
    linear_command_threshold: float = 0.05,
    yaw_command_threshold: float = 0.06,
    floor_m: float = 0.030,
    active_half_width: float = 0.30,
) -> torch.Tensor:
    return _phase_component_clearance_floor_v253(
        env, phase_mode, command_name, asset_cfg, "main", period_s,
        linear_command_threshold, yaw_command_threshold, floor_m, active_half_width,
    )


def phase_toe_clearance_floor_v253(
    env: ManagerBasedRLEnv,
    phase_mode: str,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    period_s: float = 1.12,
    linear_command_threshold: float = 0.05,
    yaw_command_threshold: float = 0.06,
    floor_m: float = 0.022,
    active_half_width: float = 0.30,
) -> torch.Tensor:
    return _phase_component_clearance_floor_v253(
        env, phase_mode, command_name, asset_cfg, "toe", period_s,
        linear_command_threshold, yaw_command_threshold, floor_m, active_half_width,
    )

# -----------------------------------------------------------------------------
# LittleGreen v2.6 canonical Walk: minimal step floors / contact requirements
# -----------------------------------------------------------------------------

def _v26_active_swing_window(data: dict[str, torch.Tensor], active_half_width: float) -> torch.Tensor:
    return (data["swing_shape"] >= float(active_half_width)) & data["moving"].bool()


def phase_scheduled_swing_whole_foot_contact_penalty_v26(
    env: ManagerBasedRLEnv,
    phase_mode: str,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    period_s: float = 1.16,
    linear_command_threshold: float = 0.05,
    yaw_command_threshold: float = 0.06,
    active_half_width: float = 0.30,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalty indicator when the scheduled swing whole foot fails to release.

    This is deliberately binary and stateless. It says only that during the central
    swing window the scheduled swing foot may not remain planted. It does not specify
    knee motion, COM motion, foot height trajectory, or landing location.
    """
    data = _phase_whole_foot_state_v243(
        env, phase_mode, command_name, asset_cfg, period_s,
        linear_command_threshold, yaw_command_threshold,
    )
    component, _, _ = _whole_foot_vertical_forces_v243(env, sensor_cfg)
    c = component > float(force_threshold)
    left_whole = c[:, 0] | c[:, 1]
    right_whole = c[:, 2] | c[:, 3]
    swing_whole = torch.where(data["left_stance"], right_whole, left_whole)
    active = _v26_active_swing_window(data, active_half_width)
    return swing_whole.float() * active.float()


def phase_scheduled_stance_missing_contact_penalty_v26(
    env: ManagerBasedRLEnv,
    phase_mode: str,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    period_s: float = 1.16,
    linear_command_threshold: float = 0.05,
    yaw_command_threshold: float = 0.06,
    active_half_width: float = 0.30,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalty indicator if the scheduled stance whole foot is not grounded in swing."""
    data = _phase_whole_foot_state_v243(
        env, phase_mode, command_name, asset_cfg, period_s,
        linear_command_threshold, yaw_command_threshold,
    )
    component, _, _ = _whole_foot_vertical_forces_v243(env, sensor_cfg)
    c = component > float(force_threshold)
    left_whole = c[:, 0] | c[:, 1]
    right_whole = c[:, 2] | c[:, 3]
    stance_whole = torch.where(data["left_stance"], left_whole, right_whole)
    active = _v26_active_swing_window(data, active_half_width)
    return (~stance_whole).float() * active.float()


def _phase_component_clearance_floor_v26(
    env: ManagerBasedRLEnv,
    phase_mode: str,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    component: str,
    period_s: float,
    linear_command_threshold: float,
    yaw_command_threshold: float,
    floor_m: float,
    active_half_width: float,
    force_threshold: float,
) -> torch.Tensor:
    """Saturating clearance floor credited only for true released swing support."""
    data = _phase_whole_foot_state_v243(
        env, phase_mode, command_name, asset_cfg, period_s,
        linear_command_threshold, yaw_command_threshold,
    )
    if component == "main":
        clearance = data["swing_main_pos"][:, 2] - data["stance_main_pos"][:, 2]
    elif component == "toe":
        clearance = data["swing_toe_pos"][:, 2] - data["stance_toe_pos"][:, 2]
    else:
        raise ValueError(f"Unknown v2.6 clearance component: {component}")

    contacts, _, _ = _whole_foot_vertical_forces_v243(env, sensor_cfg)
    c = contacts > float(force_threshold)
    left_whole = c[:, 0] | c[:, 1]
    right_whole = c[:, 2] | c[:, 3]
    stance_whole = torch.where(data["left_stance"], left_whole, right_whole)
    swing_whole = torch.where(data["left_stance"], right_whole, left_whole)
    true_release = stance_whole & (~swing_whole)

    score = torch.clamp(clearance / max(float(floor_m), 1.0e-6), 0.0, 1.0)
    active = _v26_active_swing_window(data, active_half_width)
    return score * active.float() * true_release.float()


def phase_main_foot_clearance_floor_v26(
    env: ManagerBasedRLEnv, phase_mode: str, command_name: str,
    asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
    period_s: float = 1.16, linear_command_threshold: float = 0.05,
    yaw_command_threshold: float = 0.06, floor_m: float = 0.032,
    active_half_width: float = 0.30, force_threshold: float = 1.0,
) -> torch.Tensor:
    return _phase_component_clearance_floor_v26(
        env, phase_mode, command_name, asset_cfg, sensor_cfg, "main", period_s,
        linear_command_threshold, yaw_command_threshold, floor_m,
        active_half_width, force_threshold,
    )


def phase_toe_clearance_floor_v26(
    env: ManagerBasedRLEnv, phase_mode: str, command_name: str,
    asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
    period_s: float = 1.16, linear_command_threshold: float = 0.05,
    yaw_command_threshold: float = 0.06, floor_m: float = 0.024,
    active_half_width: float = 0.30, force_threshold: float = 1.0,
) -> torch.Tensor:
    return _phase_component_clearance_floor_v26(
        env, phase_mode, command_name, asset_cfg, sensor_cfg, "toe", period_s,
        linear_command_threshold, yaw_command_threshold, floor_m,
        active_half_width, force_threshold,
    )



def sustained_max_torque_barrier_l2_v54(
    env: ManagerBasedRLEnv,
    torque_limit_nm: float,
    soft_ratio: float = 0.90,
    free_duration_s: float = 0.20,
    full_penalty_duration_s: float = 0.60,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize *sustained* near-limit actuator loading while allowing brief bursts.

    The worst selected actuator sets the amplitude, but no duration penalty is paid
    for the first ``free_duration_s`` above the soft threshold.  The duration factor
    then rises smoothly to one by ``full_penalty_duration_s``.  Dropping below the
    threshold immediately clears the timer.  This targets static bracing without
    discouraging short high-torque weight transfers needed to explore gait.
    """
    if torque_limit_nm <= 0.0:
        raise ValueError("torque_limit_nm must be positive")
    if not 0.0 <= soft_ratio < 1.0:
        raise ValueError("soft_ratio must be in [0, 1)")
    if free_duration_s < 0.0 or full_penalty_duration_s <= free_duration_s:
        raise ValueError("v5.4 torque durations must satisfy 0 <= free < full")
    asset: Articulation = env.scene[asset_cfg.name]
    utilization = torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids]) / float(torque_limit_nm)
    max_util = torch.max(utilization, dim=1).values
    above = max_util > float(soft_ratio)
    n = max_util.shape[0]
    device = max_util.device
    dtype = max_util.dtype
    timer = getattr(env, "_littlegreen_v54_high_torque_time_s", None)
    if not isinstance(timer, torch.Tensor) or timer.shape[0] != n:
        timer = torch.zeros(n, device=device, dtype=dtype)
    else:
        timer = timer.to(device=device, dtype=dtype)

    common_step = int(getattr(env, "common_step_counter", 0))
    last_step = int(getattr(env, "_littlegreen_v54_torque_last_common_step", -1))
    if common_step != last_step:
        dt = float(getattr(env, "step_dt", 0.02))
        reset_mask = torch.zeros(n, device=device, dtype=torch.bool)
        episode_length = getattr(env, "episode_length_buf", None)
        last_episode_length = getattr(env, "_littlegreen_v54_torque_last_episode_length", None)
        if isinstance(episode_length, torch.Tensor) and episode_length.shape[0] == n:
            current_episode_length = episode_length.to(device=device)
            reset_mask = current_episode_length <= 0
            if isinstance(last_episode_length, torch.Tensor) and last_episode_length.shape[0] == n:
                reset_mask = reset_mask | (current_episode_length < last_episode_length.to(device=device))
            env._littlegreen_v54_torque_last_episode_length = current_episode_length.clone()
        timer = torch.where(reset_mask, torch.zeros_like(timer), timer)
        timer = torch.where(above, timer + dt, torch.zeros_like(timer))
        env._littlegreen_v54_high_torque_time_s = timer
        env._littlegreen_v54_torque_last_common_step = common_step

    amplitude = torch.relu((max_util - float(soft_ratio)) / max(1.0 - float(soft_ratio), 1.0e-6))
    amplitude = torch.square(torch.clamp(amplitude, max=1.5))
    span = max(float(full_penalty_duration_s) - float(free_duration_s), 1.0e-6)
    x = torch.clamp((timer - float(free_duration_s)) / span, min=0.0, max=1.0)
    duration = x * x * (3.0 - 2.0 * x)
    return amplitude * duration



def sustained_max_torque_barrier_l2_v60(
    env: ManagerBasedRLEnv,
    torque_limit_nm: float,
    soft_ratio: float = 0.92,
    free_duration_s: float = 0.25,
    full_penalty_duration_s: float = 0.75,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """v6 sustained near-limit torque guard with fresh state names.

    Brief high-torque transfer bursts remain free.  Only continuous loading above the
    soft threshold accumulates duration, reaching full influence at the configured
    duration.  This is gait-agnostic and does not alter physical actuator authority.
    """
    if torque_limit_nm <= 0.0:
        raise ValueError("torque_limit_nm must be positive")
    if not 0.0 <= soft_ratio < 1.0:
        raise ValueError("soft_ratio must be in [0, 1)")
    if free_duration_s < 0.0 or full_penalty_duration_s <= free_duration_s:
        raise ValueError("v6.0 torque durations must satisfy 0 <= free < full")
    asset: Articulation = env.scene[asset_cfg.name]
    utilization = torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids]) / float(torque_limit_nm)
    max_util = torch.max(utilization, dim=1).values
    above = max_util > float(soft_ratio)
    n = max_util.shape[0]
    device = max_util.device
    dtype = max_util.dtype
    timer = getattr(env, "_littlegreen_v60_high_torque_time_s", None)
    if not isinstance(timer, torch.Tensor) or timer.shape[0] != n:
        timer = torch.zeros(n, device=device, dtype=dtype)
    else:
        timer = timer.to(device=device, dtype=dtype)

    common_step = int(getattr(env, "common_step_counter", 0))
    last_step = int(getattr(env, "_littlegreen_v60_torque_last_common_step", -1))
    if common_step != last_step:
        dt = float(getattr(env, "step_dt", 0.02))
        reset_mask = torch.zeros(n, device=device, dtype=torch.bool)
        episode_length = getattr(env, "episode_length_buf", None)
        last_episode_length = getattr(env, "_littlegreen_v60_torque_last_episode_length", None)
        if isinstance(episode_length, torch.Tensor) and episode_length.shape[0] == n:
            current_episode_length = episode_length.to(device=device)
            reset_mask = current_episode_length <= 0
            if isinstance(last_episode_length, torch.Tensor) and last_episode_length.shape[0] == n:
                reset_mask = reset_mask | (current_episode_length < last_episode_length.to(device=device))
            env._littlegreen_v60_torque_last_episode_length = current_episode_length.clone()
        timer = torch.where(reset_mask, torch.zeros_like(timer), timer)
        timer = torch.where(above, timer + dt, torch.zeros_like(timer))
        env._littlegreen_v60_high_torque_time_s = timer
        env._littlegreen_v60_torque_last_common_step = common_step

    amplitude = torch.relu((max_util - float(soft_ratio)) / max(1.0 - float(soft_ratio), 1.0e-6))
    amplitude = torch.square(torch.clamp(amplitude, max=1.5))
    span = max(float(full_penalty_duration_s) - float(free_duration_s), 1.0e-6)
    x = torch.clamp((timer - float(free_duration_s)) / span, min=0.0, max=1.0)
    duration = x * x * (3.0 - 2.0 * x)
    return amplitude * duration
