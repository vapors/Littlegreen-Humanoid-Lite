"""Selective MDP facade for the active LittleGreen v2.5 foundation lineage.

The active path exposes canonical Stand v24, Transfer/March v24, and Walk v25.
It preserves the measured ST3215 action/actuator stack and the exact 47-D actor
observation contract while keeping the locomotion objective explicit and stateless.
"""

import torch

from isaaclab.envs.mdp import *  # noqa: F401,F403

from littlegreen_humanoid_lite.tasks.locomotion.velocity.mdp.commands_v101 import (
    TurnBiasedUniformVelocityCommandCfgV101,
    TurnBiasedUniformVelocityCommandV101,
)
from littlegreen_humanoid_lite.tasks.locomotion.velocity.mdp.diagnostics import PolicyDiagnostics
from littlegreen_humanoid_lite.tasks.locomotion.velocity.mdp.events import set_joint_position_limits
from littlegreen_humanoid_lite.tasks.locomotion.velocity.mdp.foundation_phase import (
    command_synchronized_phase_fraction_v24,
    continuous_transfer_phase_fraction_v24,
    phase_sin_cos,
    random_static_phase_fraction_v24,
    reset_command_synchronized_phase_v24,
    reset_continuous_transfer_phase_v24,
    reset_random_static_phase_v24,
)
from littlegreen_humanoid_lite.tasks.locomotion.velocity.mdp.hardware_actions import *  # noqa: F401,F403
from littlegreen_humanoid_lite.tasks.locomotion.velocity.mdp.hardware_rewards import (
    bounded_action_l2,
    bounded_action_rate_l2,
    base_com_height_exp_v24,
    bounded_last_action,
    normalized_torque_barrier_l2,
    max_torque_barrier_l2,
    sustained_max_torque_barrier_l2_v54,
    sustained_max_torque_barrier_l2_v60,
    pair_max_torque_barrier_l2,
    phase_com_lateral_progress_v243,
    phase_com_lateral_target_exp_v243,
    phase_main_foot_clearance_trajectory_v243,
    phase_main_foot_clearance_floor_v253,
    phase_main_foot_clearance_floor_v26,
    phase_toe_clearance_floor_v26,
    phase_scheduled_swing_whole_foot_contact_penalty_v26,
    phase_scheduled_stance_missing_contact_penalty_v26,
    phase_stance_whole_foot_slide_l2_v243,
    phase_support_load_target_exp_v243,
    phase_support_progress_v243,
    phase_toe_clearance_trajectory_v243,
    phase_toe_clearance_floor_v253,
    phase_whole_foot_contact_schedule_reward_v243,
    phase_zero_command_foot_alignment_l2_v243,
    transfer_sagittal_com_readiness_band_l2_v243,
    transfer_static_both_whole_feet_contact_v243,
    phase_lateral_foot_crossing_l2_v250,
    phase_swing_placement_along_command_exp_v250,
    standing_both_whole_feet_contact_v250,
    walk_sagittal_com_readiness_band_l2_v250,
    phase_com_lateral_progress_v24,
    phase_com_lateral_target_exp_v24,
    phase_contact_schedule_reward_v24,
    phase_stance_foot_slide_l2_v24,
    phase_support_load_target_exp_v24,
    phase_support_progress_v24,
    phase_swing_clearance_trajectory_v24,
    phase_swing_progress_along_command_v24,
    raw_action_excess_l2,
    soft_torque_utilization_l2,
    standing_base_height_exp,
    standing_base_xy_speed_l2,
    standing_both_feet_contact,
    standing_com_forward_over_feet_band_l2,
    standing_com_lateral_over_feet_l2,
    standing_contact_force_balance_l2,
    standing_default_joint_pose_l2,
    standing_feet_slide,
    standing_forward_lean_projected_gravity_exp,
    standing_mirrored_joint_residual_l2,
    standing_pair_max_torque_barrier_l2,
    standing_soft_torque_utilization_l2,
    standing_target_limit_margin_l2,
    standing_yaw_rate_l2,
    transfer_static_both_feet_contact_v24,
    transfer_static_default_pose_l2_v24,
)
from littlegreen_humanoid_lite.tasks.locomotion.velocity.mdp.rewards import *  # noqa: F401,F403
from littlegreen_humanoid_lite.tasks.locomotion.velocity.mdp.st3215_actions import (
    ST3215MeasuredResidualJointPositionActionCfg,
)
from littlegreen_humanoid_lite.tasks.locomotion.velocity.mdp.terminations import (
    body_contact_termination,
    moving_no_support_timeout,
)


def littlegreen_phase_sin_cos_v24(
    env,
    phase_mode: str,
    command_name: str,
    asset_cfg,
    sensor_cfg,
    linear_command_threshold: float = 0.08,
    yaw_command_threshold: float = 0.08,
    period_s: float = 0.90,
    transition_fraction: float = 0.08,
):
    """Shared 2-D phase tail for the v2.4 Stand/Transfer/Walk contract.

    ``asset_cfg``, ``sensor_cfg``, and ``transition_fraction`` remain in the
    signature so the same observation configuration stays compatible with the
    previous manager-based task plumbing.  v2.4's phase clocks do not wait for
    contact or reward success.
    """
    del asset_cfg, sensor_cfg, transition_fraction
    if phase_mode == "neutral_static":
        return torch.stack((torch.zeros(env.num_envs, device=env.device), torch.ones(env.num_envs, device=env.device)), dim=1)
    if phase_mode == "random_static":
        return phase_sin_cos(random_static_phase_fraction_v24(env))
    if phase_mode == "continuous_transfer":
        return phase_sin_cos(continuous_transfer_phase_fraction_v24(env, period_s=period_s))
    if phase_mode == "command_synchronized":
        phase, _ = command_synchronized_phase_fraction_v24(
            env,
            command_name=command_name,
            linear_command_threshold=linear_command_threshold,
            yaw_command_threshold=yaw_command_threshold,
            period_s=period_s,
        )
        return phase_sin_cos(phase)
    raise ValueError(f"Unsupported LittleGreen v2.4 phase mode: {phase_mode}")

# v10.3.1 continuous, symmetric pre-handoff reward (kept out of v10.2 base).
from littlegreen_humanoid_lite.tasks.locomotion.velocity.mdp.bilateral_acquisition_v1031 import (
    balanced_swing_discovery_v1031,
)
