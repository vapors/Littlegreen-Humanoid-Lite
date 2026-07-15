"""Velocity-Lilgreen-Hardware-ST3215-Loaded-v9.

A behavior-focused gait-acquisition task built from the successful v5s3 Stand
checkpoint.  It preserves the 47-D phase-guided observation contract, action
contract v4, v5s3 q_default, vector residual scale, physical limits, ST3215 Stage-A
model, and loaded Stage-B envelope.

The v9 change is deliberately narrow:
* reset-safe no-progress/no-support terminations;
* zero-column 45-D -> 47-D actor warm start (implemented in train_eval.py);
* an explicit 22-term reward set instead of the inherited v5-v8 reward stack;
* a gentle positive-forward command curriculum with no external pushes.
"""

from __future__ import annotations

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import littlegreen_humanoid_lite.tasks.locomotion.velocity.mdp as mdp
from littlegreen_humanoid_lite.tasks.locomotion.velocity.mdp.hardware_contract import (
    ACTIONABLE_JOINTS_V1_2_3,
    HARDWARE_LOWER_LIMIT_RAD,
    HARDWARE_UPPER_LIMIT_RAD,
    ST3215_NO_LOAD_SPEED_RAD_S,
    ST3215_PEAK_TORQUE_NM,
    V1_4_5_STABILIZED_FORWARD2_COM_BAND_HALF_WIDTH_M,
    V1_4_5_STABILIZED_FORWARD2_COM_TARGET_X_M,
    V1_4_5_STABILIZED_FORWARD2_MOVING_BASE_COM_HEIGHT_M,
    V1_4_5_STABILIZED_FORWARD2_STAND_BASE_COM_HEIGHT_M,
)

from .env_cfg_hardware_st3215_loaded_v145 import (
    HardwareV147PhaseGuidedObservationsCfg,
    LilgreenHardwareST3215LoadedV145StabilizedForward2EnvCfg,
)
from .v1_2_3_common import FEET_BODY_PATTERN
from .v1_3_0_st3215_common import ST3215HardwareEventsCfg


@configclass
class V9GaitAcquisitionCommandsCfg:
    """Positive-forward acquisition commands with explicit yaw-rate sampling."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        resampling_time_range=(7.0, 11.0),
        debug_vis=True,
        asset_name="robot",
        heading_command=False,
        heading_control_stiffness=0.5,
        rel_standing_envs=0.25,
        rel_heading_envs=0.0,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.25, 0.36),
            lin_vel_y=(-0.04, 0.04),
            ang_vel_z=(-0.08, 0.08),
            heading=(0.0, 0.0),
        ),
    )


@configclass
class V9GaitAcquisitionEventsCfg(ST3215HardwareEventsCfg):
    """Narrow reset/domain randomization with pushes disabled for gait acquisition."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.85, 1.05),
            "dynamic_friction_range": (0.80, 1.00),
            "restitution_range": (0.0, 0.01),
            "num_buckets": 32,
            "make_consistent": True,
        },
        mode="startup",
    )
    base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (0.995, 1.005),
            "operation": "scale",
        },
        mode="reset",
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        params={
            "pose_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (-0.01, 0.01),
                "y": (-0.01, 0.01),
                "z": (0.0, 0.0),
                "roll": (-0.01, 0.01),
                "pitch": (-0.01, 0.01),
                "yaw": (-0.01, 0.01),
            },
        },
        mode="reset",
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        params={"position_range": (-0.004, 0.004), "velocity_range": (0.0, 0.0)},
        mode="reset",
    )
    push_robot = None


@configclass
class V9SlimGaitRewardsCfg:
    """Explicit, non-inherited 22-term gait acquisition objective."""

    # Command following: one tracking pair plus one mild directional-progress cue.
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        params={"command_name": "base_velocity", "std": 0.30},
        weight=3.2,
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        params={"command_name": "base_velocity", "std": 0.32},
        weight=1.2,
    )
    moving_velocity_along_command = RewTerm(
        func=mdp.moving_velocity_along_command,
        params={"command_name": "base_velocity", "command_threshold": 0.20},
        weight=0.8,
    )

    # Balance and failure handling.
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-10.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.15)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.08)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.25)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["base", ".*_hip_.*", ".*_knee_.*"]
            ),
            "threshold": 1.0,
        },
        weight=-2.0,
    )

    # Actuator/action health without suppressing the movement authority needed to step.
    bounded_action_rate_l2 = RewTerm(
        func=mdp.bounded_action_rate_l2,
        params={"action_name": "joint_pos"},
        weight=-0.012,
    )
    raw_action_excess_l2 = RewTerm(
        func=mdp.raw_action_excess_l2,
        params={"action_name": "joint_pos"},
        weight=-0.080,
    )
    soft_torque_utilization = RewTerm(
        func=mdp.soft_torque_utilization_l2,
        params={
            "torque_limit_nm": ST3215_PEAK_TORQUE_NM,
            "soft_ratio": 0.74,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=ACTIONABLE_JOINTS_V1_2_3, preserve_order=True
            ),
        },
        weight=-0.012,
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=ACTIONABLE_JOINTS_V1_2_3, preserve_order=True
            )
        },
        weight=-0.15,
    )

    # Preserve the successful v5s3 standing posture without a posture-symmetry lock.
    stand_base_height = RewTerm(
        func=mdp.standing_base_height_exp,
        params={
            "command_name": "base_velocity",
            "desired_height": V1_4_5_STABILIZED_FORWARD2_STAND_BASE_COM_HEIGHT_M,
            "std": 0.090,
        },
        weight=0.65,
    )
    stand_com_over_feet = RewTerm(
        func=mdp.standing_com_forward_over_feet_band_l2,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.05,
            "target_forward_m": V1_4_5_STABILIZED_FORWARD2_COM_TARGET_X_M,
            "band_half_width_m": V1_4_5_STABILIZED_FORWARD2_COM_BAND_HALF_WIDTH_M,
            "lateral_weight": 0.25,
        },
        weight=-0.85,
    )
    stand_both_feet_contact = RewTerm(
        func=mdp.standing_both_feet_contact,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "force_threshold": 1.0,
        },
        weight=0.60,
    )
    stand_feet_slide = RewTerm(
        func=mdp.standing_feet_slide,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
        },
        weight=-0.65,
    )

    # Moving height plus four direct phase-guided gait terms.
    moving_base_height = RewTerm(
        func=mdp.moving_base_height_exp,
        params={
            "command_name": "base_velocity",
            "desired_height": V1_4_5_STABILIZED_FORWARD2_MOVING_BASE_COM_HEIGHT_M,
            "std": 0.115,
            "command_threshold": 0.20,
        },
        weight=0.35,
    )
    phase_contact = RewTerm(
        func=mdp.phase_guided_contact_reward,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.20,
            "period_s": 0.72,
            "transition_width": 0.10,
            "force_threshold": 1.0,
        },
        weight=1.8,
    )
    phase_swing_clearance = RewTerm(
        func=mdp.phase_guided_expected_swing_clearance_reward,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.20,
            "period_s": 0.72,
            "transition_width": 0.10,
            "target_clearance_m": 0.028,
            "clearance_std_m": 0.018,
            "force_threshold": 1.0,
        },
        weight=1.2,
    )
    phase_foot_placement = RewTerm(
        func=mdp.phase_guided_step_placement_target_reward,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.20,
            "period_s": 0.72,
            "transition_width": 0.10,
            "target_step_m": 0.045,
            "step_std_m": 0.040,
            "force_threshold": 1.0,
        },
        weight=1.0,
    )
    moving_stance_foot_slide = RewTerm(
        func=mdp.moving_stance_foot_slide_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.20,
            "force_threshold": 1.0,
        },
        weight=-0.90,
    )
    moving_no_support = RewTerm(
        func=mdp.moving_no_support_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.20,
            "force_threshold": 1.0,
        },
        weight=-1.25,
    )


@configclass
class V9ResetSafeTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.0, "asset_cfg": SceneEntityCfg("robot", body_names="base")},
    )
    moving_no_progress = DoneTerm(
        func=mdp.moving_no_progress_timeout,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.20,
            "min_progress_fraction": 0.08,
            "grace_time_s": 2.50,
            "timeout_s": 4.00,
        },
    )
    moving_no_support = DoneTerm(
        func=mdp.moving_no_support_timeout,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.20,
            "max_no_support_s": 0.18,
            "force_threshold": 1.0,
            "arming_timeout_s": 0.40,
        },
    )


@configclass
class V9GaitAcquisitionCurriculumsCfg:
    hardware_stage = CurrTerm(func=mdp.st3215_loaded_v9_gait_acquisition_curriculum)
    policy_diagnostics = CurrTerm(
        func=mdp.PolicyDiagnostics,
        params={
            "command_name": "base_velocity",
            "action_name": "joint_pos",
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=ACTIONABLE_JOINTS_V1_2_3, preserve_order=True
            ),
            "foot_asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "joint_velocity_limit_rad_s": ST3215_NO_LOAD_SPEED_RAD_S,
            "torque_limit_nm": ST3215_PEAK_TORQUE_NM,
            "torque_soft_ratio": 0.74,
            "desired_base_com_height_m": V1_4_5_STABILIZED_FORWARD2_STAND_BASE_COM_HEIGHT_M,
            "update_interval_steps": 25,
            "standing_command_threshold": 0.05,
        },
    )


@configclass
class LilgreenHardwareST3215LoadedV9GaitAcquisitionEnvCfg(
    LilgreenHardwareST3215LoadedV145StabilizedForward2EnvCfg
):
    """v9 forward gait acquisition from the v5s3 Stand model_3500 policy."""

    commands: V9GaitAcquisitionCommandsCfg = V9GaitAcquisitionCommandsCfg()
    observations: HardwareV147PhaseGuidedObservationsCfg = HardwareV147PhaseGuidedObservationsCfg()
    rewards: V9SlimGaitRewardsCfg = V9SlimGaitRewardsCfg()
    terminations: V9ResetSafeTerminationsCfg = V9ResetSafeTerminationsCfg()
    events: V9GaitAcquisitionEventsCfg = V9GaitAcquisitionEventsCfg()
    curriculum: V9GaitAcquisitionCurriculumsCfg = V9GaitAcquisitionCurriculumsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Contract lock: v9 must remain the v5s3 action profile and 12-joint limits.
        action_cfg = self.actions.joint_pos
        if list(action_cfg.joint_names) != list(ACTIONABLE_JOINTS_V1_2_3):
            raise ValueError("v9 canonical joint order differs from the v5s3 deployment contract")
        if list(action_cfg.lower_limits) != list(HARDWARE_LOWER_LIMIT_RAD):
            raise ValueError("v9 lower physical limits differ from the v5s3 deployment contract")
        if list(action_cfg.upper_limits) != list(HARDWARE_UPPER_LIMIT_RAD):
            raise ValueError("v9 upper physical limits differ from the v5s3 deployment contract")
