"""Velocity-Lilgreen-Hardware-ST3215-Loaded-v10.

A condensed 5,000-iteration go/no-go gait-acquisition task built from the
successful v5s3 Stand model_3500 checkpoint.  It preserves action contract v4,
the canonical 12-joint order, q_default, vector residual scales, physical limits,
and both ST3215 actuator-model stages.

v10 changes only the locomotion scaffold:
* command-synchronized 47-D phase semantics with alternating first swing side;
* short one-sided double-support/weight-transfer windows;
* signed contact matching, alternating support-force and COM transfer;
* baseline-corrected swing-height trajectory with zero reward for zero lift;
* placement reward gated by real clearance;
* taller 0.455 m moving COM target and 0.080-0.085 m forward COM target;
* a condensed transfer -> lift -> place -> translate curriculum;
* no no-progress termination during the 5k acquisition decision run.
"""

from __future__ import annotations

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
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
    V1_4_5_STABILIZED_FORWARD2_STAND_BASE_COM_HEIGHT_M,
)

from .env_cfg_hardware import HardwareObservationsCfg
from .env_cfg_hardware_st3215_loaded_v145 import (
    LilgreenHardwareST3215LoadedV145StabilizedForward2EnvCfg,
)
from .v1_2_3_common import FEET_BODY_PATTERN
from .v1_3_0_st3215_common import ST3215HardwareEventsCfg


V10_MOVING_BASE_COM_HEIGHT_M = 0.455
V10_INITIAL_COM_TARGET_FORWARD_M = 0.080


@configclass
class V10TransferLiftPlaceCommandsCfg:
    """Stage-A straight-forward command envelope."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        resampling_time_range=(7.0, 11.0),
        debug_vis=True,
        asset_name="robot",
        heading_command=False,
        heading_control_stiffness=0.5,
        rel_standing_envs=0.28,
        rel_heading_envs=0.0,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.25, 0.34),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )


@configclass
class V10TransferLiftPlaceEventsCfg(ST3215HardwareEventsCfg):
    """Very narrow reset/domain randomization; pushes stay disabled."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.88, 1.02),
            "dynamic_friction_range": (0.83, 0.98),
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
            "mass_distribution_params": (0.997, 1.003),
            "operation": "scale",
        },
        mode="reset",
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        params={
            "pose_range": {"x": (-0.025, 0.025), "y": (-0.025, 0.025), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (-0.008, 0.008),
                "y": (-0.008, 0.008),
                "z": (0.0, 0.0),
                "roll": (-0.008, 0.008),
                "pitch": (-0.008, 0.008),
                "yaw": (-0.008, 0.008),
            },
        },
        mode="reset",
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        params={"position_range": (-0.003, 0.003), "velocity_range": (0.0, 0.0)},
        mode="reset",
    )
    push_robot = None


@configclass
class V10CommandSynchronizedObservationsCfg(HardwareObservationsCfg):
    """47-D observation contract with revised command-synchronized phase semantics."""

    @configclass
    class PolicyCfg(HardwareObservationsCfg.PolicyCfg):
        gait_phase = ObsTerm(
            func=mdp.command_synchronized_gait_phase_sin_cos,
            params={
                "command_name": "base_velocity",
                "linear_command_threshold": 0.20,
                "yaw_command_threshold": 0.08,
                "period_s": 0.90,
            },
        )

    @configclass
    class CriticCfg(PolicyCfg):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        def __post_init__(self):
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class V10TransferLiftPlaceRewardsCfg:
    """Standalone 24-term objective with no duplicate gait proxies."""

    # Command following is intentionally reduced during Stage A by a curriculum scale.
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.v10_track_lin_vel_xy_yaw_frame_exp,
        params={"command_name": "base_velocity", "std": 0.30, "default_stage_scale": 0.55},
        weight=3.0,
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        params={"command_name": "base_velocity", "std": 0.34},
        weight=0.8,
    )
    moving_velocity_along_command = RewTerm(
        func=mdp.moving_velocity_along_command,
        params={"command_name": "base_velocity", "command_threshold": 0.20},
        weight=0.4,
    )

    # Balance/failure protection.
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-10.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.14)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.08)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.24)
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

    # Actuator/action health.  Movement authority remains available for first-step learning.
    bounded_action_rate_l2 = RewTerm(
        func=mdp.bounded_action_rate_l2,
        params={"action_name": "joint_pos"},
        weight=-0.010,
    )
    raw_action_excess_l2 = RewTerm(
        func=mdp.raw_action_excess_l2,
        params={"action_name": "joint_pos"},
        weight=-0.070,
    )
    soft_torque_utilization = RewTerm(
        func=mdp.soft_torque_utilization_l2,
        params={
            "torque_limit_nm": ST3215_PEAK_TORQUE_NM,
            "soft_ratio": 0.75,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=ACTIONABLE_JOINTS_V1_2_3, preserve_order=True
            ),
        },
        weight=-0.010,
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

    # Preserve the mature v5s3 stand without constraining gait-time asymmetry.
    stand_base_height = RewTerm(
        func=mdp.standing_base_height_exp,
        params={
            "command_name": "base_velocity",
            "desired_height": V1_4_5_STABILIZED_FORWARD2_STAND_BASE_COM_HEIGHT_M,
            "std": 0.085,
        },
        weight=0.55,
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
        weight=-0.75,
    )
    stand_both_feet_contact = RewTerm(
        func=mdp.standing_both_feet_contact,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "force_threshold": 1.0,
        },
        weight=0.50,
    )
    stand_feet_slide = RewTerm(
        func=mdp.standing_feet_slide,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
        },
        weight=-0.60,
    )

    # Taller moving posture and explicit transfer -> lift -> place sequence.
    moving_base_height = RewTerm(
        func=mdp.moving_base_height_exp,
        params={
            "command_name": "base_velocity",
            "desired_height": V10_MOVING_BASE_COM_HEIGHT_M,
            "std": 0.060,
            "command_threshold": 0.20,
        },
        weight=0.70,
    )
    signed_phase_contact = RewTerm(
        func=mdp.v10_signed_phase_contact_reward,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "linear_command_threshold": 0.20,
            "yaw_command_threshold": 0.08,
            "period_s": 0.90,
            "transition_fraction": 0.08,
            "force_threshold": 1.0,
            "double_support_swing_score": -0.40,
        },
        weight=1.6,
    )
    support_force_transfer = RewTerm(
        func=mdp.v10_support_force_transfer_reward,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "linear_command_threshold": 0.20,
            "yaw_command_threshold": 0.08,
            "period_s": 0.90,
            "transition_fraction": 0.08,
            "target_stance_ratio": 0.75,
            "early_swing_fraction": 0.20,
            "force_threshold": 1.0,
        },
        weight=1.8,
    )
    phase_com_shift = RewTerm(
        func=mdp.v10_phase_com_shift_reward,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "linear_command_threshold": 0.20,
            "yaw_command_threshold": 0.08,
            "period_s": 0.90,
            "transition_fraction": 0.08,
            "target_forward_m": V10_INITIAL_COM_TARGET_FORWARD_M,
            "neutral_forward_m": V1_4_5_STABILIZED_FORWARD2_COM_TARGET_X_M,
            "lateral_stance_fraction": 0.75,
            "forward_std_m": 0.035,
            "lateral_std_m": 0.030,
            "early_swing_fraction": 0.30,
        },
        weight=1.5,
    )
    swing_clearance_trajectory = RewTerm(
        func=mdp.v10_swing_clearance_trajectory_reward,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "linear_command_threshold": 0.20,
            "yaw_command_threshold": 0.08,
            "period_s": 0.90,
            "transition_fraction": 0.08,
            "target_clearance_m": 0.018,
            "clearance_std_m": 0.009,
        },
        weight=2.4,
    )
    clearance_gated_placement = RewTerm(
        func=mdp.v10_clearance_gated_placement_reward,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "linear_command_threshold": 0.20,
            "yaw_command_threshold": 0.08,
            "period_s": 0.90,
            "transition_fraction": 0.08,
            "target_step_m": 0.025,
            "step_std_m": 0.025,
            "clearance_gate_m": 0.008,
            "gate_width_m": 0.006,
            "placement_start_progress": 0.35,
            "placement_scale": 0.15,
        },
        weight=1.4,
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
        weight=-0.70,
    )
    moving_no_support = RewTerm(
        func=mdp.moving_no_support_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.20,
            "force_threshold": 1.0,
        },
        weight=-1.50,
    )


@configclass
class V10TransferLiftPlaceTerminationsCfg:
    """Reset-safe safety terminations; no-progress is intentionally absent."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.0, "asset_cfg": SceneEntityCfg("robot", body_names="base")},
    )
    moving_no_support = DoneTerm(
        func=mdp.moving_no_support_timeout,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.20,
            "max_no_support_s": 0.22,
            "force_threshold": 1.0,
            "arming_timeout_s": 0.40,
        },
    )


@configclass
class V10TransferLiftPlaceCurriculumsCfg:
    hardware_stage = CurrTerm(func=mdp.st3215_loaded_v10_transfer_lift_place_curriculum)
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
            "torque_soft_ratio": 0.75,
            "desired_base_com_height_m": V1_4_5_STABILIZED_FORWARD2_STAND_BASE_COM_HEIGHT_M,
            "update_interval_steps": 25,
            "standing_command_threshold": 0.05,
        },
    )


@configclass
class LilgreenHardwareST3215LoadedV10TransferLiftPlaceEnvCfg(
    LilgreenHardwareST3215LoadedV145StabilizedForward2EnvCfg
):
    """v10 transfer/lift/place acquisition from the v5s3 Stand model_3500 policy."""

    commands: V10TransferLiftPlaceCommandsCfg = V10TransferLiftPlaceCommandsCfg()
    observations: V10CommandSynchronizedObservationsCfg = V10CommandSynchronizedObservationsCfg()
    rewards: V10TransferLiftPlaceRewardsCfg = V10TransferLiftPlaceRewardsCfg()
    terminations: V10TransferLiftPlaceTerminationsCfg = V10TransferLiftPlaceTerminationsCfg()
    events: V10TransferLiftPlaceEventsCfg = V10TransferLiftPlaceEventsCfg()
    curriculum: V10TransferLiftPlaceCurriculumsCfg = V10TransferLiftPlaceCurriculumsCfg()

    def __post_init__(self):
        super().__post_init__()
        action_cfg = self.actions.joint_pos
        if list(action_cfg.joint_names) != list(ACTIONABLE_JOINTS_V1_2_3):
            raise ValueError("v10 canonical joint order differs from the v5s3 deployment contract")
        if list(action_cfg.lower_limits) != list(HARDWARE_LOWER_LIMIT_RAD):
            raise ValueError("v10 lower physical limits differ from the v5s3 deployment contract")
        if list(action_cfg.upper_limits) != list(HARDWARE_UPPER_LIMIT_RAD):
            raise ValueError("v10 upper physical limits differ from the v5s3 deployment contract")
