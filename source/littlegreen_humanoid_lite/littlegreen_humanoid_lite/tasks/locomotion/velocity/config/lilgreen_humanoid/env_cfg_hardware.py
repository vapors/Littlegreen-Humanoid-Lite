"""Velocity-Lilgreen-Hardware-v0: v1.2.3 residual-action hardware locomotion curriculum."""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import littlegreen_humanoid_lite.tasks.locomotion.velocity.mdp as mdp
from littlegreen_humanoid_lite.tasks.locomotion.velocity.mdp.hardware_contract import (
    ACTIONABLE_JOINTS_V1_2_3,
    NOMINAL_QDEFAULT_BASE_COM_HEIGHT_M,
    ST3215_NO_LOAD_SPEED_RAD_S,
    ST3215_PEAK_TORQUE_NM,
)

from .v1_2_3_common import (
    FEET_BODY_PATTERN,
    HardwareCommandsCfg,
    HardwareEventsCfg,
    HardwareObservationsCfg,
    V123ResidualActionsCfg,
    V123HardwareAlignedEnvCfg,
    V123TerminationsCfg,
)


@configclass
class HardwareRewardsCfg:
    """Locomotion rewards with persistent standing behavior and bounded-action health."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        params={"command_name": "base_velocity", "std": 0.5},
        weight=2.5,
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        params={"command_name": "base_velocity", "std": 0.5},
        weight=1.5,
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-10.0)

    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.15)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.08)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.25)

    bounded_action_rate_l2 = RewTerm(
        func=mdp.bounded_action_rate_l2,
        params={"action_name": "joint_pos"},
        weight=-0.015,
    )
    bounded_action_l2 = RewTerm(
        func=mdp.bounded_action_l2,
        params={"action_name": "joint_pos"},
        weight=-0.005,
    )
    raw_action_excess_l2 = RewTerm(
        func=mdp.raw_action_excess_l2,
        params={"action_name": "joint_pos"},
        weight=-0.080,
    )
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTIONABLE_JOINTS_V1_2_3, preserve_order=True)},
        weight=-1.0e-4,
    )
    soft_torque_utilization = RewTerm(
        func=mdp.soft_torque_utilization_l2,
        params={
            "torque_limit_nm": ST3215_PEAK_TORQUE_NM,
            "soft_ratio": 0.70,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=ACTIONABLE_JOINTS_V1_2_3, preserve_order=True
            ),
        },
        weight=-0.015,
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTIONABLE_JOINTS_V1_2_3, preserve_order=True)},
        weight=-1.0e-7,
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.20)

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "threshold": 0.20,
        },
        weight=1.5,
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
        },
        weight=-0.80,
    )

    stand_base_xy_speed = RewTerm(
        func=mdp.standing_base_xy_speed_l2,
        params={"command_name": "base_velocity", "command_threshold": 0.05},
        weight=-1.5,
    )
    stand_yaw_rate = RewTerm(
        func=mdp.standing_yaw_rate_l2,
        params={"command_name": "base_velocity", "command_threshold": 0.05},
        weight=-0.35,
    )
    stand_default_pose = RewTerm(
        func=mdp.standing_default_joint_pose_l2,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.05,
            "asset_cfg": SceneEntityCfg("robot", joint_names=ACTIONABLE_JOINTS_V1_2_3, preserve_order=True),
        },
        weight=-0.40,
    )
    stand_base_height = RewTerm(
        func=mdp.standing_base_height_exp,
        params={
            "command_name": "base_velocity",
            "desired_height": NOMINAL_QDEFAULT_BASE_COM_HEIGHT_M,
            "std": 0.06,
        },
        weight=0.75,
    )
    stand_both_feet_contact = RewTerm(
        func=mdp.standing_both_feet_contact,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "force_threshold": 1.0,
        },
        weight=0.75,
    )
    stand_feet_slide = RewTerm(
        func=mdp.standing_feet_slide,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
        },
        weight=-0.80,
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*_hip_.*", ".*_knee_.*"]),
            "threshold": 1.0,
        },
        weight=-2.0,
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
        weight=-0.30,
    )
    joint_deviation_ankle_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_roll_joint"])},
        weight=-0.25,
    )


@configclass
class HardwareCurriculumsCfg:
    hardware_stage = CurrTerm(func=mdp.hardware_stage_curriculum)
    policy_diagnostics = CurrTerm(
        func=mdp.PolicyDiagnostics,
        params={
            "command_name": "base_velocity",
            "action_name": "joint_pos",
            "asset_cfg": SceneEntityCfg("robot", joint_names=ACTIONABLE_JOINTS_V1_2_3, preserve_order=True),
            "foot_asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "joint_velocity_limit_rad_s": ST3215_NO_LOAD_SPEED_RAD_S,
            "torque_limit_nm": ST3215_PEAK_TORQUE_NM,
            "torque_soft_ratio": 0.70,
            "desired_base_com_height_m": NOMINAL_QDEFAULT_BASE_COM_HEIGHT_M,
            "update_interval_steps": 25,
            "standing_command_threshold": 0.05,
        },
    )


@configclass
class LilgreenHardwareEnvCfg(V123HardwareAlignedEnvCfg):
    commands: HardwareCommandsCfg = HardwareCommandsCfg()
    observations: HardwareObservationsCfg = HardwareObservationsCfg()
    actions: V123ResidualActionsCfg = V123ResidualActionsCfg()
    rewards: HardwareRewardsCfg = HardwareRewardsCfg()
    terminations: V123TerminationsCfg = V123TerminationsCfg()
    events: HardwareEventsCfg = HardwareEventsCfg()
    curriculum: HardwareCurriculumsCfg = HardwareCurriculumsCfg()
