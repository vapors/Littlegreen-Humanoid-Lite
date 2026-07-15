"""Velocity-Lilgreen-Hardware-ST3215-Loaded-v2: v1.4.2 locomotion-pressure curriculum.

This task is intentionally additive. It preserves the v1.4.0 loaded-ST3215 actuator
model, action contract v3, 45-D observation contract, q_default, physical limits,
nominal COM height, and residual scale. Only the Hardware command curriculum and
Hardware-specific reward pressure are adjusted.
"""

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

from .env_cfg_hardware import HardwareRewardsCfg
from .v1_2_3_common import (
    FEET_BODY_PATTERN,
    HardwareCommandsCfg,
    HardwareObservationsCfg,
    V123TerminationsCfg,
)
from .v1_3_0_st3215_common import ST3215HardwareEventsCfg
from .v1_4_0_st3215_loaded_common import (
    V140ST3215LoadedActionsCfg,
    V140ST3215LoadedHardwareAlignedEnvCfg,
)


@configclass
class HardwareV142LocomotionPressureRewardsCfg(HardwareRewardsCfg):
    """Hardware rewards for the v1.4.2 locomotion-pressure experiment.

    Compared with v1.4.1, this task asks for more motion and reduces standing
    stickiness. Compared with v1.4.0, it keeps stronger raw-action and knee-health
    safeguards so the policy does not solve the harder command distribution by
    slamming against the residual clamp.
    """

    # Sharper velocity tracking, closer to the older locomotion-driven setup.
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        params={"command_name": "base_velocity", "std": 0.35},
        weight=3.0,
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        params={"command_name": "base_velocity", "std": 0.40},
        weight=1.6,
    )

    # Keep action-health safeguards, but make them slightly less sticky than v1.4.1.
    raw_action_excess_l2 = RewTerm(
        func=mdp.raw_action_excess_l2,
        params={"action_name": "joint_pos"},
        weight=-0.160,
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
        weight=-0.016,
    )
    knee_soft_torque_utilization = RewTerm(
        func=mdp.soft_torque_utilization_l2,
        params={
            "torque_limit_nm": ST3215_PEAK_TORQUE_NM,
            "soft_ratio": 0.65,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_knee_pitch_joint"], preserve_order=True
            ),
        },
        weight=-0.012,
    )

    # Less standing stickiness than v1.4.1; closer to the original Hardware rewards.
    stand_base_xy_speed = RewTerm(
        func=mdp.standing_base_xy_speed_l2,
        params={"command_name": "base_velocity", "command_threshold": 0.05},
        weight=-1.35,
    )
    stand_yaw_rate = RewTerm(
        func=mdp.standing_yaw_rate_l2,
        params={"command_name": "base_velocity", "command_threshold": 0.05},
        weight=-0.30,
    )
    stand_default_pose = RewTerm(
        func=mdp.standing_default_joint_pose_l2,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.05,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=ACTIONABLE_JOINTS_V1_2_3, preserve_order=True
            ),
        },
        weight=-0.45,
    )
    stand_base_height = RewTerm(
        func=mdp.standing_base_height_exp,
        params={
            "command_name": "base_velocity",
            "desired_height": NOMINAL_QDEFAULT_BASE_COM_HEIGHT_M,
            "std": 0.06,
        },
        weight=0.85,
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


@configclass
class ST3215LoadedV142HardwareCurriculumsCfg:
    hardware_stage = CurrTerm(func=mdp.st3215_loaded_v142_hardware_stage_curriculum)
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
            "torque_soft_ratio": 0.70,
            "desired_base_com_height_m": NOMINAL_QDEFAULT_BASE_COM_HEIGHT_M,
            "update_interval_steps": 25,
            "standing_command_threshold": 0.05,
        },
    )


@configclass
class LilgreenHardwareST3215LoadedV142EnvCfg(V140ST3215LoadedHardwareAlignedEnvCfg):
    """v1.4.2 Hardware locomotion-pressure task.

    Use this when resuming from the successful v1.4.0 Stand model_5000.pt to test
    whether stronger continuous command pressure produces decisive stepping without
    the command-bin complexity of a staged gait-seed curriculum.
    """

    commands: HardwareCommandsCfg = HardwareCommandsCfg()
    observations: HardwareObservationsCfg = HardwareObservationsCfg()
    actions: V140ST3215LoadedActionsCfg = V140ST3215LoadedActionsCfg()
    rewards: HardwareV142LocomotionPressureRewardsCfg = HardwareV142LocomotionPressureRewardsCfg()
    terminations: V123TerminationsCfg = V123TerminationsCfg()
    events: ST3215HardwareEventsCfg = ST3215HardwareEventsCfg()
    curriculum: ST3215LoadedV142HardwareCurriculumsCfg = ST3215LoadedV142HardwareCurriculumsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Keep generic gain randomization disabled; the measured ST3215 response model remains authoritative.
        self.events.actuator_gains = None
