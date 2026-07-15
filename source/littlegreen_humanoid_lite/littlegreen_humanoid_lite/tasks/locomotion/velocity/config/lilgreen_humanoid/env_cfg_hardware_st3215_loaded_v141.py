"""Velocity-Lilgreen-Hardware-ST3215-Loaded-v1: v1.4.1 safer Hardware curriculum.

This task is intentionally additive. It preserves the v1.4.0 loaded-ST3215 actuator
model, action contract v3, 45-D observation contract, q_default, and residual scale.
Only the Hardware locomotion curriculum and Hardware-specific retention/action-health
reward weights are adjusted.
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
class HardwareV141RetentionRewardsCfg(HardwareRewardsCfg):
    """Hardware rewards for the v1.4.1 slow-retention diagnostic curriculum.

    Compared with v1.4.0 Hardware-ST3215-Loaded-v0 this keeps locomotion rewards
    unchanged, but strengthens standing retention and discourages the raw-action and
    knee-bracing pattern that appeared after the first curriculum escalation.
    """

    raw_action_excess_l2 = RewTerm(
        func=mdp.raw_action_excess_l2,
        params={"action_name": "joint_pos"},
        weight=-0.180,
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
        weight=-0.018,
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
        weight=-0.015,
    )

    stand_base_xy_speed = RewTerm(
        func=mdp.standing_base_xy_speed_l2,
        params={"command_name": "base_velocity", "command_threshold": 0.05},
        weight=-2.20,
    )
    stand_yaw_rate = RewTerm(
        func=mdp.standing_yaw_rate_l2,
        params={"command_name": "base_velocity", "command_threshold": 0.05},
        weight=-0.50,
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
        weight=-0.75,
    )
    stand_base_height = RewTerm(
        func=mdp.standing_base_height_exp,
        params={
            "command_name": "base_velocity",
            "desired_height": NOMINAL_QDEFAULT_BASE_COM_HEIGHT_M,
            "std": 0.06,
        },
        weight=1.00,
    )
    stand_both_feet_contact = RewTerm(
        func=mdp.standing_both_feet_contact,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "force_threshold": 1.0,
        },
        weight=1.00,
    )
    stand_feet_slide = RewTerm(
        func=mdp.standing_feet_slide,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
        },
        weight=-1.00,
    )


@configclass
class ST3215LoadedV141HardwareCurriculumsCfg:
    hardware_stage = CurrTerm(func=mdp.st3215_loaded_v141_hardware_stage_curriculum)
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
class LilgreenHardwareST3215LoadedV141EnvCfg(V140ST3215LoadedHardwareAlignedEnvCfg):
    """v1.4.1 Hardware diagnostic task.

    Use this when resuming from the best v1.4.0 Hardware diagnostic checkpoint
    (currently model_6500.pt) or when restarting Hardware from the v1.4.0 Stand
    model_5000.pt under the slower curriculum.
    """

    commands: HardwareCommandsCfg = HardwareCommandsCfg()
    observations: HardwareObservationsCfg = HardwareObservationsCfg()
    actions: V140ST3215LoadedActionsCfg = V140ST3215LoadedActionsCfg()
    rewards: HardwareV141RetentionRewardsCfg = HardwareV141RetentionRewardsCfg()
    terminations: V123TerminationsCfg = V123TerminationsCfg()
    events: ST3215HardwareEventsCfg = ST3215HardwareEventsCfg()
    curriculum: ST3215LoadedV141HardwareCurriculumsCfg = ST3215LoadedV141HardwareCurriculumsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Keep generic gain randomization disabled; the measured ST3215 response model remains authoritative.
        self.events.actuator_gains = None
