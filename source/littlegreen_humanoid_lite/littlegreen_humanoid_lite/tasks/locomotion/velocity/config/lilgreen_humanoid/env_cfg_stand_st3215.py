"""Velocity-Lilgreen-Stand-ST3215-v0: v1.3.0 measured-actuator standing task."""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import littlegreen_humanoid_lite.tasks.locomotion.velocity.mdp as mdp
from littlegreen_humanoid_lite.tasks.locomotion.velocity.mdp.hardware_contract import (
    ACTIONABLE_JOINTS_V1_2_3,
    NOMINAL_QDEFAULT_BASE_COM_HEIGHT_M,
    ST3215_NO_LOAD_SPEED_RAD_S,
    ST3215_PEAK_TORQUE_NM,
)

from .env_cfg_stand import StandRewardsCfg
from .v1_2_3_common import (
    FEET_BODY_PATTERN,
    StandCommandsCfg,
    StandObservationsCfg,
    V123TerminationsCfg,
)
from .v1_3_0_st3215_common import (
    ST3215StandEventsCfg,
    V130ST3215ActionsCfg,
    V130ST3215HardwareAlignedEnvCfg,
)


@configclass
class ST3215StandCurriculumsCfg:
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
class LilgreenStandST3215EnvCfg(V130ST3215HardwareAlignedEnvCfg):
    commands: StandCommandsCfg = StandCommandsCfg()
    observations: StandObservationsCfg = StandObservationsCfg()
    actions: V130ST3215ActionsCfg = V130ST3215ActionsCfg()
    rewards: StandRewardsCfg = StandRewardsCfg()
    terminations: V123TerminationsCfg = V123TerminationsCfg()
    events: ST3215StandEventsCfg = ST3215StandEventsCfg()
    curriculum: ST3215StandCurriculumsCfg = ST3215StandCurriculumsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Track 2 did not identify physical stiffness/damping; do not layer generic
        # gain randomization on top of the measured response model.
        self.events.actuator_gains = None
