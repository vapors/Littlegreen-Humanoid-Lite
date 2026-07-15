"""Velocity-Lilgreen-Hardware-ST3215-Loaded-v4: v1.4.4 alternating-step curriculum.

This task is additive. It preserves the v1.4.0-v1.4.3 policy interface, action
contract v3, q_default, physical limits, and loaded ST3215 data, but makes the
Hardware transition more explicitly gait-seeking:

* fewer standing environments than v1.4.1;
* clear continuous movement commands without introducing command bins;
* stronger command-aligned progress pressure;
* alternating-support, air-time-balance, long-hold, and stance-foot-slide terms;
* relaxed moving height to allow knee bend while preserving standing target;
* a still-quick actuator response proxy, backed off slightly from v1.4.3.
"""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import littlegreen_humanoid_lite.tasks.locomotion.velocity.mdp as mdp
from littlegreen_humanoid_lite.tasks.locomotion.velocity.mdp.hardware_contract import (
    ACTIONABLE_JOINTS_V1_2_3,
    HARDWARE_LOWER_LIMIT_RAD,
    HARDWARE_UPPER_LIMIT_RAD,
    NOMINAL_QDEFAULT_BASE_COM_HEIGHT_M,
    RESIDUAL_ACTION_SCALE_RAD_V1_2_3,
    ST3215_NO_LOAD_SPEED_RAD_S,
    ST3215_PEAK_TORQUE_NM,
)
from littlegreen_humanoid_lite.tasks.locomotion.velocity.mdp.st3215_actuator_model import (
    DATASET_NAME,
    ST3215_BUS_PHASE_WAIT_RANGE_S,
    ST3215_CENTER_HYSTERESIS_SPAN_RAD,
    ST3215_FIRST_ENCODER_DELAY_MEDIAN_S,
    ST3215_FIRST_ENCODER_DELAY_RANGE_S,
    ST3215_PEAK_VELOCITY_CURVES_RAD_S,
    ST3215_SMALL_SIGNAL_ERROR_FLOOR_RAD,
    ST3215_STATIC_GAIN_MEDIAN,
    ST3215_STEP_AMPLITUDE_KNOTS_RAD,
    ST3215_TAU_MEDIAN_S,
    ST3215_TAU_P10_S,
    ST3215_TAU_P90_S,
    ST3215_VELOCITY_SCALE_RANGE,
)
from littlegreen_humanoid_lite.tasks.locomotion.velocity.mdp.st3215_loaded_actuator_model import (
    LOADED_CROUCH_DIRECTION_SIGN,
    LOADED_CROUCH_LOW_DEMAND_GAIN,
    LOADED_CROUCH_VMAX_RAD_S,
    LOADED_DATASET_NAME,
    LOADED_DIRECTION_CONDITIONING_WEIGHT,
    LOADED_ENVELOPE_COMBINATION_RULE,
    LOADED_RETURN_LOW_DEMAND_GAIN,
    LOADED_RETURN_TAU_SCALE,
    LOADED_RETURN_VMAX_RAD_S,
)

from .env_cfg_hardware import HardwareRewardsCfg
from .v1_2_3_common import (
    FEET_BODY_PATTERN,
    HardwareCommandsCfg,
    HardwareObservationsCfg,
    V123TerminationsCfg,
)
from .v1_3_0_st3215_common import ST3215HardwareEventsCfg
from .v1_4_0_st3215_loaded_common import V140ST3215LoadedHardwareAlignedEnvCfg


def _scaled(values: list[float], scale: float) -> list[float]:
    return [float(v) * scale for v in values]


def _scaled_curves(values: list[list[float]], scale: float) -> list[list[float]]:
    return [[float(v) * scale for v in row] for row in values]


@configclass
class V144ST3215LoadedQuickActionsCfg:
    """Action contract v3 with a quick-but-less-exploitable training response model.

    This does not change the ONNX/deployment action semantics. It only keeps a quicker-than-v1.4.0 response proxy while backing off the v1.4.3 speed boost slightly so gait-quality rewards, not one-foot bracing, drive movement.
    """

    joint_pos = mdp.ST3215MeasuredResidualJointPositionActionCfg(
        asset_name="robot",
        joint_names=ACTIONABLE_JOINTS_V1_2_3,
        lower_limits=HARDWARE_LOWER_LIMIT_RAD,
        upper_limits=HARDWARE_UPPER_LIMIT_RAD,
        residual_scale_rad=RESIDUAL_ACTION_SCALE_RAD_V1_2_3,
        preserve_order=True,
        actuator_model_name=(
            f"{DATASET_NAME}:stage_a_quick_train+{LOADED_DATASET_NAME}:stage_b_loaded_quick_train"
        ),
        actuator_model_stage="stage_b_loaded_v144_quick_train",
        velocity_amplitude_knots_rad=ST3215_STEP_AMPLITUDE_KNOTS_RAD,
        velocity_curves_rad_s=_scaled_curves(ST3215_PEAK_VELOCITY_CURVES_RAD_S, 1.08),
        tau_median_s=_scaled(ST3215_TAU_MEDIAN_S, 0.85),
        tau_p10_s=_scaled(ST3215_TAU_P10_S, 0.85),
        tau_p90_s=_scaled(ST3215_TAU_P90_S, 0.85),
        static_gain=ST3215_STATIC_GAIN_MEDIAN,
        small_signal_error_floor_rad=ST3215_SMALL_SIGNAL_ERROR_FLOOR_RAD,
        center_hysteresis_span_rad=ST3215_CENTER_HYSTERESIS_SPAN_RAD,
        bus_phase_delay_s_range=ST3215_BUS_PHASE_WAIT_RANGE_S,
        response_delay_s_range=ST3215_FIRST_ENCODER_DELAY_RANGE_S,
        response_delay_s_nominal=ST3215_FIRST_ENCODER_DELAY_MEDIAN_S,
        response_delay_scale=0.70,
        velocity_scale_range=(1.02, 1.14),
        randomize_tau=True,
        randomize_velocity_scale=True,
        randomize_response_delay=True,
        randomize_bus_phase=True,
        loaded_envelope_enabled=True,
        loaded_dataset_name=LOADED_DATASET_NAME,
        loaded_envelope_combination_rule=LOADED_ENVELOPE_COMBINATION_RULE,
        loaded_crouch_direction_sign=LOADED_CROUCH_DIRECTION_SIGN,
        loaded_crouch_low_demand_gain=LOADED_CROUCH_LOW_DEMAND_GAIN,
        loaded_crouch_vmax_rad_s=LOADED_CROUCH_VMAX_RAD_S,
        loaded_return_low_demand_gain=LOADED_RETURN_LOW_DEMAND_GAIN,
        loaded_return_vmax_rad_s=LOADED_RETURN_VMAX_RAD_S,
        loaded_direction_conditioning_weight=LOADED_DIRECTION_CONDITIONING_WEIGHT,
        loaded_return_tau_scale=LOADED_RETURN_TAU_SCALE,
        loaded_velocity_scale_range=(1.02, 1.12),
        randomize_loaded_velocity_scale=True,
    )


@configclass
class HardwareV144AlternatingStepRewardsCfg(HardwareRewardsCfg):
    """Hardware rewards for the v1.4.4 alternating-step experiment."""

    # Make commanded translation matter more than standing still under a nonzero command.
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        params={"command_name": "base_velocity", "std": 0.28},
        weight=3.6,
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        params={"command_name": "base_velocity", "std": 0.40},
        weight=1.6,
    )
    moving_velocity_along_command = RewTerm(
        func=mdp.moving_velocity_along_command,
        params={"command_name": "base_velocity", "command_threshold": 0.12},
        weight=1.4,
    )
    moving_no_progress_l1 = RewTerm(
        func=mdp.moving_no_progress_l1,
        params={"command_name": "base_velocity", "command_threshold": 0.12, "min_fraction": 0.45},
        weight=-1.25,
    )

    # Anti-brace / step-discovery terms. These are only active for clear movement commands.
    moving_single_support_time = RewTerm(
        func=mdp.moving_single_support_time,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.12,
            "max_reward_time_s": 0.16,
        },
        weight=0.25,
    )
    moving_alternating_single_support = RewTerm(
        func=mdp.moving_alternating_single_support_reward,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.12,
            "force_threshold": 1.0,
        },
        weight=1.0,
    )
    moving_contact_switch = RewTerm(
        func=mdp.moving_contact_switch_reward,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.12,
            "force_threshold": 1.0,
        },
        weight=0.15,
    )
    moving_foot_air_balance = RewTerm(
        func=mdp.moving_foot_air_time_balance_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.12,
            "max_unpenalized_s": 0.12,
        },
        weight=-0.65,
    )
    moving_long_single_support = RewTerm(
        func=mdp.moving_long_single_support_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.12,
            "max_air_time_s": 0.28,
        },
        weight=-0.75,
    )
    moving_stance_foot_slide = RewTerm(
        func=mdp.moving_stance_foot_slide_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.12,
            "force_threshold": 1.0,
        },
        weight=-0.85,
    )
    moving_relaxed_base_height = RewTerm(
        func=mdp.moving_base_height_exp,
        params={
            "command_name": "base_velocity",
            "desired_height": NOMINAL_QDEFAULT_BASE_COM_HEIGHT_M - 0.040,
            "std": 0.085,
            "command_threshold": 0.12,
        },
        weight=0.35,
    )
    swing_foot_clearance = RewTerm(
        func=mdp.swing_foot_clearance,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.12,
            "clearance_target_m": 0.030,
            "max_reward": 1.0,
        },
        weight=0.35,
    )
    swing_foot_velocity_along_command = RewTerm(
        func=mdp.swing_foot_velocity_along_command,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.12,
        },
        weight=0.45,
    )
    moving_double_support_penalty = RewTerm(
        func=mdp.moving_double_support_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.12,
            "force_threshold": 1.0,
        },
        weight=-0.18,
    )

    # Keep safety terms, but do not make standing so sticky that it wins over moving.
    raw_action_excess_l2 = RewTerm(
        func=mdp.raw_action_excess_l2,
        params={"action_name": "joint_pos"},
        weight=-0.150,
    )
    soft_torque_utilization = RewTerm(
        func=mdp.soft_torque_utilization_l2,
        params={
            "torque_limit_nm": ST3215_PEAK_TORQUE_NM,
            "soft_ratio": 0.72,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=ACTIONABLE_JOINTS_V1_2_3, preserve_order=True
            ),
        },
        weight=-0.015,
    )
    knee_soft_torque_utilization = RewTerm(
        func=mdp.soft_torque_utilization_l2,
        params={
            "torque_limit_nm": ST3215_PEAK_TORQUE_NM,
            "soft_ratio": 0.68,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_pitch_joint"], preserve_order=True),
        },
        weight=-0.012,
    )

    stand_base_xy_speed = RewTerm(
        func=mdp.standing_base_xy_speed_l2,
        params={"command_name": "base_velocity", "command_threshold": 0.05},
        weight=-0.85,
    )
    stand_yaw_rate = RewTerm(
        func=mdp.standing_yaw_rate_l2,
        params={"command_name": "base_velocity", "command_threshold": 0.05},
        weight=-0.25,
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
        weight=-0.30,
    )
    stand_base_height = RewTerm(
        func=mdp.standing_base_height_exp,
        params={
            "command_name": "base_velocity",
            "desired_height": NOMINAL_QDEFAULT_BASE_COM_HEIGHT_M - 0.015,
            "std": 0.075,
        },
        weight=0.55,
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
        weight=-0.75,
    )


@configclass
class ST3215LoadedV144HardwareCurriculumsCfg:
    hardware_stage = CurrTerm(func=mdp.st3215_loaded_v144_hardware_stage_curriculum)
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
            "torque_soft_ratio": 0.72,
            "desired_base_com_height_m": NOMINAL_QDEFAULT_BASE_COM_HEIGHT_M,
            "update_interval_steps": 25,
            "standing_command_threshold": 0.05,
        },
    )


@configclass
class LilgreenHardwareST3215LoadedV144EnvCfg(V140ST3215LoadedHardwareAlignedEnvCfg):
    """v1.4.4 Hardware alternating-step task.

    Resume this task from the successful v1.4.0 Stand model_5000.pt when testing
    whether alternating, command-aligned gait rewards plus a still-quick actuator
    response proxy gets the robot to place both feet and translate.
    """

    commands: HardwareCommandsCfg = HardwareCommandsCfg()
    observations: HardwareObservationsCfg = HardwareObservationsCfg()
    actions: V144ST3215LoadedQuickActionsCfg = V144ST3215LoadedQuickActionsCfg()
    rewards: HardwareV144AlternatingStepRewardsCfg = HardwareV144AlternatingStepRewardsCfg()
    terminations: V123TerminationsCfg = V123TerminationsCfg()
    events: ST3215HardwareEventsCfg = ST3215HardwareEventsCfg()
    curriculum: ST3215LoadedV144HardwareCurriculumsCfg = ST3215LoadedV144HardwareCurriculumsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.events.actuator_gains = None
