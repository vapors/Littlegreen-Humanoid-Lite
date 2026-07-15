"""Velocity-Lilgreen-*-ST3215-Loaded-v5: v1.4.5 athletic vector-residual profile.

This branch is intentionally deployment-impacting. It keeps the policy I/O shape
(45-D observation, 12-D action) and the residual target equation, but changes the
q_default profile and uses a per-joint vector residual scale:

* a lower athletic q_default with more knee bend;
* more residual authority for hip/knee/ankle pitch joints;
* lower stand and moving COM-height targets;
* v1.4.4 alternating-step rewards made more grounded with no-flight, clearance-window,
  COM-over-stance-foot, and knee-flexion shaping.

Older v1.4.0-v1.4.5s2 tasks are left untouched for reproducibility.
"""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import littlegreen_humanoid_lite.tasks.locomotion.velocity.mdp as mdp
from littlegreen_humanoid_lite.tasks.locomotion.velocity.mdp.hardware_contract import (
    ACTIONABLE_JOINTS_V1_2_3,
    HARDWARE_LOWER_LIMIT_RAD,
    HARDWARE_UPPER_LIMIT_RAD,
    RESIDUAL_ACTION_SCALE_RAD_V1_4_5_ATHLETIC,
    RESIDUAL_ACTION_SCALE_RAD_V1_4_5_STABILIZED,
    ST3215_NO_LOAD_SPEED_RAD_S,
    ST3215_PEAK_TORQUE_NM,
    TRAINING_DEFAULT_RAD_V1_4_5_ATHLETIC,
    TRAINING_DEFAULT_RAD_V1_4_5_STABILIZED,
    V1_4_5_ATHLETIC_MOVING_BASE_COM_HEIGHT_M,
    V1_4_5_STABILIZED_MOVING_BASE_COM_HEIGHT_M,
    V1_4_5_STABILIZED_FORWARD_MOVING_BASE_COM_HEIGHT_M,
    V1_4_5_ATHLETIC_STAND_BASE_COM_HEIGHT_M,
    V1_4_5_STABILIZED_STAND_BASE_COM_HEIGHT_M,
    V1_4_5_STABILIZED_FORWARD_STAND_BASE_COM_HEIGHT_M,
    V1_4_5_STABILIZED_FORWARD_COM_TARGET_X_M,
    V1_4_5_STABILIZED_FORWARD_COM_BAND_HALF_WIDTH_M,
    V1_4_5_STABILIZED_FORWARD_LEAN_PROJECTED_GRAVITY_X,
    V1_4_5_STABILIZED_FORWARD2_MOVING_BASE_COM_HEIGHT_M,
    V1_4_5_STABILIZED_FORWARD2_STAND_BASE_COM_HEIGHT_M,
    V1_4_5_STABILIZED_FORWARD2_COM_TARGET_X_M,
    V1_4_5_STABILIZED_FORWARD2_COM_BAND_HALF_WIDTH_M,
    V1_4_5_STABILIZED_FORWARD2_LEAN_PROJECTED_GRAVITY_X,
    athletic_default_joint_pos_dict,
    athletic_stabilized_default_joint_pos_dict,
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
from .env_cfg_stand import StandRewardsCfg
from .v1_2_3_common import (
    FEET_BODY_PATTERN,
    HardwareCommandsCfg,
    HardwareObservationsCfg,
    StandCommandsCfg,
    StandObservationsCfg,
    V123TerminationsCfg,
)
from .v1_3_0_st3215_common import ST3215HardwareEventsCfg, ST3215StandEventsCfg
from .v1_4_0_st3215_loaded_common import V140ST3215LoadedHardwareAlignedEnvCfg


def _scaled(values: list[float], scale: float) -> list[float]:
    return [float(v) * scale for v in values]


def _scaled_curves(values: list[list[float]], scale: float) -> list[list[float]]:
    return [[float(v) * scale for v in row] for row in values]


def _apply_athletic_default(scene_robot_cfg) -> None:
    """Patch the robot init-state q_default for the v1.4.5 athletic profile."""
    scene_robot_cfg.init_state.joint_pos.update(athletic_default_joint_pos_dict(include_toes=True))


def _apply_stabilized_athletic_default(scene_robot_cfg) -> None:
    """Patch q_default for the v1.4.5 Stand-stabilized athletic profile."""
    scene_robot_cfg.init_state.joint_pos.update(athletic_stabilized_default_joint_pos_dict(include_toes=True))


@configclass
class V145ST3215LoadedAthleticActionsCfg:
    """Loaded ST3215 action model with v1.4.5 athletic q_default/vector residual.

    This keeps the residual target equation but changes the action profile:
    q_target = clip(q_default_athletic + clip(a_raw,-1,1) * residual_scale_vector, limits).
    Exported policies should be treated as action_contract_version 4 / athletic
    vector residual profile by deployment code.
    """

    joint_pos = mdp.ST3215MeasuredResidualJointPositionActionCfg(
        asset_name="robot",
        joint_names=ACTIONABLE_JOINTS_V1_2_3,
        lower_limits=HARDWARE_LOWER_LIMIT_RAD,
        upper_limits=HARDWARE_UPPER_LIMIT_RAD,
        residual_scale_rad=RESIDUAL_ACTION_SCALE_RAD_V1_4_5_ATHLETIC,
        preserve_order=True,
        actuator_model_name=(
            f"{DATASET_NAME}:stage_a_athletic_quick+{LOADED_DATASET_NAME}:stage_b_loaded_athletic_quick"
        ),
        actuator_model_stage="stage_b_loaded_v145_athletic_vector_residual",
        velocity_amplitude_knots_rad=ST3215_STEP_AMPLITUDE_KNOTS_RAD,
        velocity_curves_rad_s=_scaled_curves(ST3215_PEAK_VELOCITY_CURVES_RAD_S, 1.08),
        tau_median_s=_scaled(ST3215_TAU_MEDIAN_S, 0.84),
        tau_p10_s=_scaled(ST3215_TAU_P10_S, 0.84),
        tau_p90_s=_scaled(ST3215_TAU_P90_S, 0.84),
        static_gain=ST3215_STATIC_GAIN_MEDIAN,
        small_signal_error_floor_rad=ST3215_SMALL_SIGNAL_ERROR_FLOOR_RAD,
        center_hysteresis_span_rad=ST3215_CENTER_HYSTERESIS_SPAN_RAD,
        bus_phase_delay_s_range=ST3215_BUS_PHASE_WAIT_RANGE_S,
        response_delay_s_range=ST3215_FIRST_ENCODER_DELAY_RANGE_S,
        response_delay_s_nominal=ST3215_FIRST_ENCODER_DELAY_MEDIAN_S,
        response_delay_scale=0.68,
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
class HardwareV145AthleticStandRewardsCfg(StandRewardsCfg):
    """Standing rewards for rebuilding a lower athletic Stand checkpoint."""

    # The new default pose itself is knee-bent; keep posture but don't force the old tall COM.
    stand_base_height = RewTerm(
        func=mdp.standing_base_height_exp,
        params={
            "command_name": "base_velocity",
            "desired_height": V1_4_5_ATHLETIC_STAND_BASE_COM_HEIGHT_M,
            "std": 0.090,
        },
        weight=1.05,
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
        weight=-0.70,
    )
    raw_action_excess_l2 = RewTerm(func=mdp.raw_action_excess_l2, params={"action_name": "joint_pos"}, weight=-0.120)
    soft_torque_utilization = RewTerm(
        func=mdp.soft_torque_utilization_l2,
        params={
            "torque_limit_nm": ST3215_PEAK_TORQUE_NM,
            "soft_ratio": 0.72,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=ACTIONABLE_JOINTS_V1_2_3, preserve_order=True
            ),
        },
        weight=-0.018,
    )


@configclass
class HardwareV145GroundedStepRewardsCfg(HardwareRewardsCfg):
    """v1.4.5 Hardware rewards: lower, grounded, alternating command-aligned steps."""

    # Tracking and progress: keep the v1.4.4 movement pressure, but reward the body
    # moving in the command direction more than swing motion alone.
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        params={"command_name": "base_velocity", "std": 0.28},
        weight=3.7,
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        params={"command_name": "base_velocity", "std": 0.40},
        weight=1.6,
    )
    moving_velocity_along_command = RewTerm(
        func=mdp.moving_velocity_along_command,
        params={"command_name": "base_velocity", "command_threshold": 0.12},
        weight=1.8,
    )
    moving_no_progress_l1 = RewTerm(
        func=mdp.moving_no_progress_l1,
        params={"command_name": "base_velocity", "command_threshold": 0.12, "min_fraction": 0.50},
        weight=-1.45,
    )

    # Grounded alternating support. v1.4.4 alternated but hopped; v1.4.5 makes
    # single support valuable only when at least one foot remains grounded.
    moving_single_support_time = RewTerm(
        func=mdp.moving_single_support_time,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.12,
            "max_reward_time_s": 0.14,
        },
        weight=0.20,
    )
    moving_alternating_single_support = RewTerm(
        func=mdp.moving_alternating_single_support_reward,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.12,
            "force_threshold": 1.0,
        },
        weight=0.90,
    )
    moving_contact_switch = RewTerm(
        func=mdp.moving_contact_switch_reward,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.12,
            "force_threshold": 1.0,
        },
        weight=0.10,
    )
    moving_no_support = RewTerm(
        func=mdp.moving_no_support_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.12,
            "force_threshold": 1.0,
        },
        weight=-1.80,
    )
    moving_double_support_penalty = RewTerm(
        func=mdp.moving_double_support_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.12,
            "force_threshold": 1.0,
        },
        weight=-0.10,
    )
    moving_foot_air_balance = RewTerm(
        func=mdp.moving_foot_air_time_balance_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.12,
            "max_unpenalized_s": 0.10,
        },
        weight=-0.75,
    )
    moving_long_single_support = RewTerm(
        func=mdp.moving_long_single_support_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.12,
            "max_air_time_s": 0.22,
        },
        weight=-1.25,
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
        weight=-1.05,
    )
    moving_com_over_stance_foot = RewTerm(
        func=mdp.moving_com_over_stance_foot_reward,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.12,
            "force_threshold": 1.0,
            "std": 0.14,
        },
        weight=0.45,
    )

    # Lower athletic height and knee flexion permission.
    moving_relaxed_base_height = RewTerm(
        func=mdp.moving_base_height_exp,
        params={
            "command_name": "base_velocity",
            "desired_height": V1_4_5_ATHLETIC_MOVING_BASE_COM_HEIGHT_M,
            "std": 0.105,
            "command_threshold": 0.12,
        },
        weight=0.50,
    )
    moving_knee_flexion_band = RewTerm(
        func=mdp.moving_knee_flexion_band_reward,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_pitch_joint"], preserve_order=True),
            "command_threshold": 0.12,
            "lower_rad": 0.55,
            "upper_rad": 1.22,
            "std": 0.22,
        },
        weight=0.35,
    )

    # Step clearance: allow a small foot lift, discourage the v1.4.4 hopping solution.
    swing_foot_clearance = RewTerm(
        func=mdp.swing_foot_clearance,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.12,
            "clearance_target_m": 0.028,
            "max_reward": 1.0,
        },
        weight=0.25,
    )
    moving_swing_clearance_window = RewTerm(
        func=mdp.moving_swing_clearance_window_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.12,
            "max_clearance_m": 0.060,
        },
        weight=-4.0,
    )
    swing_foot_velocity_along_command = RewTerm(
        func=mdp.swing_foot_velocity_along_command,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.12,
        },
        weight=0.35,
    )

    # Keep safety, but do not over-penalize knees now that we need them to work.
    raw_action_excess_l2 = RewTerm(func=mdp.raw_action_excess_l2, params={"action_name": "joint_pos"}, weight=-0.140)
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
    knee_soft_torque_utilization = RewTerm(
        func=mdp.moving_soft_torque_utilization_l2,
        params={
            "command_name": "base_velocity",
            "torque_limit_nm": ST3215_PEAK_TORQUE_NM,
            "soft_ratio": 0.78,
            "command_threshold": 0.12,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_pitch_joint"], preserve_order=True),
        },
        weight=-0.006,
    )
    ankle_pitch_soft_torque_utilization = RewTerm(
        func=mdp.moving_soft_torque_utilization_l2,
        params={
            "command_name": "base_velocity",
            "torque_limit_nm": ST3215_PEAK_TORQUE_NM,
            "soft_ratio": 0.72,
            "command_threshold": 0.12,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint"], preserve_order=True),
        },
        weight=-0.020,
    )

    # Standing is still preserved, but around the lower athletic q_default.
    stand_base_xy_speed = RewTerm(
        func=mdp.standing_base_xy_speed_l2,
        params={"command_name": "base_velocity", "command_threshold": 0.05},
        weight=-0.90,
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
        weight=-0.35,
    )
    stand_base_height = RewTerm(
        func=mdp.standing_base_height_exp,
        params={
            "command_name": "base_velocity",
            "desired_height": V1_4_5_ATHLETIC_STAND_BASE_COM_HEIGHT_M,
            "std": 0.095,
        },
        weight=0.60,
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
class ST3215LoadedV145HardwareCurriculumsCfg:
    hardware_stage = CurrTerm(func=mdp.st3215_loaded_v145_hardware_stage_curriculum)
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
            "desired_base_com_height_m": V1_4_5_ATHLETIC_STAND_BASE_COM_HEIGHT_M,
            "update_interval_steps": 25,
            "standing_command_threshold": 0.05,
        },
    )


@configclass
class ST3215LoadedV145StandCurriculumsCfg:
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
            "desired_base_com_height_m": V1_4_5_ATHLETIC_STAND_BASE_COM_HEIGHT_M,
            "update_interval_steps": 25,
            "standing_command_threshold": 0.05,
        },
    )


@configclass
class LilgreenStandST3215LoadedV145EnvCfg(V140ST3215LoadedHardwareAlignedEnvCfg):
    """v1.4.5 athletic standing task. Train this before Hardware-v5."""

    commands: StandCommandsCfg = StandCommandsCfg()
    observations: StandObservationsCfg = StandObservationsCfg()
    actions: V145ST3215LoadedAthleticActionsCfg = V145ST3215LoadedAthleticActionsCfg()
    rewards: HardwareV145AthleticStandRewardsCfg = HardwareV145AthleticStandRewardsCfg()
    terminations: V123TerminationsCfg = V123TerminationsCfg()
    events: ST3215StandEventsCfg = ST3215StandEventsCfg()
    curriculum: ST3215LoadedV145StandCurriculumsCfg = ST3215LoadedV145StandCurriculumsCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_athletic_default(self.scene.robot)
        self.events.actuator_gains = None


@configclass
class LilgreenHardwareST3215LoadedV145EnvCfg(V140ST3215LoadedHardwareAlignedEnvCfg):
    """v1.4.5 athletic Hardware task for lower, grounded, alternating stepping."""

    commands: HardwareCommandsCfg = HardwareCommandsCfg()
    observations: HardwareObservationsCfg = HardwareObservationsCfg()
    actions: V145ST3215LoadedAthleticActionsCfg = V145ST3215LoadedAthleticActionsCfg()
    rewards: HardwareV145GroundedStepRewardsCfg = HardwareV145GroundedStepRewardsCfg()
    terminations: V123TerminationsCfg = V123TerminationsCfg()
    events: ST3215HardwareEventsCfg = ST3215HardwareEventsCfg()
    curriculum: ST3215LoadedV145HardwareCurriculumsCfg = ST3215LoadedV145HardwareCurriculumsCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_athletic_default(self.scene.robot)
        self.events.actuator_gains = None


@configclass
class V145ST3215LoadedAthleticStabilizedActionsCfg(V145ST3215LoadedAthleticActionsCfg):
    """v1.4.5 stabilized athletic action profile.

    Same contract v4/vector residual idea as v5, but with a moderated q_default
    applied by the environment. The residual vector is kept large for later gait.
    """

    joint_pos = mdp.ST3215MeasuredResidualJointPositionActionCfg(
        asset_name="robot",
        joint_names=ACTIONABLE_JOINTS_V1_2_3,
        lower_limits=HARDWARE_LOWER_LIMIT_RAD,
        upper_limits=HARDWARE_UPPER_LIMIT_RAD,
        residual_scale_rad=RESIDUAL_ACTION_SCALE_RAD_V1_4_5_STABILIZED,
        preserve_order=True,
        actuator_model_name=(
            f"{DATASET_NAME}:stage_a_athletic_stabilized+{LOADED_DATASET_NAME}:stage_b_loaded_athletic_stabilized"
        ),
        actuator_model_stage="stage_b_loaded_v145_stabilized_vector_residual",
        velocity_amplitude_knots_rad=ST3215_STEP_AMPLITUDE_KNOTS_RAD,
        velocity_curves_rad_s=_scaled_curves(ST3215_PEAK_VELOCITY_CURVES_RAD_S, 1.06),
        tau_median_s=_scaled(ST3215_TAU_MEDIAN_S, 0.88),
        tau_p10_s=_scaled(ST3215_TAU_P10_S, 0.88),
        tau_p90_s=_scaled(ST3215_TAU_P90_S, 0.88),
        static_gain=ST3215_STATIC_GAIN_MEDIAN,
        small_signal_error_floor_rad=ST3215_SMALL_SIGNAL_ERROR_FLOOR_RAD,
        center_hysteresis_span_rad=ST3215_CENTER_HYSTERESIS_SPAN_RAD,
        bus_phase_delay_s_range=ST3215_BUS_PHASE_WAIT_RANGE_S,
        response_delay_s_range=ST3215_FIRST_ENCODER_DELAY_RANGE_S,
        response_delay_s_nominal=ST3215_FIRST_ENCODER_DELAY_MEDIAN_S,
        response_delay_scale=0.72,
        velocity_scale_range=(1.00, 1.10),
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
        loaded_velocity_scale_range=(1.00, 1.10),
        randomize_loaded_velocity_scale=True,
    )


@configclass
class ST3215LoadedV145StabilizedStandEventsCfg(ST3215StandEventsCfg):
    """Stand-v5 stabilization events with gentle external balance perturbations.

    The interval push is deliberately small. It encourages both legs to participate
    in recovery without becoming a Hardware locomotion disturbance curriculum.
    """

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        params={"velocity_range": {"x": (-0.055, 0.055), "y": (-0.055, 0.055)}},
        mode="interval",
        interval_range_s=(5.0, 8.0),
    )


@configclass
class HardwareV145StabilizedStandRewardsCfg(HardwareV145AthleticStandRewardsCfg):
    """Stabilized Stand-v5 rewards.

    Targets a moderate 0.43-0.44 m athletic stance, reduces the incentive for a
    deep right-leg brace, and avoids knee-symmetry shaping that might be harmful
    for later gait transfer.
    """

    stand_base_height = RewTerm(
        func=mdp.standing_base_height_exp,
        params={
            "command_name": "base_velocity",
            "desired_height": V1_4_5_STABILIZED_STAND_BASE_COM_HEIGHT_M,
            "std": 0.075,
        },
        weight=0.95,
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
        weight=-0.90,
    )
    raw_action_excess_l2 = RewTerm(func=mdp.raw_action_excess_l2, params={"action_name": "joint_pos"}, weight=-0.140)
    soft_torque_utilization = RewTerm(
        func=mdp.soft_torque_utilization_l2,
        params={
            "torque_limit_nm": ST3215_PEAK_TORQUE_NM,
            "soft_ratio": 0.68,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=ACTIONABLE_JOINTS_V1_2_3, preserve_order=True
            ),
        },
        weight=-0.026,
    )
    stand_sagittal_soft_torque = RewTerm(
        func=mdp.standing_soft_torque_utilization_l2,
        params={
            "command_name": "base_velocity",
            "torque_limit_nm": ST3215_PEAK_TORQUE_NM,
            "soft_ratio": 0.62,
            "command_threshold": 0.05,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_pitch_joint", ".*_knee_pitch_joint", ".*_ankle_pitch_joint"],
                preserve_order=True,
            ),
        },
        weight=-0.045,
    )
    stand_contact_force_balance = RewTerm(
        func=mdp.standing_contact_force_balance_l2,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.05,
            "force_threshold": 1.0,
        },
        weight=-0.35,
    )
    stand_com_over_feet = RewTerm(
        func=mdp.standing_com_over_feet_l2,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.05,
        },
        weight=-1.10,
    )
    stand_both_feet_contact = RewTerm(
        func=mdp.standing_both_feet_contact,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "force_threshold": 1.0,
        },
        weight=0.80,
    )


@configclass
class HardwareV145StabilizedGroundedStepRewardsCfg(HardwareV145GroundedStepRewardsCfg):
    """Hardware-v5s rewards matching the stabilized standing profile."""

    moving_relaxed_base_height = RewTerm(
        func=mdp.moving_base_height_exp,
        params={
            "command_name": "base_velocity",
            "desired_height": V1_4_5_STABILIZED_MOVING_BASE_COM_HEIGHT_M,
            "std": 0.100,
            "command_threshold": 0.12,
        },
        weight=0.45,
    )
    stand_base_height = RewTerm(
        func=mdp.standing_base_height_exp,
        params={
            "command_name": "base_velocity",
            "desired_height": V1_4_5_STABILIZED_STAND_BASE_COM_HEIGHT_M,
            "std": 0.080,
        },
        weight=0.60,
    )


@configclass
class ST3215LoadedV145StabilizedStandCurriculumsCfg(ST3215LoadedV145StandCurriculumsCfg):
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
            "torque_soft_ratio": 0.68,
            "desired_base_com_height_m": V1_4_5_STABILIZED_STAND_BASE_COM_HEIGHT_M,
            "update_interval_steps": 25,
            "standing_command_threshold": 0.05,
        },
    )


@configclass
class ST3215LoadedV145StabilizedHardwareCurriculumsCfg(ST3215LoadedV145HardwareCurriculumsCfg):
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
            "desired_base_com_height_m": V1_4_5_STABILIZED_STAND_BASE_COM_HEIGHT_M,
            "update_interval_steps": 25,
            "standing_command_threshold": 0.05,
        },
    )


@configclass
class LilgreenStandST3215LoadedV145StabilizedEnvCfg(V140ST3215LoadedHardwareAlignedEnvCfg):
    """Stabilized v1.4.5 Stand task: moderate athletic stance plus anti-lean shaping."""

    commands: StandCommandsCfg = StandCommandsCfg()
    observations: StandObservationsCfg = StandObservationsCfg()
    actions: V145ST3215LoadedAthleticStabilizedActionsCfg = V145ST3215LoadedAthleticStabilizedActionsCfg()
    rewards: HardwareV145StabilizedStandRewardsCfg = HardwareV145StabilizedStandRewardsCfg()
    terminations: V123TerminationsCfg = V123TerminationsCfg()
    events: ST3215LoadedV145StabilizedStandEventsCfg = ST3215LoadedV145StabilizedStandEventsCfg()
    curriculum: ST3215LoadedV145StabilizedStandCurriculumsCfg = ST3215LoadedV145StabilizedStandCurriculumsCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_stabilized_athletic_default(self.scene.robot)
        self.events.actuator_gains = None


@configclass
class LilgreenHardwareST3215LoadedV145StabilizedEnvCfg(V140ST3215LoadedHardwareAlignedEnvCfg):
    """Hardware task matching the stabilized v1.4.5 default/profile."""

    commands: HardwareCommandsCfg = HardwareCommandsCfg()
    observations: HardwareObservationsCfg = HardwareObservationsCfg()
    actions: V145ST3215LoadedAthleticStabilizedActionsCfg = V145ST3215LoadedAthleticStabilizedActionsCfg()
    rewards: HardwareV145StabilizedGroundedStepRewardsCfg = HardwareV145StabilizedGroundedStepRewardsCfg()
    terminations: V123TerminationsCfg = V123TerminationsCfg()
    events: ST3215HardwareEventsCfg = ST3215HardwareEventsCfg()
    curriculum: ST3215LoadedV145StabilizedHardwareCurriculumsCfg = ST3215LoadedV145StabilizedHardwareCurriculumsCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_stabilized_athletic_default(self.scene.robot)
        self.events.actuator_gains = None


# v1.4.5s2: Forward-COM stand stabilization. This keeps the v5s q_default and
# contract-v4 vector residual profile, but uses the visually better 0.45 m stand
# height and a small stand-only forward COM/lean target so the policy does not
# settle with its mass slightly behind the feet.
@configclass
class HardwareV145StabilizedForwardStandRewardsCfg(HardwareV145StabilizedStandRewardsCfg):
    """Stand-v5s2 rewards: v5s plus forward COM placement and slight lean cue."""

    stand_base_height = RewTerm(
        func=mdp.standing_base_height_exp,
        params={
            "command_name": "base_velocity",
            "desired_height": V1_4_5_STABILIZED_FORWARD_STAND_BASE_COM_HEIGHT_M,
            "std": 0.080,
        },
        weight=0.90,
    )
    stand_com_over_feet = RewTerm(
        func=mdp.standing_com_forward_over_feet_band_l2,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.05,
            "target_forward_m": V1_4_5_STABILIZED_FORWARD_COM_TARGET_X_M,
            "band_half_width_m": V1_4_5_STABILIZED_FORWARD_COM_BAND_HALF_WIDTH_M,
            "lateral_weight": 0.25,
        },
        weight=-1.35,
    )
    stand_forward_lean = RewTerm(
        func=mdp.standing_forward_lean_projected_gravity_exp,
        params={
            "command_name": "base_velocity",
            "target_projected_gravity_x": V1_4_5_STABILIZED_FORWARD_LEAN_PROJECTED_GRAVITY_X,
            "std": 0.075,
            "command_threshold": 0.05,
        },
        weight=0.16,
    )


@configclass
class HardwareV145StabilizedForwardGroundedStepRewardsCfg(HardwareV145StabilizedGroundedStepRewardsCfg):
    """Hardware-v5s2 rewards matching the forward-COM standing profile."""

    moving_relaxed_base_height = RewTerm(
        func=mdp.moving_base_height_exp,
        params={
            "command_name": "base_velocity",
            "desired_height": V1_4_5_STABILIZED_FORWARD_MOVING_BASE_COM_HEIGHT_M,
            "std": 0.105,
            "command_threshold": 0.12,
        },
        weight=0.45,
    )
    stand_base_height = RewTerm(
        func=mdp.standing_base_height_exp,
        params={
            "command_name": "base_velocity",
            "desired_height": V1_4_5_STABILIZED_FORWARD_STAND_BASE_COM_HEIGHT_M,
            "std": 0.085,
        },
        weight=0.60,
    )
    stand_com_over_feet = RewTerm(
        func=mdp.standing_com_forward_over_feet_band_l2,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.05,
            "target_forward_m": V1_4_5_STABILIZED_FORWARD_COM_TARGET_X_M,
            "band_half_width_m": V1_4_5_STABILIZED_FORWARD_COM_BAND_HALF_WIDTH_M,
            "lateral_weight": 0.25,
        },
        weight=-0.75,
    )


@configclass
class ST3215LoadedV145StabilizedForwardStandCurriculumsCfg(ST3215LoadedV145StabilizedStandCurriculumsCfg):
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
            "torque_soft_ratio": 0.68,
            "desired_base_com_height_m": V1_4_5_STABILIZED_FORWARD_STAND_BASE_COM_HEIGHT_M,
            "update_interval_steps": 25,
            "standing_command_threshold": 0.05,
        },
    )


@configclass
class ST3215LoadedV145StabilizedForwardHardwareCurriculumsCfg(ST3215LoadedV145StabilizedHardwareCurriculumsCfg):
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
            "desired_base_com_height_m": V1_4_5_STABILIZED_FORWARD_STAND_BASE_COM_HEIGHT_M,
            "update_interval_steps": 25,
            "standing_command_threshold": 0.05,
        },
    )


@configclass
class LilgreenStandST3215LoadedV145StabilizedForwardEnvCfg(LilgreenStandST3215LoadedV145StabilizedEnvCfg):
    """Stand-v5s2: stabilized athletic default plus slight forward COM/lean target."""

    rewards: HardwareV145StabilizedForwardStandRewardsCfg = HardwareV145StabilizedForwardStandRewardsCfg()
    curriculum: ST3215LoadedV145StabilizedForwardStandCurriculumsCfg = ST3215LoadedV145StabilizedForwardStandCurriculumsCfg()


@configclass
class LilgreenHardwareST3215LoadedV145StabilizedForwardEnvCfg(LilgreenHardwareST3215LoadedV145StabilizedEnvCfg):
    """Hardware-v5s2 matching the forward-COM Stand seed profile."""

    rewards: HardwareV145StabilizedForwardGroundedStepRewardsCfg = HardwareV145StabilizedForwardGroundedStepRewardsCfg()
    curriculum: ST3215LoadedV145StabilizedForwardHardwareCurriculumsCfg = ST3215LoadedV145StabilizedForwardHardwareCurriculumsCfg()


# v1.4.5s3: Forward-COM + height refinement. This keeps the v5s/v5s2
# q_default and contract-v4 vector residual profile, but raises the explicit
# stand height to 0.460 m and moves the COM-over-feet target to 7 cm forward so
# the policy can shift over the feet without solving through ankle-pitch bracing.
@configclass
class HardwareV145StabilizedForward2StandRewardsCfg(HardwareV145StabilizedForwardStandRewardsCfg):
    """Stand-v5s3 rewards: v5s2 plus slightly taller height and more forward COM."""

    stand_base_height = RewTerm(
        func=mdp.standing_base_height_exp,
        params={
            "command_name": "base_velocity",
            "desired_height": V1_4_5_STABILIZED_FORWARD2_STAND_BASE_COM_HEIGHT_M,
            "std": 0.085,
        },
        weight=0.92,
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
        weight=-1.55,
    )
    stand_forward_lean = RewTerm(
        func=mdp.standing_forward_lean_projected_gravity_exp,
        params={
            "command_name": "base_velocity",
            "target_projected_gravity_x": V1_4_5_STABILIZED_FORWARD2_LEAN_PROJECTED_GRAVITY_X,
            "std": 0.080,
            "command_threshold": 0.05,
        },
        weight=0.18,
    )


@configclass
class HardwareV145StabilizedForward2GroundedStepRewardsCfg(HardwareV145StabilizedForwardGroundedStepRewardsCfg):
    """Hardware-v5s3 rewards matching the taller forward-COM Stand profile."""

    moving_relaxed_base_height = RewTerm(
        func=mdp.moving_base_height_exp,
        params={
            "command_name": "base_velocity",
            "desired_height": V1_4_5_STABILIZED_FORWARD2_MOVING_BASE_COM_HEIGHT_M,
            "std": 0.110,
            "command_threshold": 0.12,
        },
        weight=0.45,
    )
    stand_base_height = RewTerm(
        func=mdp.standing_base_height_exp,
        params={
            "command_name": "base_velocity",
            "desired_height": V1_4_5_STABILIZED_FORWARD2_STAND_BASE_COM_HEIGHT_M,
            "std": 0.090,
        },
        weight=0.62,
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


@configclass
class ST3215LoadedV145StabilizedForward2StandCurriculumsCfg(ST3215LoadedV145StabilizedForwardStandCurriculumsCfg):
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
            "torque_soft_ratio": 0.68,
            "desired_base_com_height_m": V1_4_5_STABILIZED_FORWARD2_STAND_BASE_COM_HEIGHT_M,
            "update_interval_steps": 25,
            "standing_command_threshold": 0.05,
        },
    )


@configclass
class ST3215LoadedV145StabilizedForward2HardwareCurriculumsCfg(ST3215LoadedV145StabilizedForwardHardwareCurriculumsCfg):
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
            "desired_base_com_height_m": V1_4_5_STABILIZED_FORWARD2_STAND_BASE_COM_HEIGHT_M,
            "update_interval_steps": 25,
            "standing_command_threshold": 0.05,
        },
    )


@configclass
class LilgreenStandST3215LoadedV145StabilizedForward2EnvCfg(LilgreenStandST3215LoadedV145StabilizedForwardEnvCfg):
    """Stand-v5s3: taller forward-COM stabilized athletic default."""

    rewards: HardwareV145StabilizedForward2StandRewardsCfg = HardwareV145StabilizedForward2StandRewardsCfg()
    curriculum: ST3215LoadedV145StabilizedForward2StandCurriculumsCfg = ST3215LoadedV145StabilizedForward2StandCurriculumsCfg()


@configclass
class LilgreenHardwareST3215LoadedV145StabilizedForward2EnvCfg(LilgreenHardwareST3215LoadedV145StabilizedForwardEnvCfg):
    """Hardware-v5s3 matching the taller forward-COM Stand seed profile."""

    rewards: HardwareV145StabilizedForward2GroundedStepRewardsCfg = HardwareV145StabilizedForward2GroundedStepRewardsCfg()
    curriculum: ST3215LoadedV145StabilizedForward2HardwareCurriculumsCfg = ST3215LoadedV145StabilizedForward2HardwareCurriculumsCfg()


# v1.4.6: Anti-planted locomotion. This branch keeps the v5s3 standing seed
# profile but makes "not moving under a moving command" a losing strategy.
@configclass
class HardwareV146AntiPlantedRewardsCfg(HardwareV145StabilizedForward2GroundedStepRewardsCfg):
    """Hardware-v1.4.6 rewards for escaping planted double-support bracing."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        params={"command_name": "base_velocity", "std": 0.24},
        weight=4.4,
    )
    moving_velocity_along_command = RewTerm(
        func=mdp.moving_velocity_along_command,
        params={"command_name": "base_velocity", "command_threshold": 0.22},
        weight=3.2,
    )
    moving_no_progress_l1 = RewTerm(
        func=mdp.moving_no_progress_l1,
        params={"command_name": "base_velocity", "command_threshold": 0.22, "min_fraction": 0.70},
        weight=-4.2,
    )
    moving_planted_no_progress = RewTerm(
        func=mdp.moving_planted_no_progress_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
            "min_fraction": 0.35,
            "max_double_support_s": 0.18,
        },
        weight=-5.0,
    )
    moving_long_double_support = RewTerm(
        func=mdp.moving_long_double_support_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
            "max_double_support_s": 0.28,
        },
        weight=-2.8,
    )
    moving_double_support_penalty = RewTerm(
        func=mdp.moving_double_support_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
            "force_threshold": 1.0,
        },
        weight=-0.35,
    )
    moving_single_support_time = RewTerm(
        func=mdp.moving_single_support_time,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
            "max_reward_time_s": 0.16,
        },
        weight=0.35,
    )
    moving_alternating_single_support = RewTerm(
        func=mdp.moving_alternating_single_support_reward,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
            "force_threshold": 1.0,
        },
        weight=1.10,
    )
    moving_contact_switch = RewTerm(
        func=mdp.moving_contact_switch_reward,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
            "force_threshold": 1.0,
        },
        weight=0.22,
    )
    moving_no_support = RewTerm(
        func=mdp.moving_no_support_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
            "force_threshold": 1.0,
        },
        weight=-1.25,
    )
    moving_stance_foot_slide = RewTerm(
        func=mdp.moving_stance_foot_slide_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
            "force_threshold": 1.0,
        },
        weight=-1.15,
    )
    swing_foot_velocity_along_command = RewTerm(
        func=mdp.swing_foot_velocity_along_command,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
        },
        weight=0.55,
    )
    swing_foot_clearance = RewTerm(
        func=mdp.swing_foot_clearance,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
            "clearance_target_m": 0.026,
            "max_reward": 1.0,
        },
        weight=0.28,
    )
    moving_swing_clearance_window = RewTerm(
        func=mdp.moving_swing_clearance_window_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
            "max_clearance_m": 0.055,
        },
        weight=-3.0,
    )
    # Keep no-flight / torque safety, but do not let torque penalties make bracing
    # preferable to taking a first risky step.
    raw_action_excess_l2 = RewTerm(func=mdp.raw_action_excess_l2, params={"action_name": "joint_pos"}, weight=-0.125)
    ankle_pitch_soft_torque_utilization = RewTerm(
        func=mdp.moving_soft_torque_utilization_l2,
        params={
            "command_name": "base_velocity",
            "torque_limit_nm": ST3215_PEAK_TORQUE_NM,
            "soft_ratio": 0.74,
            "command_threshold": 0.22,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint"], preserve_order=True),
        },
        weight=-0.014,
    )


@configclass
class V146AntiPlantedTerminationsCfg(V123TerminationsCfg):
    moving_no_progress = DoneTerm(
        func=mdp.moving_no_progress_timeout,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.22,
            "min_progress_fraction": 0.20,
            "grace_time_s": 0.80,
            "timeout_s": 1.20,
        },
    )


@configclass
class ST3215LoadedV146AntiPlantedHardwareCurriculumsCfg(ST3215LoadedV145StabilizedForward2HardwareCurriculumsCfg):
    hardware_stage = CurrTerm(func=mdp.st3215_loaded_v146_anti_planted_hardware_stage_curriculum)
    command_floor = CurrTerm(
        func=mdp.enforce_nonstanding_command_floor,
        params={"command_name": "base_velocity", "floor_mps": 0.22},
    )
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
            "desired_base_com_height_m": V1_4_5_STABILIZED_FORWARD2_STAND_BASE_COM_HEIGHT_M,
            "update_interval_steps": 25,
            "standing_command_threshold": 0.05,
        },
    )


@configclass
class LilgreenHardwareST3215LoadedV146AntiPlantedEnvCfg(LilgreenHardwareST3215LoadedV145StabilizedForward2EnvCfg):
    """v1.4.6 Hardware: anti-planted locomotion from the v5s3 stand seed.

    Intended warm-start path: load the Stand-v5s3 actor/normalization only, reset
    critic and optimizer, and make planted no-progress episodes terminate early.
    """

    rewards: HardwareV146AntiPlantedRewardsCfg = HardwareV146AntiPlantedRewardsCfg()
    terminations: V146AntiPlantedTerminationsCfg = V146AntiPlantedTerminationsCfg()
    curriculum: ST3215LoadedV146AntiPlantedHardwareCurriculumsCfg = ST3215LoadedV146AntiPlantedHardwareCurriculumsCfg()



@configclass
class HardwareV147PhaseGuidedObservationsCfg(HardwareObservationsCfg):
    """Hardware observations with a two-value gait phase scaffold.

    Policy observation dimension becomes 47-D:
    45-D v4 deployment observation + [sin(phase), cos(phase)].
    This is an intentional v1.4.7 training/deployment contract change for
    phase-guided alternating stepping.
    """

    @configclass
    class PolicyCfg(HardwareObservationsCfg.PolicyCfg):
        gait_phase = ObsTerm(func=mdp.gait_phase_sin_cos, params={"period_s": 0.72})

    @configclass
    class CriticCfg(PolicyCfg):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        def __post_init__(self):
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class HardwareV147PhaseGuidedRewardsCfg(HardwareV146AntiPlantedRewardsCfg):
    """v1.4.7 rewards: phase-guided grounded alternating gait scaffold.

    v1.4.6 made no-progress costly, but produced one-sided hopping/falling.  This
    reward set keeps command-aligned progress while adding an explicit left/right
    phase scaffold and stronger grounded/no-flight structure.
    """

    # Keep progress pressure, but soften the blunt v1.4.6 anti-stall terms so the
    # policy can learn a rhythm instead of falling forward to escape termination.
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        params={"command_name": "base_velocity", "std": 0.30},
        weight=3.6,
    )
    moving_velocity_along_command = RewTerm(
        func=mdp.moving_velocity_along_command,
        params={"command_name": "base_velocity", "command_threshold": 0.22},
        weight=2.2,
    )
    moving_no_progress_l1 = RewTerm(
        func=mdp.moving_no_progress_l1,
        params={"command_name": "base_velocity", "command_threshold": 0.22, "min_fraction": 0.45},
        weight=-2.6,
    )
    moving_planted_no_progress = RewTerm(
        func=mdp.moving_planted_no_progress_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
            "min_fraction": 0.30,
            "max_double_support_s": 0.24,
        },
        weight=-3.2,
    )
    moving_long_double_support = RewTerm(
        func=mdp.moving_long_double_support_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
            "max_double_support_s": 0.36,
        },
        weight=-1.8,
    )

    # Phase scaffold: reward the correct stance/swing foot for the current half-cycle.
    phase_contact = RewTerm(
        func=mdp.phase_guided_contact_reward,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
            "period_s": 0.72,
            "transition_width": 0.09,
            "force_threshold": 1.0,
        },
        weight=2.8,
    )
    phase_contact_mismatch = RewTerm(
        func=mdp.phase_guided_contact_mismatch_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
            "period_s": 0.72,
            "transition_width": 0.09,
            "force_threshold": 1.0,
        },
        weight=-1.4,
    )
    phase_swing_clearance = RewTerm(
        func=mdp.phase_guided_swing_clearance_reward,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
            "period_s": 0.72,
            "transition_width": 0.09,
            "target_clearance_m": 0.030,
            "clearance_std_m": 0.020,
            "force_threshold": 1.0,
        },
        weight=0.65,
    )
    phase_swing_velocity_along_command = RewTerm(
        func=mdp.phase_guided_swing_velocity_along_command,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
            "period_s": 0.72,
            "transition_width": 0.09,
            "force_threshold": 1.0,
        },
        weight=0.80,
    )
    phase_foot_placement = RewTerm(
        func=mdp.phase_guided_foot_placement_reward,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
            "period_s": 0.72,
            "transition_width": 0.09,
            "target_step_m": 0.060,
            "step_std_m": 0.055,
            "force_threshold": 1.0,
        },
        weight=0.90,
    )
    phase_long_air_hold = RewTerm(
        func=mdp.phase_guided_long_air_hold_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
            "max_air_time_s": 0.28,
        },
        weight=-2.2,
    )

    # Existing generic alternation/lift terms become secondary to the phase scaffold.
    moving_single_support_time = RewTerm(
        func=mdp.moving_single_support_time,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
            "max_reward_time_s": 0.12,
        },
        weight=0.15,
    )
    moving_alternating_single_support = RewTerm(
        func=mdp.moving_alternating_single_support_reward,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
            "force_threshold": 1.0,
        },
        weight=0.35,
    )
    moving_contact_switch = RewTerm(
        func=mdp.moving_contact_switch_reward,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
            "force_threshold": 1.0,
        },
        weight=0.06,
    )
    moving_no_support = RewTerm(
        func=mdp.moving_no_support_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
            "force_threshold": 1.0,
        },
        weight=-3.0,
    )
    moving_stance_foot_slide = RewTerm(
        func=mdp.moving_stance_foot_slide_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
            "force_threshold": 1.0,
        },
        weight=-1.25,
    )
    moving_swing_clearance_window = RewTerm(
        func=mdp.moving_swing_clearance_window_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
            "max_clearance_m": 0.052,
        },
        weight=-3.8,
    )
    raw_action_excess_l2 = RewTerm(func=mdp.raw_action_excess_l2, params={"action_name": "joint_pos"}, weight=-0.105)


@configclass
class V147PhaseGuidedTerminationsCfg(V146AntiPlantedTerminationsCfg):
    # Keep no-progress from becoming a planted exploit, but give the phase scaffold
    # more time than v1.4.6 before terminating.
    moving_no_progress = DoneTerm(
        func=mdp.moving_no_progress_timeout,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.22,
            "min_progress_fraction": 0.16,
            "grace_time_s": 1.20,
            "timeout_s": 2.00,
        },
    )
    moving_no_support = DoneTerm(
        func=mdp.moving_no_support_timeout,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
            "max_no_support_s": 0.12,
            "force_threshold": 1.0,
        },
    )


@configclass
class ST3215LoadedV147PhaseGuidedHardwareCurriculumsCfg(ST3215LoadedV146AntiPlantedHardwareCurriculumsCfg):
    hardware_stage = CurrTerm(func=mdp.st3215_loaded_v147_phase_guided_hardware_stage_curriculum)
    command_floor = CurrTerm(
        func=mdp.enforce_nonstanding_command_floor,
        params={"command_name": "base_velocity", "floor_mps": 0.22},
    )
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
            "desired_base_com_height_m": V1_4_5_STABILIZED_FORWARD2_STAND_BASE_COM_HEIGHT_M,
            "update_interval_steps": 25,
            "standing_command_threshold": 0.05,
        },
    )


@configclass
class LilgreenHardwareST3215LoadedV147PhaseGuidedEnvCfg(LilgreenHardwareST3215LoadedV146AntiPlantedEnvCfg):
    """v1.4.7 Hardware: phase-guided grounded alternating stepping.

    This intentionally changes the policy observation contract to 47-D by adding
    a [sin, cos] gait phase.  Use --policy_only_warm_start from Stand-v5s3; the
    loader partially copies the old 45-D actor input layer and leaves the two new
    phase columns freshly initialized.
    """

    observations: HardwareV147PhaseGuidedObservationsCfg = HardwareV147PhaseGuidedObservationsCfg()
    rewards: HardwareV147PhaseGuidedRewardsCfg = HardwareV147PhaseGuidedRewardsCfg()
    terminations: V147PhaseGuidedTerminationsCfg = V147PhaseGuidedTerminationsCfg()
    curriculum: ST3215LoadedV147PhaseGuidedHardwareCurriculumsCfg = ST3215LoadedV147PhaseGuidedHardwareCurriculumsCfg()



@configclass
class HardwareV148PhaseLiftStepRewardsCfg(HardwareV147PhaseGuidedRewardsCfg):
    """v1.4.8 rewards: convert phase rhythm into foot lift and placement.

    v1.4.7 learned a useful alternating rhythm but found a rocking shortcut with
    millimeter-scale clearance.  This reward set keeps the phase scaffold and
    adds explicit swing-foot clearance, commanded-direction placement, anti-rocking,
    and yaw stability shaping.
    """

    # Let clearance/placement shape the step before hard velocity pressure takes over.
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        params={"command_name": "base_velocity", "std": 0.38},
        weight=2.8,
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        params={"command_name": "base_velocity", "std": 0.34},
        weight=1.9,
    )
    moving_velocity_along_command = RewTerm(
        func=mdp.moving_velocity_along_command,
        params={"command_name": "base_velocity", "command_threshold": 0.22},
        weight=1.45,
    )
    moving_no_progress_l1 = RewTerm(
        func=mdp.moving_no_progress_l1,
        params={"command_name": "base_velocity", "command_threshold": 0.22, "min_fraction": 0.32},
        weight=-1.55,
    )
    moving_planted_no_progress = RewTerm(
        func=mdp.moving_planted_no_progress_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
            "min_fraction": 0.22,
            "max_double_support_s": 0.28,
        },
        weight=-2.2,
    )
    moving_long_double_support = RewTerm(
        func=mdp.moving_long_double_support_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
            "max_double_support_s": 0.42,
        },
        weight=-1.4,
    )

    # Keep the v1.4.7 phase clock, but make contact matching insufficient by itself.
    phase_contact = RewTerm(
        func=mdp.phase_guided_contact_reward,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
            "period_s": 0.72,
            "transition_width": 0.09,
            "force_threshold": 1.0,
        },
        weight=2.15,
    )
    phase_contact_mismatch = RewTerm(
        func=mdp.phase_guided_contact_mismatch_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
            "period_s": 0.72,
            "transition_width": 0.09,
            "force_threshold": 1.0,
        },
        weight=-1.15,
    )

    phase_swing_clearance = RewTerm(
        func=mdp.phase_guided_expected_swing_clearance_reward,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
            "period_s": 0.72,
            "transition_width": 0.09,
            "target_clearance_m": 0.030,
            "clearance_std_m": 0.017,
            "force_threshold": 1.0,
        },
        weight=2.15,
    )
    phase_swing_clearance_shortfall = RewTerm(
        func=mdp.phase_guided_swing_clearance_shortfall_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
            "period_s": 0.72,
            "transition_width": 0.09,
            "min_clearance_m": 0.016,
            "force_threshold": 1.0,
        },
        weight=-1.85,
    )
    phase_foot_placement = RewTerm(
        func=mdp.phase_guided_step_placement_target_reward,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
            "period_s": 0.72,
            "transition_width": 0.09,
            "target_step_m": 0.050,
            "step_std_m": 0.036,
            "force_threshold": 1.0,
        },
        weight=1.85,
    )
    phase_step_shortfall = RewTerm(
        func=mdp.phase_guided_step_shortfall_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
            "period_s": 0.72,
            "transition_width": 0.09,
            "min_step_m": 0.024,
            "force_threshold": 1.0,
        },
        weight=-1.55,
    )
    phase_swing_velocity_along_command = RewTerm(
        func=mdp.phase_guided_swing_velocity_along_command,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
            "period_s": 0.72,
            "transition_width": 0.09,
            "force_threshold": 1.0,
        },
        weight=0.35,
    )
    phase_rocking_no_step = RewTerm(
        func=mdp.phase_guided_rocking_no_step_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
            "period_s": 0.72,
            "transition_width": 0.09,
            "min_clearance_m": 0.014,
            "min_step_m": 0.022,
            "min_progress_fraction": 0.16,
            "force_threshold": 1.0,
        },
        weight=-2.05,
    )
    phase_long_air_hold = RewTerm(
        func=mdp.phase_guided_long_air_hold_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
            "max_air_time_s": 0.24,
        },
        weight=-2.55,
    )
    moving_yaw_stability = RewTerm(
        func=mdp.moving_yaw_stability_penalty,
        params={
            "command_name": "base_velocity",
            "lin_command_threshold": 0.12,
            "yaw_command_threshold": 0.12,
            "yaw_error_std": 0.50,
            "yaw_overspeed_margin": 0.30,
            "lateral_velocity_weight": 0.55,
            "yaw_only_linear_drift_weight": 0.40,
        },
        weight=-1.85,
    )

    moving_no_support = RewTerm(
        func=mdp.moving_no_support_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
            "force_threshold": 1.0,
        },
        weight=-3.35,
    )
    moving_swing_clearance_window = RewTerm(
        func=mdp.moving_swing_clearance_window_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
            "max_clearance_m": 0.060,
        },
        weight=-2.30,
    )
    moving_stance_foot_slide = RewTerm(
        func=mdp.moving_stance_foot_slide_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
            "force_threshold": 1.0,
        },
        weight=-1.20,
    )
    raw_action_excess_l2 = RewTerm(func=mdp.raw_action_excess_l2, params={"action_name": "joint_pos"}, weight=-0.105)


@configclass
class V148PhaseLiftStepTerminationsCfg(V147PhaseGuidedTerminationsCfg):
    # Let clearance and placement rewards shape the gait before no-progress stops
    # the episode.  Keep hard protection against sustained no-support/hopping.
    moving_no_progress = DoneTerm(
        func=mdp.moving_no_progress_timeout,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.22,
            "min_progress_fraction": 0.10,
            "grace_time_s": 1.80,
            "timeout_s": 3.00,
        },
    )
    moving_no_support = DoneTerm(
        func=mdp.moving_no_support_timeout,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.22,
            "max_no_support_s": 0.14,
            "force_threshold": 1.0,
        },
    )


@configclass
class ST3215LoadedV148PhaseLiftStepHardwareCurriculumsCfg(ST3215LoadedV147PhaseGuidedHardwareCurriculumsCfg):
    hardware_stage = CurrTerm(func=mdp.st3215_loaded_v148_phase_lift_step_hardware_stage_curriculum)
    command_floor = CurrTerm(
        func=mdp.enforce_nonstanding_command_floor,
        params={"command_name": "base_velocity", "floor_mps": 0.22},
    )


@configclass
class LilgreenHardwareST3215LoadedV148PhaseLiftStepEnvCfg(LilgreenHardwareST3215LoadedV147PhaseGuidedEnvCfg):
    """v1.4.8 Hardware: phase-guided lift-and-place stepping.

    Observation contract remains the v1.4.7 47-D phase-guided contract.  This is
    a reward/curriculum refinement that teaches the phase-selected swing foot to
    clear and place, rather than merely unweighting during the correct half-cycle.
    """

    rewards: HardwareV148PhaseLiftStepRewardsCfg = HardwareV148PhaseLiftStepRewardsCfg()
    terminations: V148PhaseLiftStepTerminationsCfg = V148PhaseLiftStepTerminationsCfg()
    curriculum: ST3215LoadedV148PhaseLiftStepHardwareCurriculumsCfg = ST3215LoadedV148PhaseLiftStepHardwareCurriculumsCfg()

