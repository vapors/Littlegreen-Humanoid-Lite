"""Shared configuration blocks for Berkeley Humanoid Lite v1.4.0 loaded ST3215 tasks.

v1.4.0 preserves the v1.3.1 Stage-A suspended single-joint ST3215 response model
and adds the v2 loaded standing-transition calibration layer. The policy action
contract, observation vector, residual scale, physical limits, and nominal COM
height are unchanged.
"""

from __future__ import annotations

from isaaclab.utils import configclass

import littlegreen_humanoid_lite.tasks.locomotion.velocity.mdp as mdp
from littlegreen_humanoid_lite.tasks.locomotion.velocity.mdp.hardware_contract import (
    ACTIONABLE_JOINTS_V1_2_3,
    HARDWARE_LOWER_LIMIT_RAD,
    HARDWARE_UPPER_LIMIT_RAD,
    RESIDUAL_ACTION_SCALE_RAD_V1_2_3,
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
    validate_st3215_model_constants,
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
    LOADED_VELOCITY_SCALE_RANGE,
    validate_loaded_model_constants,
)

from .v1_3_0_st3215_common import (
    ST3215HardwareEventsCfg,
    ST3215StandEventsCfg,
    V130ST3215HardwareAlignedEnvCfg,
)

validate_st3215_model_constants(ACTIONABLE_JOINTS_V1_2_3)
validate_loaded_model_constants(ACTIONABLE_JOINTS_V1_2_3)


@configclass
class V140ST3215LoadedActionsCfg:
    """Action contract v3 + Stage-A response + Stage-B loaded envelope."""

    joint_pos = mdp.ST3215MeasuredResidualJointPositionActionCfg(
        asset_name="robot",
        joint_names=ACTIONABLE_JOINTS_V1_2_3,
        lower_limits=HARDWARE_LOWER_LIMIT_RAD,
        upper_limits=HARDWARE_UPPER_LIMIT_RAD,
        residual_scale_rad=RESIDUAL_ACTION_SCALE_RAD_V1_2_3,
        preserve_order=True,
        actuator_model_name=(
            f"{DATASET_NAME}:stage_a+{LOADED_DATASET_NAME}:stage_b_loaded"
        ),
        actuator_model_stage="stage_b_loaded",
        # Preserve the v1.3.1 suspended single-joint response model.
        velocity_amplitude_knots_rad=ST3215_STEP_AMPLITUDE_KNOTS_RAD,
        velocity_curves_rad_s=ST3215_PEAK_VELOCITY_CURVES_RAD_S,
        tau_median_s=ST3215_TAU_MEDIAN_S,
        tau_p10_s=ST3215_TAU_P10_S,
        tau_p90_s=ST3215_TAU_P90_S,
        static_gain=ST3215_STATIC_GAIN_MEDIAN,
        small_signal_error_floor_rad=ST3215_SMALL_SIGNAL_ERROR_FLOOR_RAD,
        center_hysteresis_span_rad=ST3215_CENTER_HYSTERESIS_SPAN_RAD,
        bus_phase_delay_s_range=ST3215_BUS_PHASE_WAIT_RANGE_S,
        response_delay_s_range=ST3215_FIRST_ENCODER_DELAY_RANGE_S,
        response_delay_s_nominal=ST3215_FIRST_ENCODER_DELAY_MEDIAN_S,
        velocity_scale_range=ST3215_VELOCITY_SCALE_RANGE,
        randomize_tau=True,
        randomize_velocity_scale=True,
        randomize_response_delay=True,
        randomize_bus_phase=True,
        # v1.4.0 loaded standing-transition calibration layer.
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
        loaded_velocity_scale_range=LOADED_VELOCITY_SCALE_RANGE,
        randomize_loaded_velocity_scale=True,
    )


@configclass
class V140ST3215LoadedHardwareAlignedEnvCfg(V130ST3215HardwareAlignedEnvCfg):
    """50 Hz v1.4 base preserving the v1.3.1 policy/deployment contract."""

    def __post_init__(self):
        super().__post_init__()
