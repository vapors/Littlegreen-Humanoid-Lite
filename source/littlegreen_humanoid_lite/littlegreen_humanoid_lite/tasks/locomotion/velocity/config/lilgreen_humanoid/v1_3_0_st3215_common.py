"""Shared configuration blocks for Berkeley Humanoid Lite v1.3.0 ST3215 tasks."""

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

from .v1_2_3_common import (
    HardwareEventsCfg,
    StandEventsCfg,
    V123HardwareAlignedEnvCfg,
)

validate_st3215_model_constants(ACTIONABLE_JOINTS_V1_2_3)


@configclass
class V130ST3215ActionsCfg:
    """Action contract v3 followed by the Track 2 measured-response Stage-A model."""

    joint_pos = mdp.ST3215MeasuredResidualJointPositionActionCfg(
        asset_name="robot",
        joint_names=ACTIONABLE_JOINTS_V1_2_3,
        lower_limits=HARDWARE_LOWER_LIMIT_RAD,
        upper_limits=HARDWARE_UPPER_LIMIT_RAD,
        residual_scale_rad=RESIDUAL_ACTION_SCALE_RAD_V1_2_3,
        preserve_order=True,
        actuator_model_name=f"{DATASET_NAME}:stage_a",
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
    )


@configclass
class ST3215StandEventsCfg(StandEventsCfg):
    """Stand randomization without invented stiffness/damping scaling."""

    actuator_gains = None


@configclass
class ST3215HardwareEventsCfg(HardwareEventsCfg):
    """Hardware curriculum events without generic actuator-gain randomization."""

    actuator_gains = None


@configclass
class V130ST3215HardwareAlignedEnvCfg(V123HardwareAlignedEnvCfg):
    """50 Hz task base preserving v1.2.3 policy observations and residual contract."""

    def __post_init__(self):
        super().__post_init__()
