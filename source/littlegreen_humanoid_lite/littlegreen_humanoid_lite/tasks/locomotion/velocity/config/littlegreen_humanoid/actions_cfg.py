"""Canonical action contract for LittleGreen v2.2."""
from __future__ import annotations
from isaaclab.utils import configclass
from littlegreen_humanoid_lite.tasks.locomotion.velocity.mdp.hardware_contract import (
 ACTIONABLE_JOINTS_V1_2_3, HARDWARE_LOWER_LIMIT_RAD, HARDWARE_UPPER_LIMIT_RAD,
 RESIDUAL_ACTION_SCALE_RAD_V1_4_5_STABILIZED,
)
from littlegreen_humanoid_lite.tasks.locomotion.velocity.mdp.st3215_actuator_model import (
 DATASET_NAME, ST3215_BUS_PHASE_WAIT_RANGE_S, ST3215_CENTER_HYSTERESIS_SPAN_RAD,
 ST3215_FIRST_ENCODER_DELAY_MEDIAN_S, ST3215_FIRST_ENCODER_DELAY_RANGE_S,
 ST3215_PEAK_VELOCITY_CURVES_RAD_S, ST3215_SMALL_SIGNAL_ERROR_FLOOR_RAD,
 ST3215_STATIC_GAIN_MEDIAN, ST3215_STEP_AMPLITUDE_KNOTS_RAD, ST3215_TAU_MEDIAN_S,
 ST3215_TAU_P10_S, ST3215_TAU_P90_S, validate_st3215_model_constants,
)
from littlegreen_humanoid_lite.tasks.locomotion.velocity.mdp.st3215_loaded_actuator_model import (
 LOADED_CROUCH_DIRECTION_SIGN, LOADED_CROUCH_LOW_DEMAND_GAIN, LOADED_CROUCH_VMAX_RAD_S,
 LOADED_DATASET_NAME, LOADED_DIRECTION_CONDITIONING_WEIGHT, LOADED_ENVELOPE_COMBINATION_RULE,
 LOADED_RETURN_LOW_DEMAND_GAIN, LOADED_RETURN_TAU_SCALE, LOADED_RETURN_VMAX_RAD_S,
 validate_loaded_model_constants,
)
from . import mdp

def _scaled(values, scale): return [float(v)*scale for v in values]
def _scaled_curves(values, scale): return [[float(v)*scale for v in row] for row in values]
validate_st3215_model_constants(ACTIONABLE_JOINTS_V1_2_3)
validate_loaded_model_constants(ACTIONABLE_JOINTS_V1_2_3)

@configclass
class LittlegreenActionsCfg:
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
