"""Loaded standing-response calibration data for Berkeley Humanoid Lite v1.4.0.

The v1.4 model preserves the suspended single-joint ST3215 Stage-A response model
and adds a conservative loaded whole-body trajectory envelope calibrated from
``track1_loaded_actuator_extension_v2``.

The empirical fit is:

    v_loaded = min(gain * abs(v_ref), vmax)

These are loaded trajectory envelopes, not intrinsic motor free-speed limits or
torque-speed constants. The dataset does not identify physical stiffness, physical
damping, joint torque from load_ratio, or balance stability.

Direction conditioning is intentionally conservative. Only knee pitch uses the full
crouch/stand-return distinction by default; the other families use the average of
the two fitted envelopes. A modest 1.10x response-tau proxy is applied to knee
stand-return motion as a first-pass representation of the persistent loaded return
asymmetry. This multiplier is a modeling choice, not an identified damping constant.
"""

from __future__ import annotations

LOADED_DATASET_NAME = 'track1_loaded_actuator_extension_v2'
LOADED_DATASET_SCHEMA_VERSION = 2
LOADED_SOURCE_ARCHIVE = 'new_standing_pose_tests.zip'
LOADED_SOURCE_ARCHIVE_SHA256 = 'c40f1cf2d3adb9aa44cd8b84699d5c613440d3bb4828873087297ac7167370ba'
LOADED_COMMAND_RATE_HZ = 50.0
LOADED_MINIMUM_TRANSITION_DURATION_S = 0.25
LOADED_COMPLETED_REQUESTED_SPEEDS_RAD_S = [1.0, 2.0, 3.0, 4.5]

LOADED_MODEL_JOINT_NAMES = ['leg_left_hip_roll_joint',
 'leg_left_hip_yaw_joint',
 'leg_left_hip_pitch_joint',
 'leg_left_knee_pitch_joint',
 'leg_left_ankle_pitch_joint',
 'leg_left_ankle_roll_joint',
 'leg_right_hip_roll_joint',
 'leg_right_hip_yaw_joint',
 'leg_right_hip_pitch_joint',
 'leg_right_knee_pitch_joint',
 'leg_right_ankle_pitch_joint',
 'leg_right_ankle_roll_joint']
LOADED_JOINT_FAMILIES = ['hip_roll',
 'hip_yaw',
 'hip_pitch',
 'knee_pitch',
 'ankle_pitch',
 'ankle_roll',
 'hip_roll',
 'hip_yaw',
 'hip_pitch',
 'knee_pitch',
 'ankle_pitch',
 'ankle_roll']

# Sign of joint motion from normal stand toward deep crouch for the captured pose
# ladder. +1 means increasing joint position is crouch direction; -1 means decreasing.
LOADED_CROUCH_DIRECTION_SIGN = [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1]

# Per-joint loaded trajectory fit parameters, expanded from family fits.
LOADED_CROUCH_LOW_DEMAND_GAIN = [0.853240864877988,
 0.7742168144674261,
 0.5561976464574944,
 0.7093610570990286,
 0.835729227211154,
 0.7068385177542612,
 0.853240864877988,
 0.7742168144674261,
 0.5561976464574944,
 0.7093610570990286,
 0.835729227211154,
 0.7068385177542612]
LOADED_CROUCH_VMAX_RAD_S = [0.40581949464787187,
 0.6443286768088556,
 1.70053602694104,
 1.1776129510006368,
 0.7697119853572694,
 0.7221514635426171,
 0.40581949464787187,
 0.6443286768088556,
 1.70053602694104,
 1.1776129510006368,
 0.7697119853572694,
 0.7221514635426171]
LOADED_RETURN_LOW_DEMAND_GAIN = [0.7717192457086172,
 0.7080706538042757,
 0.5687273072895153,
 0.7146960927876614,
 0.8157768251787225,
 0.7100238552497466,
 0.7717192457086172,
 0.7080706538042757,
 0.5687273072895153,
 0.7146960927876614,
 0.8157768251787225,
 0.7100238552497466]
LOADED_RETURN_VMAX_RAD_S = [0.39511696307122546,
 0.49366834121343106,
 1.7304575268159563,
 1.1630749813090666,
 0.700353006241388,
 0.757723806246477,
 0.39511696307122546,
 0.49366834121343106,
 1.7304575268159563,
 1.1630749813090666,
 0.700353006241388,
 0.757723806246477]

# Data-informed modeling policy:
# - knee pitch: full direction conditioning
# - hip pitch: speed-conditioned but not direction-conditioned
# - ankle pitch: speed-sensitive, no robust broad direction penalty
# - roll families: conservative symmetric loaded envelope
# - hip yaw: cautious symmetric treatment because pose-path excitation is limited
LOADED_DIRECTION_CONDITIONING_WEIGHT = [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]

# Conservative first-pass response modifier for persistent knee stand-return
# asymmetry. Do not interpret as physical damping.
LOADED_RETURN_TAU_SCALE = [1, 1, 1, 1.1, 1, 1, 1, 1, 1, 1.1, 1, 1]

# Conservative episode-level envelope scale randomization. The v2 handoff has one
# completed run per speed condition, so this is intentionally narrow.
LOADED_VELOCITY_SCALE_RANGE = (0.95, 1.05)

LOADED_ENVELOPE_COMBINATION_RULE = (
    "min(stage_a_unloaded_velocity_cap, loaded_trajectory_envelope)"
)


def loaded_velocity_envelope_scalar(v_ref_rad_s: float, gain: float, vmax_rad_s: float) -> float:
    """Return the empirical loaded trajectory envelope for one reference speed."""
    if gain < 0.0 or vmax_rad_s <= 0.0:
        raise ValueError("gain must be non-negative and vmax_rad_s must be positive")
    return min(float(gain) * abs(float(v_ref_rad_s)), float(vmax_rad_s))


def validate_loaded_model_constants(expected_joint_names: list[str] | None = None) -> None:
    """Validate loaded-model lengths, signs, weights, and fit parameters."""
    n = len(LOADED_MODEL_JOINT_NAMES)
    arrays = [
        LOADED_JOINT_FAMILIES,
        LOADED_CROUCH_DIRECTION_SIGN,
        LOADED_CROUCH_LOW_DEMAND_GAIN,
        LOADED_CROUCH_VMAX_RAD_S,
        LOADED_RETURN_LOW_DEMAND_GAIN,
        LOADED_RETURN_VMAX_RAD_S,
        LOADED_DIRECTION_CONDITIONING_WEIGHT,
        LOADED_RETURN_TAU_SCALE,
    ]
    if any(len(values) != n for values in arrays):
        raise ValueError("Loaded actuator-model arrays must match canonical joint count")
    if any(sign not in (-1, 1) for sign in LOADED_CROUCH_DIRECTION_SIGN):
        raise ValueError("Crouch direction signs must be +/-1")
    if any(not 0.0 <= weight <= 1.0 for weight in LOADED_DIRECTION_CONDITIONING_WEIGHT):
        raise ValueError("Direction-conditioning weights must be in [0, 1]")
    if any(gain < 0.0 for gain in LOADED_CROUCH_LOW_DEMAND_GAIN + LOADED_RETURN_LOW_DEMAND_GAIN):
        raise ValueError("Loaded trajectory gains must be non-negative")
    if any(vmax <= 0.0 for vmax in LOADED_CROUCH_VMAX_RAD_S + LOADED_RETURN_VMAX_RAD_S):
        raise ValueError("Loaded trajectory vmax values must be positive")
    if any(scale < 1.0 for scale in LOADED_RETURN_TAU_SCALE):
        raise ValueError("Return tau scale must be >= 1.0")
    if expected_joint_names is not None and list(expected_joint_names) != LOADED_MODEL_JOINT_NAMES:
        raise ValueError(
            "Loaded actuator-model joint order mismatch\n"
            + "expected=" + repr(list(expected_joint_names)) + "\n"
            + "model=" + repr(LOADED_MODEL_JOINT_NAMES)
        )


validate_loaded_model_constants()
