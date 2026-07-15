"""Measured ST3215 response data used by Berkeley Humanoid Lite v1.3.0 tasks.

Generated from Track 2 ``track1_actuator_model.json`` dataset v2. The constants are
empirical command-to-motion response descriptors. ``tau`` is an equivalent
first-order response proxy, not physical damping. The unloaded dataset does not
identify closed-loop stiffness in N*m/rad or physical damping in N*m*s/rad.
"""

from __future__ import annotations

SOURCE_DATASET_NAME = 'berkeley_humanoid_lite_st3215_track1_actuator_identification_v2'
DATASET_NAME = 'littlegreen_humanoid_lite_st3215_track1_actuator_identification_v2'
DATASET_REVISION_NOTE = 'Nominal left-knee source replaced with final post-mechanical-correction run after removal of an interfering long servo-horn screw.'

ST3215_MODEL_JOINT_NAMES = ['leg_left_hip_roll_joint', 'leg_left_hip_yaw_joint', 'leg_left_hip_pitch_joint', 'leg_left_knee_pitch_joint', 'leg_left_ankle_pitch_joint', 'leg_left_ankle_roll_joint', 'leg_right_hip_roll_joint', 'leg_right_hip_yaw_joint', 'leg_right_hip_pitch_joint', 'leg_right_knee_pitch_joint', 'leg_right_ankle_pitch_joint', 'leg_right_ankle_roll_joint']
ST3215_JOINT_FAMILIES = ['hip_roll', 'hip_yaw', 'hip_pitch', 'knee_pitch', 'ankle_pitch', 'ankle_roll', 'hip_roll', 'hip_yaw', 'hip_pitch', 'knee_pitch', 'ankle_pitch', 'ankle_roll']
ST3215_JOINT_SIDES = ['left', 'left', 'left', 'left', 'left', 'left', 'right', 'right', 'right', 'right', 'right', 'right']

ST3215_STEP_AMPLITUDE_KNOTS_RAD = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]
ST3215_PEAK_VELOCITY_CURVES_RAD_S = [[0.0, 0.1599211289934907, 0.3672376275589496, 0.5687369042201524, 0.7686608842873164, 0.9278029172887974], [0.0, 0.13745068365579424, 0.33239427999779997, 0.546476567624137, 0.7352050720699749, 0.892482880747503], [0.0, 0.12853602466007175, 0.3069279932357105, 0.5509577424978689, 0.7374106173824361, 0.9022010275417774], [0.0, 0.13980651141134964, 0.36131082189030383, 0.5853147895159424, 0.7605717261407328, 0.9302084371979238], [0.0, 0.13714341402743835, 0.35533336679811617, 0.5725799760907168, 0.7473755911667506, 0.9137536826264377], [0.0, 0.12984002709616457, 0.3502538441729349, 0.5667218916905497, 0.7440841461432939, 0.9038385895905592], [0.0, 0.1281854130965801, 0.34261049149232703, 0.556569987959203, 0.738029962332148, 0.9040483930682879], [0.0, 0.13196393794187988, 0.30715494388243114, 0.5254822498244924, 0.7200175606553139, 0.8633272842093787], [0.0, 0.14699364806686577, 0.3281470989646674, 0.572014175464848, 0.7518813778305018, 0.9010160816263557], [0.0, 0.1048669833337526, 0.3213336012953922, 0.5668004815771661, 0.7285791609061356, 0.8999017647606035], [0.0, 0.15498037426360534, 0.33017642600271396, 0.5664044722668659, 0.7422732660608355, 0.8972155485683265], [0.0, 0.1585722448084566, 0.3432934798029378, 0.5502552420952519, 0.7622520219036586, 0.9258548212101252]]

ST3215_TAU_MEDIAN_S = [0.07848864530228243, 0.07376044996569338, 0.07368140888732962, 0.06798579082029894, 0.06932546111634515, 0.07100886732714053, 0.06947962651368048, 0.0753084480788965, 0.07369036518620001, 0.06483566790096076, 0.06810058313720711, 0.06986317105832676]
ST3215_TAU_P10_S = [0.06040907100216242, 0.06313900906214533, 0.06194785885975092, 0.0590360456723359, 0.054964899239573595, 0.060247315802595666, 0.060415182348329995, 0.06733297361863677, 0.053209684256189646, 0.0610385976851731, 0.052149315860518144, 0.06474366410577062]
ST3215_TAU_P90_S = [0.0825934080074831, 0.07820908102545067, 0.07973455276583304, 0.07737895536655637, 0.08129130146384135, 0.08299801207443641, 0.07909514748405559, 0.08253020713970093, 0.07809094403450424, 0.08262641487475925, 0.08270744550851263, 0.08277440384382229]
ST3215_STATIC_GAIN_MEDIAN = [0.9817477042468103, 0.958737992428526, 0.966407896367954, 0.96257294439824, 0.9664078963679539, 0.9689645310144301, 0.9702428483376679, 0.970242848337668, 0.9625729443982397, 0.9549030404588121, 0.9638512617214778, 0.9817477042468103]
ST3215_SMALL_SIGNAL_ERROR_FLOOR_RAD = [0.0038932017272007673, 0.004660192121143575, 0.0038932017272007707, 0.005368932757599754, 0.005427182515086414, 0.00466019212114359, 0.005427182515086409, 0.005427182515086407, 0.001592230545372303, 0.006194172909029294, 0.003126211333257939, 0.003893201727200769]
ST3215_CENTER_HYSTERESIS_SPAN_RAD = [0.006135923151542565, 0.01073786551519948, 0.01073786551519948, 0.010737865515199507, 0.010737865515199493, 0.007669903939428206, 0.01227184630308513, 0.012271846303085115, 0.006135923151542572, 0.01380582709097078, 0.007669903939428208, 0.006135923151542565]
ST3215_SUSTAINED_MOTION_MEDIAN_S = [0.0642758755, 0.065867671, 0.07254573500000001, 0.07476643999999999, 0.0675801395, 0.06527400499999998, 0.06356524449999999, 0.062876084, 0.06778885600000001, 0.064948665, 0.0643943015, 0.0625720365]
ST3215_OVERSHOOT_PERCENT_MAX = [0.0, 16.582539879308733, 0.0, 15.048559091422863, 0.0, 0.0, 0.0, 13.514578303537439, 15.048559091423176, 0.0, 0.0, 0.0]
ST3215_RESPONSE_CLASS = ['monotonic_nonoscillatory_over_tested_range', 'underdamped_or_hysteretic_overshoot_observed', 'monotonic_nonoscillatory_over_tested_range', 'underdamped_or_hysteretic_overshoot_observed', 'monotonic_nonoscillatory_over_tested_range', 'monotonic_nonoscillatory_over_tested_range', 'monotonic_nonoscillatory_over_tested_range', 'underdamped_or_hysteretic_overshoot_observed', 'underdamped_or_hysteretic_overshoot_observed', 'monotonic_nonoscillatory_over_tested_range', 'monotonic_nonoscillatory_over_tested_range', 'monotonic_nonoscillatory_over_tested_range']

# Keep transport phase separate from the response-delay proxy. Track 2 can revise
# either component without changing policy action contract v3.
ST3215_BUS_PHASE_WAIT_RANGE_S = (0.0, 0.020)
'''ST3215_FIRST_ENCODER_DELAY_RANGE_S = (0.043442205050, 0.068951098900)
ST3215_FIRST_ENCODER_DELAY_MEDIAN_S = 0.062590821000'''
ST3215_FIRST_ENCODER_DELAY_RANGE_S = (0.02442205050, 0.048951098900)
ST3215_FIRST_ENCODER_DELAY_MEDIAN_S = 0.032590821000

ST3215_VELOCITY_SCALE_RANGE = (0.95, 1.05)

# Stage A carries overshoot/tail data as metadata but does not inject special
# transient branches until the monotonic response model is validated.
ST3215_SPECIAL_TRANSIENTS_ENABLED_BY_DEFAULT = False


def validate_st3215_model_constants(expected_joint_names: list[str] | None = None) -> None:
    """Validate array lengths, velocity curves, and canonical joint order."""
    n = len(ST3215_MODEL_JOINT_NAMES)
    arrays = [
        ST3215_JOINT_FAMILIES,
        ST3215_JOINT_SIDES,
        ST3215_PEAK_VELOCITY_CURVES_RAD_S,
        ST3215_TAU_MEDIAN_S,
        ST3215_TAU_P10_S,
        ST3215_TAU_P90_S,
        ST3215_STATIC_GAIN_MEDIAN,
        ST3215_SMALL_SIGNAL_ERROR_FLOOR_RAD,
        ST3215_CENTER_HYSTERESIS_SPAN_RAD,
        ST3215_SUSTAINED_MOTION_MEDIAN_S,
        ST3215_OVERSHOOT_PERCENT_MAX,
        ST3215_RESPONSE_CLASS,
    ]
    if any(len(values) != n for values in arrays):
        raise ValueError("ST3215 actuator-model arrays must match canonical joint count")
    curve_len = len(ST3215_STEP_AMPLITUDE_KNOTS_RAD)
    if any(len(curve) != curve_len for curve in ST3215_PEAK_VELOCITY_CURVES_RAD_S):
        raise ValueError("Each ST3215 velocity curve must match the amplitude knots")
    if expected_joint_names is not None and list(expected_joint_names) != ST3215_MODEL_JOINT_NAMES:
        raise ValueError(
            "ST3215 actuator-model joint order mismatch\n"
            + "expected=" + repr(list(expected_joint_names)) + "\n"
            + "model=" + repr(ST3215_MODEL_JOINT_NAMES)
        )


validate_st3215_model_constants()
