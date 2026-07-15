"""Hardware-aligned action contracts and constants for Berkeley Humanoid Lite v1.2.3.

The order in this file is the canonical 12-action order shared with the Orange Pi
ROS 2 deployment stack.  The limits are the calibrated physical joint envelope,
not the wider legacy URDF limits.
"""

from __future__ import annotations

ACTIONABLE_JOINTS_V1_2_3 = [
    "leg_left_hip_roll_joint",
    "leg_left_hip_yaw_joint",
    "leg_left_hip_pitch_joint",
    "leg_left_knee_pitch_joint",
    "leg_left_ankle_pitch_joint",
    "leg_left_ankle_roll_joint",
    "leg_right_hip_roll_joint",
    "leg_right_hip_yaw_joint",
    "leg_right_hip_pitch_joint",
    "leg_right_knee_pitch_joint",
    "leg_right_ankle_pitch_joint",
    "leg_right_ankle_roll_joint",
]

TRAINING_DEFAULT_RAD = [
    0.0,
    0.0,
    -0.1,
    0.4,
    -0.3,
    0.0,
    0.0,
    0.0,
    -0.1,
    0.4,
    -0.3,
    0.0,
]
'''
HARDWARE_LOWER_LIMIT_RAD = [
    -0.698,
    -0.087,
    -1.483,
    0.0,
    -0.384,
    -0.524,
    -0.698,
    -0.087,
    -1.483,
    0.0,
    -0.384,
    -0.524,
]

HARDWARE_UPPER_LIMIT_RAD = [
    0.524,
    1.221,
    0.785,
    1.483,
    1.134,
    0.785,
    0.524,
    1.221,
    0.785,
    1.483,
    1.134,
    0.785,
]
'''
# New captured limits with shim
HARDWARE_LOWER_LIMIT_RAD = [
    -0.694893296912195,
    -0.0889708856973672,
    -1.92207792722071,
    0.134990309333936,
    -0.809941856003619,
    -0.51388356394169,
    -0.874369049094815,
    -0.0567572891517687,
    -1.99110706267556,
    0.171805848243192,
    -0.845223414124988,
    -0.44332044769895,
]

HARDWARE_UPPER_LIMIT_RAD = [
    0.780796221033791,
    0.644271930911969,
    0.681087469821225,
    2.23501000794938,
    0.710233104791052,
    0.912718568791956,
    0.642737950124084,
    0.701029220063738,
    0.546097160487288,
    2.24114593110092,
    0.819145740730932,
    1.06151470521686,
]
# ST3215 12 V / 30 kg.cm hardware-oriented first-pass limits.
# Keep these in one place so Track 2 measurements can refine them later.
ST3215_PEAK_TORQUE_NM = 2.94
ST3215_NO_LOAD_SPEED_RAD_S = 4.72

# v1.2.3 first residual-action experiment. Network output remains normalized,
# but the physical target is constrained to a symmetric +/-0.20 rad residual
# around q_default before the final physical-limit clip.
RESIDUAL_ACTION_SCALE_RAD_V1_2_3 = 0.20

# Deterministic q_default root-body COM height measured with
# scripts/rsl_rl/measure_nominal_pose.py on Velocity-Lilgreen-Stand-v0.
NOMINAL_QDEFAULT_BASE_COM_HEIGHT_M = 0.4899105727672577


def map_bounded_action_scalar(action: float, default: float, lower: float, upper: float) -> float:
    """Map one normalized action in [-1, 1] around the training default.

    Negative action uses the available range from default toward ``lower``;
    positive action uses the available range from default toward ``upper``.
    """
    a = max(-1.0, min(1.0, float(action)))
    if a >= 0.0:
        return default + a * (upper - default)
    return default + (-a) * (lower - default)


def map_bounded_residual_action_scalar(
    action: float,
    default: float,
    lower: float,
    upper: float,
    residual_scale_rad: float = RESIDUAL_ACTION_SCALE_RAD_V1_2_3,
) -> float:
    """Map one normalized action to a symmetric residual target around ``default``.

    The network action is clamped to ``[-1, 1]``, multiplied by the residual
    scale in radians, added to ``default``, then clipped to the physical joint
    limits. This is the scalar reference implementation for action contract v3.
    """
    if residual_scale_rad <= 0.0:
        raise ValueError("residual_scale_rad must be positive")
    a = max(-1.0, min(1.0, float(action)))
    target = default + a * float(residual_scale_rad)
    return max(lower, min(upper, target))


def validate_contract() -> None:
    """Validate lengths and that the training default lies inside each limit pair."""
    n = len(ACTIONABLE_JOINTS_V1_2_3)
    if not (
        len(TRAINING_DEFAULT_RAD)
        == len(HARDWARE_LOWER_LIMIT_RAD)
        == len(HARDWARE_UPPER_LIMIT_RAD)
        == n
    ):
        raise ValueError("v1.2.3 action-contract arrays must have identical lengths")
    for name, default, lower, upper in zip(
        ACTIONABLE_JOINTS_V1_2_3,
        TRAINING_DEFAULT_RAD,
        HARDWARE_LOWER_LIMIT_RAD,
        HARDWARE_UPPER_LIMIT_RAD,
    ):
        if not lower <= default <= upper:
            raise ValueError(
                f"Training default for {name} ({default}) is outside [{lower}, {upper}]"
            )


# v1.4.5 athletic q_default + vector residual profile.
# This keeps the 45-D observation / 12-D action I/O and the residual equation,
# but it is intentionally deployment-impacting: exported policies should be
# treated as a new contract profile because q_default and the residual authority
# are no longer the old scalar +/-0.20 rad standing pose.
TRAINING_DEFAULT_RAD_V1_4_5_ATHLETIC = [
    0.0,
    0.0,
    -0.30,
    0.78,
    -0.30,
    0.0,
    0.0,
    0.0,
    -0.30,
    0.78,
    -0.30,
    0.0,
]

# More authority for sagittal walking joints, while roll/yaw axes remain
# conservative. Order matches ACTIONABLE_JOINTS_V1_2_3.
RESIDUAL_ACTION_SCALE_RAD_V1_4_5_ATHLETIC = [
    0.24,
    0.16,
    0.42,
    0.58,
    0.48,
    0.26,
    0.24,
    0.16,
    0.42,
    0.58,
    0.48,
    0.26,
]

# Height targets for the athletic profile. The old q_default geometry was about
# 0.4899 m, but real/learned behavior showed the robot was too tall and
# ankle-dominant. These values intentionally create a lower, knee-friendly stance.
V1_4_5_ATHLETIC_STAND_BASE_COM_HEIGHT_M = NOMINAL_QDEFAULT_BASE_COM_HEIGHT_M - 0.045
V1_4_5_ATHLETIC_MOVING_BASE_COM_HEIGHT_M = NOMINAL_QDEFAULT_BASE_COM_HEIGHT_M - 0.080


def validate_v1_4_5_athletic_contract() -> None:
    """Validate the v1.4.5 athletic q_default and vector residual profile."""
    n = len(ACTIONABLE_JOINTS_V1_2_3)
    if len(TRAINING_DEFAULT_RAD_V1_4_5_ATHLETIC) != n:
        raise ValueError("v1.4.5 athletic q_default length mismatch")
    if len(RESIDUAL_ACTION_SCALE_RAD_V1_4_5_ATHLETIC) != n:
        raise ValueError("v1.4.5 residual scale length mismatch")
    for name, default, scale, lower, upper in zip(
        ACTIONABLE_JOINTS_V1_2_3,
        TRAINING_DEFAULT_RAD_V1_4_5_ATHLETIC,
        RESIDUAL_ACTION_SCALE_RAD_V1_4_5_ATHLETIC,
        HARDWARE_LOWER_LIMIT_RAD,
        HARDWARE_UPPER_LIMIT_RAD,
    ):
        if scale <= 0.0:
            raise ValueError(f"v1.4.5 residual scale for {name} must be positive")
        if not lower <= default <= upper:
            raise ValueError(
                f"v1.4.5 athletic default for {name} ({default}) is outside [{lower}, {upper}]"
            )


def athletic_default_joint_pos_dict(include_toes: bool = True) -> dict[str, float]:
    """Return v1.4.5 athletic q_default keyed by joint name."""
    values = dict(zip(ACTIONABLE_JOINTS_V1_2_3, TRAINING_DEFAULT_RAD_V1_4_5_ATHLETIC))
    if include_toes:
        values["leg_left_toe_pitch_joint"] = 0.0
        values["leg_right_toe_pitch_joint"] = 0.0
    return values



# v1.4.5 Stand-stabilized athletic profile. This keeps contract v4/vector residual,
# but backs the default posture away from the too-deep v1.4.5 crouch that loaded the
# right knee/ankle in the first Stand-v5 run. The intent is a moderate athletic
# stance around 0.43-0.44 m COM height while preserving the larger sagittal motion
# authority needed later for walking.
TRAINING_DEFAULT_RAD_V1_4_5_STABILIZED = [
    0.0,
    0.0,
    -0.24,
    0.62,
    -0.22,
    0.0,
    0.0,
    0.0,
    -0.24,
    0.62,
    -0.22,
    0.0,
]

# Use the same vector residual authority as the athletic v5 profile: the stabilizer
# changes the center/default and rewards, not the amount of hip/knee/ankle pitch
# authority available to the policy.
RESIDUAL_ACTION_SCALE_RAD_V1_4_5_STABILIZED = RESIDUAL_ACTION_SCALE_RAD_V1_4_5_ATHLETIC

# Desired height targets for the stabilized profile. 0.435 m is deliberately in the
# visually identified sweet spot between the old tall stance and the first v1.4.5
# deep crouch. The moving target remains lower to allow knee bend during gait.
V1_4_5_STABILIZED_STAND_BASE_COM_HEIGHT_M = 0.435
V1_4_5_STABILIZED_MOVING_BASE_COM_HEIGHT_M = 0.415

# v1.4.5s2 forward-COM stabilizer. The 2000/5000-iteration v5s runs showed that
# the useful posture sits closer to 0.45-0.465 m than the original 0.435 target,
# but visually still wants a slight sagittal COM shift forward over the feet.
# Keep q_default and the vector residual contract unchanged; only the standing
# target geometry/rewards are adjusted.
V1_4_5_STABILIZED_FORWARD_STAND_BASE_COM_HEIGHT_M = 0.450
V1_4_5_STABILIZED_FORWARD_MOVING_BASE_COM_HEIGHT_M = 0.420
V1_4_5_STABILIZED_FORWARD_COM_TARGET_X_M = 0.055
V1_4_5_STABILIZED_FORWARD_COM_BAND_HALF_WIDTH_M = 0.010
V1_4_5_STABILIZED_FORWARD_LEAN_PROJECTED_GRAVITY_X = 0.052

# v1.4.5s3 forward-COM/height refinement. The v5s2 analysis showed the best
# posture in the 0.463-0.467 m realized-COM range but still rear-biased relative
# to the desired support polygon. Raise the explicit height target slightly so
# forward COM placement can be achieved with less ankle pitch bracing, then move
# the preferred COM-over-feet band farther forward and nudge the weak lean cue.
V1_4_5_STABILIZED_FORWARD2_STAND_BASE_COM_HEIGHT_M = 0.460
V1_4_5_STABILIZED_FORWARD2_MOVING_BASE_COM_HEIGHT_M = 0.430
V1_4_5_STABILIZED_FORWARD2_COM_TARGET_X_M = 0.070
V1_4_5_STABILIZED_FORWARD2_COM_BAND_HALF_WIDTH_M = 0.010
V1_4_5_STABILIZED_FORWARD2_LEAN_PROJECTED_GRAVITY_X = 0.065


def validate_v1_4_5_stabilized_contract() -> None:
    """Validate the v1.4.5 stabilized q_default and vector residual profile."""
    n = len(ACTIONABLE_JOINTS_V1_2_3)
    if len(TRAINING_DEFAULT_RAD_V1_4_5_STABILIZED) != n:
        raise ValueError("v1.4.5 stabilized q_default length mismatch")
    if len(RESIDUAL_ACTION_SCALE_RAD_V1_4_5_STABILIZED) != n:
        raise ValueError("v1.4.5 stabilized residual scale length mismatch")
    for name, default, scale, lower, upper in zip(
        ACTIONABLE_JOINTS_V1_2_3,
        TRAINING_DEFAULT_RAD_V1_4_5_STABILIZED,
        RESIDUAL_ACTION_SCALE_RAD_V1_4_5_STABILIZED,
        HARDWARE_LOWER_LIMIT_RAD,
        HARDWARE_UPPER_LIMIT_RAD,
    ):
        if scale <= 0.0:
            raise ValueError(f"v1.4.5 stabilized residual scale for {name} must be positive")
        if not lower <= default <= upper:
            raise ValueError(
                f"v1.4.5 stabilized default for {name} ({default}) is outside [{lower}, {upper}]"
            )


def athletic_stabilized_default_joint_pos_dict(include_toes: bool = True) -> dict[str, float]:
    """Return v1.4.5 stabilized athletic q_default keyed by joint name."""
    values = dict(zip(ACTIONABLE_JOINTS_V1_2_3, TRAINING_DEFAULT_RAD_V1_4_5_STABILIZED))
    if include_toes:
        values["leg_left_toe_pitch_joint"] = 0.0
        values["leg_right_toe_pitch_joint"] = 0.0
    return values

validate_contract()
validate_v1_4_5_athletic_contract()
validate_v1_4_5_stabilized_contract()
