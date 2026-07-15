"""Structural tests for v1.3.0 ST3215 model constants."""

from _load_st3215_actuator_model import load_st3215_actuator_model

model = load_st3215_actuator_model()


def test_st3215_model_lengths():
    model.validate_st3215_model_constants()
    assert len(model.ST3215_MODEL_JOINT_NAMES) == 12
    assert all(len(curve) == 6 for curve in model.ST3215_PEAK_VELOCITY_CURVES_RAD_S)


def test_tau_ranges_are_ordered():
    assert all(
        low > 0.0 and high >= low
        for low, high in zip(model.ST3215_TAU_P10_S, model.ST3215_TAU_P90_S)
    )


def test_velocity_curves_are_monotonic_non_decreasing():
    for curve in model.ST3215_PEAK_VELOCITY_CURVES_RAD_S:
        assert curve[0] == 0.0
        assert all(b >= a for a, b in zip(curve, curve[1:]))
