"""Structural tests for v1.4.0 loaded ST3215 model constants."""

import math

from _load_st3215_loaded_actuator_model import load_st3215_loaded_actuator_model

model = load_st3215_loaded_actuator_model()


def test_loaded_model_lengths_and_joint_count():
    model.validate_loaded_model_constants()
    assert len(model.LOADED_MODEL_JOINT_NAMES) == 12
    assert len(model.LOADED_CROUCH_LOW_DEMAND_GAIN) == 12
    assert len(model.LOADED_RETURN_VMAX_RAD_S) == 12


def test_loaded_envelope_scalar_piecewise_behavior():
    # Knee-pitch crouch fit from the v2 loaded synthesis.
    gain = 0.7093610570990286
    vmax = 1.1776129510006368
    assert math.isclose(
        model.loaded_velocity_envelope_scalar(1.0, gain, vmax), gain, rel_tol=1.0e-12
    )
    assert math.isclose(
        model.loaded_velocity_envelope_scalar(10.0, gain, vmax), vmax, rel_tol=1.0e-12
    )


def test_only_knees_use_full_direction_conditioning():
    active = [i for i, weight in enumerate(model.LOADED_DIRECTION_CONDITIONING_WEIGHT) if weight > 0]
    assert active == [3, 9]
    assert model.LOADED_RETURN_TAU_SCALE[3] == 1.1
    assert model.LOADED_RETURN_TAU_SCALE[9] == 1.1
    assert all(
        scale == 1.0
        for i, scale in enumerate(model.LOADED_RETURN_TAU_SCALE)
        if i not in active
    )


def test_loaded_randomization_range_is_conservative():
    low, high = model.LOADED_VELOCITY_SCALE_RANGE
    assert math.isclose(low, 0.95)
    assert math.isclose(high, 1.05)
