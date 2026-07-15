"""Pure-Python compatibility tests for historical action contract v2."""

import math

from _load_hardware_contract import load_hardware_contract

contract = load_hardware_contract()


def test_contract_lengths_and_bounds():
    assert len(contract.ACTIONABLE_JOINTS_V1_2_3) == 12
    assert len(contract.TRAINING_DEFAULT_RAD) == 12
    assert len(contract.HARDWARE_LOWER_LIMIT_RAD) == 12
    assert len(contract.HARDWARE_UPPER_LIMIT_RAD) == 12
    for default, lower, upper in zip(
        contract.TRAINING_DEFAULT_RAD,
        contract.HARDWARE_LOWER_LIMIT_RAD,
        contract.HARDWARE_UPPER_LIMIT_RAD,
    ):
        assert lower <= default <= upper


def test_asymmetric_mapping_endpoints_and_default():
    for default, lower, upper in zip(
        contract.TRAINING_DEFAULT_RAD,
        contract.HARDWARE_LOWER_LIMIT_RAD,
        contract.HARDWARE_UPPER_LIMIT_RAD,
    ):
        assert math.isclose(contract.map_bounded_action_scalar(-1.0, default, lower, upper), lower, abs_tol=1e-9)
        assert math.isclose(contract.map_bounded_action_scalar(0.0, default, lower, upper), default, abs_tol=1e-9)
        assert math.isclose(contract.map_bounded_action_scalar(1.0, default, lower, upper), upper, abs_tol=1e-9)


def test_mapping_clamps_raw_action_before_transform():
    default, lower, upper = -0.1, -1.483, 0.785
    assert contract.map_bounded_action_scalar(-50.0, default, lower, upper) == lower
    assert contract.map_bounded_action_scalar(50.0, default, lower, upper) == upper
