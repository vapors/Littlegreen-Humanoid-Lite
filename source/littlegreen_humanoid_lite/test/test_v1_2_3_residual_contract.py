"""Pure-Python reference tests for the v1.2.3 residual action contract."""

import math

from _load_hardware_contract import load_hardware_contract

contract = load_hardware_contract()


def test_residual_mapping_center_and_endpoints():
    default, lower, upper = -0.1, -1.483, 0.785
    scale = contract.RESIDUAL_ACTION_SCALE_RAD_V1_2_3
    assert math.isclose(contract.map_bounded_residual_action_scalar(0.0, default, lower, upper), default)
    assert math.isclose(contract.map_bounded_residual_action_scalar(-1.0, default, lower, upper), default - scale)
    assert math.isclose(contract.map_bounded_residual_action_scalar(1.0, default, lower, upper), default + scale)


def test_residual_mapping_clamps_raw_action_and_physical_limit():
    assert math.isclose(
        contract.map_bounded_residual_action_scalar(50.0, 0.4, 0.0, 0.5, 0.20),
        0.5,
    )
    assert math.isclose(
        contract.map_bounded_residual_action_scalar(-50.0, 0.1, 0.0, 1.0, 0.20),
        0.0,
    )


def test_nominal_height_constant_matches_measurement():
    assert math.isclose(
        contract.NOMINAL_QDEFAULT_BASE_COM_HEIGHT_M,
        0.4899105727672577,
        abs_tol=1.0e-12,
    )
