"""Load hardware_contract.py without importing Isaac Lab task registration."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def load_hardware_contract():
    path = (
        Path(__file__).resolve().parents[1]
        / "littlegreen_humanoid_lite"
        / "tasks"
        / "locomotion"
        / "velocity"
        / "mdp"
        / "hardware_contract.py"
    )
    spec = importlib.util.spec_from_file_location("littlegreen_hardware_contract_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load hardware contract from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
