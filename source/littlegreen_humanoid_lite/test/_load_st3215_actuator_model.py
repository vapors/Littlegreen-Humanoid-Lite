"""Load st3215_actuator_model.py without importing Isaac Lab task registration."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def load_st3215_actuator_model():
    path = (
        Path(__file__).resolve().parents[1]
        / "littlegreen_humanoid_lite"
        / "tasks"
        / "locomotion"
        / "velocity"
        / "mdp"
        / "st3215_actuator_model.py"
    )
    spec = importlib.util.spec_from_file_location("littlegreen_st3215_actuator_model_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load ST3215 actuator model from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
