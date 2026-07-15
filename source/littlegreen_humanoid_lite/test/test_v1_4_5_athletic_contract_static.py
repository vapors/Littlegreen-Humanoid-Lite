"""Static checks for v1.4.5 athletic vector-residual tasks."""

from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / "source/littlegreen_humanoid_lite/littlegreen_humanoid_lite/tasks/locomotion/velocity/mdp/hardware_contract.py"
CONFIG = ROOT / "source/littlegreen_humanoid_lite/littlegreen_humanoid_lite/tasks/locomotion/velocity/config/lilgreen_humanoid/env_cfg_hardware_st3215_loaded_v145.py"
REGISTRY = ROOT / "source/littlegreen_humanoid_lite/littlegreen_humanoid_lite/tasks/locomotion/velocity/config/lilgreen_humanoid/__init__.py"
EXPORT = ROOT / "scripts/rsl_rl/export_policy_littlegreen.py"


def _literal_list(name: str):
    tree = ast.parse(CONTRACT.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"Missing {name}")


def test_athletic_default_and_scales_are_12d():
    q_default = _literal_list("TRAINING_DEFAULT_RAD_V1_4_5_ATHLETIC")
    scales = _literal_list("RESIDUAL_ACTION_SCALE_RAD_V1_4_5_ATHLETIC")
    assert len(q_default) == 12
    assert len(scales) == 12
    assert max(scales) > 0.20
    assert scales[2] >= 0.40
    assert scales[3] >= 0.55
    assert scales[4] >= 0.45
    assert q_default[3] > 0.60
    assert q_default[9] > 0.60


def test_v145_tasks_registered_and_profile_is_new():
    text = REGISTRY.read_text()
    assert "Velocity-Lilgreen-Stand-ST3215-Loaded-v5" in text
    assert "Velocity-Lilgreen-Hardware-ST3215-Loaded-v5" in text
    cfg = CONFIG.read_text()
    assert "V1_4_5_ATHLETIC_MOVING_BASE_COM_HEIGHT_M" in cfg
    assert "moving_no_support" in cfg
    assert "moving_com_over_stance_foot" in cfg
    assert "moving_knee_flexion_band" in cfg


def test_export_marks_nonuniform_scale_as_contract_v4():
    text = EXPORT.read_text()
    assert "nonuniform_residual_scale" in text
    assert "action_contract_version = 4" in text
    assert "deployment_requires_action_contract_v4_transform" in text
    assert "v1_4_5_athletic_vector_residual" in text
