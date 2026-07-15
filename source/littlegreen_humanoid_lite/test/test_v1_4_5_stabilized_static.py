from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / "source/littlegreen_humanoid_lite/littlegreen_humanoid_lite/tasks/locomotion/velocity/mdp/hardware_contract.py"
ENV = ROOT / "source/littlegreen_humanoid_lite/littlegreen_humanoid_lite/tasks/locomotion/velocity/config/lilgreen_humanoid/env_cfg_hardware_st3215_loaded_v145.py"
INIT = ROOT / "source/littlegreen_humanoid_lite/littlegreen_humanoid_lite/tasks/locomotion/velocity/config/lilgreen_humanoid/__init__.py"
EXPORT = ROOT / "scripts/rsl_rl/export_policy_littlegreen.py"


def _literal_list(name: str):
    tree = ast.parse(CONTRACT.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"missing {name}")


def test_stabilized_q_default_is_moderate_athletic():
    q = _literal_list("TRAINING_DEFAULT_RAD_V1_4_5_STABILIZED")
    assert len(q) == 12
    assert q[2] == q[8] == -0.24
    assert q[3] == q[9] == 0.62
    assert q[4] == q[10] == -0.22


def test_stabilized_height_and_tasks_registered():
    contract = CONTRACT.read_text()
    assert "V1_4_5_STABILIZED_STAND_BASE_COM_HEIGHT_M = 0.435" in contract
    assert "V1_4_5_STABILIZED_MOVING_BASE_COM_HEIGHT_M = 0.415" in contract
    init = INIT.read_text()
    assert "Velocity-Lilgreen-Stand-ST3215-Loaded-v5s" in init
    assert "Velocity-Lilgreen-Hardware-ST3215-Loaded-v5s" in init


def test_stabilization_terms_are_stand_only_and_export_profile_is_distinct():
    env = ENV.read_text()
    assert "standing_contact_force_balance_l2" in env
    assert "standing_com_over_feet_l2" in env
    assert "ST3215LoadedV145StabilizedStandEventsCfg" in env
    export = EXPORT.read_text()
    assert "v1_4_5_stabilized_vector_residual" in export
