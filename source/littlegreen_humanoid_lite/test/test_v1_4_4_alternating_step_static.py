"""Static checks for the additive v1.4.4 alternating-step task."""

from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[3]
ENV = ROOT / "source/littlegreen_humanoid_lite/littlegreen_humanoid_lite/tasks/locomotion/velocity/config/lilgreen_humanoid/env_cfg_hardware_st3215_loaded_v144.py"
CURR = ROOT / "source/littlegreen_humanoid_lite/littlegreen_humanoid_lite/tasks/locomotion/velocity/mdp/curriculums.py"


def _function_defaults(name: str):
    tree = ast.parse(CURR.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            arg_names = [arg.arg for arg in node.args.args]
            default_names = arg_names[-len(node.args.defaults):]
            return {key: ast.literal_eval(value) for key, value in zip(default_names, node.args.defaults)}
    raise AssertionError(f"missing function {name}")


def test_v144_task_contract_is_unchanged():
    text = ENV.read_text()
    assert "residual_scale_rad=RESIDUAL_ACTION_SCALE_RAD_V1_2_3" in text
    assert "joint_names=ACTIONABLE_JOINTS_V1_2_3" in text
    assert "preserve_order=True" in text
    assert 'actuator_model_stage="stage_b_loaded_v144_quick_train"' in text


def test_v144_keeps_continuous_command_curriculum():
    defaults = _function_defaults("st3215_loaded_v144_hardware_stage_curriculum")
    assert defaults["stage_step_boundaries"] == (32000, 128000, 256000)
    assert defaults["standing_fractions"] == (0.40, 0.30, 0.25, 0.20)
    assert defaults["lin_vel_x_ranges"][0] == (-0.28, 0.28)
    assert defaults["lin_vel_y_ranges"][1] == (-0.16, 0.16)


def test_v144_reward_terms_exist_and_relax_moving_height():
    text = ENV.read_text()
    for term in (
        "moving_alternating_single_support",
        "moving_foot_air_balance",
        "moving_long_single_support",
        "moving_stance_foot_slide",
        "moving_relaxed_base_height",
    ):
        assert term in text
    assert '"std": 0.075' in text
