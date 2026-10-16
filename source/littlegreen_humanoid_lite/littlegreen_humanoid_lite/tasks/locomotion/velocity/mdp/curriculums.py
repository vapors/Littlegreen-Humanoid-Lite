"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import RLTaskEnv


def terrain_levels_vel(
    env: RLTaskEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def hardware_stage_curriculum(
    env: RLTaskEnv,
    env_ids: Sequence[int],
    stage_step_boundaries: tuple[int, int, int] = (32000, 96000, 224000),
    standing_fractions: tuple[float, float, float, float] = (0.75, 0.50, 0.30, 0.25),
    lin_vel_x_ranges: tuple[tuple[float, float], ...] = ((-0.15, 0.15), (-0.4, 0.4), (-0.7, 0.7), (-1.0, 1.0)),
    lin_vel_y_ranges: tuple[tuple[float, float], ...] = ((-0.08, 0.08), (-0.2, 0.2), (-0.35, 0.35), (-0.5, 0.5)),
    ang_vel_z_ranges: tuple[tuple[float, float], ...] = ((-0.25, 0.25), (-0.6, 0.6), (-1.0, 1.0), (-1.5, 1.5)),
    root_velocity_ranges: tuple[float, float, float, float] = (0.10, 0.20, 0.35, 0.50),
    joint_reset_offsets: tuple[float, float, float, float] = (0.03, 0.04, 0.06, 0.08),
    push_xy_ranges: tuple[float, float, float, float] = (0.0, 0.0, 0.25, 0.50),
    mass_scale_ranges: tuple[tuple[float, float], ...] = ((0.95, 1.05), (0.92, 1.08), (0.90, 1.10), (0.88, 1.12)),
    actuator_gain_scale_ranges: tuple[tuple[float, float], ...] = ((0.95, 1.05), (0.92, 1.08), (0.90, 1.10), (0.88, 1.12)),
    one_step_delay_probabilities: tuple[float, float, float, float] = (0.0, 0.05, 0.15, 0.25),
) -> dict[str, torch.Tensor | float]:
    """Advance the Hardware-v0 command and disturbance curriculum by global policy step.

    The curriculum is intentionally deterministic in stage selection.  Stage boundaries
    are policy/environment steps, not physics substeps.  With the v1.2.3 runner default of
    64 steps per PPO iteration, the default boundaries are approximately 500, 1500 and
    3500 total PPO iterations.
    """
    del env_ids  # curriculum state is global across vectorized environments

    step = int(env.common_step_counter)
    b0, b1, b2 = stage_step_boundaries
    if step < b0:
        stage = 0
    elif step < b1:
        stage = 1
    elif step < b2:
        stage = 2
    else:
        stage = 3

    previous_stage = getattr(env, "_littlegreen_v12_hardware_stage", None)
    if previous_stage != stage:
        command_term = env.command_manager.get_term("base_velocity")
        command_term.cfg.rel_standing_envs = standing_fractions[stage]
        command_term.cfg.ranges.lin_vel_x = lin_vel_x_ranges[stage]
        command_term.cfg.ranges.lin_vel_y = lin_vel_y_ranges[stage]
        command_term.cfg.ranges.ang_vel_z = ang_vel_z_ranges[stage]

        # Reset perturbation envelope.
        reset_base_cfg = env.event_manager.get_term_cfg("reset_base")
        root_vel = root_velocity_ranges[stage]
        reset_base_cfg.params["velocity_range"] = {
            "x": (-root_vel, root_vel),
            "y": (-root_vel, root_vel),
            "z": (0.0, 0.0),
            "roll": (-root_vel, root_vel),
            "pitch": (-root_vel, root_vel),
            "yaw": (-root_vel, root_vel),
        }
        env.event_manager.set_term_cfg("reset_base", reset_base_cfg)

        reset_joint_cfg = env.event_manager.get_term_cfg("reset_robot_joints")
        offset = joint_reset_offsets[stage]
        reset_joint_cfg.params["position_range"] = (-offset, offset)
        env.event_manager.set_term_cfg("reset_robot_joints", reset_joint_cfg)

        # Reset-time mass and actuator-gain envelopes expand from +/-5% to +/-12%.
        mass_cfg = env.event_manager.get_term_cfg("base_mass")
        mass_cfg.params["mass_distribution_params"] = mass_scale_ranges[stage]
        env.event_manager.set_term_cfg("base_mass", mass_cfg)

        gain_cfg = env.event_manager.get_term_cfg("actuator_gains")
        gain_cfg.params["stiffness_distribution_params"] = actuator_gain_scale_ranges[stage]
        gain_cfg.params["damping_distribution_params"] = actuator_gain_scale_ranges[stage]
        env.event_manager.set_term_cfg("actuator_gains", gain_cfg)

        # Periodic push envelope.  Stage 0/1 use zero range, so the event is harmless.
        push_cfg = env.event_manager.get_term_cfg("push_robot")
        push = push_xy_ranges[stage]
        push_cfg.params["velocity_range"] = {"x": (-push, push), "y": (-push, push)}
        env.event_manager.set_term_cfg("push_robot", push_cfg)

        # Optional 0/1 policy-step command latency is sampled per environment on reset.
        action_term = env.action_manager.get_term("joint_pos")
        action_term.cfg.one_step_delay_probability = one_step_delay_probabilities[stage]

        # Force a fresh command draw so new bounds take effect immediately at stage transition.
        command_term.time_left[:] = 0.0
        env._littlegreen_v12_hardware_stage = stage

    return {
        "stage": float(stage),
        "standing_fraction": float(standing_fractions[stage]),
        "cmd_x_abs_max": float(max(abs(v) for v in lin_vel_x_ranges[stage])),
        "cmd_y_abs_max": float(max(abs(v) for v in lin_vel_y_ranges[stage])),
        "cmd_yaw_abs_max": float(max(abs(v) for v in ang_vel_z_ranges[stage])),
        "root_reset_velocity_abs_max": float(root_velocity_ranges[stage]),
        "joint_reset_offset_abs_max": float(joint_reset_offsets[stage]),
        "push_xy_abs_max": float(push_xy_ranges[stage]),
        "mass_scale_min": float(mass_scale_ranges[stage][0]),
        "mass_scale_max": float(mass_scale_ranges[stage][1]),
        "actuator_gain_scale_min": float(actuator_gain_scale_ranges[stage][0]),
        "actuator_gain_scale_max": float(actuator_gain_scale_ranges[stage][1]),
        "one_step_delay_probability": float(one_step_delay_probabilities[stage]),
    }



def st3215_hardware_stage_curriculum(
    env: RLTaskEnv,
    env_ids: Sequence[int],
    stage_step_boundaries: tuple[int, int, int] = (32000, 96000, 224000),
    standing_fractions: tuple[float, float, float, float] = (0.75, 0.50, 0.30, 0.25),
    lin_vel_x_ranges: tuple[tuple[float, float], ...] = ((-0.15, 0.15), (-0.4, 0.4), (-0.7, 0.7), (-1.0, 1.0)),
    lin_vel_y_ranges: tuple[tuple[float, float], ...] = ((-0.08, 0.08), (-0.2, 0.2), (-0.35, 0.35), (-0.5, 0.5)),
    ang_vel_z_ranges: tuple[tuple[float, float], ...] = ((-0.25, 0.25), (-0.6, 0.6), (-1.0, 1.0), (-1.5, 1.5)),
    root_velocity_ranges: tuple[float, float, float, float] = (0.10, 0.20, 0.35, 0.50),
    joint_reset_offsets: tuple[float, float, float, float] = (0.03, 0.04, 0.06, 0.08),
    push_xy_ranges: tuple[float, float, float, float] = (0.0, 0.0, 0.25, 0.50),
    mass_scale_ranges: tuple[tuple[float, float], ...] = ((0.95, 1.05), (0.92, 1.08), (0.90, 1.10), (0.88, 1.12)),
) -> dict[str, torch.Tensor | float]:
    """Hardware curriculum that leaves measured ST3215 response randomization intact.

    Unlike v1.2.3 Hardware-v0, this curriculum does not expand generic stiffness or
    damping gains and does not add the legacy 0/1-step action delay. The ST3215
    action term samples evidence-based response delay, tau, and velocity scale at reset.
    """
    del env_ids
    step = int(env.common_step_counter)
    b0, b1, b2 = stage_step_boundaries
    stage = 0 if step < b0 else 1 if step < b1 else 2 if step < b2 else 3

    previous_stage = getattr(env, "_littlegreen_v130_st3215_hardware_stage", None)
    if previous_stage != stage:
        command_term = env.command_manager.get_term("base_velocity")
        command_term.cfg.rel_standing_envs = standing_fractions[stage]
        command_term.cfg.ranges.lin_vel_x = lin_vel_x_ranges[stage]
        command_term.cfg.ranges.lin_vel_y = lin_vel_y_ranges[stage]
        command_term.cfg.ranges.ang_vel_z = ang_vel_z_ranges[stage]

        reset_base_cfg = env.event_manager.get_term_cfg("reset_base")
        root_vel = root_velocity_ranges[stage]
        reset_base_cfg.params["velocity_range"] = {
            "x": (-root_vel, root_vel),
            "y": (-root_vel, root_vel),
            "z": (0.0, 0.0),
            "roll": (-root_vel, root_vel),
            "pitch": (-root_vel, root_vel),
            "yaw": (-root_vel, root_vel),
        }
        env.event_manager.set_term_cfg("reset_base", reset_base_cfg)

        reset_joint_cfg = env.event_manager.get_term_cfg("reset_robot_joints")
        offset = joint_reset_offsets[stage]
        reset_joint_cfg.params["position_range"] = (-offset, offset)
        env.event_manager.set_term_cfg("reset_robot_joints", reset_joint_cfg)

        mass_cfg = env.event_manager.get_term_cfg("base_mass")
        mass_cfg.params["mass_distribution_params"] = mass_scale_ranges[stage]
        env.event_manager.set_term_cfg("base_mass", mass_cfg)

        push_cfg = env.event_manager.get_term_cfg("push_robot")
        push = push_xy_ranges[stage]
        push_cfg.params["velocity_range"] = {"x": (-push, push), "y": (-push, push)}
        env.event_manager.set_term_cfg("push_robot", push_cfg)

        command_term.time_left[:] = 0.0
        env._littlegreen_v130_st3215_hardware_stage = stage

    action_term = env.action_manager.get_term("joint_pos")
    return {
        "stage": float(stage),
        "standing_fraction": float(standing_fractions[stage]),
        "cmd_x_abs_max": float(max(abs(v) for v in lin_vel_x_ranges[stage])),
        "cmd_y_abs_max": float(max(abs(v) for v in lin_vel_y_ranges[stage])),
        "cmd_yaw_abs_max": float(max(abs(v) for v in ang_vel_z_ranges[stage])),
        "root_reset_velocity_abs_max": float(root_velocity_ranges[stage]),
        "joint_reset_offset_abs_max": float(joint_reset_offsets[stage]),
        "push_xy_abs_max": float(push_xy_ranges[stage]),
        "mass_scale_min": float(mass_scale_ranges[stage][0]),
        "mass_scale_max": float(mass_scale_ranges[stage][1]),
        "actuator_delay_mean_ms": float(torch.mean(action_term.sampled_total_delay_s).item() * 1000.0),
        "actuator_tau_mean_ms": float(torch.mean(action_term.sampled_tau_s).item() * 1000.0),
        "actuator_velocity_scale_mean": float(torch.mean(action_term.sampled_velocity_scale).item()),
    }


def st3215_loaded_v141_hardware_stage_curriculum(
    env: RLTaskEnv,
    env_ids: Sequence[int],
    stage_step_boundaries: tuple[int, int, int] = (64000, 192000, 320000),
    standing_fractions: tuple[float, float, float, float] = (0.85, 0.70, 0.55, 0.45),
    lin_vel_x_ranges: tuple[tuple[float, float], ...] = ((-0.12, 0.12), (-0.25, 0.25), (-0.40, 0.40), (-0.60, 0.60)),
    lin_vel_y_ranges: tuple[tuple[float, float], ...] = ((-0.05, 0.05), (-0.10, 0.10), (-0.18, 0.18), (-0.25, 0.25)),
    ang_vel_z_ranges: tuple[tuple[float, float], ...] = ((-0.15, 0.15), (-0.30, 0.30), (-0.55, 0.55), (-0.80, 0.80)),
    root_velocity_ranges: tuple[float, float, float, float] = (0.06, 0.10, 0.16, 0.22),
    joint_reset_offsets: tuple[float, float, float, float] = (0.02, 0.03, 0.04, 0.05),
    push_xy_ranges: tuple[float, float, float, float] = (0.0, 0.0, 0.05, 0.10),
    mass_scale_ranges: tuple[tuple[float, float], ...] = ((0.97, 1.03), (0.95, 1.05), (0.93, 1.07), (0.92, 1.08)),
) -> dict[str, torch.Tensor | float]:
    """v1.4.1 slower loaded-ST3215 Hardware curriculum.

    This is a targeted recovery curriculum after the first v1.4.0 Hardware attempt.
    It preserves the v1.4.0 actuator model and policy contract, but delays and softens
    the locomotion/disturbance expansion so the Stand-ST3215-Loaded checkpoint can
    retain its standing controller while learning low/medium-speed walking, reverse,
    strafing, and turning.

    Stage boundaries are policy/environment steps. With the current PPO rollout length
    of 64 steps per iteration, the default boundaries correspond to roughly +1000,
    +3000, and +5000 Hardware training iterations from the start of this environment.
    """
    del env_ids
    step = int(env.common_step_counter)
    b0, b1, b2 = stage_step_boundaries
    stage = 0 if step < b0 else 1 if step < b1 else 2 if step < b2 else 3

    previous_stage = getattr(env, "_littlegreen_v141_st3215_loaded_hardware_stage", None)
    if previous_stage != stage:
        command_term = env.command_manager.get_term("base_velocity")
        command_term.cfg.rel_standing_envs = standing_fractions[stage]
        command_term.cfg.ranges.lin_vel_x = lin_vel_x_ranges[stage]
        command_term.cfg.ranges.lin_vel_y = lin_vel_y_ranges[stage]
        command_term.cfg.ranges.ang_vel_z = ang_vel_z_ranges[stage]

        reset_base_cfg = env.event_manager.get_term_cfg("reset_base")
        root_vel = root_velocity_ranges[stage]
        reset_base_cfg.params["velocity_range"] = {
            "x": (-root_vel, root_vel),
            "y": (-root_vel, root_vel),
            "z": (0.0, 0.0),
            "roll": (-root_vel, root_vel),
            "pitch": (-root_vel, root_vel),
            "yaw": (-root_vel, root_vel),
        }
        env.event_manager.set_term_cfg("reset_base", reset_base_cfg)

        reset_joint_cfg = env.event_manager.get_term_cfg("reset_robot_joints")
        offset = joint_reset_offsets[stage]
        reset_joint_cfg.params["position_range"] = (-offset, offset)
        env.event_manager.set_term_cfg("reset_robot_joints", reset_joint_cfg)

        mass_cfg = env.event_manager.get_term_cfg("base_mass")
        mass_cfg.params["mass_distribution_params"] = mass_scale_ranges[stage]
        env.event_manager.set_term_cfg("base_mass", mass_cfg)

        push_cfg = env.event_manager.get_term_cfg("push_robot")
        push = push_xy_ranges[stage]
        push_cfg.params["velocity_range"] = {"x": (-push, push), "y": (-push, push)}
        env.event_manager.set_term_cfg("push_robot", push_cfg)

        # Force a fresh command draw so the new stage bounds take effect immediately.
        command_term.time_left[:] = 0.0
        env._littlegreen_v141_st3215_loaded_hardware_stage = stage

    action_term = env.action_manager.get_term("joint_pos")
    return {
        "stage": float(stage),
        "standing_fraction": float(standing_fractions[stage]),
        "cmd_x_abs_max": float(max(abs(v) for v in lin_vel_x_ranges[stage])),
        "cmd_y_abs_max": float(max(abs(v) for v in lin_vel_y_ranges[stage])),
        "cmd_yaw_abs_max": float(max(abs(v) for v in ang_vel_z_ranges[stage])),
        "root_reset_velocity_abs_max": float(root_velocity_ranges[stage]),
        "joint_reset_offset_abs_max": float(joint_reset_offsets[stage]),
        "push_xy_abs_max": float(push_xy_ranges[stage]),
        "mass_scale_min": float(mass_scale_ranges[stage][0]),
        "mass_scale_max": float(mass_scale_ranges[stage][1]),
        "actuator_delay_mean_ms": float(torch.mean(action_term.sampled_total_delay_s).item() * 1000.0),
        "actuator_tau_mean_ms": float(torch.mean(action_term.sampled_tau_s).item() * 1000.0),
        "actuator_velocity_scale_mean": float(torch.mean(action_term.sampled_velocity_scale).item()),
        # RSL-RL episode logging requires scalar numeric values here; keep profile names in docs/config, not episode extras.
        "curriculum_profile_id": 141.0,
    }


def st3215_loaded_v142_hardware_stage_curriculum(
    env: RLTaskEnv,
    env_ids: Sequence[int],
    stage_step_boundaries: tuple[int, int, int] = (32000, 128000, 256000),
    standing_fractions: tuple[float, float, float, float] = (0.50, 0.38, 0.30, 0.25),
    lin_vel_x_ranges: tuple[tuple[float, float], ...] = ((-0.22, 0.22), (-0.40, 0.40), (-0.60, 0.60), (-0.80, 0.80)),
    lin_vel_y_ranges: tuple[tuple[float, float], ...] = ((-0.08, 0.08), (-0.15, 0.15), (-0.25, 0.25), (-0.35, 0.35)),
    ang_vel_z_ranges: tuple[tuple[float, float], ...] = ((-0.25, 0.25), (-0.45, 0.45), (-0.75, 0.75), (-1.00, 1.00)),
    root_velocity_ranges: tuple[float, float, float, float] = (0.06, 0.12, 0.20, 0.28),
    joint_reset_offsets: tuple[float, float, float, float] = (0.02, 0.03, 0.05, 0.06),
    push_xy_ranges: tuple[float, float, float, float] = (0.0, 0.0, 0.05, 0.10),
    mass_scale_ranges: tuple[tuple[float, float], ...] = ((0.97, 1.03), (0.95, 1.05), (0.93, 1.07), (0.92, 1.08)),
) -> dict[str, torch.Tensor | float]:
    """v1.4.2 locomotion-pressure loaded-ST3215 Hardware curriculum.

    This curriculum keeps the v1.4.0/v1.4.1 actuator model and policy contract, but
    intentionally asks for more motion earlier. It does not split commands into bins;
    it simply reduces standing-env probability, increases the early continuous command
    ranges, and keeps disturbance growth moderate.

    Stage boundaries are policy/environment steps. With the current PPO rollout length
    of 64 steps per iteration, the default boundaries correspond to roughly +500,
    +2000, and +4000 Hardware iterations from the start of this environment.
    """
    del env_ids
    step = int(env.common_step_counter)
    b0, b1, b2 = stage_step_boundaries
    stage = 0 if step < b0 else 1 if step < b1 else 2 if step < b2 else 3

    previous_stage = getattr(env, "_littlegreen_v142_st3215_loaded_hardware_stage", None)
    if previous_stage != stage:
        command_term = env.command_manager.get_term("base_velocity")
        command_term.cfg.rel_standing_envs = standing_fractions[stage]
        command_term.cfg.ranges.lin_vel_x = lin_vel_x_ranges[stage]
        command_term.cfg.ranges.lin_vel_y = lin_vel_y_ranges[stage]
        command_term.cfg.ranges.ang_vel_z = ang_vel_z_ranges[stage]

        reset_base_cfg = env.event_manager.get_term_cfg("reset_base")
        root_vel = root_velocity_ranges[stage]
        reset_base_cfg.params["velocity_range"] = {
            "x": (-root_vel, root_vel),
            "y": (-root_vel, root_vel),
            "z": (0.0, 0.0),
            "roll": (-root_vel, root_vel),
            "pitch": (-root_vel, root_vel),
            "yaw": (-root_vel, root_vel),
        }
        env.event_manager.set_term_cfg("reset_base", reset_base_cfg)

        reset_joint_cfg = env.event_manager.get_term_cfg("reset_robot_joints")
        offset = joint_reset_offsets[stage]
        reset_joint_cfg.params["position_range"] = (-offset, offset)
        env.event_manager.set_term_cfg("reset_robot_joints", reset_joint_cfg)

        mass_cfg = env.event_manager.get_term_cfg("base_mass")
        mass_cfg.params["mass_distribution_params"] = mass_scale_ranges[stage]
        env.event_manager.set_term_cfg("base_mass", mass_cfg)

        push_cfg = env.event_manager.get_term_cfg("push_robot")
        push = push_xy_ranges[stage]
        push_cfg.params["velocity_range"] = {"x": (-push, push), "y": (-push, push)}
        env.event_manager.set_term_cfg("push_robot", push_cfg)

        # Force a fresh command draw so the new stage bounds take effect immediately.
        command_term.time_left[:] = 0.0
        env._littlegreen_v142_st3215_loaded_hardware_stage = stage

    action_term = env.action_manager.get_term("joint_pos")
    return {
        "stage": float(stage),
        "standing_fraction": float(standing_fractions[stage]),
        "cmd_x_abs_max": float(max(abs(v) for v in lin_vel_x_ranges[stage])),
        "cmd_y_abs_max": float(max(abs(v) for v in lin_vel_y_ranges[stage])),
        "cmd_yaw_abs_max": float(max(abs(v) for v in ang_vel_z_ranges[stage])),
        "root_reset_velocity_abs_max": float(root_velocity_ranges[stage]),
        "joint_reset_offset_abs_max": float(joint_reset_offsets[stage]),
        "push_xy_abs_max": float(push_xy_ranges[stage]),
        "mass_scale_min": float(mass_scale_ranges[stage][0]),
        "mass_scale_max": float(mass_scale_ranges[stage][1]),
        "actuator_delay_mean_ms": float(torch.mean(action_term.sampled_total_delay_s).item() * 1000.0),
        "actuator_tau_mean_ms": float(torch.mean(action_term.sampled_tau_s).item() * 1000.0),
        "actuator_velocity_scale_mean": float(torch.mean(action_term.sampled_velocity_scale).item()),
        "curriculum_profile_id": 142.0,
    }

def st3215_loaded_v143_hardware_stage_curriculum(
    env: RLTaskEnv,
    env_ids: Sequence[int],
    stage_step_boundaries: tuple[int, int, int] = (32000, 128000, 256000),
    standing_fractions: tuple[float, float, float, float] = (0.45, 0.35, 0.28, 0.22),
    lin_vel_x_ranges: tuple[tuple[float, float], ...] = ((-0.25, 0.25), (-0.45, 0.45), (-0.65, 0.65), (-0.85, 0.85)),
    lin_vel_y_ranges: tuple[tuple[float, float], ...] = ((-0.08, 0.08), (-0.16, 0.16), (-0.26, 0.26), (-0.35, 0.35)),
    ang_vel_z_ranges: tuple[tuple[float, float], ...] = ((-0.25, 0.25), (-0.50, 0.50), (-0.75, 0.75), (-1.00, 1.00)),
    root_velocity_ranges: tuple[float, float, float, float] = (0.04, 0.08, 0.14, 0.20),
    joint_reset_offsets: tuple[float, float, float, float] = (0.015, 0.025, 0.04, 0.055),
    push_xy_ranges: tuple[float, float, float, float] = (0.0, 0.0, 0.02, 0.06),
    mass_scale_ranges: tuple[tuple[float, float], ...] = ((0.98, 1.02), (0.96, 1.04), (0.94, 1.06), (0.93, 1.07)),
) -> dict[str, torch.Tensor | float]:
    """v1.4.3 move-now loaded-ST3215 Hardware curriculum.

    This curriculum remains continuous and does not introduce command bins. Compared
    with v1.4.2, it keeps similar command pressure but slightly lowers standing
    fraction, reduces reset/push disturbances, and relies on new anti-bracing gait
    rewards plus the faster v1.4.3 actuator training response to encourage swing
    discovery instead of crouch/bracing.
    """
    del env_ids
    step = int(env.common_step_counter)
    b0, b1, b2 = stage_step_boundaries
    stage = 0 if step < b0 else 1 if step < b1 else 2 if step < b2 else 3

    previous_stage = getattr(env, "_littlegreen_v143_st3215_loaded_hardware_stage", None)
    if previous_stage != stage:
        command_term = env.command_manager.get_term("base_velocity")
        command_term.cfg.rel_standing_envs = standing_fractions[stage]
        command_term.cfg.ranges.lin_vel_x = lin_vel_x_ranges[stage]
        command_term.cfg.ranges.lin_vel_y = lin_vel_y_ranges[stage]
        command_term.cfg.ranges.ang_vel_z = ang_vel_z_ranges[stage]

        reset_base_cfg = env.event_manager.get_term_cfg("reset_base")
        root_vel = root_velocity_ranges[stage]
        reset_base_cfg.params["velocity_range"] = {
            "x": (-root_vel, root_vel),
            "y": (-root_vel, root_vel),
            "z": (0.0, 0.0),
            "roll": (-root_vel, root_vel),
            "pitch": (-root_vel, root_vel),
            "yaw": (-root_vel, root_vel),
        }
        env.event_manager.set_term_cfg("reset_base", reset_base_cfg)

        reset_joint_cfg = env.event_manager.get_term_cfg("reset_robot_joints")
        offset = joint_reset_offsets[stage]
        reset_joint_cfg.params["position_range"] = (-offset, offset)
        env.event_manager.set_term_cfg("reset_robot_joints", reset_joint_cfg)

        mass_cfg = env.event_manager.get_term_cfg("base_mass")
        mass_cfg.params["mass_distribution_params"] = mass_scale_ranges[stage]
        env.event_manager.set_term_cfg("base_mass", mass_cfg)

        push_cfg = env.event_manager.get_term_cfg("push_robot")
        push = push_xy_ranges[stage]
        push_cfg.params["velocity_range"] = {"x": (-push, push), "y": (-push, push)}
        env.event_manager.set_term_cfg("push_robot", push_cfg)

        command_term.time_left[:] = 0.0
        env._littlegreen_v143_st3215_loaded_hardware_stage = stage

    action_term = env.action_manager.get_term("joint_pos")
    return {
        "stage": float(stage),
        "standing_fraction": float(standing_fractions[stage]),
        "cmd_x_abs_max": float(max(abs(v) for v in lin_vel_x_ranges[stage])),
        "cmd_y_abs_max": float(max(abs(v) for v in lin_vel_y_ranges[stage])),
        "cmd_yaw_abs_max": float(max(abs(v) for v in ang_vel_z_ranges[stage])),
        "root_reset_velocity_abs_max": float(root_velocity_ranges[stage]),
        "joint_reset_offset_abs_max": float(joint_reset_offsets[stage]),
        "push_xy_abs_max": float(push_xy_ranges[stage]),
        "mass_scale_min": float(mass_scale_ranges[stage][0]),
        "mass_scale_max": float(mass_scale_ranges[stage][1]),
        "actuator_delay_mean_ms": float(torch.mean(action_term.sampled_total_delay_s).item() * 1000.0),
        "actuator_tau_mean_ms": float(torch.mean(action_term.sampled_tau_s).item() * 1000.0),
        "actuator_velocity_scale_mean": float(torch.mean(action_term.sampled_velocity_scale).item()),
        "curriculum_profile_id": 143.0,
    }


def st3215_loaded_v144_hardware_stage_curriculum(
    env: RLTaskEnv,
    env_ids: Sequence[int],
    stage_step_boundaries: tuple[int, int, int] = (32000, 128000, 256000),
    standing_fractions: tuple[float, float, float, float] = (0.40, 0.30, 0.25, 0.20),
    lin_vel_x_ranges: tuple[tuple[float, float], ...] = ((-0.28, 0.28), (-0.50, 0.50), (-0.70, 0.70), (-0.90, 0.90)),
    lin_vel_y_ranges: tuple[tuple[float, float], ...] = ((-0.08, 0.08), (-0.16, 0.16), (-0.26, 0.26), (-0.36, 0.36)),
    ang_vel_z_ranges: tuple[tuple[float, float], ...] = ((-0.22, 0.22), (-0.45, 0.45), (-0.70, 0.70), (-0.95, 0.95)),
    root_velocity_ranges: tuple[float, float, float, float] = (0.035, 0.07, 0.12, 0.18),
    joint_reset_offsets: tuple[float, float, float, float] = (0.012, 0.022, 0.035, 0.050),
    push_xy_ranges: tuple[float, float, float, float] = (0.0, 0.0, 0.02, 0.05),
    mass_scale_ranges: tuple[tuple[float, float], ...] = ((0.98, 1.02), (0.96, 1.04), (0.94, 1.06), (0.93, 1.07)),
) -> dict[str, torch.Tensor | float]:
    """v1.4.4 alternating-step loaded-ST3215 Hardware curriculum.

    This keeps continuous velocity commands and avoids command bins. Compared with
    v1.4.3, it keeps similar locomotion pressure but relies on gait-quality rewards
    to favor alternating, command-aligned stepping over one-foot lift/bracing.
    Reset/push disturbances remain gentle so failures primarily reflect gait
    learning, not disturbance survival.
    """
    del env_ids
    step = int(env.common_step_counter)
    b0, b1, b2 = stage_step_boundaries
    stage = 0 if step < b0 else 1 if step < b1 else 2 if step < b2 else 3

    previous_stage = getattr(env, "_littlegreen_v144_st3215_loaded_hardware_stage", None)
    if previous_stage != stage:
        command_term = env.command_manager.get_term("base_velocity")
        command_term.cfg.rel_standing_envs = standing_fractions[stage]
        command_term.cfg.ranges.lin_vel_x = lin_vel_x_ranges[stage]
        command_term.cfg.ranges.lin_vel_y = lin_vel_y_ranges[stage]
        command_term.cfg.ranges.ang_vel_z = ang_vel_z_ranges[stage]

        reset_base_cfg = env.event_manager.get_term_cfg("reset_base")
        root_vel = root_velocity_ranges[stage]
        reset_base_cfg.params["velocity_range"] = {
            "x": (-root_vel, root_vel),
            "y": (-root_vel, root_vel),
            "z": (0.0, 0.0),
            "roll": (-root_vel, root_vel),
            "pitch": (-root_vel, root_vel),
            "yaw": (-root_vel, root_vel),
        }
        env.event_manager.set_term_cfg("reset_base", reset_base_cfg)

        reset_joint_cfg = env.event_manager.get_term_cfg("reset_robot_joints")
        offset = joint_reset_offsets[stage]
        reset_joint_cfg.params["position_range"] = (-offset, offset)
        env.event_manager.set_term_cfg("reset_robot_joints", reset_joint_cfg)

        mass_cfg = env.event_manager.get_term_cfg("base_mass")
        mass_cfg.params["mass_distribution_params"] = mass_scale_ranges[stage]
        env.event_manager.set_term_cfg("base_mass", mass_cfg)

        push_cfg = env.event_manager.get_term_cfg("push_robot")
        push = push_xy_ranges[stage]
        push_cfg.params["velocity_range"] = {"x": (-push, push), "y": (-push, push)}
        env.event_manager.set_term_cfg("push_robot", push_cfg)

        command_term.time_left[:] = 0.0
        env._littlegreen_v144_st3215_loaded_hardware_stage = stage

    action_term = env.action_manager.get_term("joint_pos")
    return {
        "stage": float(stage),
        "standing_fraction": float(standing_fractions[stage]),
        "cmd_x_abs_max": float(max(abs(v) for v in lin_vel_x_ranges[stage])),
        "cmd_y_abs_max": float(max(abs(v) for v in lin_vel_y_ranges[stage])),
        "cmd_yaw_abs_max": float(max(abs(v) for v in ang_vel_z_ranges[stage])),
        "root_reset_velocity_abs_max": float(root_velocity_ranges[stage]),
        "joint_reset_offset_abs_max": float(joint_reset_offsets[stage]),
        "push_xy_abs_max": float(push_xy_ranges[stage]),
        "mass_scale_min": float(mass_scale_ranges[stage][0]),
        "mass_scale_max": float(mass_scale_ranges[stage][1]),
        "actuator_delay_mean_ms": float(torch.mean(action_term.sampled_total_delay_s).item() * 1000.0),
        "actuator_tau_mean_ms": float(torch.mean(action_term.sampled_tau_s).item() * 1000.0),
        "actuator_velocity_scale_mean": float(torch.mean(action_term.sampled_velocity_scale).item()),
        "curriculum_profile_id": 144.0,
    }

def st3215_loaded_v145_hardware_stage_curriculum(
    env: RLTaskEnv,
    env_ids: Sequence[int],
    stage_step_boundaries: tuple[int, int, int] = (32000, 128000, 256000),
    standing_fractions: tuple[float, float, float, float] = (0.35, 0.28, 0.22, 0.18),
    lin_vel_x_ranges: tuple[tuple[float, float], ...] = ((-0.30, 0.30), (-0.55, 0.55), (-0.75, 0.75), (-0.95, 0.95)),
    lin_vel_y_ranges: tuple[tuple[float, float], ...] = ((-0.08, 0.08), (-0.17, 0.17), (-0.28, 0.28), (-0.38, 0.38)),
    ang_vel_z_ranges: tuple[tuple[float, float], ...] = ((-0.22, 0.22), (-0.45, 0.45), (-0.70, 0.70), (-0.95, 0.95)),
    root_velocity_ranges: tuple[float, float, float, float] = (0.030, 0.060, 0.100, 0.150),
    joint_reset_offsets: tuple[float, float, float, float] = (0.012, 0.022, 0.035, 0.050),
    push_xy_ranges: tuple[float, float, float, float] = (0.0, 0.0, 0.02, 0.05),
    mass_scale_ranges: tuple[tuple[float, float], ...] = ((0.98, 1.02), (0.96, 1.04), (0.94, 1.06), (0.93, 1.07)),
) -> dict[str, torch.Tensor | float]:
    """v1.4.5 athletic loaded-ST3215 Hardware curriculum.

    Continuous commands only: no command bins. The curriculum keeps v1.4.4's
    locomotion pressure but uses the lower athletic q_default/vector residual and
    grounded-step rewards to favor knee-bent, command-aligned walking over hopping.
    """
    del env_ids
    step = int(env.common_step_counter)
    b0, b1, b2 = stage_step_boundaries
    stage = 0 if step < b0 else 1 if step < b1 else 2 if step < b2 else 3

    previous_stage = getattr(env, "_littlegreen_v145_st3215_loaded_hardware_stage", None)
    if previous_stage != stage:
        command_term = env.command_manager.get_term("base_velocity")
        command_term.cfg.rel_standing_envs = standing_fractions[stage]
        command_term.cfg.ranges.lin_vel_x = lin_vel_x_ranges[stage]
        command_term.cfg.ranges.lin_vel_y = lin_vel_y_ranges[stage]
        command_term.cfg.ranges.ang_vel_z = ang_vel_z_ranges[stage]

        reset_base_cfg = env.event_manager.get_term_cfg("reset_base")
        root_vel = root_velocity_ranges[stage]
        reset_base_cfg.params["velocity_range"] = {
            "x": (-root_vel, root_vel),
            "y": (-root_vel, root_vel),
            "z": (0.0, 0.0),
            "roll": (-root_vel, root_vel),
            "pitch": (-root_vel, root_vel),
            "yaw": (-root_vel, root_vel),
        }
        env.event_manager.set_term_cfg("reset_base", reset_base_cfg)

        reset_joint_cfg = env.event_manager.get_term_cfg("reset_robot_joints")
        offset = joint_reset_offsets[stage]
        reset_joint_cfg.params["position_range"] = (-offset, offset)
        env.event_manager.set_term_cfg("reset_robot_joints", reset_joint_cfg)

        mass_cfg = env.event_manager.get_term_cfg("base_mass")
        mass_cfg.params["mass_distribution_params"] = mass_scale_ranges[stage]
        env.event_manager.set_term_cfg("base_mass", mass_cfg)

        push_cfg = env.event_manager.get_term_cfg("push_robot")
        push = push_xy_ranges[stage]
        push_cfg.params["velocity_range"] = {"x": (-push, push), "y": (-push, push)}
        env.event_manager.set_term_cfg("push_robot", push_cfg)

        command_term.time_left[:] = 0.0
        env._littlegreen_v145_st3215_loaded_hardware_stage = stage

    action_term = env.action_manager.get_term("joint_pos")
    return {
        "stage": float(stage),
        "standing_fraction": float(standing_fractions[stage]),
        "cmd_x_abs_max": float(max(abs(v) for v in lin_vel_x_ranges[stage])),
        "cmd_y_abs_max": float(max(abs(v) for v in lin_vel_y_ranges[stage])),
        "cmd_yaw_abs_max": float(max(abs(v) for v in ang_vel_z_ranges[stage])),
        "root_reset_velocity_abs_max": float(root_velocity_ranges[stage]),
        "joint_reset_offset_abs_max": float(joint_reset_offsets[stage]),
        "push_xy_abs_max": float(push_xy_ranges[stage]),
        "mass_scale_min": float(mass_scale_ranges[stage][0]),
        "mass_scale_max": float(mass_scale_ranges[stage][1]),
        "actuator_delay_mean_ms": float(torch.mean(action_term.sampled_total_delay_s).item() * 1000.0),
        "actuator_tau_mean_ms": float(torch.mean(action_term.sampled_tau_s).item() * 1000.0),
        "actuator_velocity_scale_mean": float(torch.mean(action_term.sampled_velocity_scale).item()),
        "curriculum_profile_id": 145.0,
    }


def _find_command_tensor_for_floor(command_term):
    """Best-effort lookup of mutable command tensor inside UniformVelocityCommand."""
    for attr in ("vel_command_b", "command", "_command", "commands", "_commands"):
        if not hasattr(command_term, attr):
            continue
        try:
            target = getattr(command_term, attr)
            if torch.is_tensor(target) and target.ndim == 2 and target.shape[1] >= 3:
                return target
        except Exception:
            pass
    return None


def enforce_nonstanding_command_floor(
    env: RLTaskEnv,
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
    floor_mps: float = 0.22,
    eps: float = 1.0e-6,
) -> dict[str, torch.Tensor | float]:
    """Raise non-standing horizontal commands below a minimum speed floor.

    This keeps the command distribution continuous, but removes the ambiguous
    near-zero locomotion samples that let the policy treat Hardware training as
    standing with tiny disturbances. True standing environments remain zero.
    """
    del env_ids
    command_term = env.command_manager.get_term(command_name)
    command = _find_command_tensor_for_floor(command_term)
    if command is None:
        command = env.command_manager.get_command(command_name)

    xy = command[:, :2]
    norm = torch.linalg.vector_norm(xy, dim=1)

    standing_mask = None
    for attr in ("is_standing_env", "_is_standing_env"):
        if hasattr(command_term, attr):
            candidate = getattr(command_term, attr)
            if torch.is_tensor(candidate):
                standing_mask = candidate.bool()
                break
    if standing_mask is None:
        standing_mask = norm <= eps

    needs_floor = (~standing_mask) & (norm > eps) & (norm < floor_mps)
    scale = floor_mps / torch.clamp(norm, min=eps)
    command[:, :2] = torch.where(needs_floor.unsqueeze(-1), xy * scale.unsqueeze(-1), xy)

    return {
        "command_floor_mps": float(floor_mps),
        "command_floor_adjusted_fraction": float(torch.mean(needs_floor.float()).item()),
    }


def st3215_loaded_v146_anti_planted_hardware_stage_curriculum(
    env: RLTaskEnv,
    env_ids: Sequence[int],
    stage_step_boundaries: tuple[int, int, int] = (32000, 96000, 192000),
    standing_fractions: tuple[float, float, float, float] = (0.25, 0.18, 0.15, 0.12),
    lin_vel_x_ranges: tuple[tuple[float, float], ...] = ((-0.42, 0.42), (-0.60, 0.60), (-0.80, 0.80), (-1.00, 1.00)),
    lin_vel_y_ranges: tuple[tuple[float, float], ...] = ((-0.10, 0.10), (-0.18, 0.18), (-0.30, 0.30), (-0.40, 0.40)),
    ang_vel_z_ranges: tuple[tuple[float, float], ...] = ((-0.25, 0.25), (-0.50, 0.50), (-0.75, 0.75), (-1.00, 1.00)),
    root_velocity_ranges: tuple[float, float, float, float] = (0.025, 0.050, 0.090, 0.140),
    joint_reset_offsets: tuple[float, float, float, float] = (0.010, 0.020, 0.032, 0.045),
    push_xy_ranges: tuple[float, float, float, float] = (0.0, 0.0, 0.02, 0.04),
    mass_scale_ranges: tuple[tuple[float, float], ...] = ((0.98, 1.02), (0.96, 1.04), (0.94, 1.06), (0.93, 1.07)),
) -> dict[str, torch.Tensor | float]:
    """v1.4.6 anti-planted Hardware curriculum.

    This keeps continuous commands/no command bins, but lowers standing exposure
    and starts locomotion samples above the ambiguous micro-command region through
    ``enforce_nonstanding_command_floor``. Disturbances remain gentle so the main
    pressure is: move under a move command, do not brace in double support.
    """
    del env_ids
    step = int(env.common_step_counter)
    b0, b1, b2 = stage_step_boundaries
    stage = 0 if step < b0 else 1 if step < b1 else 2 if step < b2 else 3

    previous_stage = getattr(env, "_littlegreen_v146_st3215_loaded_hardware_stage", None)
    if previous_stage != stage:
        command_term = env.command_manager.get_term("base_velocity")
        command_term.cfg.rel_standing_envs = standing_fractions[stage]
        command_term.cfg.ranges.lin_vel_x = lin_vel_x_ranges[stage]
        command_term.cfg.ranges.lin_vel_y = lin_vel_y_ranges[stage]
        command_term.cfg.ranges.ang_vel_z = ang_vel_z_ranges[stage]

        reset_base_cfg = env.event_manager.get_term_cfg("reset_base")
        root_vel = root_velocity_ranges[stage]
        reset_base_cfg.params["velocity_range"] = {
            "x": (-root_vel, root_vel),
            "y": (-root_vel, root_vel),
            "z": (0.0, 0.0),
            "roll": (-root_vel, root_vel),
            "pitch": (-root_vel, root_vel),
            "yaw": (-root_vel, root_vel),
        }
        env.event_manager.set_term_cfg("reset_base", reset_base_cfg)

        reset_joint_cfg = env.event_manager.get_term_cfg("reset_robot_joints")
        offset = joint_reset_offsets[stage]
        reset_joint_cfg.params["position_range"] = (-offset, offset)
        env.event_manager.set_term_cfg("reset_robot_joints", reset_joint_cfg)

        mass_cfg = env.event_manager.get_term_cfg("base_mass")
        mass_cfg.params["mass_distribution_params"] = mass_scale_ranges[stage]
        env.event_manager.set_term_cfg("base_mass", mass_cfg)

        push_cfg = env.event_manager.get_term_cfg("push_robot")
        push = push_xy_ranges[stage]
        push_cfg.params["velocity_range"] = {"x": (-push, push), "y": (-push, push)}
        env.event_manager.set_term_cfg("push_robot", push_cfg)

        command_term.time_left[:] = 0.0
        env._littlegreen_v146_st3215_loaded_hardware_stage = stage

    action_term = env.action_manager.get_term("joint_pos")
    return {
        "stage": float(stage),
        "standing_fraction": float(standing_fractions[stage]),
        "cmd_x_abs_max": float(max(abs(v) for v in lin_vel_x_ranges[stage])),
        "cmd_y_abs_max": float(max(abs(v) for v in lin_vel_y_ranges[stage])),
        "cmd_yaw_abs_max": float(max(abs(v) for v in ang_vel_z_ranges[stage])),
        "root_reset_velocity_abs_max": float(root_velocity_ranges[stage]),
        "joint_reset_offset_abs_max": float(joint_reset_offsets[stage]),
        "push_xy_abs_max": float(push_xy_ranges[stage]),
        "mass_scale_min": float(mass_scale_ranges[stage][0]),
        "mass_scale_max": float(mass_scale_ranges[stage][1]),
        "actuator_delay_mean_ms": float(torch.mean(action_term.sampled_total_delay_s).item() * 1000.0),
        "actuator_tau_mean_ms": float(torch.mean(action_term.sampled_tau_s).item() * 1000.0),
        "actuator_velocity_scale_mean": float(torch.mean(action_term.sampled_velocity_scale).item()),
        "curriculum_profile_id": 146.0,
    }




def st3215_loaded_v147_phase_guided_hardware_stage_curriculum(
    env: RLTaskEnv,
    env_ids: Sequence[int],
    stage_step_boundaries: tuple[int, int, int] = (40000, 120000, 240000),
    standing_fractions: tuple[float, float, float, float] = (0.22, 0.16, 0.12, 0.10),
    lin_vel_x_ranges: tuple[tuple[float, float], ...] = ((-0.40, 0.40), (-0.55, 0.55), (-0.75, 0.75), (-0.95, 0.95)),
    lin_vel_y_ranges: tuple[tuple[float, float], ...] = ((-0.08, 0.08), (-0.14, 0.14), (-0.24, 0.24), (-0.34, 0.34)),
    ang_vel_z_ranges: tuple[tuple[float, float], ...] = ((-0.18, 0.18), (-0.35, 0.35), (-0.60, 0.60), (-0.85, 0.85)),
    root_velocity_ranges: tuple[float, float, float, float] = (0.020, 0.040, 0.070, 0.110),
    joint_reset_offsets: tuple[float, float, float, float] = (0.008, 0.016, 0.026, 0.038),
    push_xy_ranges: tuple[float, float, float, float] = (0.0, 0.0, 0.015, 0.030),
    mass_scale_ranges: tuple[tuple[float, float], ...] = ((0.98, 1.02), (0.97, 1.03), (0.95, 1.05), (0.94, 1.06)),
) -> dict[str, torch.Tensor | float]:
    """v1.4.7 phase-guided Hardware curriculum.

    Continuous commands are preserved.  The main new scaffold is the phase input
    and phase-conditioned contact shaping, so the command curriculum is less
    aggressive than v1.4.6 and lets the policy learn rhythm before broad stress.
    """
    del env_ids
    step = int(env.common_step_counter)
    b0, b1, b2 = stage_step_boundaries
    stage = 0 if step < b0 else 1 if step < b1 else 2 if step < b2 else 3

    previous_stage = getattr(env, "_littlegreen_v147_st3215_loaded_hardware_stage", None)
    if previous_stage != stage:
        command_term = env.command_manager.get_term("base_velocity")
        command_term.cfg.rel_standing_envs = standing_fractions[stage]
        command_term.cfg.ranges.lin_vel_x = lin_vel_x_ranges[stage]
        command_term.cfg.ranges.lin_vel_y = lin_vel_y_ranges[stage]
        command_term.cfg.ranges.ang_vel_z = ang_vel_z_ranges[stage]

        reset_base_cfg = env.event_manager.get_term_cfg("reset_base")
        root_vel = root_velocity_ranges[stage]
        reset_base_cfg.params["velocity_range"] = {
            "x": (-root_vel, root_vel),
            "y": (-root_vel, root_vel),
            "z": (0.0, 0.0),
            "roll": (-root_vel, root_vel),
            "pitch": (-root_vel, root_vel),
            "yaw": (-root_vel, root_vel),
        }
        env.event_manager.set_term_cfg("reset_base", reset_base_cfg)

        reset_joint_cfg = env.event_manager.get_term_cfg("reset_robot_joints")
        offset = joint_reset_offsets[stage]
        reset_joint_cfg.params["position_range"] = (-offset, offset)
        env.event_manager.set_term_cfg("reset_robot_joints", reset_joint_cfg)

        mass_cfg = env.event_manager.get_term_cfg("base_mass")
        mass_cfg.params["mass_distribution_params"] = mass_scale_ranges[stage]
        env.event_manager.set_term_cfg("base_mass", mass_cfg)

        push_cfg = env.event_manager.get_term_cfg("push_robot")
        push = push_xy_ranges[stage]
        push_cfg.params["velocity_range"] = {"x": (-push, push), "y": (-push, push)}
        env.event_manager.set_term_cfg("push_robot", push_cfg)

        command_term.time_left[:] = 0.0
        env._littlegreen_v147_st3215_loaded_hardware_stage = stage

    action_term = env.action_manager.get_term("joint_pos")
    return {
        "stage": float(stage),
        "standing_fraction": float(standing_fractions[stage]),
        "cmd_x_abs_max": float(max(abs(v) for v in lin_vel_x_ranges[stage])),
        "cmd_y_abs_max": float(max(abs(v) for v in lin_vel_y_ranges[stage])),
        "cmd_yaw_abs_max": float(max(abs(v) for v in ang_vel_z_ranges[stage])),
        "root_reset_velocity_abs_max": float(root_velocity_ranges[stage]),
        "joint_reset_offset_abs_max": float(joint_reset_offsets[stage]),
        "push_xy_abs_max": float(push_xy_ranges[stage]),
        "mass_scale_min": float(mass_scale_ranges[stage][0]),
        "mass_scale_max": float(mass_scale_ranges[stage][1]),
        "actuator_delay_mean_ms": float(torch.mean(action_term.sampled_total_delay_s).item() * 1000.0),
        "actuator_tau_mean_ms": float(torch.mean(action_term.sampled_tau_s).item() * 1000.0),
        "actuator_velocity_scale_mean": float(torch.mean(action_term.sampled_velocity_scale).item()),
        "curriculum_profile_id": 147.0,
    }



def st3215_loaded_v148_phase_lift_step_hardware_stage_curriculum(
    env: RLTaskEnv,
    env_ids: Sequence[int],
    stage_step_boundaries: tuple[int, int, int] = (48000, 120000, 216000),
    standing_fractions: tuple[float, float, float, float] = (0.18, 0.14, 0.10, 0.08),
    lin_vel_x_ranges: tuple[tuple[float, float], ...] = ((-0.30, 0.45), (-0.45, 0.60), (-0.60, 0.75), (-0.75, 0.90)),
    lin_vel_y_ranges: tuple[tuple[float, float], ...] = ((-0.06, 0.06), (-0.10, 0.10), (-0.18, 0.18), (-0.28, 0.28)),
    ang_vel_z_ranges: tuple[tuple[float, float], ...] = ((-0.12, 0.12), (-0.22, 0.22), (-0.45, 0.45), (-0.70, 0.70)),
    root_velocity_ranges: tuple[float, float, float, float] = (0.015, 0.030, 0.055, 0.085),
    joint_reset_offsets: tuple[float, float, float, float] = (0.006, 0.012, 0.022, 0.034),
    push_xy_ranges: tuple[float, float, float, float] = (0.0, 0.0, 0.010, 0.020),
    mass_scale_ranges: tuple[tuple[float, float], ...] = ((0.99, 1.01), (0.98, 1.02), (0.96, 1.04), (0.94, 1.06)),
) -> dict[str, torch.Tensor | float]:
    """v1.4.8 phase-lift/foot-placement Hardware curriculum.

    Stage 1 emphasizes phase alternation plus actual foot clearance at modest
    commands.  Stage 2 adds more placement pressure.  Stage 3 increases tracking
    pressure through larger commands.  Stage 4 broadens the command envelope.  The
    reward weights are fixed in the config; this curriculum mainly controls the
    command and disturbance envelope so early learning is not dominated by
    high-speed yaw/forward breakdowns.
    """
    del env_ids
    step = int(env.common_step_counter)
    b0, b1, b2 = stage_step_boundaries
    stage = 0 if step < b0 else 1 if step < b1 else 2 if step < b2 else 3

    previous_stage = getattr(env, "_littlegreen_v148_st3215_loaded_hardware_stage", None)
    if previous_stage != stage:
        command_term = env.command_manager.get_term("base_velocity")
        command_term.cfg.rel_standing_envs = standing_fractions[stage]
        command_term.cfg.ranges.lin_vel_x = lin_vel_x_ranges[stage]
        command_term.cfg.ranges.lin_vel_y = lin_vel_y_ranges[stage]
        command_term.cfg.ranges.ang_vel_z = ang_vel_z_ranges[stage]

        reset_base_cfg = env.event_manager.get_term_cfg("reset_base")
        root_vel = root_velocity_ranges[stage]
        reset_base_cfg.params["velocity_range"] = {
            "x": (-root_vel, root_vel),
            "y": (-root_vel, root_vel),
            "z": (0.0, 0.0),
            "roll": (-root_vel, root_vel),
            "pitch": (-root_vel, root_vel),
            "yaw": (-root_vel, root_vel),
        }
        env.event_manager.set_term_cfg("reset_base", reset_base_cfg)

        reset_joint_cfg = env.event_manager.get_term_cfg("reset_robot_joints")
        offset = joint_reset_offsets[stage]
        reset_joint_cfg.params["position_range"] = (-offset, offset)
        env.event_manager.set_term_cfg("reset_robot_joints", reset_joint_cfg)

        mass_cfg = env.event_manager.get_term_cfg("base_mass")
        mass_cfg.params["mass_distribution_params"] = mass_scale_ranges[stage]
        env.event_manager.set_term_cfg("base_mass", mass_cfg)

        push_cfg = env.event_manager.get_term_cfg("push_robot")
        push = push_xy_ranges[stage]
        push_cfg.params["velocity_range"] = {"x": (-push, push), "y": (-push, push)}
        env.event_manager.set_term_cfg("push_robot", push_cfg)

        command_term.time_left[:] = 0.0
        env._littlegreen_v148_st3215_loaded_hardware_stage = stage

    action_term = env.action_manager.get_term("joint_pos")
    return {
        "stage": float(stage),
        "standing_fraction": float(standing_fractions[stage]),
        "cmd_x_abs_max": float(max(abs(v) for v in lin_vel_x_ranges[stage])),
        "cmd_y_abs_max": float(max(abs(v) for v in lin_vel_y_ranges[stage])),
        "cmd_yaw_abs_max": float(max(abs(v) for v in ang_vel_z_ranges[stage])),
        "root_reset_velocity_abs_max": float(root_velocity_ranges[stage]),
        "joint_reset_offset_abs_max": float(joint_reset_offsets[stage]),
        "push_xy_abs_max": float(push_xy_ranges[stage]),
        "mass_scale_min": float(mass_scale_ranges[stage][0]),
        "mass_scale_max": float(mass_scale_ranges[stage][1]),
        "actuator_delay_mean_ms": float(torch.mean(action_term.sampled_total_delay_s).item() * 1000.0),
        "actuator_tau_mean_ms": float(torch.mean(action_term.sampled_tau_s).item() * 1000.0),
        "actuator_velocity_scale_mean": float(torch.mean(action_term.sampled_velocity_scale).item()),
        "curriculum_profile_id": 148.0,
    }

# -----------------------------------------------------------------------------
# Littlegreen v2.0.0 / Hardware-ST3215-Loaded-v9 gait acquisition
# -----------------------------------------------------------------------------

def st3215_loaded_v9_gait_acquisition_curriculum(
    env: RLTaskEnv,
    env_ids: Sequence[int],
    stage_step_boundaries: tuple[int, int, int] = (128000, 320000, 512000),
    standing_fractions: tuple[float, float, float, float] = (0.25, 0.23, 0.21, 0.20),
    lin_vel_x_ranges: tuple[tuple[float, float], ...] = (
        (0.25, 0.36),
        (0.25, 0.42),
        (0.25, 0.50),
        (0.25, 0.55),
    ),
    lin_vel_y_ranges: tuple[tuple[float, float], ...] = (
        (-0.04, 0.04),
        (-0.05, 0.05),
        (-0.065, 0.065),
        (-0.08, 0.08),
    ),
    ang_vel_z_ranges: tuple[tuple[float, float], ...] = (
        (-0.08, 0.08),
        (-0.12, 0.12),
        (-0.17, 0.17),
        (-0.22, 0.22),
    ),
    root_velocity_ranges: tuple[float, float, float, float] = (0.010, 0.015, 0.020, 0.025),
    joint_reset_offsets: tuple[float, float, float, float] = (0.004, 0.006, 0.008, 0.010),
    mass_scale_ranges: tuple[tuple[float, float], ...] = (
        (0.995, 1.005),
        (0.990, 1.010),
        (0.985, 1.015),
        (0.980, 1.020),
    ),
) -> dict[str, torch.Tensor | float]:
    """Gentle four-stage forward-gait acquisition curriculum for v9.

    With the v9 runner's 64 policy steps per PPO iteration, the default stage
    changes occur near iterations 2000, 5000, and 8000.  Only command/reset/mass
    ranges broaden.  External pushes remain disabled for the complete 10k run.
    """
    del env_ids
    step = int(env.common_step_counter)
    b0, b1, b2 = stage_step_boundaries
    stage = 0 if step < b0 else 1 if step < b1 else 2 if step < b2 else 3

    previous_stage = getattr(env, "_littlegreen_v9_gait_acquisition_stage", None)
    if previous_stage != stage:
        command_term = env.command_manager.get_term("base_velocity")
        command_term.cfg.rel_standing_envs = standing_fractions[stage]
        command_term.cfg.ranges.lin_vel_x = lin_vel_x_ranges[stage]
        command_term.cfg.ranges.lin_vel_y = lin_vel_y_ranges[stage]
        command_term.cfg.ranges.ang_vel_z = ang_vel_z_ranges[stage]

        reset_base_cfg = env.event_manager.get_term_cfg("reset_base")
        root_vel = root_velocity_ranges[stage]
        reset_base_cfg.params["velocity_range"] = {
            "x": (-root_vel, root_vel),
            "y": (-root_vel, root_vel),
            "z": (0.0, 0.0),
            "roll": (-root_vel, root_vel),
            "pitch": (-root_vel, root_vel),
            "yaw": (-root_vel, root_vel),
        }
        env.event_manager.set_term_cfg("reset_base", reset_base_cfg)

        reset_joint_cfg = env.event_manager.get_term_cfg("reset_robot_joints")
        offset = joint_reset_offsets[stage]
        reset_joint_cfg.params["position_range"] = (-offset, offset)
        env.event_manager.set_term_cfg("reset_robot_joints", reset_joint_cfg)

        mass_cfg = env.event_manager.get_term_cfg("base_mass")
        mass_cfg.params["mass_distribution_params"] = mass_scale_ranges[stage]
        env.event_manager.set_term_cfg("base_mass", mass_cfg)

        # Force immediate resampling when the envelope changes.
        command_term.time_left[:] = 0.0
        env._littlegreen_v9_gait_acquisition_stage = stage

    action_term = env.action_manager.get_term("joint_pos")
    return {
        "stage": float(stage),
        "standing_fraction": float(standing_fractions[stage]),
        "cmd_x_min": float(lin_vel_x_ranges[stage][0]),
        "cmd_x_max": float(lin_vel_x_ranges[stage][1]),
        "cmd_y_abs_max": float(max(abs(v) for v in lin_vel_y_ranges[stage])),
        "cmd_yaw_abs_max": float(max(abs(v) for v in ang_vel_z_ranges[stage])),
        "root_reset_velocity_abs_max": float(root_velocity_ranges[stage]),
        "joint_reset_offset_abs_max": float(joint_reset_offsets[stage]),
        "push_xy_abs_max": 0.0,
        "mass_scale_min": float(mass_scale_ranges[stage][0]),
        "mass_scale_max": float(mass_scale_ranges[stage][1]),
        "actuator_delay_mean_ms": float(torch.mean(action_term.sampled_total_delay_s).item() * 1000.0),
        "actuator_tau_mean_ms": float(torch.mean(action_term.sampled_tau_s).item() * 1000.0),
        "actuator_velocity_scale_mean": float(torch.mean(action_term.sampled_velocity_scale).item()),
        "curriculum_profile_id": 209.0,
    }


def st3215_loaded_v10_transfer_lift_place_curriculum(
    env,
    env_ids,
    stage_step_boundaries: tuple[int, int, int] = (96000, 192000, 268800),
    standing_fractions: tuple[float, float, float, float] = (0.28, 0.26, 0.23, 0.20),
    lin_vel_x_ranges: tuple[tuple[float, float], ...] = (
        (0.25, 0.34),
        (0.25, 0.40),
        (0.25, 0.48),
        (0.25, 0.55),
    ),
    lin_vel_y_ranges: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (-0.02, 0.02),
        (-0.05, 0.05),
        (-0.08, 0.08),
    ),
    ang_vel_z_ranges: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (-0.04, 0.04),
        (-0.12, 0.12),
        (-0.22, 0.22),
    ),
    gait_period_s: tuple[float, float, float, float] = (0.90, 0.86, 0.82, 0.78),
    transition_fractions: tuple[float, float, float, float] = (0.08, 0.07, 0.06, 0.05),
    target_clearance_m: tuple[float, float, float, float] = (0.018, 0.024, 0.028, 0.030),
    clearance_std_m: tuple[float, float, float, float] = (0.009, 0.010, 0.011, 0.012),
    clearance_gate_m: tuple[float, float, float, float] = (0.008, 0.010, 0.012, 0.012),
    target_step_m: tuple[float, float, float, float] = (0.025, 0.040, 0.050, 0.055),
    placement_scales: tuple[float, float, float, float] = (0.15, 0.70, 1.00, 1.00),
    tracking_scales: tuple[float, float, float, float] = (0.55, 0.70, 0.90, 1.00),
    target_stance_force_ratios: tuple[float, float, float, float] = (0.75, 0.78, 0.80, 0.80),
    com_target_forward_m: tuple[float, float, float, float] = (0.080, 0.082, 0.085, 0.085),
    com_lateral_stance_fractions: tuple[float, float, float, float] = (0.75, 0.80, 0.85, 0.85),
    root_velocity_ranges: tuple[float, float, float, float] = (0.008, 0.012, 0.018, 0.022),
    joint_reset_offsets: tuple[float, float, float, float] = (0.003, 0.005, 0.007, 0.009),
    mass_scale_ranges: tuple[tuple[float, float], ...] = (
        (0.997, 1.003),
        (0.995, 1.005),
        (0.990, 1.010),
        (0.985, 1.015),
    ),
) -> dict[str, torch.Tensor | float]:
    """Condensed 5k v10 transfer -> lift -> place -> translate curriculum.

    With 64 policy steps per PPO iteration, stages change near iterations 1500,
    3000, and 4200.  The first stage is deliberately straight-forward and puts
    reduced pressure on base-velocity tracking while the policy learns alternating
    load transfer and real foot clearance.  Placement is then gated in gradually.
    External pushes remain disabled for the entire go/no-go run.
    """
    del env_ids
    step = int(env.common_step_counter)
    b0, b1, b2 = stage_step_boundaries
    stage = 0 if step < b0 else 1 if step < b1 else 2 if step < b2 else 3

    previous_stage = getattr(env, "_littlegreen_v10_transfer_lift_place_stage", None)
    if previous_stage != stage:
        command_term = env.command_manager.get_term("base_velocity")
        command_term.cfg.rel_standing_envs = standing_fractions[stage]
        command_term.cfg.ranges.lin_vel_x = lin_vel_x_ranges[stage]
        command_term.cfg.ranges.lin_vel_y = lin_vel_y_ranges[stage]
        command_term.cfg.ranges.ang_vel_z = ang_vel_z_ranges[stage]

        reset_base_cfg = env.event_manager.get_term_cfg("reset_base")
        root_vel = root_velocity_ranges[stage]
        reset_base_cfg.params["velocity_range"] = {
            "x": (-root_vel, root_vel),
            "y": (-root_vel, root_vel),
            "z": (0.0, 0.0),
            "roll": (-root_vel, root_vel),
            "pitch": (-root_vel, root_vel),
            "yaw": (-root_vel, root_vel),
        }
        env.event_manager.set_term_cfg("reset_base", reset_base_cfg)

        reset_joint_cfg = env.event_manager.get_term_cfg("reset_robot_joints")
        joint_offset = joint_reset_offsets[stage]
        reset_joint_cfg.params["position_range"] = (-joint_offset, joint_offset)
        env.event_manager.set_term_cfg("reset_robot_joints", reset_joint_cfg)

        mass_cfg = env.event_manager.get_term_cfg("base_mass")
        mass_cfg.params["mass_distribution_params"] = mass_scale_ranges[stage]
        env.event_manager.set_term_cfg("base_mass", mass_cfg)

        # Reward/phase functions read these scalars directly.  This avoids
        # rebuilding manager terms and keeps stage changes atomic.
        env._littlegreen_v10_gait_period_s = float(gait_period_s[stage])
        env._littlegreen_v10_transition_fraction = float(transition_fractions[stage])
        env._littlegreen_v10_target_clearance_m = float(target_clearance_m[stage])
        env._littlegreen_v10_clearance_std_m = float(clearance_std_m[stage])
        env._littlegreen_v10_clearance_gate_m = float(clearance_gate_m[stage])
        env._littlegreen_v10_target_step_m = float(target_step_m[stage])
        env._littlegreen_v10_placement_scale = float(placement_scales[stage])
        env._littlegreen_v10_tracking_scale = float(tracking_scales[stage])
        env._littlegreen_v10_target_stance_force_ratio = float(
            target_stance_force_ratios[stage]
        )
        env._littlegreen_v10_com_target_forward_m = float(com_target_forward_m[stage])
        env._littlegreen_v10_com_lateral_stance_fraction = float(
            com_lateral_stance_fractions[stage]
        )

        # Start the new skill stage from fresh commands and synchronized phase.
        command_term.time_left[:] = 0.0
        env._littlegreen_v10_was_moving = torch.zeros(
            env.num_envs, device=command_term.time_left.device, dtype=torch.bool
        )
        env._littlegreen_v10_transfer_lift_place_stage = stage

    action_term = env.action_manager.get_term("joint_pos")
    return {
        "stage": float(stage),
        "standing_fraction": float(standing_fractions[stage]),
        "cmd_x_min": float(lin_vel_x_ranges[stage][0]),
        "cmd_x_max": float(lin_vel_x_ranges[stage][1]),
        "cmd_y_abs_max": float(max(abs(v) for v in lin_vel_y_ranges[stage])),
        "cmd_yaw_abs_max": float(max(abs(v) for v in ang_vel_z_ranges[stage])),
        "gait_period_s": float(gait_period_s[stage]),
        "transition_fraction_per_half": float(transition_fractions[stage]),
        "double_support_fraction_total": float(2.0 * transition_fractions[stage]),
        "target_clearance_m": float(target_clearance_m[stage]),
        "clearance_gate_m": float(clearance_gate_m[stage]),
        "target_step_m": float(target_step_m[stage]),
        "placement_scale": float(placement_scales[stage]),
        "tracking_scale": float(tracking_scales[stage]),
        "target_stance_force_ratio": float(target_stance_force_ratios[stage]),
        "com_target_forward_m": float(com_target_forward_m[stage]),
        "com_lateral_stance_fraction": float(com_lateral_stance_fractions[stage]),
        "moving_height_target_m": 0.455,
        "root_reset_velocity_abs_max": float(root_velocity_ranges[stage]),
        "joint_reset_offset_abs_max": float(joint_reset_offsets[stage]),
        "push_xy_abs_max": 0.0,
        "mass_scale_min": float(mass_scale_ranges[stage][0]),
        "mass_scale_max": float(mass_scale_ranges[stage][1]),
        "actuator_delay_mean_ms": float(torch.mean(action_term.sampled_total_delay_s).item() * 1000.0),
        "actuator_tau_mean_ms": float(torch.mean(action_term.sampled_tau_s).item() * 1000.0),
        "actuator_velocity_scale_mean": float(torch.mean(action_term.sampled_velocity_scale).item()),
        "curriculum_profile_id": 210.0,
    }


def st3215_loaded_v11_step_takeover_curriculum(
    env,
    env_ids,
    stage_step_boundaries: tuple[int, int, int] = (80000, 176000, 256000),
    standing_fractions: tuple[float, float, float, float] = (0.22, 0.20, 0.18, 0.18),
    lin_vel_x_ranges: tuple[tuple[float, float], ...] = (
        (0.25, 0.34),
        (0.25, 0.40),
        (0.25, 0.48),
        (0.25, 0.55),
    ),
    lin_vel_y_ranges: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.0, 0.0),
        (-0.025, 0.025),
        (-0.06, 0.06),
    ),
    ang_vel_z_ranges: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.0, 0.0),
        (-0.06, 0.06),
        (-0.16, 0.16),
    ),
    gait_period_s: tuple[float, float, float, float] = (0.82, 0.80, 0.78, 0.76),
    transition_fractions: tuple[float, float, float, float] = (0.050, 0.045, 0.040, 0.040),
    takeover_fractions: tuple[float, float, float, float] = (0.050, 0.045, 0.040, 0.040),
    target_clearance_m: tuple[float, float, float, float] = (0.026, 0.028, 0.030, 0.030),
    clearance_std_m: tuple[float, float, float, float] = (0.012, 0.012, 0.013, 0.013),
    clearance_gate_m: tuple[float, float, float, float] = (0.010, 0.011, 0.012, 0.012),
    min_step_m: tuple[float, float, float, float] = (0.022, 0.024, 0.025, 0.025),
    max_step_m: tuple[float, float, float, float] = (0.040, 0.048, 0.058, 0.065),
    step_length_scales: tuple[float, float, float, float] = (0.12, 0.14, 0.16, 0.17),
    tracking_scales: tuple[float, float, float, float] = (0.65, 0.75, 0.90, 1.00),
    target_stance_force_ratios: tuple[float, float, float, float] = (0.78, 0.80, 0.82, 0.82),
    target_takeover_force_ratios: tuple[float, float, float, float] = (0.74, 0.76, 0.78, 0.80),
    com_lateral_stance_fractions: tuple[float, float, float, float] = (0.82, 0.85, 0.88, 0.90),
    com_support_forward_offsets_m: tuple[float, float, float, float] = (-0.018, -0.016, -0.015, -0.012),
    transfer_max_s: tuple[float, float, float, float] = (0.22, 0.20, 0.18, 0.17),
    swing_max_s: tuple[float, float, float, float] = (0.60, 0.58, 0.55, 0.52),
    takeover_max_s: tuple[float, float, float, float] = (0.26, 0.24, 0.22, 0.20),
    root_velocity_ranges: tuple[float, float, float, float] = (0.010, 0.012, 0.016, 0.020),
    joint_reset_offsets: tuple[float, float, float, float] = (0.004, 0.005, 0.007, 0.008),
    mass_scale_ranges: tuple[tuple[float, float], ...] = (
        (0.995, 1.005),
        (0.992, 1.008),
        (0.988, 1.012),
        (0.985, 1.015),
    ),
) -> dict[str, torch.Tensor | float]:
    """Condensed 5k v11 touchdown/takeover curriculum from v10 model_4000.

    With 64 policy steps per PPO iteration, stages change near iterations 1250,
    2750, and 4000.  The first two stages remain straight-forward and convert the
    v10 rocking oscillator into trailing-foot recovery, advancing touchdown, and
    load/COM takeover.  Small lateral and yaw ranges appear only after consecutive
    forward half-cycle progression has had time to emerge.
    """
    del env_ids
    step = int(env.common_step_counter)
    b0, b1, b2 = stage_step_boundaries
    stage = 0 if step < b0 else 1 if step < b1 else 2 if step < b2 else 3

    previous_stage = getattr(env, "_littlegreen_v11_step_takeover_stage", None)
    if previous_stage != stage:
        command_term = env.command_manager.get_term("base_velocity")
        command_term.cfg.rel_standing_envs = standing_fractions[stage]
        command_term.cfg.ranges.lin_vel_x = lin_vel_x_ranges[stage]
        command_term.cfg.ranges.lin_vel_y = lin_vel_y_ranges[stage]
        command_term.cfg.ranges.ang_vel_z = ang_vel_z_ranges[stage]

        reset_base_cfg = env.event_manager.get_term_cfg("reset_base")
        root_vel = root_velocity_ranges[stage]
        reset_base_cfg.params["velocity_range"] = {
            "x": (-root_vel, root_vel),
            "y": (-root_vel, root_vel),
            "z": (0.0, 0.0),
            "roll": (-root_vel, root_vel),
            "pitch": (-root_vel, root_vel),
            "yaw": (-root_vel, root_vel),
        }
        env.event_manager.set_term_cfg("reset_base", reset_base_cfg)

        reset_joint_cfg = env.event_manager.get_term_cfg("reset_robot_joints")
        joint_offset = joint_reset_offsets[stage]
        reset_joint_cfg.params["position_range"] = (-joint_offset, joint_offset)
        env.event_manager.set_term_cfg("reset_robot_joints", reset_joint_cfg)

        mass_cfg = env.event_manager.get_term_cfg("base_mass")
        mass_cfg.params["mass_distribution_params"] = mass_scale_ranges[stage]
        env.event_manager.set_term_cfg("base_mass", mass_cfg)

        env._littlegreen_v11_gait_period_s = float(gait_period_s[stage])
        env._littlegreen_v11_transition_fraction = float(transition_fractions[stage])
        env._littlegreen_v11_takeover_fraction = float(takeover_fractions[stage])
        env._littlegreen_v11_target_clearance_m = float(target_clearance_m[stage])
        env._littlegreen_v11_clearance_std_m = float(clearance_std_m[stage])
        env._littlegreen_v11_clearance_gate_m = float(clearance_gate_m[stage])
        env._littlegreen_v11_min_step_m = float(min_step_m[stage])
        env._littlegreen_v11_max_step_m = float(max_step_m[stage])
        env._littlegreen_v11_step_length_scale = float(step_length_scales[stage])
        env._littlegreen_v11_tracking_scale = float(tracking_scales[stage])
        env._littlegreen_v11_target_stance_force_ratio = float(
            target_stance_force_ratios[stage]
        )
        env._littlegreen_v11_target_takeover_force_ratio = float(
            target_takeover_force_ratios[stage]
        )
        env._littlegreen_v11_reward_force_ratio = float(
            target_takeover_force_ratios[stage]
        )
        env._littlegreen_v11_com_lateral_stance_fraction = float(
            com_lateral_stance_fractions[stage]
        )
        env._littlegreen_v11_com_support_forward_offset_m = float(
            com_support_forward_offsets_m[stage]
        )
        env._littlegreen_v11_transfer_max_s = float(transfer_max_s[stage])
        env._littlegreen_v11_swing_max_s = float(swing_max_s[stage])
        env._littlegreen_v11_takeover_max_s = float(takeover_max_s[stage])

        # Force a new command and a fresh event-gated half-cycle at every skill stage.
        command_term.time_left[:] = 0.0
        env._littlegreen_v11_was_moving = torch.zeros(
            env.num_envs, device=command_term.time_left.device, dtype=torch.bool
        )
        env._littlegreen_v11_step_takeover_stage = stage

    action_term = env.action_manager.get_term("joint_pos")
    return {
        "stage": float(stage),
        "standing_fraction": float(standing_fractions[stage]),
        "cmd_x_min": float(lin_vel_x_ranges[stage][0]),
        "cmd_x_max": float(lin_vel_x_ranges[stage][1]),
        "cmd_y_abs_max": float(max(abs(v) for v in lin_vel_y_ranges[stage])),
        "cmd_yaw_abs_max": float(max(abs(v) for v in ang_vel_z_ranges[stage])),
        "gait_period_s": float(gait_period_s[stage]),
        "transfer_fraction_per_half": float(transition_fractions[stage]),
        "takeover_fraction_per_half": float(takeover_fractions[stage]),
        "scheduled_double_support_fraction_total": float(
            2.0 * (transition_fractions[stage] + takeover_fractions[stage])
        ),
        "target_clearance_m": float(target_clearance_m[stage]),
        "clearance_gate_m": float(clearance_gate_m[stage]),
        "min_step_m": float(min_step_m[stage]),
        "max_step_m": float(max_step_m[stage]),
        "step_length_scale": float(step_length_scales[stage]),
        "tracking_scale": float(tracking_scales[stage]),
        "target_stance_force_ratio": float(target_stance_force_ratios[stage]),
        "target_takeover_force_ratio": float(target_takeover_force_ratios[stage]),
        "com_lateral_stance_fraction": float(com_lateral_stance_fractions[stage]),
        "com_support_forward_offset_m": float(com_support_forward_offsets_m[stage]),
        "moving_height_target_m": 0.457,
        "moving_height_std_m": 0.075,
        "root_reset_velocity_abs_max": float(root_velocity_ranges[stage]),
        "joint_reset_offset_abs_max": float(joint_reset_offsets[stage]),
        "push_xy_abs_max": 0.0,
        "mass_scale_min": float(mass_scale_ranges[stage][0]),
        "mass_scale_max": float(mass_scale_ranges[stage][1]),
        "actuator_delay_mean_ms": float(torch.mean(action_term.sampled_total_delay_s).item() * 1000.0),
        "actuator_tau_mean_ms": float(torch.mean(action_term.sampled_tau_s).item() * 1000.0),
        "actuator_velocity_scale_mean": float(torch.mean(action_term.sampled_velocity_scale).item()),
        "curriculum_profile_id": 211.0,
    }


def st3215_loaded_v11a_hybrid_progression_curriculum(
    env,
    env_ids,
    stage_step_boundaries: tuple[int, int, int] = (48000, 128000, 224000),
    standing_fractions: tuple[float, float, float, float] = (0.25, 0.23, 0.20, 0.18),
    lin_vel_x_ranges: tuple[tuple[float, float], ...] = (
        (0.30, 0.40),
        (0.30, 0.48),
        (0.32, 0.56),
        (0.35, 0.65),
    ),
    lin_vel_y_ranges: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.0, 0.0),
        (-0.02, 0.02),
        (-0.05, 0.05),
    ),
    ang_vel_z_ranges: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.0, 0.0),
        (-0.05, 0.05),
        (-0.12, 0.12),
    ),
    gait_period_s: tuple[float, float, float, float] = (0.82, 0.82, 0.80, 0.78),
    transition_fractions: tuple[float, float, float, float] = (0.060, 0.055, 0.050, 0.050),
    soft_gate_floors: tuple[float, float, float, float] = (1.00, 0.75, 0.55, 0.45),
    late_swing_starts: tuple[float, float, float, float] = (0.82, 0.80, 0.78, 0.76),
    target_clearance_m: tuple[float, float, float, float] = (0.028, 0.028, 0.030, 0.030),
    clearance_std_m: tuple[float, float, float, float] = (0.012, 0.012, 0.013, 0.013),
    clearance_gate_m: tuple[float, float, float, float] = (0.010, 0.010, 0.011, 0.012),
    min_step_m: tuple[float, float, float, float] = (0.018, 0.020, 0.022, 0.025),
    max_step_m: tuple[float, float, float, float] = (0.035, 0.040, 0.045, 0.050),
    step_length_scales: tuple[float, float, float, float] = (0.10, 0.11, 0.12, 0.13),
    progression_scales: tuple[float, float, float, float] = (0.20, 0.55, 0.85, 1.00),
    contact_scales: tuple[float, float, float, float] = (1.00, 0.90, 0.75, 0.65),
    tracking_scales: tuple[float, float, float, float] = (0.70, 0.78, 0.90, 1.00),
    target_stance_force_ratios: tuple[float, float, float, float] = (0.76, 0.78, 0.80, 0.80),
    com_lateral_stance_fractions: tuple[float, float, float, float] = (0.85, 0.86, 0.88, 0.90),
    com_support_forward_offsets_m: tuple[float, float, float, float] = (-0.015, -0.014, -0.012, -0.010),
    root_velocity_ranges: tuple[float, float, float, float] = (0.008, 0.010, 0.014, 0.018),
    joint_reset_offsets: tuple[float, float, float, float] = (0.003, 0.004, 0.006, 0.008),
    mass_scale_ranges: tuple[tuple[float, float], ...] = (
        (0.997, 1.003),
        (0.995, 1.005),
        (0.990, 1.010),
        (0.985, 1.015),
    ),
) -> dict[str, torch.Tensor | float]:
    """5k v11a hybrid curriculum from the clean v10 model_4000 actor.

    Stage 0 (iterations 0-750 at 64 steps/env) preserves the exact v10 clock:
    ``soft_gate_floor=1.0``.  Progression rewards start at 20 percent so the
    bilateral v10 lift/alternation skill is not erased.  Later stages introduce
    bounded phase slowing, touchdown/takeover pressure, and modestly stronger
    command tracking.  The phase speed is never zero.
    """
    del env_ids
    step = int(env.common_step_counter)
    b0, b1, b2 = stage_step_boundaries
    stage = 0 if step < b0 else 1 if step < b1 else 2 if step < b2 else 3

    previous_stage = getattr(env, "_littlegreen_v11a_hybrid_stage", None)
    if previous_stage != stage:
        command_term = env.command_manager.get_term("base_velocity")
        command_term.cfg.rel_standing_envs = standing_fractions[stage]
        command_term.cfg.ranges.lin_vel_x = lin_vel_x_ranges[stage]
        command_term.cfg.ranges.lin_vel_y = lin_vel_y_ranges[stage]
        command_term.cfg.ranges.ang_vel_z = ang_vel_z_ranges[stage]

        reset_base_cfg = env.event_manager.get_term_cfg("reset_base")
        root_vel = root_velocity_ranges[stage]
        reset_base_cfg.params["velocity_range"] = {
            "x": (-root_vel, root_vel),
            "y": (-root_vel, root_vel),
            "z": (0.0, 0.0),
            "roll": (-root_vel, root_vel),
            "pitch": (-root_vel, root_vel),
            "yaw": (-root_vel, root_vel),
        }
        env.event_manager.set_term_cfg("reset_base", reset_base_cfg)

        reset_joint_cfg = env.event_manager.get_term_cfg("reset_robot_joints")
        joint_offset = joint_reset_offsets[stage]
        reset_joint_cfg.params["position_range"] = (-joint_offset, joint_offset)
        env.event_manager.set_term_cfg("reset_robot_joints", reset_joint_cfg)

        mass_cfg = env.event_manager.get_term_cfg("base_mass")
        mass_cfg.params["mass_distribution_params"] = mass_scale_ranges[stage]
        env.event_manager.set_term_cfg("base_mass", mass_cfg)

        env._littlegreen_v11a_gait_period_s = float(gait_period_s[stage])
        env._littlegreen_v11a_transition_fraction = float(transition_fractions[stage])
        env._littlegreen_v11a_soft_gate_floor = float(soft_gate_floors[stage])
        env._littlegreen_v11a_late_swing_start = float(late_swing_starts[stage])
        env._littlegreen_v11a_event_hold_s = 0.10
        env._littlegreen_v11a_target_clearance_m = float(target_clearance_m[stage])
        env._littlegreen_v11a_clearance_std_m = float(clearance_std_m[stage])
        env._littlegreen_v11a_clearance_gate_m = float(clearance_gate_m[stage])
        env._littlegreen_v11a_min_step_m = float(min_step_m[stage])
        env._littlegreen_v11a_max_step_m = float(max_step_m[stage])
        env._littlegreen_v11a_step_length_scale = float(step_length_scales[stage])
        env._littlegreen_v11a_progression_scale = float(progression_scales[stage])
        env._littlegreen_v11a_contact_scale = float(contact_scales[stage])
        env._littlegreen_v11a_tracking_scale = float(tracking_scales[stage])
        env._littlegreen_v11a_target_stance_force_ratio = float(
            target_stance_force_ratios[stage]
        )
        env._littlegreen_v11a_com_lateral_stance_fraction = float(
            com_lateral_stance_fractions[stage]
        )
        env._littlegreen_v11a_com_support_forward_offset_m = float(
            com_support_forward_offsets_m[stage]
        )

        # New command onset synchronizes the v10-compatible clock and clears only
        # v11a progression state; the actor weights remain untouched.
        command_term.time_left[:] = 0.0
        env._littlegreen_v11a_was_moving = torch.zeros(
            env.num_envs, device=command_term.time_left.device, dtype=torch.bool
        )
        env._littlegreen_v11a_hybrid_stage = stage

    action_term = env.action_manager.get_term("joint_pos")
    return {
        "stage": float(stage),
        "standing_fraction": float(standing_fractions[stage]),
        "cmd_x_min": float(lin_vel_x_ranges[stage][0]),
        "cmd_x_max": float(lin_vel_x_ranges[stage][1]),
        "cmd_y_abs_max": float(max(abs(v) for v in lin_vel_y_ranges[stage])),
        "cmd_yaw_abs_max": float(max(abs(v) for v in ang_vel_z_ranges[stage])),
        "gait_period_s": float(gait_period_s[stage]),
        "transition_fraction_per_half": float(transition_fractions[stage]),
        "double_support_fraction_total": float(2.0 * transition_fractions[stage]),
        "soft_gate_floor": float(soft_gate_floors[stage]),
        "phase_can_freeze": 0.0,
        "target_clearance_m": float(target_clearance_m[stage]),
        "clearance_gate_m": float(clearance_gate_m[stage]),
        "min_step_m": float(min_step_m[stage]),
        "max_step_m": float(max_step_m[stage]),
        "step_length_scale": float(step_length_scales[stage]),
        "progression_scale": float(progression_scales[stage]),
        "contact_scale": float(contact_scales[stage]),
        "tracking_scale": float(tracking_scales[stage]),
        "target_stance_force_ratio": float(target_stance_force_ratios[stage]),
        "moving_height_target_m": 0.457,
        "moving_height_width_m": 0.075,
        "root_reset_velocity_abs_max": float(root_velocity_ranges[stage]),
        "joint_reset_offset_abs_max": float(joint_reset_offsets[stage]),
        "push_xy_abs_max": 0.0,
        "mass_scale_min": float(mass_scale_ranges[stage][0]),
        "mass_scale_max": float(mass_scale_ranges[stage][1]),
        "actuator_delay_mean_ms": float(torch.mean(action_term.sampled_total_delay_s).item() * 1000.0),
        "actuator_tau_mean_ms": float(torch.mean(action_term.sampled_tau_s).item() * 1000.0),
        "actuator_velocity_scale_mean": float(torch.mean(action_term.sampled_velocity_scale).item()),
        "curriculum_profile_id": 211.1,
    }


def st3215_loaded_v11b_confirmed_takeover_curriculum(
    env,
    env_ids,
    stage_step_boundaries: tuple[int, int, int] = (48000, 128000, 224000),
    standing_fractions: tuple[float, float, float, float] = (0.22, 0.20, 0.18, 0.16),
    lin_vel_x_ranges: tuple[tuple[float, float], ...] = (
        (0.30, 0.42),
        (0.30, 0.48),
        (0.32, 0.55),
        (0.35, 0.62),
    ),
    lin_vel_y_ranges: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.0, 0.0),
        (-0.015, 0.015),
        (-0.04, 0.04),
    ),
    ang_vel_z_ranges: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.0, 0.0),
        (-0.04, 0.04),
        (-0.10, 0.10),
    ),
    gait_period_s: tuple[float, float, float, float] = (0.82, 0.82, 0.80, 0.78),
    transition_fractions: tuple[float, float, float, float] = (0.055, 0.050, 0.050, 0.045),
    target_clearance_m: tuple[float, float, float, float] = (0.026, 0.028, 0.029, 0.030),
    clearance_std_m: tuple[float, float, float, float] = (0.012, 0.012, 0.013, 0.013),
    clearance_gate_m: tuple[float, float, float, float] = (0.010, 0.011, 0.012, 0.012),
    clean_flight_s: tuple[float, float, float, float] = (0.040, 0.050, 0.060, 0.060),
    min_step_m: tuple[float, float, float, float] = (0.018, 0.020, 0.022, 0.024),
    max_step_m: tuple[float, float, float, float] = (0.034, 0.038, 0.042, 0.045),
    step_length_scales: tuple[float, float, float, float] = (0.105, 0.110, 0.120, 0.125),
    min_pass_m: tuple[float, float, float, float] = (0.010, 0.012, 0.014, 0.015),
    valid_step_fractions: tuple[float, float, float, float] = (0.45, 0.52, 0.58, 0.62),
    progression_scales: tuple[float, float, float, float] = (0.70, 0.85, 1.00, 1.10),
    contact_scales: tuple[float, float, float, float] = (0.78, 0.70, 0.60, 0.52),
    tracking_scales: tuple[float, float, float, float] = (0.85, 0.92, 1.00, 1.08),
    target_stance_force_ratios: tuple[float, float, float, float] = (0.78, 0.80, 0.82, 0.84),
    takeover_gate_hold_s: tuple[float, float, float, float] = (0.060, 0.080, 0.090, 0.100),
    takeover_max_s: tuple[float, float, float, float] = (0.120, 0.140, 0.150, 0.160),
    recovery_contact_hold_s: tuple[float, float, float, float] = (0.040, 0.050, 0.060, 0.060),
    recovery_max_s: tuple[float, float, float, float] = (0.180, 0.180, 0.200, 0.200),
    com_lateral_stance_fractions: tuple[float, float, float, float] = (0.88, 0.90, 0.92, 0.92),
    support_forward_offsets_m: tuple[float, float, float, float] = (0.002, 0.004, 0.006, 0.008),
    com_takeover_forward_offsets_m: tuple[float, float, float, float] = (0.005, 0.008, 0.010, 0.012),
    com_takeover_forward_tolerances_m: tuple[float, float, float, float] = (0.032, 0.029, 0.026, 0.024),
    com_takeover_lateral_tolerances_m: tuple[float, float, float, float] = (0.034, 0.032, 0.030, 0.028),
    root_velocity_ranges: tuple[float, float, float, float] = (0.006, 0.008, 0.012, 0.016),
    joint_reset_offsets: tuple[float, float, float, float] = (0.002, 0.003, 0.005, 0.007),
    mass_scale_ranges: tuple[tuple[float, float], ...] = (
        (0.998, 1.002),
        (0.996, 1.004),
        (0.992, 1.008),
        (0.988, 1.012),
    ),
) -> dict[str, torch.Tensor | float]:
    """5k v11b consolidation curriculum from the clean v11a model_2000 actor.

    The early stage keeps command pressure and gait timing near the v11a model's
    learned distribution while immediately enforcing confirmed-stance promotion.
    Later stages tighten clean-flight, forward-pass, COM/load takeover, and
    full-cycle progression requirements.  Failed touchdowns recover and retry
    the same trailing foot instead of promoting it or redirecting COM backward.
    """
    del env_ids
    step = int(env.common_step_counter)
    b0, b1, b2 = stage_step_boundaries
    stage = 0 if step < b0 else 1 if step < b1 else 2 if step < b2 else 3

    previous_stage = getattr(env, "_littlegreen_v11b_confirmed_takeover_stage", None)
    if previous_stage != stage:
        command_term = env.command_manager.get_term("base_velocity")
        command_term.cfg.rel_standing_envs = standing_fractions[stage]
        command_term.cfg.ranges.lin_vel_x = lin_vel_x_ranges[stage]
        command_term.cfg.ranges.lin_vel_y = lin_vel_y_ranges[stage]
        command_term.cfg.ranges.ang_vel_z = ang_vel_z_ranges[stage]

        reset_base_cfg = env.event_manager.get_term_cfg("reset_base")
        root_vel = root_velocity_ranges[stage]
        reset_base_cfg.params["velocity_range"] = {
            "x": (-root_vel, root_vel),
            "y": (-root_vel, root_vel),
            "z": (0.0, 0.0),
            "roll": (-root_vel, root_vel),
            "pitch": (-root_vel, root_vel),
            "yaw": (-root_vel, root_vel),
        }
        env.event_manager.set_term_cfg("reset_base", reset_base_cfg)

        reset_joint_cfg = env.event_manager.get_term_cfg("reset_robot_joints")
        joint_offset = joint_reset_offsets[stage]
        reset_joint_cfg.params["position_range"] = (-joint_offset, joint_offset)
        env.event_manager.set_term_cfg("reset_robot_joints", reset_joint_cfg)

        mass_cfg = env.event_manager.get_term_cfg("base_mass")
        mass_cfg.params["mass_distribution_params"] = mass_scale_ranges[stage]
        env.event_manager.set_term_cfg("base_mass", mass_cfg)

        env._littlegreen_v11b_gait_period_s = float(gait_period_s[stage])
        env._littlegreen_v11b_transition_fraction = float(transition_fractions[stage])
        env._littlegreen_v11b_target_clearance_m = float(target_clearance_m[stage])
        env._littlegreen_v11b_clearance_std_m = float(clearance_std_m[stage])
        env._littlegreen_v11b_clearance_gate_m = float(clearance_gate_m[stage])
        env._littlegreen_v11b_clean_flight_s = float(clean_flight_s[stage])
        env._littlegreen_v11b_min_step_m = float(min_step_m[stage])
        env._littlegreen_v11b_max_step_m = float(max_step_m[stage])
        env._littlegreen_v11b_step_length_scale = float(step_length_scales[stage])
        env._littlegreen_v11b_min_pass_m = float(min_pass_m[stage])
        env._littlegreen_v11b_valid_step_fraction = float(valid_step_fractions[stage])
        env._littlegreen_v11b_progression_scale = float(progression_scales[stage])
        env._littlegreen_v11b_contact_scale = float(contact_scales[stage])
        env._littlegreen_v11b_tracking_scale = float(tracking_scales[stage])
        env._littlegreen_v11b_target_stance_force_ratio = float(
            target_stance_force_ratios[stage]
        )
        env._littlegreen_v11b_takeover_gate_hold_s = float(takeover_gate_hold_s[stage])
        env._littlegreen_v11b_takeover_max_s = float(takeover_max_s[stage])
        env._littlegreen_v11b_recovery_contact_hold_s = float(
            recovery_contact_hold_s[stage]
        )
        env._littlegreen_v11b_recovery_max_s = float(recovery_max_s[stage])
        env._littlegreen_v11b_com_lateral_stance_fraction = float(
            com_lateral_stance_fractions[stage]
        )
        env._littlegreen_v11b_support_forward_offset_m = float(
            support_forward_offsets_m[stage]
        )
        env._littlegreen_v11b_com_takeover_forward_offset_m = float(
            com_takeover_forward_offsets_m[stage]
        )
        env._littlegreen_v11b_com_takeover_forward_tolerance_m = float(
            com_takeover_forward_tolerances_m[stage]
        )
        env._littlegreen_v11b_com_takeover_lateral_tolerance_m = float(
            com_takeover_lateral_tolerances_m[stage]
        )
        env._littlegreen_v11b_event_hold_s = 0.10

        command_term.time_left[:] = 0.0
        env._littlegreen_v11b_was_moving = torch.zeros(
            env.num_envs, device=command_term.time_left.device, dtype=torch.bool
        )
        env._littlegreen_v11b_confirmed_takeover_stage = stage

    action_term = env.action_manager.get_term("joint_pos")
    return {
        "stage": float(stage),
        "standing_fraction": float(standing_fractions[stage]),
        "cmd_x_min": float(lin_vel_x_ranges[stage][0]),
        "cmd_x_max": float(lin_vel_x_ranges[stage][1]),
        "cmd_y_abs_max": float(max(abs(v) for v in lin_vel_y_ranges[stage])),
        "cmd_yaw_abs_max": float(max(abs(v) for v in ang_vel_z_ranges[stage])),
        "gait_period_s": float(gait_period_s[stage]),
        "transition_fraction_per_half": float(transition_fractions[stage]),
        "clean_flight_s": float(clean_flight_s[stage]),
        "clearance_gate_m": float(clearance_gate_m[stage]),
        "min_step_m": float(min_step_m[stage]),
        "max_step_m": float(max_step_m[stage]),
        "min_pass_m": float(min_pass_m[stage]),
        "valid_step_fraction": float(valid_step_fractions[stage]),
        "progression_scale": float(progression_scales[stage]),
        "contact_scale": float(contact_scales[stage]),
        "tracking_scale": float(tracking_scales[stage]),
        "target_stance_force_ratio": float(target_stance_force_ratios[stage]),
        "takeover_hold_s": float(takeover_gate_hold_s[stage]),
        "takeover_max_s": float(takeover_max_s[stage]),
        "recovery_max_s": float(recovery_max_s[stage]),
        "com_takeover_forward_offset_m": float(com_takeover_forward_offsets_m[stage]),
        "com_takeover_forward_tolerance_m": float(com_takeover_forward_tolerances_m[stage]),
        "moving_height_target_m": 0.461,
        "moving_height_width_m": 0.070,
        "push_xy_abs_max": 0.0,
        "mass_scale_min": float(mass_scale_ranges[stage][0]),
        "mass_scale_max": float(mass_scale_ranges[stage][1]),
        "actuator_delay_mean_ms": float(torch.mean(action_term.sampled_total_delay_s).item() * 1000.0),
        "actuator_tau_mean_ms": float(torch.mean(action_term.sampled_tau_s).item() * 1000.0),
        "actuator_velocity_scale_mean": float(torch.mean(action_term.sampled_velocity_scale).item()),
        "curriculum_profile_id": 2112.0,
    }


def st3215_loaded_v11c_soft_takeover_curriculum(
    env,
    env_ids,
    stage_step_boundaries: tuple[int, int, int] = (48000, 128000, 224000),
    standing_fractions: tuple[float, float, float, float] = (0.22, 0.20, 0.18, 0.16),
    lin_vel_x_ranges: tuple[tuple[float, float], ...] = (
        (0.30, 0.42),
        (0.30, 0.48),
        (0.32, 0.55),
        (0.35, 0.62),
    ),
    lin_vel_y_ranges: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.0, 0.0),
        (-0.015, 0.015),
        (-0.04, 0.04),
    ),
    ang_vel_z_ranges: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.0, 0.0),
        (-0.04, 0.04),
        (-0.10, 0.10),
    ),
    gait_period_s: tuple[float, float, float, float] = (0.82, 0.82, 0.80, 0.78),
    transition_fractions: tuple[float, float, float, float] = (0.060, 0.055, 0.050, 0.050),
    soft_gate_floors: tuple[float, float, float, float] = (1.00, 0.90, 0.80, 0.72),
    late_swing_starts: tuple[float, float, float, float] = (0.82, 0.80, 0.78, 0.76),
    target_clearance_m: tuple[float, float, float, float] = (0.028, 0.028, 0.030, 0.030),
    clearance_std_m: tuple[float, float, float, float] = (0.012, 0.012, 0.013, 0.013),
    clearance_gate_m: tuple[float, float, float, float] = (0.010, 0.010, 0.011, 0.012),
    clean_flight_s: tuple[float, float, float, float] = (0.040, 0.050, 0.055, 0.060),
    min_step_m: tuple[float, float, float, float] = (0.018, 0.020, 0.022, 0.024),
    max_step_m: tuple[float, float, float, float] = (0.035, 0.040, 0.045, 0.050),
    step_length_scales: tuple[float, float, float, float] = (0.10, 0.11, 0.12, 0.13),
    progression_scales: tuple[float, float, float, float] = (0.25, 0.55, 0.85, 1.00),
    contact_scales: tuple[float, float, float, float] = (1.00, 0.90, 0.75, 0.65),
    tracking_scales: tuple[float, float, float, float] = (0.72, 0.82, 0.94, 1.05),
    target_pre_swing_force_ratios: tuple[float, float, float, float] = (0.74, 0.76, 0.78, 0.80),
    target_landed_force_ratios: tuple[float, float, float, float] = (0.56, 0.60, 0.64, 0.68),
    takeover_window_s: tuple[float, float, float, float] = (0.40, 0.38, 0.36, 0.34),
    com_lateral_stance_fractions: tuple[float, float, float, float] = (0.84, 0.86, 0.88, 0.90),
    com_support_forward_offsets_m: tuple[float, float, float, float] = (-0.008, -0.004, 0.000, 0.004),
    com_takeover_forward_offsets_m: tuple[float, float, float, float] = (0.004, 0.006, 0.009, 0.012),
    com_takeover_forward_tolerances_m: tuple[float, float, float, float] = (0.050, 0.045, 0.040, 0.035),
    com_takeover_lateral_tolerances_m: tuple[float, float, float, float] = (0.050, 0.045, 0.040, 0.035),
    root_velocity_ranges: tuple[float, float, float, float] = (0.008, 0.010, 0.014, 0.018),
    joint_reset_offsets: tuple[float, float, float, float] = (0.003, 0.004, 0.006, 0.008),
    mass_scale_ranges: tuple[tuple[float, float], ...] = (
        (0.997, 1.003),
        (0.995, 1.005),
        (0.990, 1.010),
        (0.985, 1.015),
    ),
) -> dict[str, torch.Tensor | float]:
    """5k soft-takeover curriculum from v11a model_2000.

    Stage 0 preserves the v11a/v10 bilateral rhythm and introduces only low
    progression pressure.  Later stages increase trailing-foot pass, landed-foot
    load/COM takeover, and full-cycle translation.  Phase speed remains strictly
    positive and no touchdown or takeover condition can block the next swing.
    """
    del env_ids
    step = int(env.common_step_counter)
    b0, b1, b2 = stage_step_boundaries
    stage = 0 if step < b0 else 1 if step < b1 else 2 if step < b2 else 3

    previous_stage = getattr(env, "_littlegreen_v11c_soft_takeover_stage", None)
    if previous_stage != stage:
        command_term = env.command_manager.get_term("base_velocity")
        command_term.cfg.rel_standing_envs = standing_fractions[stage]
        command_term.cfg.ranges.lin_vel_x = lin_vel_x_ranges[stage]
        command_term.cfg.ranges.lin_vel_y = lin_vel_y_ranges[stage]
        command_term.cfg.ranges.ang_vel_z = ang_vel_z_ranges[stage]

        reset_base_cfg = env.event_manager.get_term_cfg("reset_base")
        root_vel = root_velocity_ranges[stage]
        reset_base_cfg.params["velocity_range"] = {
            "x": (-root_vel, root_vel),
            "y": (-root_vel, root_vel),
            "z": (0.0, 0.0),
            "roll": (-root_vel, root_vel),
            "pitch": (-root_vel, root_vel),
            "yaw": (-root_vel, root_vel),
        }
        env.event_manager.set_term_cfg("reset_base", reset_base_cfg)

        reset_joint_cfg = env.event_manager.get_term_cfg("reset_robot_joints")
        joint_offset = joint_reset_offsets[stage]
        reset_joint_cfg.params["position_range"] = (-joint_offset, joint_offset)
        env.event_manager.set_term_cfg("reset_robot_joints", reset_joint_cfg)

        mass_cfg = env.event_manager.get_term_cfg("base_mass")
        mass_cfg.params["mass_distribution_params"] = mass_scale_ranges[stage]
        env.event_manager.set_term_cfg("base_mass", mass_cfg)

        # v11c intentionally reuses the proven v11a continuous phase generator.
        env._littlegreen_v11a_gait_period_s = float(gait_period_s[stage])
        env._littlegreen_v11a_transition_fraction = float(transition_fractions[stage])
        env._littlegreen_v11a_soft_gate_floor = float(soft_gate_floors[stage])
        env._littlegreen_v11a_late_swing_start = float(late_swing_starts[stage])
        env._littlegreen_v11a_event_hold_s = 0.10
        env._littlegreen_v11a_target_clearance_m = float(target_clearance_m[stage])
        env._littlegreen_v11a_clearance_std_m = float(clearance_std_m[stage])
        env._littlegreen_v11a_clearance_gate_m = float(clearance_gate_m[stage])
        env._littlegreen_v11a_min_step_m = float(min_step_m[stage])
        env._littlegreen_v11a_max_step_m = float(max_step_m[stage])
        env._littlegreen_v11a_step_length_scale = float(step_length_scales[stage])
        env._littlegreen_v11a_progression_scale = float(progression_scales[stage])
        env._littlegreen_v11a_contact_scale = float(contact_scales[stage])
        env._littlegreen_v11a_tracking_scale = float(tracking_scales[stage])
        env._littlegreen_v11a_target_stance_force_ratio = float(
            target_pre_swing_force_ratios[stage]
        )
        env._littlegreen_v11a_com_lateral_stance_fraction = float(
            com_lateral_stance_fractions[stage]
        )
        env._littlegreen_v11a_com_support_forward_offset_m = float(
            com_support_forward_offsets_m[stage]
        )

        env._littlegreen_v11c_clean_flight_s = float(clean_flight_s[stage])
        env._littlegreen_v11c_event_hold_s = 0.10
        env._littlegreen_v11c_takeover_window_s = float(takeover_window_s[stage])
        env._littlegreen_v11c_target_landed_force_ratio = float(
            target_landed_force_ratios[stage]
        )
        env._littlegreen_v11c_com_takeover_forward_offset_m = float(
            com_takeover_forward_offsets_m[stage]
        )
        env._littlegreen_v11c_com_takeover_forward_tolerance_m = float(
            com_takeover_forward_tolerances_m[stage]
        )
        env._littlegreen_v11c_com_takeover_lateral_tolerance_m = float(
            com_takeover_lateral_tolerances_m[stage]
        )
        env._littlegreen_v11c_clearance_gate_m = float(clearance_gate_m[stage])
        env._littlegreen_v11c_min_step_m = float(min_step_m[stage])
        env._littlegreen_v11c_max_step_m = float(max_step_m[stage])
        env._littlegreen_v11c_step_length_scale = float(step_length_scales[stage])
        env._littlegreen_v11c_progression_scale = float(progression_scales[stage])
        env._littlegreen_v11c_tracking_scale = float(tracking_scales[stage])

        command_term.time_left[:] = 0.0
        env._littlegreen_v11a_was_moving = torch.zeros(
            env.num_envs, device=command_term.time_left.device, dtype=torch.bool
        )
        env._littlegreen_v11c_was_moving = torch.zeros(
            env.num_envs, device=command_term.time_left.device, dtype=torch.bool
        )
        env._littlegreen_v11c_soft_takeover_stage = stage

    action_term = env.action_manager.get_term("joint_pos")
    return {
        "stage": float(stage),
        "standing_fraction": float(standing_fractions[stage]),
        "cmd_x_min": float(lin_vel_x_ranges[stage][0]),
        "cmd_x_max": float(lin_vel_x_ranges[stage][1]),
        "cmd_y_abs_max": float(max(abs(v) for v in lin_vel_y_ranges[stage])),
        "cmd_yaw_abs_max": float(max(abs(v) for v in ang_vel_z_ranges[stage])),
        "gait_period_s": float(gait_period_s[stage]),
        "transition_fraction_per_half": float(transition_fractions[stage]),
        "soft_gate_floor": float(soft_gate_floors[stage]),
        "phase_can_freeze": 0.0,
        "takeover_is_hard_gate": 0.0,
        "failed_step_blocks_alternation": 0.0,
        "target_clearance_m": float(target_clearance_m[stage]),
        "clean_flight_s": float(clean_flight_s[stage]),
        "min_step_m": float(min_step_m[stage]),
        "max_step_m": float(max_step_m[stage]),
        "step_length_scale": float(step_length_scales[stage]),
        "progression_scale": float(progression_scales[stage]),
        "contact_scale": float(contact_scales[stage]),
        "tracking_scale": float(tracking_scales[stage]),
        "target_pre_swing_force_ratio": float(target_pre_swing_force_ratios[stage]),
        "target_landed_force_ratio": float(target_landed_force_ratios[stage]),
        "takeover_window_s": float(takeover_window_s[stage]),
        "com_takeover_forward_offset_m": float(com_takeover_forward_offsets_m[stage]),
        "moving_height_target_m": 0.461,
        "moving_height_width_m": 0.070,
        "root_reset_velocity_abs_max": float(root_velocity_ranges[stage]),
        "joint_reset_offset_abs_max": float(joint_reset_offsets[stage]),
        "push_xy_abs_max": 0.0,
        "mass_scale_min": float(mass_scale_ranges[stage][0]),
        "mass_scale_max": float(mass_scale_ranges[stage][1]),
        "actuator_delay_mean_ms": float(
            torch.mean(action_term.sampled_total_delay_s).item() * 1000.0
        ),
        "actuator_tau_mean_ms": float(
            torch.mean(action_term.sampled_tau_s).item() * 1000.0
        ),
        "actuator_velocity_scale_mean": float(
            torch.mean(action_term.sampled_velocity_scale).item()
        ),
        "curriculum_profile_id": 211.3,
    }


def st3215_loaded_v11d_true_step_curriculum(
    env,
    env_ids,
    stage_step_boundaries: tuple[int, int, int] = (48000, 128000, 224000),
    standing_fractions: tuple[float, float, float, float] = (0.20, 0.18, 0.16, 0.14),
    lin_vel_x_ranges: tuple[tuple[float, float], ...] = (
        (0.30, 0.44),
        (0.30, 0.50),
        (0.32, 0.56),
        (0.35, 0.62),
    ),
    lin_vel_y_ranges: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.0, 0.0),
        (-0.012, 0.012),
        (-0.035, 0.035),
    ),
    ang_vel_z_ranges: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.0, 0.0),
        (-0.035, 0.035),
        (-0.09, 0.09),
    ),
    gait_period_s: tuple[float, float, float, float] = (0.82, 0.82, 0.80, 0.78),
    transition_fractions: tuple[float, float, float, float] = (0.055, 0.050, 0.048, 0.045),
    soft_gate_floors: tuple[float, float, float, float] = (1.00, 0.92, 0.84, 0.78),
    late_swing_starts: tuple[float, float, float, float] = (0.82, 0.80, 0.78, 0.76),
    target_clearance_m: tuple[float, float, float, float] = (0.028, 0.028, 0.030, 0.030),
    clearance_std_m: tuple[float, float, float, float] = (0.012, 0.012, 0.013, 0.013),
    clearance_gate_m: tuple[float, float, float, float] = (0.010, 0.011, 0.012, 0.013),
    clean_flight_s: tuple[float, float, float, float] = (0.040, 0.045, 0.050, 0.055),
    min_step_m: tuple[float, float, float, float] = (0.014, 0.016, 0.018, 0.020),
    max_step_m: tuple[float, float, float, float] = (0.032, 0.036, 0.040, 0.045),
    step_length_scales: tuple[float, float, float, float] = (0.085, 0.095, 0.105, 0.115),
    lead_margin_m: tuple[float, float, float, float] = (0.008, 0.010, 0.012, 0.014),
    valid_step_fractions: tuple[float, float, float, float] = (0.45, 0.50, 0.55, 0.60),
    stance_drift_limits_m: tuple[float, float, float, float] = (0.020, 0.018, 0.016, 0.015),
    takeover_window_s: tuple[float, float, float, float] = (0.42, 0.40, 0.38, 0.36),
    takeover_progress_ratios: tuple[float, float, float, float] = (0.35, 0.40, 0.45, 0.50),
    takeover_progress_min_m: tuple[float, float, float, float] = (0.006, 0.007, 0.008, 0.010),
    takeover_progress_max_m: tuple[float, float, float, float] = (0.016, 0.018, 0.020, 0.022),
    support_zone_forward_m: tuple[float, float, float, float] = (0.024, 0.021, 0.018, 0.015),
    old_foot_behind_m: tuple[float, float, float, float] = (0.000, 0.002, 0.004, 0.006),
    progression_scales: tuple[float, float, float, float] = (0.60, 0.82, 1.00, 1.15),
    contact_scales: tuple[float, float, float, float] = (0.85, 0.75, 0.65, 0.58),
    tracking_scales: tuple[float, float, float, float] = (0.80, 0.90, 1.00, 1.08),
    target_pre_swing_force_ratios: tuple[float, float, float, float] = (0.72, 0.74, 0.76, 0.78),
    com_lateral_stance_fractions: tuple[float, float, float, float] = (0.84, 0.86, 0.88, 0.90),
    com_support_forward_offsets_m: tuple[float, float, float, float] = (-0.004, 0.000, 0.004, 0.008),
    root_velocity_ranges: tuple[float, float, float, float] = (0.006, 0.008, 0.012, 0.016),
    joint_reset_offsets: tuple[float, float, float, float] = (0.003, 0.004, 0.006, 0.008),
    mass_scale_ranges: tuple[tuple[float, float], ...] = (
        (0.997, 1.003),
        (0.995, 1.005),
        (0.990, 1.010),
        (0.985, 1.015),
    ),
) -> dict[str, torch.Tensor | float]:
    """5k true-step curriculum with unchanged 47-D observations.

    Stage 0 retains the v11c rhythm and makes the physical true-step event
    attainable.  Later stages tighten lead reversal, stance anchoring, and base
    takeover while reducing clock-only contact pressure.  Failed steps never gate
    or terminate the alternating phase.
    """
    del env_ids
    step = int(env.common_step_counter)
    b0, b1, b2 = stage_step_boundaries
    stage = 0 if step < b0 else 1 if step < b1 else 2 if step < b2 else 3

    previous_stage = getattr(env, "_littlegreen_v11d_true_step_stage", None)
    if previous_stage != stage:
        command_term = env.command_manager.get_term("base_velocity")
        command_term.cfg.rel_standing_envs = standing_fractions[stage]
        command_term.cfg.ranges.lin_vel_x = lin_vel_x_ranges[stage]
        command_term.cfg.ranges.lin_vel_y = lin_vel_y_ranges[stage]
        command_term.cfg.ranges.ang_vel_z = ang_vel_z_ranges[stage]

        reset_base_cfg = env.event_manager.get_term_cfg("reset_base")
        root_vel = root_velocity_ranges[stage]
        reset_base_cfg.params["velocity_range"] = {
            "x": (-root_vel, root_vel),
            "y": (-root_vel, root_vel),
            "z": (0.0, 0.0),
            "roll": (-root_vel, root_vel),
            "pitch": (-root_vel, root_vel),
            "yaw": (-root_vel, root_vel),
        }
        env.event_manager.set_term_cfg("reset_base", reset_base_cfg)

        reset_joint_cfg = env.event_manager.get_term_cfg("reset_robot_joints")
        joint_offset = joint_reset_offsets[stage]
        reset_joint_cfg.params["position_range"] = (-joint_offset, joint_offset)
        env.event_manager.set_term_cfg("reset_robot_joints", reset_joint_cfg)

        mass_cfg = env.event_manager.get_term_cfg("base_mass")
        mass_cfg.params["mass_distribution_params"] = mass_scale_ranges[stage]
        env.event_manager.set_term_cfg("base_mass", mass_cfg)

        # Preserve the existing 47-D phase semantics exactly.
        env._littlegreen_v11a_gait_period_s = float(gait_period_s[stage])
        env._littlegreen_v11a_transition_fraction = float(transition_fractions[stage])
        env._littlegreen_v11a_soft_gate_floor = float(soft_gate_floors[stage])
        env._littlegreen_v11a_late_swing_start = float(late_swing_starts[stage])
        env._littlegreen_v11a_event_hold_s = 0.10
        env._littlegreen_v11a_target_clearance_m = float(target_clearance_m[stage])
        env._littlegreen_v11a_clearance_std_m = float(clearance_std_m[stage])
        env._littlegreen_v11a_clearance_gate_m = float(clearance_gate_m[stage])
        env._littlegreen_v11a_min_step_m = float(min_step_m[stage])
        env._littlegreen_v11a_max_step_m = float(max_step_m[stage])
        env._littlegreen_v11a_step_length_scale = float(step_length_scales[stage])
        env._littlegreen_v11a_progression_scale = float(progression_scales[stage])
        env._littlegreen_v11a_contact_scale = float(contact_scales[stage])
        env._littlegreen_v11a_tracking_scale = float(tracking_scales[stage])
        env._littlegreen_v11a_target_stance_force_ratio = float(
            target_pre_swing_force_ratios[stage]
        )
        env._littlegreen_v11a_com_lateral_stance_fraction = float(
            com_lateral_stance_fractions[stage]
        )
        env._littlegreen_v11a_com_support_forward_offset_m = float(
            com_support_forward_offsets_m[stage]
        )

        # Keep v11c's phase observation/state usable by the shared phase term.
        env._littlegreen_v11c_clean_flight_s = float(clean_flight_s[stage])
        env._littlegreen_v11c_event_hold_s = 0.10
        env._littlegreen_v11c_takeover_window_s = 0.36
        env._littlegreen_v11c_target_landed_force_ratio = 0.62
        env._littlegreen_v11c_com_takeover_forward_offset_m = float(
            com_support_forward_offsets_m[stage]
        )
        env._littlegreen_v11c_com_takeover_forward_tolerance_m = 0.045
        env._littlegreen_v11c_com_takeover_lateral_tolerance_m = 0.045
        env._littlegreen_v11c_clearance_gate_m = float(clearance_gate_m[stage])
        env._littlegreen_v11c_min_step_m = float(min_step_m[stage])
        env._littlegreen_v11c_max_step_m = float(max_step_m[stage])
        env._littlegreen_v11c_step_length_scale = float(step_length_scales[stage])
        env._littlegreen_v11c_progression_scale = 0.0
        env._littlegreen_v11c_tracking_scale = float(tracking_scales[stage])

        env._littlegreen_v11d_clean_flight_s = float(clean_flight_s[stage])
        env._littlegreen_v11d_clearance_gate_m = float(clearance_gate_m[stage])
        env._littlegreen_v11d_lead_margin_m = float(lead_margin_m[stage])
        env._littlegreen_v11d_valid_step_fraction = float(valid_step_fractions[stage])
        env._littlegreen_v11d_stance_drift_limit_m = float(
            stance_drift_limits_m[stage]
        )
        env._littlegreen_v11d_takeover_window_s = float(takeover_window_s[stage])
        env._littlegreen_v11d_takeover_progress_ratio = float(
            takeover_progress_ratios[stage]
        )
        env._littlegreen_v11d_takeover_progress_min_m = float(
            takeover_progress_min_m[stage]
        )
        env._littlegreen_v11d_takeover_progress_max_m = float(
            takeover_progress_max_m[stage]
        )
        env._littlegreen_v11d_support_zone_forward_m = float(
            support_zone_forward_m[stage]
        )
        env._littlegreen_v11d_old_foot_behind_m = float(old_foot_behind_m[stage])
        env._littlegreen_v11d_event_hold_s = 0.12
        env._littlegreen_v11d_progression_scale = float(progression_scales[stage])

        command_term.time_left[:] = 0.0
        env._littlegreen_v11a_was_moving = torch.zeros(
            env.num_envs, device=command_term.time_left.device, dtype=torch.bool
        )
        env._littlegreen_v11c_was_moving = torch.zeros(
            env.num_envs, device=command_term.time_left.device, dtype=torch.bool
        )
        env._littlegreen_v11d_was_moving = torch.zeros(
            env.num_envs, device=command_term.time_left.device, dtype=torch.bool
        )
        env._littlegreen_v11d_true_step_stage = stage

    action_term = env.action_manager.get_term("joint_pos")
    return {
        "stage": float(stage),
        "standing_fraction": float(standing_fractions[stage]),
        "cmd_x_min": float(lin_vel_x_ranges[stage][0]),
        "cmd_x_max": float(lin_vel_x_ranges[stage][1]),
        "cmd_y_abs_max": float(max(abs(v) for v in lin_vel_y_ranges[stage])),
        "cmd_yaw_abs_max": float(max(abs(v) for v in ang_vel_z_ranges[stage])),
        "gait_period_s": float(gait_period_s[stage]),
        "transition_fraction_per_half": float(transition_fractions[stage]),
        "observation_count": 47.0,
        "observation_contract_changed": 0.0,
        "true_step_is_phase_gate": 0.0,
        "failed_step_blocks_alternation": 0.0,
        "lead_margin_m": float(lead_margin_m[stage]),
        "valid_step_fraction": float(valid_step_fractions[stage]),
        "stance_drift_limit_m": float(stance_drift_limits_m[stage]),
        "takeover_window_s": float(takeover_window_s[stage]),
        "takeover_progress_target_min_m": float(takeover_progress_min_m[stage]),
        "takeover_progress_target_max_m": float(takeover_progress_max_m[stage]),
        "support_zone_forward_m": float(support_zone_forward_m[stage]),
        "old_foot_behind_m": float(old_foot_behind_m[stage]),
        "min_step_m": float(min_step_m[stage]),
        "max_step_m": float(max_step_m[stage]),
        "progression_scale": float(progression_scales[stage]),
        "contact_scale": float(contact_scales[stage]),
        "tracking_scale": float(tracking_scales[stage]),
        "moving_height_target_m": 0.461,
        "moving_height_width_m": 0.070,
        "push_xy_abs_max": 0.0,
        "mass_scale_min": float(mass_scale_ranges[stage][0]),
        "mass_scale_max": float(mass_scale_ranges[stage][1]),
        "actuator_delay_mean_ms": float(
            torch.mean(action_term.sampled_total_delay_s).item() * 1000.0
        ),
        "actuator_tau_mean_ms": float(
            torch.mean(action_term.sampled_tau_s).item() * 1000.0
        ),
        "actuator_velocity_scale_mean": float(
            torch.mean(action_term.sampled_velocity_scale).item()
        ),
        "curriculum_profile_id": 211.4,
    }


def st3215_loaded_v11e_swing_freedom_curriculum(
    env,
    env_ids,
    stage_step_boundaries: tuple[int, int, int] = (48000, 128000, 224000),
    standing_fractions: tuple[float, float, float, float] = (0.18, 0.16, 0.14, 0.12),
    lin_vel_x_ranges: tuple[tuple[float, float], ...] = (
        (0.28, 0.44),
        (0.30, 0.50),
        (0.32, 0.56),
        (0.35, 0.62),
    ),
    lin_vel_y_ranges: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.0, 0.0),
        (-0.015, 0.015),
        (-0.040, 0.040),
    ),
    ang_vel_z_ranges: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.0, 0.0),
        (-0.040, 0.040),
        (-0.10, 0.10),
    ),
    gait_period_s: tuple[float, float, float, float] = (0.84, 0.82, 0.80, 0.78),
    transition_fractions: tuple[float, float, float, float] = (0.055, 0.052, 0.048, 0.045),
    soft_gate_floors: tuple[float, float, float, float] = (1.00, 0.94, 0.88, 0.82),
    late_swing_starts: tuple[float, float, float, float] = (0.84, 0.82, 0.80, 0.78),
    target_clearance_m: tuple[float, float, float, float] = (0.018, 0.022, 0.026, 0.030),
    clearance_std_m: tuple[float, float, float, float] = (0.010, 0.011, 0.012, 0.013),
    clearance_gate_start_m: tuple[float, float, float, float] = (0.004, 0.004, 0.005, 0.006),
    clearance_gate_full_m: tuple[float, float, float, float] = (0.010, 0.011, 0.012, 0.013),
    placement_start_progress: tuple[float, float, float, float] = (0.10, 0.12, 0.15, 0.18),
    clean_flight_s: tuple[float, float, float, float] = (0.035, 0.040, 0.045, 0.050),
    min_step_m: tuple[float, float, float, float] = (0.012, 0.014, 0.016, 0.018),
    max_step_m: tuple[float, float, float, float] = (0.030, 0.034, 0.040, 0.045),
    step_length_scales: tuple[float, float, float, float] = (0.080, 0.090, 0.105, 0.115),
    lead_margin_m: tuple[float, float, float, float] = (0.006, 0.008, 0.010, 0.012),
    valid_step_fractions: tuple[float, float, float, float] = (0.35, 0.42, 0.50, 0.58),
    stance_drift_limits_m: tuple[float, float, float, float] = (0.045, 0.040, 0.035, 0.030),
    stance_drift_deadzones_m: tuple[float, float, float, float] = (0.025, 0.023, 0.021, 0.019),
    stance_slide_deadzones_m_s: tuple[float, float, float, float] = (0.045, 0.040, 0.035, 0.030),
    takeover_window_s: tuple[float, float, float, float] = (0.44, 0.42, 0.40, 0.38),
    takeover_progress_ratios: tuple[float, float, float, float] = (0.25, 0.32, 0.40, 0.48),
    takeover_progress_min_m: tuple[float, float, float, float] = (0.004, 0.005, 0.007, 0.009),
    takeover_progress_max_m: tuple[float, float, float, float] = (0.014, 0.016, 0.019, 0.022),
    support_zone_forward_m: tuple[float, float, float, float] = (0.028, 0.025, 0.021, 0.018),
    old_foot_behind_m: tuple[float, float, float, float] = (0.000, 0.000, 0.002, 0.004),
    swing_progress_scales: tuple[float, float, float, float] = (1.00, 1.00, 0.90, 0.82),
    true_step_scales: tuple[float, float, float, float] = (0.20, 0.45, 0.80, 1.00),
    failure_penalty_scales: tuple[float, float, float, float] = (0.00, 0.15, 0.45, 0.75),
    stance_drift_scales: tuple[float, float, float, float] = (0.00, 0.15, 0.40, 0.65),
    stance_slide_scales: tuple[float, float, float, float] = (0.10, 0.18, 0.28, 0.40),
    airborne_persistence_scales: tuple[float, float, float, float] = (0.10, 0.20, 0.35, 0.50),
    contact_scales: tuple[float, float, float, float] = (1.00, 0.92, 0.82, 0.72),
    tracking_scales: tuple[float, float, float, float] = (0.72, 0.82, 0.95, 1.08),
    target_pre_swing_force_ratios: tuple[float, float, float, float] = (0.68, 0.70, 0.72, 0.74),
    com_lateral_stance_fractions: tuple[float, float, float, float] = (0.82, 0.84, 0.86, 0.88),
    com_support_forward_offsets_m: tuple[float, float, float, float] = (0.000, 0.004, 0.008, 0.012),
    root_velocity_ranges: tuple[float, float, float, float] = (0.005, 0.007, 0.010, 0.014),
    joint_reset_offsets: tuple[float, float, float, float] = (0.003, 0.004, 0.005, 0.007),
    mass_scale_ranges: tuple[tuple[float, float], ...] = (
        (0.998, 1.002),
        (0.996, 1.004),
        (0.992, 1.008),
        (0.988, 1.012),
    ),
) -> dict[str, torch.Tensor | float]:
    """5k v11e curriculum: restore lift freedom, then reward physical steps.

    The 47-D policy observation and continuous v11c phase remain unchanged.
    Dense command-direction swing progress is available from the first stage.
    True-step and true-stride events are bonuses, while failure penalties ramp
    only after bilateral lift and forward swing have had time to re-emerge.
    """
    del env_ids
    step = int(env.common_step_counter)
    b0, b1, b2 = stage_step_boundaries
    stage = 0 if step < b0 else 1 if step < b1 else 2 if step < b2 else 3

    previous_stage = getattr(env, "_littlegreen_v11e_swing_freedom_stage", None)
    if previous_stage != stage:
        command_term = env.command_manager.get_term("base_velocity")
        command_term.cfg.rel_standing_envs = standing_fractions[stage]
        command_term.cfg.ranges.lin_vel_x = lin_vel_x_ranges[stage]
        command_term.cfg.ranges.lin_vel_y = lin_vel_y_ranges[stage]
        command_term.cfg.ranges.ang_vel_z = ang_vel_z_ranges[stage]

        reset_base_cfg = env.event_manager.get_term_cfg("reset_base")
        root_vel = root_velocity_ranges[stage]
        reset_base_cfg.params["velocity_range"] = {
            "x": (-root_vel, root_vel),
            "y": (-root_vel, root_vel),
            "z": (0.0, 0.0),
            "roll": (-root_vel, root_vel),
            "pitch": (-root_vel, root_vel),
            "yaw": (-root_vel, root_vel),
        }
        env.event_manager.set_term_cfg("reset_base", reset_base_cfg)

        reset_joint_cfg = env.event_manager.get_term_cfg("reset_robot_joints")
        joint_offset = joint_reset_offsets[stage]
        reset_joint_cfg.params["position_range"] = (-joint_offset, joint_offset)
        env.event_manager.set_term_cfg("reset_robot_joints", reset_joint_cfg)

        mass_cfg = env.event_manager.get_term_cfg("base_mass")
        mass_cfg.params["mass_distribution_params"] = mass_scale_ranges[stage]
        env.event_manager.set_term_cfg("base_mass", mass_cfg)

        env._littlegreen_v11a_gait_period_s = float(gait_period_s[stage])
        env._littlegreen_v11a_transition_fraction = float(transition_fractions[stage])
        env._littlegreen_v11a_soft_gate_floor = float(soft_gate_floors[stage])
        env._littlegreen_v11a_late_swing_start = float(late_swing_starts[stage])
        env._littlegreen_v11a_event_hold_s = 0.10
        env._littlegreen_v11a_target_clearance_m = float(target_clearance_m[stage])
        env._littlegreen_v11a_clearance_std_m = float(clearance_std_m[stage])
        env._littlegreen_v11a_clearance_gate_m = float(clearance_gate_start_m[stage])
        env._littlegreen_v11a_min_step_m = float(min_step_m[stage])
        env._littlegreen_v11a_max_step_m = float(max_step_m[stage])
        env._littlegreen_v11a_step_length_scale = float(step_length_scales[stage])
        env._littlegreen_v11a_progression_scale = 1.0
        env._littlegreen_v11a_contact_scale = float(contact_scales[stage])
        env._littlegreen_v11a_tracking_scale = float(tracking_scales[stage])
        env._littlegreen_v11a_target_stance_force_ratio = float(
            target_pre_swing_force_ratios[stage]
        )
        env._littlegreen_v11a_com_lateral_stance_fraction = float(
            com_lateral_stance_fractions[stage]
        )
        env._littlegreen_v11a_com_support_forward_offset_m = float(
            com_support_forward_offsets_m[stage]
        )

        env._littlegreen_v11c_clean_flight_s = float(clean_flight_s[stage])
        env._littlegreen_v11c_event_hold_s = 0.10
        env._littlegreen_v11c_takeover_window_s = float(takeover_window_s[stage])
        env._littlegreen_v11c_target_landed_force_ratio = 0.58 + 0.03 * stage
        env._littlegreen_v11c_com_takeover_forward_offset_m = float(
            com_support_forward_offsets_m[stage]
        )
        env._littlegreen_v11c_com_takeover_forward_tolerance_m = 0.050
        env._littlegreen_v11c_com_takeover_lateral_tolerance_m = 0.050
        env._littlegreen_v11c_clearance_gate_m = float(clearance_gate_start_m[stage])
        env._littlegreen_v11c_min_step_m = float(min_step_m[stage])
        env._littlegreen_v11c_max_step_m = float(max_step_m[stage])
        env._littlegreen_v11c_step_length_scale = float(step_length_scales[stage])
        env._littlegreen_v11c_progression_scale = 0.0
        env._littlegreen_v11c_tracking_scale = float(tracking_scales[stage])

        env._littlegreen_v11d_clean_flight_s = float(clean_flight_s[stage])
        env._littlegreen_v11d_clearance_gate_m = float(clearance_gate_full_m[stage])
        env._littlegreen_v11d_lead_margin_m = float(lead_margin_m[stage])
        env._littlegreen_v11d_valid_step_fraction = float(valid_step_fractions[stage])
        env._littlegreen_v11d_stance_drift_limit_m = float(
            stance_drift_limits_m[stage]
        )
        env._littlegreen_v11d_takeover_window_s = float(takeover_window_s[stage])
        env._littlegreen_v11d_takeover_progress_ratio = float(
            takeover_progress_ratios[stage]
        )
        env._littlegreen_v11d_takeover_progress_min_m = float(
            takeover_progress_min_m[stage]
        )
        env._littlegreen_v11d_takeover_progress_max_m = float(
            takeover_progress_max_m[stage]
        )
        env._littlegreen_v11d_support_zone_forward_m = float(
            support_zone_forward_m[stage]
        )
        env._littlegreen_v11d_old_foot_behind_m = float(old_foot_behind_m[stage])
        env._littlegreen_v11d_event_hold_s = 0.12
        env._littlegreen_v11d_progression_scale = float(true_step_scales[stage])

        env._littlegreen_v11e_clearance_gate_start_m = float(
            clearance_gate_start_m[stage]
        )
        env._littlegreen_v11e_clearance_gate_full_m = float(
            clearance_gate_full_m[stage]
        )
        env._littlegreen_v11e_placement_start_progress = float(
            placement_start_progress[stage]
        )
        env._littlegreen_v11e_swing_progress_scale = float(
            swing_progress_scales[stage]
        )
        env._littlegreen_v11e_failure_penalty_scale = float(
            failure_penalty_scales[stage]
        )
        env._littlegreen_v11e_stance_drift_deadzone_m = float(
            stance_drift_deadzones_m[stage]
        )
        env._littlegreen_v11e_stance_drift_scale = float(
            stance_drift_scales[stage]
        )
        env._littlegreen_v11e_stance_slide_deadzone_m_s = float(
            stance_slide_deadzones_m_s[stage]
        )
        env._littlegreen_v11e_stance_slide_scale = float(
            stance_slide_scales[stage]
        )
        env._littlegreen_v11e_airborne_grace_s = 0.56
        env._littlegreen_v11e_airborne_ramp_s = 0.24
        env._littlegreen_v11e_airborne_persistence_scale = float(
            airborne_persistence_scales[stage]
        )

        command_term.time_left[:] = 0.0
        env._littlegreen_v11a_was_moving = torch.zeros(
            env.num_envs, device=command_term.time_left.device, dtype=torch.bool
        )
        env._littlegreen_v11c_was_moving = torch.zeros(
            env.num_envs, device=command_term.time_left.device, dtype=torch.bool
        )
        env._littlegreen_v11d_was_moving = torch.zeros(
            env.num_envs, device=command_term.time_left.device, dtype=torch.bool
        )
        env._littlegreen_v11e_prev_anchored_advance_m = torch.zeros(
            env.num_envs, device=command_term.time_left.device
        )
        env._littlegreen_v11e_prev_left_stance = torch.zeros(
            env.num_envs, device=command_term.time_left.device, dtype=torch.bool
        )
        env._littlegreen_v11e_airborne_side = torch.zeros(
            env.num_envs, device=command_term.time_left.device, dtype=torch.long
        )
        env._littlegreen_v11e_airborne_persistence_s = torch.zeros(
            env.num_envs, device=command_term.time_left.device
        )
        env._littlegreen_v11e_swing_freedom_stage = stage

    action_term = env.action_manager.get_term("joint_pos")
    return {
        "stage": float(stage),
        "standing_fraction": float(standing_fractions[stage]),
        "cmd_x_min": float(lin_vel_x_ranges[stage][0]),
        "cmd_x_max": float(lin_vel_x_ranges[stage][1]),
        "cmd_y_abs_max": float(max(abs(v) for v in lin_vel_y_ranges[stage])),
        "cmd_yaw_abs_max": float(max(abs(v) for v in ang_vel_z_ranges[stage])),
        "gait_period_s": float(gait_period_s[stage]),
        "transition_fraction_per_half": float(transition_fractions[stage]),
        "observation_count": 47.0,
        "observation_contract_changed": 0.0,
        "dense_swing_progress_enabled": 1.0,
        "dense_swing_progress_uses_command_direction": 1.0,
        "true_step_is_bonus_not_gate": 1.0,
        "failed_step_blocks_alternation": 0.0,
        "target_clearance_m": float(target_clearance_m[stage]),
        "clearance_gate_start_m": float(clearance_gate_start_m[stage]),
        "clearance_gate_full_m": float(clearance_gate_full_m[stage]),
        "placement_start_progress": float(placement_start_progress[stage]),
        "swing_progress_scale": float(swing_progress_scales[stage]),
        "true_step_scale": float(true_step_scales[stage]),
        "failure_penalty_scale": float(failure_penalty_scales[stage]),
        "stance_drift_deadzone_m": float(stance_drift_deadzones_m[stage]),
        "stance_drift_limit_m": float(stance_drift_limits_m[stage]),
        "stance_slide_deadzone_m_s": float(stance_slide_deadzones_m_s[stage]),
        "airborne_persistence_scale": float(airborne_persistence_scales[stage]),
        "moving_height_target_m": 0.461,
        "moving_height_width_m": 0.070,
        "push_xy_abs_max": 0.0,
        "mass_scale_min": float(mass_scale_ranges[stage][0]),
        "mass_scale_max": float(mass_scale_ranges[stage][1]),
        "actuator_delay_mean_ms": float(
            torch.mean(action_term.sampled_total_delay_s).item() * 1000.0
        ),
        "actuator_tau_mean_ms": float(
            torch.mean(action_term.sampled_tau_s).item() * 1000.0
        ),
        "actuator_velocity_scale_mean": float(
            torch.mean(action_term.sampled_velocity_scale).item()
        ),
        "curriculum_profile_id": 211.5,
    }


def st3215_loaded_v11f_stride_chain_curriculum(
    env,
    env_ids,
    stage_step_boundaries: tuple[int, int, int] = (48000, 128000, 224000),
    standing_fractions: tuple[float, float, float, float] = (0.16, 0.14, 0.12, 0.10),
    lin_vel_x_ranges: tuple[tuple[float, float], ...] = (
        (0.30, 0.44),
        (0.30, 0.50),
        (0.32, 0.56),
        (0.35, 0.62),
    ),
    lin_vel_y_ranges: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.0, 0.0),
        (-0.012, 0.012),
        (-0.035, 0.035),
    ),
    ang_vel_z_ranges: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.0, 0.0),
        (-0.035, 0.035),
        (-0.09, 0.09),
    ),
    gait_period_s: tuple[float, float, float, float] = (0.84, 0.83, 0.81, 0.80),
    transition_fractions: tuple[float, float, float, float] = (0.050, 0.048, 0.045, 0.042),
    target_clearance_m: tuple[float, float, float, float] = (0.020, 0.024, 0.028, 0.030),
    clearance_std_m: tuple[float, float, float, float] = (0.011, 0.012, 0.013, 0.014),
    clearance_gate_start_m: tuple[float, float, float, float] = (0.006, 0.007, 0.008, 0.009),
    clearance_gate_full_m: tuple[float, float, float, float] = (0.012, 0.013, 0.014, 0.015),
    clean_flight_s: tuple[float, float, float, float] = (0.035, 0.040, 0.045, 0.050),
    min_step_m: tuple[float, float, float, float] = (0.012, 0.014, 0.016, 0.018),
    max_step_m: tuple[float, float, float, float] = (0.030, 0.035, 0.041, 0.046),
    step_length_scales: tuple[float, float, float, float] = (0.080, 0.092, 0.105, 0.118),
    lead_margin_m: tuple[float, float, float, float] = (0.006, 0.008, 0.010, 0.012),
    valid_step_fractions: tuple[float, float, float, float] = (0.30, 0.38, 0.46, 0.54),
    stance_drift_limits_m: tuple[float, float, float, float] = (0.045, 0.040, 0.035, 0.030),
    stance_drift_deadzones_m: tuple[float, float, float, float] = (0.027, 0.024, 0.021, 0.019),
    stance_slide_deadzones_m_s: tuple[float, float, float, float] = (0.050, 0.045, 0.038, 0.032),
    takeover_window_s: tuple[float, float, float, float] = (0.46, 0.44, 0.42, 0.40),
    takeover_progress_ratios: tuple[float, float, float, float] = (0.20, 0.28, 0.36, 0.45),
    takeover_progress_min_m: tuple[float, float, float, float] = (0.004, 0.005, 0.006, 0.008),
    takeover_progress_max_m: tuple[float, float, float, float] = (0.012, 0.015, 0.018, 0.022),
    support_zone_forward_m: tuple[float, float, float, float] = (0.030, 0.027, 0.023, 0.020),
    old_foot_behind_m: tuple[float, float, float, float] = (0.000, 0.000, 0.002, 0.004),
    first_step_scales: tuple[float, float, float, float] = (0.50, 0.30, 0.15, 0.08),
    trailing_pass_scales: tuple[float, float, float, float] = (0.90, 1.00, 1.10, 1.00),
    lead_reversal_scales: tuple[float, float, float, float] = (0.45, 0.75, 1.00, 1.15),
    takeover_scales: tuple[float, float, float, float] = (0.35, 0.65, 0.95, 1.10),
    stride_scales: tuple[float, float, float, float] = (0.35, 0.65, 1.00, 1.20),
    leading_repeat_scales: tuple[float, float, float, float] = (0.15, 0.45, 0.75, 1.00),
    drag_scales: tuple[float, float, float, float] = (0.10, 0.35, 0.65, 0.90),
    stall_scales: tuple[float, float, float, float] = (0.00, 0.15, 0.40, 0.65),
    fullcycle_scales: tuple[float, float, float, float] = (0.20, 0.45, 0.75, 1.00),
    failure_penalty_scales: tuple[float, float, float, float] = (0.00, 0.10, 0.25, 0.40),
    stance_drift_scales: tuple[float, float, float, float] = (0.00, 0.10, 0.25, 0.40),
    stance_slide_scales: tuple[float, float, float, float] = (0.08, 0.12, 0.20, 0.30),
    contact_scales: tuple[float, float, float, float] = (0.75, 0.60, 0.45, 0.35),
    tracking_scales: tuple[float, float, float, float] = (0.75, 0.90, 1.05, 1.15),
    target_pre_swing_force_ratios: tuple[float, float, float, float] = (0.68, 0.70, 0.72, 0.74),
    com_lateral_stance_fractions: tuple[float, float, float, float] = (0.82, 0.84, 0.86, 0.88),
    com_support_forward_offsets_m: tuple[float, float, float, float] = (0.004, 0.008, 0.012, 0.016),
    root_velocity_ranges: tuple[float, float, float, float] = (0.005, 0.007, 0.010, 0.014),
    joint_reset_offsets: tuple[float, float, float, float] = (0.003, 0.004, 0.005, 0.007),
    mass_scale_ranges: tuple[tuple[float, float], ...] = (
        (0.998, 1.002),
        (0.996, 1.004),
        (0.992, 1.008),
        (0.988, 1.012),
    ),
) -> dict[str, torch.Tensor | float]:
    """5k v11f curriculum: first-step preservation to rolling true strides.

    Stage 0 introduces the stride chain gently.  Later stages reduce the value of
    another first step and increase reward for the opposite rear-foot pass, physical
    lead reversal, base takeover, and rolling full-stride completion.
    """
    del env_ids
    step = int(env.common_step_counter)
    b0, b1, b2 = stage_step_boundaries
    stage = 0 if step < b0 else 1 if step < b1 else 2 if step < b2 else 3

    previous_stage = getattr(env, "_littlegreen_v11f_stride_chain_stage", None)
    if previous_stage != stage:
        command_term = env.command_manager.get_term("base_velocity")
        command_term.cfg.rel_standing_envs = standing_fractions[stage]
        command_term.cfg.ranges.lin_vel_x = lin_vel_x_ranges[stage]
        command_term.cfg.ranges.lin_vel_y = lin_vel_y_ranges[stage]
        command_term.cfg.ranges.ang_vel_z = ang_vel_z_ranges[stage]

        reset_base_cfg = env.event_manager.get_term_cfg("reset_base")
        root_vel = root_velocity_ranges[stage]
        reset_base_cfg.params["velocity_range"] = {
            "x": (-root_vel, root_vel),
            "y": (-root_vel, root_vel),
            "z": (0.0, 0.0),
            "roll": (-root_vel, root_vel),
            "pitch": (-root_vel, root_vel),
            "yaw": (-root_vel, root_vel),
        }
        env.event_manager.set_term_cfg("reset_base", reset_base_cfg)

        reset_joint_cfg = env.event_manager.get_term_cfg("reset_robot_joints")
        joint_offset = joint_reset_offsets[stage]
        reset_joint_cfg.params["position_range"] = (-joint_offset, joint_offset)
        env.event_manager.set_term_cfg("reset_robot_joints", reset_joint_cfg)

        mass_cfg = env.event_manager.get_term_cfg("base_mass")
        mass_cfg.params["mass_distribution_params"] = mass_scale_ranges[stage]
        env.event_manager.set_term_cfg("base_mass", mass_cfg)

        env._littlegreen_v11a_gait_period_s = float(gait_period_s[stage])
        env._littlegreen_v11a_transition_fraction = float(transition_fractions[stage])
        env._littlegreen_v11a_soft_gate_floor = 1.0
        env._littlegreen_v11a_late_swing_start = 0.82
        env._littlegreen_v11a_event_hold_s = 0.10
        env._littlegreen_v11a_target_clearance_m = float(target_clearance_m[stage])
        env._littlegreen_v11a_clearance_std_m = float(clearance_std_m[stage])
        env._littlegreen_v11a_clearance_gate_m = float(clearance_gate_start_m[stage])
        env._littlegreen_v11a_min_step_m = float(min_step_m[stage])
        env._littlegreen_v11a_max_step_m = float(max_step_m[stage])
        env._littlegreen_v11a_step_length_scale = float(step_length_scales[stage])
        env._littlegreen_v11a_progression_scale = 1.0
        env._littlegreen_v11a_contact_scale = float(contact_scales[stage])
        env._littlegreen_v11a_tracking_scale = float(tracking_scales[stage])
        env._littlegreen_v11a_target_stance_force_ratio = float(
            target_pre_swing_force_ratios[stage]
        )
        env._littlegreen_v11a_com_lateral_stance_fraction = float(
            com_lateral_stance_fractions[stage]
        )
        env._littlegreen_v11a_com_support_forward_offset_m = float(
            com_support_forward_offsets_m[stage]
        )

        env._littlegreen_v11c_clean_flight_s = float(clean_flight_s[stage])
        env._littlegreen_v11c_event_hold_s = 0.10
        env._littlegreen_v11c_takeover_window_s = float(takeover_window_s[stage])
        env._littlegreen_v11c_target_landed_force_ratio = 0.56 + 0.03 * stage
        env._littlegreen_v11c_com_takeover_forward_offset_m = float(
            com_support_forward_offsets_m[stage]
        )
        env._littlegreen_v11c_com_takeover_forward_tolerance_m = 0.052
        env._littlegreen_v11c_com_takeover_lateral_tolerance_m = 0.052
        env._littlegreen_v11c_clearance_gate_m = float(clearance_gate_start_m[stage])
        env._littlegreen_v11c_min_step_m = float(min_step_m[stage])
        env._littlegreen_v11c_max_step_m = float(max_step_m[stage])
        env._littlegreen_v11c_step_length_scale = float(step_length_scales[stage])
        env._littlegreen_v11c_progression_scale = float(fullcycle_scales[stage])
        env._littlegreen_v11c_tracking_scale = float(tracking_scales[stage])

        env._littlegreen_v11d_clean_flight_s = float(clean_flight_s[stage])
        env._littlegreen_v11d_clearance_gate_m = float(clearance_gate_full_m[stage])
        env._littlegreen_v11d_lead_margin_m = float(lead_margin_m[stage])
        env._littlegreen_v11d_valid_step_fraction = float(valid_step_fractions[stage])
        env._littlegreen_v11d_stance_drift_limit_m = float(
            stance_drift_limits_m[stage]
        )
        env._littlegreen_v11d_takeover_window_s = float(takeover_window_s[stage])
        env._littlegreen_v11d_takeover_progress_ratio = float(
            takeover_progress_ratios[stage]
        )
        env._littlegreen_v11d_takeover_progress_min_m = float(
            takeover_progress_min_m[stage]
        )
        env._littlegreen_v11d_takeover_progress_max_m = float(
            takeover_progress_max_m[stage]
        )
        env._littlegreen_v11d_support_zone_forward_m = float(
            support_zone_forward_m[stage]
        )
        env._littlegreen_v11d_old_foot_behind_m = float(old_foot_behind_m[stage])
        env._littlegreen_v11d_event_hold_s = 0.14
        env._littlegreen_v11d_progression_scale = 1.0

        env._littlegreen_v11e_failure_penalty_scale = float(
            failure_penalty_scales[stage]
        )
        env._littlegreen_v11e_stance_drift_deadzone_m = float(
            stance_drift_deadzones_m[stage]
        )
        env._littlegreen_v11e_stance_drift_scale = float(
            stance_drift_scales[stage]
        )
        env._littlegreen_v11e_stance_slide_deadzone_m_s = float(
            stance_slide_deadzones_m_s[stage]
        )
        env._littlegreen_v11e_stance_slide_scale = float(
            stance_slide_scales[stage]
        )

        env._littlegreen_v11f_lead_deadband_m = float(lead_margin_m[stage])
        env._littlegreen_v11f_clearance_gate_start_m = float(
            clearance_gate_start_m[stage]
        )
        env._littlegreen_v11f_clearance_gate_full_m = float(
            clearance_gate_full_m[stage]
        )
        env._littlegreen_v11f_clean_flight_s = float(clean_flight_s[stage])
        env._littlegreen_v11f_event_hold_s = 0.14
        env._littlegreen_v11f_drag_speed_deadzone_m_s = 0.045
        env._littlegreen_v11f_stall_grace_s = 1.45 - 0.10 * stage
        env._littlegreen_v11f_stall_ramp_s = 0.90
        env._littlegreen_v11f_first_step_scale = float(first_step_scales[stage])
        env._littlegreen_v11f_trailing_pass_scale = float(trailing_pass_scales[stage])
        env._littlegreen_v11f_lead_reversal_scale = float(
            lead_reversal_scales[stage]
        )
        env._littlegreen_v11f_takeover_scale = float(takeover_scales[stage])
        env._littlegreen_v11f_stride_scale = float(stride_scales[stage])
        env._littlegreen_v11f_leading_repeat_scale = float(
            leading_repeat_scales[stage]
        )
        env._littlegreen_v11f_drag_scale = float(drag_scales[stage])
        env._littlegreen_v11f_stall_scale = float(stall_scales[stage])

        command_term.time_left[:] = 0.0
        zeros = torch.zeros(env.num_envs, device=command_term.time_left.device)
        zeros_bool = torch.zeros(
            env.num_envs, device=command_term.time_left.device, dtype=torch.bool
        )
        zeros_long = torch.zeros(
            env.num_envs, device=command_term.time_left.device, dtype=torch.long
        )
        env._littlegreen_v11a_was_moving = zeros_bool.clone()
        env._littlegreen_v11c_was_moving = zeros_bool.clone()
        env._littlegreen_v11d_was_moving = zeros_bool.clone()
        env._littlegreen_v11f_was_moving = zeros_bool.clone()
        env._littlegreen_v11f_chain_active = zeros_bool.clone()
        env._littlegreen_v11f_chain_first_side = zeros_long.clone()
        env._littlegreen_v11f_previous_due_side = zeros_long.clone()
        env._littlegreen_v11f_previous_due_gap_m = zeros.clone()
        env._littlegreen_v11f_chain_elapsed_s = zeros.clone()
        env._littlegreen_v11f_first_step_hold_s = zeros.clone()
        env._littlegreen_v11f_lead_reversal_hold_s = zeros.clone()
        env._littlegreen_v11f_stride_hold_s = zeros.clone()
        env._littlegreen_v11f_stride_chain_stage = stage

    action_term = env.action_manager.get_term("joint_pos")
    return {
        "stage": float(stage),
        "standing_fraction": float(standing_fractions[stage]),
        "cmd_x_min": float(lin_vel_x_ranges[stage][0]),
        "cmd_x_max": float(lin_vel_x_ranges[stage][1]),
        "cmd_y_abs_max": float(max(abs(v) for v in lin_vel_y_ranges[stage])),
        "cmd_yaw_abs_max": float(max(abs(v) for v in ang_vel_z_ranges[stage])),
        "gait_period_s": float(gait_period_s[stage]),
        "transition_fraction_per_half": float(transition_fractions[stage]),
        "observation_count": 47.0,
        "observation_contract_changed": 0.0,
        "stride_chain_enabled": 1.0,
        "trailing_foot_only_dense_pass": 1.0,
        "leading_foot_repeat_penalized": 1.0,
        "true_stride_requires_opposite_takeover": 1.0,
        "failed_step_blocks_alternation": 0.0,
        "target_clearance_m": float(target_clearance_m[stage]),
        "first_step_scale": float(first_step_scales[stage]),
        "trailing_pass_scale": float(trailing_pass_scales[stage]),
        "lead_reversal_scale": float(lead_reversal_scales[stage]),
        "opposite_takeover_scale": float(takeover_scales[stage]),
        "true_stride_scale": float(stride_scales[stage]),
        "leading_repeat_scale": float(leading_repeat_scales[stage]),
        "drag_scale": float(drag_scales[stage]),
        "stall_scale": float(stall_scales[stage]),
        "fullcycle_scale": float(fullcycle_scales[stage]),
        "moving_height_target_m": 0.461,
        "moving_height_width_m": 0.070,
        "push_xy_abs_max": 0.0,
        "mass_scale_min": float(mass_scale_ranges[stage][0]),
        "mass_scale_max": float(mass_scale_ranges[stage][1]),
        "actuator_delay_mean_ms": float(
            torch.mean(action_term.sampled_total_delay_s).item() * 1000.0
        ),
        "actuator_tau_mean_ms": float(
            torch.mean(action_term.sampled_tau_s).item() * 1000.0
        ),
        "actuator_velocity_scale_mean": float(
            torch.mean(action_term.sampled_velocity_scale).item()
        ),
        "curriculum_profile_id": 211.6,
    }
