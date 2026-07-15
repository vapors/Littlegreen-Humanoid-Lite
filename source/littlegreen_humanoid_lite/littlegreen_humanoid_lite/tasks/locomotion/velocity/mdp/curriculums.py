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
