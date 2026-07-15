"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_out_of_bounds(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_buffer: float = 3.0
) -> torch.Tensor:
    """Terminate when the actor move too close to the edge of the terrain.

    If the actor moves too close to the edge of the terrain, the termination is activated. The distance
    to the edge of the terrain is calculated based on the size of the terrain and the distance buffer.
    """
    if env.scene.cfg.terrain.terrain_type == "plane":
        return False  # we have infinite terrain because it is a plane
    elif env.scene.cfg.terrain.terrain_type == "generator":
        # obtain the size of the sub-terrains
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols
        border_width = terrain_gen_cfg.border_width
        # compute the size of the map
        map_width = n_rows * grid_width + 2 * border_width
        map_height = n_cols * grid_length + 2 * border_width

        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]

        # check if the agent is out of bounds
        x_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 0]) > 0.5 * map_width - distance_buffer
        y_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 1]) > 0.5 * map_height - distance_buffer
        return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
    else:
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")


def _env_step_dt(env: ManagerBasedRLEnv, fallback: float = 0.02) -> float:
    """Return environment policy-step dt with a safe fallback for static tests."""
    for attr in ("step_dt", "dt"):
        value = getattr(env, attr, None)
        if value is not None:
            try:
                return float(value)
            except Exception:
                pass
    return fallback


def moving_no_progress_timeout(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.22,
    min_progress_fraction: float = 0.20,
    grace_time_s: float = 0.80,
    timeout_s: float = 1.20,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate episodes that brace in place under clear movement commands.

    This is the v1.4.6 anti-planted mechanism. It gives each episode a short
    grace period to organize a step, then resets environments that continue to
    make too little velocity progress along the commanded horizontal direction.
    Standing-command environments are unaffected.
    """
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)[:, :2]
    command_norm = torch.linalg.vector_norm(command, dim=1)
    command_dir = command / torch.clamp(command_norm.unsqueeze(-1), min=1.0e-6)
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])[:, :2]
    projected_speed = torch.sum(vel_yaw * command_dir, dim=1)

    moving = command_norm > command_threshold
    stalled = moving & (projected_speed < min_progress_fraction * command_norm)

    dt = _env_step_dt(env)
    num_envs = command.shape[0]
    device = command.device
    active_timer = getattr(env, "_littlegreen_v146_moving_active_timer_s", None)
    stall_timer = getattr(env, "_littlegreen_v146_no_progress_timer_s", None)
    if active_timer is None or active_timer.shape[0] != num_envs:
        active_timer = torch.zeros(num_envs, device=device, dtype=torch.float32)
    if stall_timer is None or stall_timer.shape[0] != num_envs:
        stall_timer = torch.zeros(num_envs, device=device, dtype=torch.float32)

    active_timer = torch.where(moving, active_timer + dt, torch.zeros_like(active_timer))
    stall_timer = torch.where(stalled, stall_timer + dt, torch.zeros_like(stall_timer))
    done = moving & (active_timer > grace_time_s) & (stall_timer > timeout_s)

    # Clear timers for terminated environments so a reset starts cleanly.
    active_timer = torch.where(done, torch.zeros_like(active_timer), active_timer)
    stall_timer = torch.where(done, torch.zeros_like(stall_timer), stall_timer)
    env._littlegreen_v146_moving_active_timer_s = active_timer
    env._littlegreen_v146_no_progress_timer_s = stall_timer
    return done




def moving_no_support_timeout(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    command_threshold: float = 0.22,
    max_no_support_s: float = 0.12,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Terminate persistent flight/no-support under moving commands.

    v1.4.6 escaped planted bracing by allowing a hopping/falling exploit.  This
    termination keeps v1.4.7 grounded: brief contact jitter is allowed, but both
    feet off the ground for more than ``max_no_support_s`` is treated as failure.
    """
    asset = env.scene["robot"]
    command = env.command_manager.get_command(command_name)[:, :2]
    moving = torch.linalg.vector_norm(command, dim=1) > command_threshold
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    in_contact = forces.norm(dim=-1).max(dim=1)[0] > force_threshold
    no_support = torch.sum(in_contact.int(), dim=1) == 0

    timer = getattr(env, "_littlegreen_v147_no_support_timer_s", None)
    if timer is None or timer.shape[0] != asset.data.root_pos_w.shape[0]:
        timer = torch.zeros(asset.data.root_pos_w.shape[0], device=asset.data.root_pos_w.device)
    timer = torch.where(moving & no_support, timer + float(env.step_dt), torch.zeros_like(timer))
    env._littlegreen_v147_no_support_timer_s = timer
    return timer > max_no_support_s
