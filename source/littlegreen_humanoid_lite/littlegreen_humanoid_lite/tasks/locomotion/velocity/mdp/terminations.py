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


def _episode_reset_mask(env: ManagerBasedRLEnv, num_envs: int, device: torch.device) -> torch.Tensor:
    """Return a best-effort mask for environments at the start of a new episode.

    Isaac Lab resets ``episode_length_buf`` to zero.  Treating both zero and one as
    reset-adjacent keeps stateful termination buffers clean regardless of whether the
    termination manager runs before or after the first post-reset step increment.
    """
    episode_length = getattr(env, "episode_length_buf", None)
    if torch.is_tensor(episode_length) and episode_length.shape[0] == num_envs:
        return episode_length.to(device=device) <= 1
    return torch.zeros(num_envs, device=device, dtype=torch.bool)


def moving_no_progress_timeout(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.22,
    min_progress_fraction: float = 0.20,
    grace_time_s: float = 0.80,
    timeout_s: float = 1.20,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate sustained no-progress under a clear movement command.

    The timers are explicitly cleared at every episode boundary and when this
    termination fires.  This prevents a timeout accumulated in one episode from
    leaking into a newly reset episode.
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
    reset_mask = _episode_reset_mask(env, num_envs, device)

    active_timer = getattr(env, "_littlegreen_v146_moving_active_timer_s", None)
    stall_timer = getattr(env, "_littlegreen_v146_no_progress_timer_s", None)
    if active_timer is None or active_timer.shape[0] != num_envs:
        active_timer = torch.zeros(num_envs, device=device, dtype=torch.float32)
    if stall_timer is None or stall_timer.shape[0] != num_envs:
        stall_timer = torch.zeros(num_envs, device=device, dtype=torch.float32)

    # Clear state inherited from a previous episode before evaluating this step.
    active_timer = torch.where(reset_mask, torch.zeros_like(active_timer), active_timer)
    stall_timer = torch.where(reset_mask, torch.zeros_like(stall_timer), stall_timer)

    active_timer = torch.where(moving, active_timer + dt, torch.zeros_like(active_timer))
    stall_timer = torch.where(stalled, stall_timer + dt, torch.zeros_like(stall_timer))
    done = moving & (active_timer > grace_time_s) & (stall_timer > timeout_s)

    # Also clear immediately on termination so repeated manager evaluation cannot
    # emit a second stale done before the environment reset is applied.
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
    arming_timeout_s: float = 0.40,
) -> torch.Tensor:
    """Terminate persistent flight/no-support under moving commands.

    This implementation is episode-reset-aware and requires a newly reset
    environment to establish valid support before the no-support timer is normally
    armed.  A bounded arming timeout remains as a fallback so a genuinely broken
    reset pose cannot evade protection forever.
    """
    command = env.command_manager.get_command(command_name)[:, :2]
    moving = torch.linalg.vector_norm(command, dim=1) > command_threshold
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    in_contact = forces.norm(dim=-1).max(dim=1)[0] > force_threshold
    has_support = torch.sum(in_contact.int(), dim=1) > 0
    no_support = ~has_support

    num_envs = command.shape[0]
    device = command.device
    dt = _env_step_dt(env)
    reset_mask = _episode_reset_mask(env, num_envs, device)

    timer = getattr(env, "_littlegreen_v147_no_support_timer_s", None)
    had_support = getattr(env, "_littlegreen_v147_had_support", None)
    arming_timer = getattr(env, "_littlegreen_v147_support_arming_timer_s", None)
    if timer is None or timer.shape[0] != num_envs:
        timer = torch.zeros(num_envs, device=device, dtype=torch.float32)
    if had_support is None or had_support.shape[0] != num_envs:
        had_support = torch.zeros(num_envs, device=device, dtype=torch.bool)
    if arming_timer is None or arming_timer.shape[0] != num_envs:
        arming_timer = torch.zeros(num_envs, device=device, dtype=torch.float32)

    timer = torch.where(reset_mask, torch.zeros_like(timer), timer)
    had_support = torch.where(reset_mask, torch.zeros_like(had_support), had_support)
    arming_timer = torch.where(reset_mask, torch.zeros_like(arming_timer), arming_timer)

    had_support = had_support | has_support
    arming_timer = arming_timer + dt
    armed = had_support | (arming_timer >= float(arming_timeout_s))

    timer = torch.where(armed & moving & no_support, timer + dt, torch.zeros_like(timer))
    done = armed & moving & (timer > max_no_support_s)

    timer = torch.where(done, torch.zeros_like(timer), timer)
    had_support = torch.where(done, torch.zeros_like(had_support), had_support)
    arming_timer = torch.where(done, torch.zeros_like(arming_timer), arming_timer)
    env._littlegreen_v147_no_support_timer_s = timer
    env._littlegreen_v147_had_support = had_support
    env._littlegreen_v147_support_arming_timer_s = arming_timer
    return done
