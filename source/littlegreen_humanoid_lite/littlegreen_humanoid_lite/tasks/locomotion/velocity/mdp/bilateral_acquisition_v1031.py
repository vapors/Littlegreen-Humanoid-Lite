"""v10.3.1 low-weight, pre-handoff bilateral acquisition signal.

This term intentionally DOES NOT require the v6.6 support-baton swing stage,
a confirmed step, or a completed support handoff. v10.3.0's unilateral
solution rarely reached those gates, leaving important rewards at zero.

Per-side quality is independent of leg labels and uses v6.7's existing,
world-frame-corrected physical-foot geometry: coherent main+toe lift, stance
relative swing forward velocity, foot load share and settled planted stance.
A short recency buffer adds bilateral credit only when BOTH sides show useful
motion, while the single-side bootstrap is deliberately small.
"""
from __future__ import annotations
import torch
from isaaclab.managers import SceneEntityCfg
from .rewards import _v67_whole_foot_kinematics


def balanced_swing_discovery_v1031(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    command_threshold: float = 0.08,
    clearance_full_m: float = 0.022,
    rise_full_mps: float = 0.100,
    forward_full_mps: float = 0.22,
    max_stance_speed_mps: float = 0.12,
    recency_decay: float = 0.985,
    individual_bootstrap: float = 0.22,
    full_tilt_deg: float = 25.0,
    zero_tilt_deg: float = 55.0,
    full_height_m: float = 0.400,
    zero_height_m: float = 0.340,
) -> torch.Tensor:
    """Small [0,1] pre-handoff reward; high bilateral income needs both feet.

    Foot velocities are relative to the opposite *world-frame* stance foot;
    both main and toe clearance must rise. A planted, slow stance foot is
    required and uprightness gates the signal, preventing falling, flying,
    and pure foot sliding from easily earning locomotion credit.

    The recency window is reset per environment when the episode-length
    counter decreases (including simultaneous PPO resets). It is not a
    controller phase, an observation, or a persistent training checkpoint.
    """
    if not (0.0 < recency_decay < 1.0 and 0.0 < individual_bootstrap < 1.0):
        raise ValueError("v10.3.1 invalid bilateral recency/bootstrap settings")
    if min(clearance_full_m, rise_full_mps, forward_full_mps, max_stance_speed_mps) <= 0:
        raise ValueError("v10.3.1 positive physical quality scales required")

    k = _v67_whole_foot_kinematics(
        env, command_name, sensor_cfg, asset_cfg,
        full_tilt_deg, zero_tilt_deg, full_height_m, zero_height_m,
    )
    num_envs = k["load_share"].shape[0]
    ids = torch.arange(num_envs, device=k["load_share"].device)
    # Each foot is evaluated as a swing candidate, with the *other* foot
    # as stance. This is symmetric even while a learned policy is asymmetric.
    other = torch.tensor([1, 0], device=ids.device)
    stance_vel = k["center_vel"][:, other, :]
    swing_rel_vel = k["center_vel"] - stance_vel
    along_world = torch.sum(swing_rel_vel * k["command_dir_world"][:, None, :], dim=-1)
    forward = torch.clamp((along_world - 0.010) / float(forward_full_mps), 0.0, 1.0)
    rise = torch.clamp((k["coherent_rise"] - 0.005) / float(rise_full_mps), 0.0, 1.0)
    clear = torch.clamp((k["whole_clear"] - 0.002) / float(clearance_full_m), 0.0, 1.0)
    unloaded = torch.clamp((0.55 - k["load_share"]) / 0.35, 0.0, 1.0)
    stance_speed = torch.linalg.vector_norm(stance_vel, dim=-1)
    stance_anchor = torch.clamp(1.0 - stance_speed / float(max_stance_speed_mps), 0.0, 1.0)
    # Distinguish meaningful full-foot air from early unloading: a toe still
    # in contact cannot earn full forward-passage credit.
    genuinely_airborne = (~k["any_contact"]).float()
    forward_part = forward * clear * (0.25 + 0.75 * genuinely_airborne)
    precursor = 0.25 * rise + 0.20 * clear + 0.55 * forward_part
    gate = (
        (k["command_norm"][:, None] > float(command_threshold))
        & k["full_plant"][:, other]
        & (k["load_share"][:, other] >= 0.54)
    ).float()
    # Never pay this bilateral helper for a command-opposing backward swing.
    # A small onset ramp keeps early nearly-stationary liftoff discoverable.
    directional_gate = torch.clamp((along_world + 0.010) / 0.075, 0.0, 1.0)
    quality = torch.clamp(
        unloaded * precursor * directional_gate * gate * stance_anchor * k["viability"][:, None],
        0.0, 1.0,
    )

    # Recent per-side opportunity (roughly one second at 50 Hz) is NOT a
    # prescribed gait phase. Its sole purpose is to value evidence that both
    # sides can swing without requiring a hard completed-step event.
    lengths = env.episode_length_buf
    existing = getattr(env, "_littlegreen_v1031_recent_quality", None)
    prev_len = getattr(env, "_littlegreen_v1031_prev_ep_len", None)
    if existing is None or existing.shape != quality.shape:
        existing = torch.zeros_like(quality)
        prev_len = torch.full_like(lengths, -1)
    reset = lengths <= prev_len
    recent = torch.where(reset[:, None], torch.zeros_like(existing), existing)
    recent = torch.maximum(recent * float(recency_decay), quality)
    env._littlegreen_v1031_recent_quality = recent.detach()
    env._littlegreen_v1031_prev_ep_len = lengths.clone().detach()

    # A single swinging foot earns at most 22% of this reward ceiling.
    # Large credit needs BOTH recent leg qualities and current swing activity,
    # so prolonged one-sided hovering/rocking cannot farm bilateral income.
    instantaneous = torch.max(quality, dim=1).values
    paired = torch.sqrt(torch.clamp(recent[:, 0] * recent[:, 1], min=0.0))
    reward = (
        float(individual_bootstrap) * 0.5 * (quality[:, 0] + quality[:, 1])
        + (1.0 - float(individual_bootstrap)) * paired * instantaneous
    )
    env._littlegreen_v1031_swing_quality = quality.detach()
    env._littlegreen_v1031_bilateral_pair_quality = paired.detach()
    env._littlegreen_v1031_bilateral_reward = reward.detach()
    return torch.clamp(reward, 0.0, 1.0)
