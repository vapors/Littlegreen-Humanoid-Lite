"""Utilities for forcing velocity commands during playback/evaluation.

These helpers intentionally use defensive introspection because Isaac Lab's
UniformVelocityCommand internals have changed across releases. They set both the
policy observation command (handled by caller) and the environment command term
when possible so debug arrows/markers match the command seen by the policy.
"""

from __future__ import annotations

from typing import Any

import torch


def format_command_suffix(command: list[float] | tuple[float, float, float] | None) -> str:
    """Return a filename-safe suffix for a forced [vx, vy, yaw] command."""
    if command is None:
        return ""
    vx, vy, wz = (float(command[0]), float(command[1]), float(command[2]))
    return f"_cmd_vx{vx:+.2f}_vy{vy:+.2f}_yaw{wz:+.2f}".replace("+", "p").replace("-", "m").replace(".", "p")


def find_command_term(env_like: Any, command_name: str = "base_velocity") -> Any | None:
    """Best-effort lookup of a command term through common wrapper layers."""
    candidates: list[Any] = []
    obj = env_like
    for _ in range(8):
        if obj is None:
            break
        candidates.append(obj)
        next_obj = getattr(obj, "unwrapped", None)
        if next_obj is None or next_obj is obj:
            next_obj = getattr(obj, "env", None)
        if next_obj is None or next_obj is obj:
            break
        obj = next_obj

    for candidate in candidates:
        command_manager = getattr(candidate, "command_manager", None)
        if command_manager is None:
            continue
        try:
            return command_manager.get_term(command_name)
        except Exception:
            pass
        terms = getattr(command_manager, "_terms", None)
        if isinstance(terms, dict) and command_name in terms:
            return terms[command_name]
    return None


def apply_velocity_command_override(
    env_like: Any,
    command: torch.Tensor,
    command_name: str = "base_velocity",
    warn: bool = False,
) -> bool:
    """Force the environment command term to a fixed velocity command.

    Args:
        env_like: Environment or wrapper object.
        command: Tensor of shape ``(num_envs, 3)`` containing ``[vx, vy, yaw_rate]``.
        command_name: Command term name, normally ``base_velocity``.
        warn: Print a warning if no known command tensor can be updated.

    Returns:
        True if at least one known command tensor/property was updated.
    """
    term = find_command_term(env_like, command_name=command_name)
    if term is None:
        if warn:
            print(f"[WARN] Could not find command term '{command_name}' for forced-command override.")
        return False

    command = command.detach()
    updated = False

    # Common Isaac Lab UniformVelocityCommand storage. The command property returned
    # by command_manager.get_command(...) is usually backed by vel_command_b.
    for attr in ("vel_command_b", "command", "_command", "commands", "_commands"):
        if not hasattr(term, attr):
            continue
        try:
            target = getattr(term, attr)
            if torch.is_tensor(target) and target.ndim == 2 and target.shape[1] >= 3:
                n = min(target.shape[0], command.shape[0])
                target[:n, :3] = command[:n, :3].to(device=target.device, dtype=target.dtype)
                updated = True
        except Exception:
            # Some properties are read-only; keep trying other known fields.
            pass

    # Some versions expose heading/standing masks. Force them off so the velocity
    # command is not overwritten by heading control or standing resampling.
    for attr in ("is_heading_env", "_is_heading_env"):
        if hasattr(term, attr):
            try:
                mask = getattr(term, attr)
                if torch.is_tensor(mask):
                    mask[:] = False
            except Exception:
                pass
    for attr in ("is_standing_env", "_is_standing_env"):
        if hasattr(term, attr):
            try:
                mask = getattr(term, attr)
                if torch.is_tensor(mask):
                    mask[:] = False
            except Exception:
                pass

    # Prevent resampling during a forced-command clip. This is not required because
    # the override is applied every step, but it keeps markers and logs cleaner.
    if hasattr(term, "time_left"):
        try:
            time_left = getattr(term, "time_left")
            if torch.is_tensor(time_left):
                time_left[:] = 1.0e6
        except Exception:
            pass

    if warn and not updated:
        print(
            f"[WARN] Found command term '{command_name}', but no known command tensor "
            "could be updated; policy observation will still be overridden."
        )
    return updated
