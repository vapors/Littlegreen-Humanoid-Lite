"""Play an RSL-RL checkpoint in Isaac Lab with joystick/command-file override.

This script is intentionally playback-only. Policy export is handled by:
    scripts/rsl_rl/export_policy.py

The command override reads three floats from a text file:
    <linear_x> <linear_y> <yaw_rate>

Default command file:
    /tmp/joystick_cmd.txt
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Play an RSL-RL policy checkpoint with joystick command override.")
parser.add_argument("--video", action="store_true", default=False, help="Record a playback video.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in simulation steps.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the Isaac Lab task/env.")
parser.add_argument(
    "--joystick_file",
    type=str,
    default="/tmp/joystick_cmd.txt",
    help="Path to a text file containing three floats: linear_x linear_y yaw_rate.",
)
parser.add_argument(
    "--debug_joystick",
    action="store_true",
    default=False,
    help="Print joystick command and observation override debug information each step.",
)
parser.add_argument(
    "--no_command_manager_override",
    action="store_true",
    default=False,
    help=(
        "Only override the policy observation command. By default the environment command "
        "manager is also overridden so velocity arrows/markers match the policy input."
    ),
)

# RSL-RL args include --experiment_name, --load_run, --checkpoint, etc.
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch
from command_override import apply_velocity_command_override
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# Import extensions to register project tasks.
import littlegreen_humanoid_lite.tasks  # noqa: F401,E402


def get_joystick_cmd(path: str) -> list[float]:
    """Read a [linear_x, linear_y, yaw_rate] command from a text file."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            parts = file.readline().strip().split()
        if len(parts) >= 3:
            return [float(parts[0]), float(parts[1]), float(parts[2])]
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"[WARN] Could not read joystick command from {path}: {exc}")
    return [0.0, 0.0, 0.0]


def main():
    """Load a checkpoint and run it with command-file override."""
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    checkpoint_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(checkpoint_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play_joystick"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording joystick playback video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env)

    print(f"[INFO] Loading model checkpoint from: {checkpoint_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs, _ = env.get_observations()
    timestep = 0

    while simulation_app.is_running():
        joystick_cmd = get_joystick_cmd(args_cli.joystick_file)
        joystick_tensor = torch.tensor(joystick_cmd, device=env.unwrapped.device, dtype=obs.dtype).unsqueeze(0)
        joystick_tensor = joystick_tensor.repeat(env.num_envs, 1)

        if not args_cli.no_command_manager_override:
            # Keep command visualization/markers and the policy input aligned.
            # Without this, the policy can receive the joystick command while the
            # green command arrow still shows the random command generator state.
            apply_velocity_command_override(
                env, joystick_tensor, command_name="base_velocity", warn=args_cli.debug_joystick
            )

        # The current policy observation layout uses columns 0:3 for velocity command.
        # Keep this isolated here so future obs-layout changes are easy to update.
        obs = obs.clone()
        if args_cli.debug_joystick:
            print(f"[JOY] command: {joystick_cmd}")
            print(f"[JOY] obs[:, :3] before: {obs[:, :3]}")
        obs[:, :3] = joystick_tensor
        if args_cli.debug_joystick:
            print(f"[JOY] obs[:, :3] after:  {obs[:, :3]}")

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        if args_cli.video:
            timestep += 1
            if timestep >= args_cli.video_length:
                break

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
