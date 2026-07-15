"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import re
import glob
import subprocess
import threading
from datetime import datetime

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# ---------------------------
# CLI SETUP (original options)
# ---------------------------
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=4000, help="(Ignored in this mode) interval between video recordings (steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--policy_only_warm_start",
    action="store_true",
    default=False,
    help="Load actor/action-std and observation normalization from the resume checkpoint, but reset critic/optimizer.",
)
parser.add_argument(
    "--resume_log_root",
    type=str,
    default=None,
    help="Optional logs/rsl_rl/<experiment> root to search for --load_run when warm-starting from another task family.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# ---- hidden flags for eval-only subprocess ----
parser.add_argument("--eval_only", action="store_true", default=False, help=argparse.SUPPRESS)
parser.add_argument("--eval_log_dir", type=str, default="", help=argparse.SUPPRESS)
parser.add_argument("--eval_steps", type=int, default=0, help=argparse.SUPPRESS)
parser.add_argument("--eval_tag", type=str, default="", help=argparse.SUPPRESS)
parser.add_argument("--eval_target_iter", type=int, default=-1, help=argparse.SUPPRESS)
parser.add_argument(
    "--eval_force_command",
    type=float,
    nargs=3,
    default=None,
    metavar=("VX", "VY", "YAW"),
    help="Force a fixed [vx, vy, yaw_rate] command during eval-only/checkpoint video rendering.",
)
parser.add_argument(
    "--eval_force_command_name",
    type=str,
    default="base_velocity",
    help="Command term to override for --eval_force_command.",
)

args_cli, hydra_args = parser.parse_known_args()

# Only the eval-only subprocess should enable cameras/rendering.
# Training stays on the native headless kit to avoid VRAM spikes.
args_cli.enable_cameras = bool(args_cli.eval_only)

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""
import gymnasium as gym
import torch
import imageio.v2 as imageio  # used in eval-only subprocess to encode mp4

from command_override import apply_velocity_command_override, format_command_suffix
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import littlegreen_humanoid_lite.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False




def _candidate_state_dict(obj):
    """Return an object's state_dict if it behaves like a torch module."""
    if obj is None or not hasattr(obj, "state_dict"):
        return None
    try:
        state = obj.state_dict()
    except Exception:
        return None
    return state if isinstance(state, dict) else None


def _score_policy_module_candidate(obj, model_state):
    """Score likely RSL-RL ActorCritic modules across RSL-RL versions.

    Isaac Lab / RSL-RL versions differ in where the policy module is stored:
    runner.alg.actor_critic, runner.alg.policy, runner.alg.ac, or sometimes a
    private attribute.  We identify the correct module by its state_dict shape
    rather than relying on a single attribute name.
    """
    current = _candidate_state_dict(obj)
    if current is None:
        return -1
    score = 0
    actor_score = 0
    for key, value in model_state.items():
        if key not in current or not hasattr(value, "shape") or not hasattr(current[key], "shape"):
            continue
        if tuple(current[key].shape) != tuple(value.shape):
            continue
        score += 1
        if key == "std" or key.startswith("actor") or ".actor" in key:
            actor_score += 1
    if actor_score == 0:
        return -1
    return score + 100 * actor_score


def _find_actor_critic_module(runner, model_state):
    """Locate the ActorCritic-like policy module robustly across RSL-RL versions."""
    alg = getattr(runner, "alg", None) or getattr(runner, "algo", None)
    candidate_roots = [
        ("runner", runner),
        ("runner.alg", alg),
    ]
    preferred_names = (
        "actor_critic",
        "actor_critic_module",
        "policy",
        "ac",
        "model",
        "module",
        "_actor_critic",
        "_policy",
    )

    scored = []
    seen = set()

    def add_candidate(name, obj):
        if obj is None or id(obj) in seen:
            return
        seen.add(id(obj))
        score = _score_policy_module_candidate(obj, model_state)
        if score >= 0:
            scored.append((score, name, obj))

    for root_name, root in candidate_roots:
        if root is None:
            continue
        for attr in preferred_names:
            add_candidate(f"{root_name}.{attr}", getattr(root, attr, None))
        # Shallow scan as a fallback for renamed attributes.
        for attr, obj in vars(root).items():
            if attr.startswith("__"):
                continue
            add_candidate(f"{root_name}.{attr}", obj)

    if not scored:
        alg_type = type(alg).__name__ if alg is not None else "None"
        alg_attrs = sorted([name for name in vars(alg).keys() if not name.startswith("__")]) if alg is not None else []
        runner_attrs = sorted([name for name in vars(runner).keys() if not name.startswith("__")])
        raise RuntimeError(
            "Could not locate the RSL-RL ActorCritic/policy module for policy-only warm start. "
            f"runner type={type(runner).__name__}, alg type={alg_type}, "
            f"runner attrs={runner_attrs}, alg attrs={alg_attrs}"
        )

    scored.sort(key=lambda item: item[0], reverse=True)
    score, name, module = scored[0]
    print(f"[INFO] Policy-only warm start target module: {name} ({type(module).__name__})")
    return module


def _load_policy_only_warm_start(runner: OnPolicyRunner, checkpoint_path: str, device: str | torch.device):
    """Load actor-side policy weights without carrying critic/optimizer state.

    RSL-RL checkpoint formats vary slightly across versions. This helper keeps any
    matching non-critic tensors from the saved actor_critic state dict, including
    action standard deviation, and intentionally leaves critic/optimizer freshly
    initialized for a new Hardware objective.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = None
    for key in ("model_state_dict", "actor_critic_state_dict", "state_dict"):
        maybe = checkpoint.get(key) if isinstance(checkpoint, dict) else None
        if isinstance(maybe, dict):
            model_state = maybe
            break
    if model_state is None and isinstance(checkpoint, dict):
        # Some checkpoints are raw state_dicts.
        if all(isinstance(k, str) for k in checkpoint.keys()):
            model_state = checkpoint
    if model_state is None:
        raise RuntimeError(f"Could not find model state dict in checkpoint: {checkpoint_path}")

    actor_critic = _find_actor_critic_module(runner, model_state)
    current = actor_critic.state_dict()
    loaded = []
    partial_loaded = []
    skipped = []
    for key, value in model_state.items():
        if key not in current:
            skipped.append(key)
            continue
        if key.startswith("critic") or ".critic" in key or key.startswith("value") or ".value" in key:
            skipped.append(key)
            continue
        if tuple(current[key].shape) == tuple(value.shape):
            current[key] = value.to(device=current[key].device, dtype=current[key].dtype)
            loaded.append(key)
            continue
        # v1.4.7 adds two gait-phase observations.  Allow actor first-layer
        # expansion by copying the old input columns and keeping new columns from
        # the fresh initialization.  This keeps the Stand policy motor prior while
        # giving the new phase inputs learnable random weights.
        if (
            hasattr(value, "ndim")
            and value.ndim == 2
            and current[key].ndim == 2
            and current[key].shape[0] == value.shape[0]
            and current[key].shape[1] > value.shape[1]
            and (key.startswith("actor") or ".actor" in key or "mlp" in key or "policy" in key)
        ):
            expanded = current[key].clone()
            expanded[:, : value.shape[1]] = value.to(device=current[key].device, dtype=current[key].dtype)
            current[key] = expanded
            partial_loaded.append(f"{key}: {tuple(value.shape)} -> {tuple(expanded.shape)}")
            continue
        skipped.append(key)
    actor_critic.load_state_dict(current, strict=True)

    # Preserve observation normalization if present.  When the observation dimension
    # expands (45-D -> 47-D for v1.4.7 phase), copy the shared prefix and leave the
    # new phase dimensions at the freshly initialized normalization defaults.
    norm_state = None
    if isinstance(checkpoint, dict):
        for key in ("empirical_normalization_state_dict", "obs_normalizer_state_dict", "normalizer_state_dict"):
            if isinstance(checkpoint.get(key), dict):
                norm_state = checkpoint[key]
                break
    if norm_state is not None:
        for attr in ("empirical_normalization", "obs_normalizer", "normalizer"):
            module = getattr(runner, attr, None)
            if module is None or not hasattr(module, "load_state_dict") or not hasattr(module, "state_dict"):
                continue
            try:
                module.load_state_dict(norm_state)
                print(f"[INFO]: Loaded observation normalization via runner.{attr}")
                break
            except Exception as exc:
                try:
                    current_norm = module.state_dict()
                    merged = dict(current_norm)
                    norm_partial = []
                    for key, value in norm_state.items():
                        if key not in merged or not hasattr(value, "shape") or not hasattr(merged[key], "shape"):
                            continue
                        if tuple(merged[key].shape) == tuple(value.shape):
                            merged[key] = value.to(device=merged[key].device, dtype=merged[key].dtype)
                        elif (
                            value.ndim == 1
                            and merged[key].ndim == 1
                            and merged[key].shape[0] > value.shape[0]
                        ):
                            tmp = merged[key].clone()
                            tmp[: value.shape[0]] = value.to(device=tmp.device, dtype=tmp.dtype)
                            merged[key] = tmp
                            norm_partial.append(f"{key}: {tuple(value.shape)} -> {tuple(tmp.shape)}")
                    module.load_state_dict(merged, strict=True)
                    print(f"[INFO]: Partially loaded observation normalization via runner.{attr}: {norm_partial}")
                    break
                except Exception as exc2:
                    print(f"[WARN]: Could not load normalization via runner.{attr}: {exc}; partial failed: {exc2}")

    if partial_loaded:
        print(f"[INFO]: Partially expanded actor tensors: {partial_loaded}")
    print(
        f"[INFO]: Policy-only warm start loaded {len(loaded)} actor-side tensors "
        f"and {len(partial_loaded)} expanded actor tensors; "
        f"left critic/optimizer fresh; skipped {len(skipped)} tensors."
    )

# ---------- helpers for eval-only flow ----------

def _parse_ckpt_iter(path: str) -> int | None:
    m = re.search(r"model_(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else None

def _list_ckpts(log_dir: str):
    items = []
    for p in glob.glob(os.path.join(log_dir, "model_*.pt")):
        it = _parse_ckpt_iter(p)
        if it is not None:
            items.append((it, p))
    return sorted(items)

def _find_best_ckpt(log_dir: str, upto_iter: int | None):
    best = None
    for it, p in _list_ckpts(log_dir):
        if upto_iter is not None and it > upto_iter:
            break
        best = p
    return best

def _make_policy_act(runner: OnPolicyRunner):
    """Return a callable policy_act(obs)->actions, robust across rsl_rl variants."""
    alg = getattr(runner, "alg", None) or getattr(runner, "algo", None)
    candidates = []
    if alg is not None:
        for name in ("actor_critic", "policy", "ac", "actor"):
            if hasattr(alg, name):
                candidates.append(getattr(alg, name))
    if not candidates and hasattr(runner, "get_inference_policy"):
        candidates.append(runner.get_inference_policy())
    if not candidates and hasattr(runner, "policy"):
        candidates.append(getattr(runner, "policy"))
    if not candidates:
        raise RuntimeError("Could not locate a policy object.")
    pol = candidates[0]
    if hasattr(pol, "act_inference"):
        def _act(obs):
            with torch.no_grad():
                return pol.act_inference(obs)
        return _act
    if hasattr(pol, "act"):
        def _act(obs):
            with torch.no_grad():
                try:
                    return pol.act(obs, deterministic=True)[0]
                except TypeError:
                    return pol.act(obs)[0]
        return _act
    raise RuntimeError("Policy lacks act/act_inference.")

def _eval_render_clip(
    task_name,
    base_env_cfg,
    agent_cfg,
    ckpt_path,
    out_path,
    steps=500,
    force_command=None,
    force_command_name="base_velocity",
):
    """Runs in eval-only process: load checkpoint, roll out a 1-env rgb_array clip, write mp4."""
    # Clone cfg; force 1 env for lightweight checkpoint videos.
    eval_cfg = base_env_cfg.copy()
    eval_cfg.scene.num_envs = 1
    eval_cfg.viewer.resolution = (960, 540)
    eval_cfg.viewer.eye = (3, 5, 1.4)
    eval_cfg.viewer.lookat = (0.0, -1.0, 0.5)

    # Base env does the actual rendering
    base_env = gym.make(task_name, cfg=eval_cfg, render_mode="rgb_array")
    if isinstance(base_env.unwrapped, DirectMARLEnv):
        base_env = multi_agent_to_single_agent(base_env)

    # Vec wrapper is used only for reset/step compatibility with RSL-RL
    vec_env = RslRlVecEnvWrapper(base_env)

    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=os.path.dirname(out_path), device=agent_cfg.device)
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"[EVAL] Loading checkpoint: {ckpt_path}")
        runner.load(ckpt_path)
    else:
        print("[EVAL][WARN] No checkpoint found; recording with uninitialized policy.")

    policy_act = _make_policy_act(runner)

    # --- robust reset signature ---
    reset_out = vec_env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    force_command_tensor = None
    if force_command is not None:
        force_command_tensor = torch.tensor(
            force_command, device=vec_env.unwrapped.device, dtype=obs.dtype
        ).unsqueeze(0).repeat(vec_env.num_envs, 1)
        print(f"[EVAL] Forcing command {force_command_name}: {force_command}")
        apply_velocity_command_override(
            base_env, force_command_tensor, command_name=force_command_name, warn=True
        )

    frames = []
    for _ in range(int(steps)):
        if force_command_tensor is not None:
            # Keep the environment command manager/marker and the policy observation aligned.
            apply_velocity_command_override(base_env, force_command_tensor, command_name=force_command_name)
            obs = obs.clone()
            obs[:, :3] = force_command_tensor

        actions = policy_act(obs)

        # --- robust step signature (gym 4-tuple or gymnasium 5-tuple) ---
        step_out = vec_env.step(actions)
        obs = step_out[0] if isinstance(step_out, tuple) else step_out

        # Render from the base env (wrapper has no .render())
        try:
            frame = base_env.render()
        except AttributeError:
            # extra safety if a wrapper hides render()
            try:
                frame = base_env.unwrapped.render()
            except Exception:
                frame = None

        if frame is not None:
            frames.append(frame)

    # Clean up
    try:
        vec_env.close()
    except Exception:
        pass
    try:
        base_env.close()
    except Exception:
        pass

    if frames:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        policy_fps = max(1, int(round(1.0 / (float(eval_cfg.sim.dt) * int(eval_cfg.decimation)))))
        imageio.mimsave(out_path, frames, fps=policy_fps)
        print(f"[VIDEO] Saved: {out_path} at {policy_fps} FPS")
    else:
        print("[VIDEO] No frames captured (render() returned None).")




def _spawn_eval_subprocess(
    task_name,
    log_dir,
    video_steps,
    device,
    checkpoint_iter,
    extra_hydra_args=None,
    force=False,
    force_command=None,
    force_command_name="base_velocity",
):
    """Spawn one eval-only subprocess for a checkpoint iteration.

    Returns True when a subprocess was started, False when the video already exists
    or the checkpoint iteration is intentionally skipped.
    """
    if checkpoint_iter == 0:
        return False

    tag = f"{checkpoint_iter:06d}"
    cmd_suffix = format_command_suffix(force_command)
    out_path = os.path.join(log_dir, "videos", f"iter_{tag}{cmd_suffix}.mp4")
    if os.path.exists(out_path) and not force:
        return False

    print(f"[EVAL] Spawning video subprocess for checkpoint iter {checkpoint_iter}")
    print(f"[EVAL] Video target: {out_path}")
    cmd = [
        sys.executable, __file__,
        "--task", task_name,
        "--headless",
        "--eval_only",
        f"--eval_log_dir={log_dir}",
        f"--eval_steps={video_steps}",
        f"--eval_tag={tag}",
        f"--eval_target_iter={checkpoint_iter}",
    ]
    if force_command is not None:
        cmd += ["--eval_force_command", *(f"{float(v):.9g}" for v in force_command)]
        cmd += [f"--eval_force_command_name={force_command_name}"]
    if device:
        cmd += ["--device", device]
    if extra_hydra_args:
        cmd += list(extra_hydra_args)

    try:
        # Serialize videos: wait for completion; simple and robust.
        subprocess.run(cmd, check=True, env=os.environ.copy())
        return True
    except subprocess.CalledProcessError as e:
        print(f"[EVAL][WARN] Subprocess failed for iter {checkpoint_iter}: {e}")
        return False


def _render_missing_checkpoint_videos(
    task_name,
    log_dir,
    video_steps,
    device,
    extra_hydra_args=None,
    force_command=None,
    force_command_name="base_velocity",
):
    """Render videos for any checkpoints that do not already have MP4s."""
    ckpts = _list_ckpts(log_dir)
    if not ckpts:
        print(f"[EVAL][WARN] No model_*.pt checkpoints found in {log_dir}; no video can be rendered yet.")
        return 0

    rendered = 0
    for it, _ in ckpts:
        if _spawn_eval_subprocess(
            task_name,
            log_dir,
            video_steps,
            device,
            it,
            extra_hydra_args=extra_hydra_args,
            force_command=force_command,
            force_command_name=force_command_name,
        ):
            rendered += 1
    return rendered


def _ensure_at_least_one_checkpoint(runner: OnPolicyRunner, log_dir: str, fallback_iter: int):
    """For short smoke tests, force-save one checkpoint if none were created.

    RSL-RL only saves at save_interval. If max_iterations is smaller than
    save_interval, a smoke test can finish with no model_*.pt file, which means
    the eval-video subprocess has nothing to load.
    """
    if _list_ckpts(log_dir):
        return

    fallback_iter = int(fallback_iter) if fallback_iter is not None else 1
    fallback_iter = max(fallback_iter, 1)
    final_ckpt_path = os.path.join(log_dir, f"model_{fallback_iter}.pt")
    try:
        print(f"[INFO] No checkpoint was written during this run; saving final smoke-test checkpoint: {final_ckpt_path}")
        runner.save(final_ckpt_path)
    except Exception as e:
        print(f"[EVAL][WARN] Could not save a final checkpoint for video rendering: {e}")


def _checkpoint_watcher_thread(
    task_name,
    env_cfg,
    agent_cfg,
    log_dir,
    video_steps,
    device,
    stop_event,
    extra_hydra_args=None,
    force_command=None,
    force_command_name="base_velocity",
):
    """When a new checkpoint appears, spawn an eval-only subprocess to render a clip right after that save."""
    seen = set()
    while not stop_event.is_set():
        for it, _ in _list_ckpts(log_dir):
            if it in seen:
                continue
            seen.add(it)
            _spawn_eval_subprocess(
                task_name,
                log_dir,
                video_steps,
                device,
                it,
                extra_hydra_args=extra_hydra_args,
                force_command=force_command,
                force_command_name=force_command_name,
            )

        stop_event.wait(2.0)  # scan every 2s


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent (original flow), with optional video via eval-only subprocess."""
    # ---------------- EVAL-ONLY SUBPROCESS PATH ----------------
    if args_cli.eval_only:
        # Apply the same non-Hydra CLI overrides as the training path where relevant.
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        log_dir = args_cli.eval_log_dir
        tag = args_cli.eval_tag or "eval"
        target_iter = int(args_cli.eval_target_iter) if args_cli.eval_target_iter is not None else -1
        out_path = os.path.join(
            log_dir, "videos", f"iter_{tag}{format_command_suffix(args_cli.eval_force_command)}.mp4"
        )

        ckpt = _find_best_ckpt(log_dir, target_iter if target_iter >= 0 else None)
        if ckpt:
            print(f"[EVAL] Using checkpoint: {ckpt}")
        else:
            print("[EVAL][WARN] No checkpoint found; proceeding without load.")

        _eval_render_clip(
            args_cli.task,
            env_cfg,
            agent_cfg,
            ckpt,
            out_path,
            steps=args_cli.eval_steps,
            force_command=args_cli.eval_force_command,
            force_command_name=args_cli.eval_force_command_name,
        )
        return  # IMPORTANT: do not fall through to training

    # ---------------- TRAINING PATH (UNCHANGED SAVE/PROGRESS) ----------------
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed / device
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment — IMPORTANT: keep training headless even if --video
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # (Removed RecordVideo wrapper to avoid VRAM spikes)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint. --resume_log_root allows v1.4.6 Hardware
        # to warm-start from the v5s3 Stand experiment root while logging into a
        # clean v1.4.6 experiment root.
        resume_log_root_path = log_root_path
        if args_cli.resume_log_root:
            resume_log_root_path = os.path.abspath(os.path.expanduser(args_cli.resume_log_root))
        resume_path = get_checkpoint_path(resume_log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if args_cli.policy_only_warm_start:
            _load_policy_only_warm_start(runner, resume_path, agent_cfg.device)
        else:
            # load previously trained model, including optimizer/critic state
            runner.load(resume_path)

    # dump the configuration into log-directory (unchanged)
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # ---------- start checkpoint watcher (videos right after each save) ----------
    stop_evt = threading.Event()
    watcher_thread = None
    if args_cli.video:
        print("[INFO] train_eval video mode: training stays headless; MP4s are rendered from checkpoints.")
        print(f"[INFO] Videos will be written under: {os.path.join(log_dir, 'videos')}")
        print(f"[INFO] Current RSL-RL save_interval: {getattr(agent_cfg, 'save_interval', 'unknown')} iterations")
        watcher_thread = threading.Thread(
            target=_checkpoint_watcher_thread,
            args=(
                args_cli.task,
                env_cfg,
                agent_cfg,
                log_dir,
                args_cli.video_length,
                args_cli.device,
                stop_evt,
                hydra_args,
                args_cli.eval_force_command,
                args_cli.eval_force_command_name,
            ),
            daemon=True,
        )
        watcher_thread.start()

    # run training (single call — keeps original save/progress dashboard behavior)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # Stop the watcher and do one final sweep. This makes short smoke tests reliable:
    # if training finishes before the watcher sees the last checkpoint, the final
    # sweep still renders the missing video. If no checkpoint was written because
    # save_interval > max_iterations, save one final checkpoint first.
    if args_cli.video:
        stop_evt.set()
        if watcher_thread is not None:
            watcher_thread.join()
        _ensure_at_least_one_checkpoint(runner, log_dir, agent_cfg.max_iterations)
        rendered = _render_missing_checkpoint_videos(
            args_cli.task,
            log_dir,
            args_cli.video_length,
            args_cli.device,
            extra_hydra_args=hydra_args,
            force_command=args_cli.eval_force_command,
            force_command_name=args_cli.eval_force_command_name,
        )
        if rendered == 0:
            print("[INFO] No new videos were rendered in the final sweep; they may already exist.")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()