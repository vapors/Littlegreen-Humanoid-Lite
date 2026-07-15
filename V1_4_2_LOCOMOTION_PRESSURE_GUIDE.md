# Berkeley Humanoid Lite v1.4.2 — Locomotion-Pressure Hardware Curriculum

## Purpose

v1.4.2 keeps the successful v1.4.0 loaded-ST3215 Stand baseline and the v1.4.0/v1.4.1 actuator model unchanged, but changes the Hardware continuation so the policy is asked to move sooner and more clearly.

The policy/deployment contract remains unchanged:

- 45-D actor observation
- 12-D action output
- action contract v3
- `q_target = q_default + 0.20 * bounded_action`
- measured physical joint-limit clipping
- v1.4 loaded ST3215 Stage-A + Stage-B actuator response model

## New task

```text
Velocity-Lilgreen-Hardware-ST3215-Loaded-v2
```

This task is additive. The previous tasks remain registered:

```text
Velocity-Lilgreen-Hardware-ST3215-Loaded-v0  # v1.4.0 original loaded Hardware
Velocity-Lilgreen-Hardware-ST3215-Loaded-v1  # v1.4.1 slow-retention Hardware
Velocity-Lilgreen-Hardware-ST3215-Loaded-v2  # v1.4.2 locomotion-pressure Hardware
```

## Curriculum changes

v1.4.2 does not split the task into command bins. It keeps a continuous `[vx, vy, yaw]` command distribution but asks for more locomotion earlier.

| Stage | Approx. Hardware iterations | Standing envs | vx | vy | yaw | Push |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0–500 | 50% | ±0.22 | ±0.08 | ±0.25 | 0.00 |
| 1 | 500–2000 | 38% | ±0.40 | ±0.15 | ±0.45 | 0.00 |
| 2 | 2000–4000 | 30% | ±0.60 | ±0.25 | ±0.75 | 0.05 |
| 3 | 4000+ | 25% | ±0.80 | ±0.35 | ±1.00 | 0.10 |

The intent is to avoid the v1.4.1 failure mode where standing retention remained strong, but the policy did not appear to commit to foot lift or stepping.

## Reward changes

Relative to v1.4.1, v1.4.2:

- sharpens linear velocity tracking: `weight=3.0`, `std=0.35`
- sharpens yaw tracking slightly: `weight=1.6`, `std=0.40`
- reduces standing stickiness:
  - `stand_base_xy_speed = -1.35`
  - `stand_default_pose = -0.45`
  - `stand_base_height = 0.85`
  - `stand_both_feet_contact = 0.75`
- retains action/torque safeguards:
  - `raw_action_excess_l2 = -0.160`
  - `soft_torque_utilization = -0.016`
  - `knee_soft_torque_utilization = -0.012`

The foot-air-time command threshold is unchanged.

## Recommended training command

Resume from the successful v1.4.0 Stand checkpoint:

```bash
python scripts/rsl_rl/train_eval.py \
  --task Velocity-Lilgreen-Hardware-ST3215-Loaded-v2 \
  --num_envs 4096 \
  --max_iterations 5000 \
  --seed 42 \
  --run_name hardware_st3215_loaded_v142_from_stand5000_seed42 \
  --resume \
  --load_run 2026-07-10_19-48-48_stand_st3215_loaded_v140_fresh_seed42 \
  --checkpoint model_5000.pt \
  --headless
```

## Forced-command evaluation/video

`train_eval.py` now supports fixed-command checkpoint rendering:

```bash
python scripts/rsl_rl/train_eval.py \
  --task Velocity-Lilgreen-Hardware-ST3215-Loaded-v2 \
  --headless \
  --eval_only \
  --eval_log_dir logs/rsl_rl/lilgreen_v1_4_0_st3215_loaded/<run-folder> \
  --eval_steps 500 \
  --eval_tag fwd025_005500 \
  --eval_target_iter 5500 \
  --eval_force_command 0.25 0.0 0.0
```

Useful forced clips:

```bash
# forward
--eval_force_command 0.25 0.0 0.0

# reverse
--eval_force_command -0.20 0.0 0.0

# left strafe
--eval_force_command 0.0 0.12 0.0

# right strafe
--eval_force_command 0.0 -0.12 0.0

# turn in place
--eval_force_command 0.0 0.0 0.35
```

During forced-command eval, the script overrides both:

1. the policy observation command; and
2. the environment command-manager value used by the command marker.

This keeps the command marker and the policy input aligned.

## Joystick playback update

`play_joystick.py` now also overrides the environment command manager by default, so the command arrow/marker matches the command sent to the policy.

```bash
python scripts/rsl_rl/play_joystick.py \
  --task Velocity-Lilgreen-Hardware-ST3215-Loaded-v2 \
  --load_run <run-folder> \
  --checkpoint model_5500.pt \
  --num_envs 1 \
  --video \
  --video_length 500
```

Then write commands to:

```bash
echo "0.25 0.0 0.0" > /tmp/joystick_cmd.txt
```

Use `--no_command_manager_override` only for debugging old behavior where the policy command and visual marker may differ.
