# Berkeley Humanoid Lite v1.4.1 Hardware Curriculum

v1.4.1 is a targeted Hardware-only recovery patch after the first v1.4.0
Stand→Hardware continuation.  It preserves the successful v1.4.0
`Velocity-Lilgreen-Stand-ST3215-Loaded-v0` baseline and the v1.4.0 actuator
model, then adds a slower Hardware curriculum under a new task id:

```text
Velocity-Lilgreen-Hardware-ST3215-Loaded-v1
```

The original v1.4.0 Hardware task remains available and reproducible:

```text
Velocity-Lilgreen-Hardware-ST3215-Loaded-v0
```

## What changed

The policy/deployment contract did not change:

```text
45-D actor observation
12-D raw action
action contract v3
q_target = q_default + 0.20 * clamp(raw_action, -1, +1)
measured physical joint-limit clip
v1.4.0 Stage-A + Stage-B loaded ST3215 actuator model
```

v1.4.1 changes only Hardware curriculum and Hardware-specific reward shaping.

## Slower Hardware curriculum

The new stage boundaries are policy/environment steps. With the current 64-step
PPO rollout, they correspond approximately to +1000, +3000 and +5000 Hardware
training iterations.

| Stage | Approx. Hardware iterations | Standing envs | vx m/s | vy m/s | yaw rad/s | root reset | joint reset | push | mass scale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0–1000 | 85% | ±0.12 | ±0.05 | ±0.15 | ±0.06 | ±0.02 rad | 0.00 | 0.97–1.03 |
| 1 | 1000–3000 | 70% | ±0.25 | ±0.10 | ±0.30 | ±0.10 | ±0.03 rad | 0.00 | 0.95–1.05 |
| 2 | 3000–5000 | 55% | ±0.40 | ±0.18 | ±0.55 | ±0.16 | ±0.04 rad | 0.05 | 0.93–1.07 |
| 3 | 5000+ | 45% | ±0.60 | ±0.25 | ±0.80 | ±0.22 | ±0.05 rad | 0.10 | 0.92–1.08 |

This intentionally does not reach the old full v1.4.0 Hardware envelope during the
first +5000 run.

## Reward changes

Compared with v1.4.0 Hardware-ST3215-Loaded-v0:

```text
raw_action_excess_l2        -0.080 -> -0.180
soft_torque_utilization     -0.015 -> -0.018
new knee_soft_torque term    -0.015 above 65% of ST3215 peak torque

stand_base_xy_speed         -1.50  -> -2.20
stand_yaw_rate              -0.35  -> -0.50
stand_default_pose          -0.40  -> -0.75
stand_base_height            0.75  ->  1.00
stand_both_feet_contact      0.75  ->  1.00
stand_feet_slide            -0.80  -> -1.00
```

The goal is to retain the Stand checkpoint's posture and low-drift behavior while
learning low/medium-speed forward, reverse, strafe, and turning commands.

## Quick diagnostic from model_6500.pt

Use this to test whether the best v1.4.0 Hardware checkpoint can be rescued under
the safer curriculum. Replace `<hardware-v140-run-folder>` with the run folder that
contains `model_6500.pt`.

```bash
python scripts/rsl_rl/train_eval.py \
  --task Velocity-Lilgreen-Hardware-ST3215-Loaded-v1 \
  --num_envs 4096 \
  --max_iterations 1000 \
  --seed 42 \
  --run_name hardware_st3215_loaded_v141_rescue_from_6500_seed42 \
  --resume \
  --load_run <hardware-v140-run-folder> \
  --checkpoint model_6500.pt \
  --headless
```

The v1.4.1 runner intentionally keeps the same experiment root,
`lilgreen_v1_4_0_st3215_loaded`, so existing v1.4.0 checkpoints can be loaded
without copying files. Use the `v141` run name suffix to keep the new run obvious.

Analyze the resulting checkpoint with the same v1.4 analyzer:

```bash
python scripts/rsl_rl/analyze_policy_v1_4_0.py \
  --task Velocity-Lilgreen-Hardware-ST3215-Loaded-v1 \
  --load_run <v141-rescue-run-folder> \
  --checkpoint model_7500.pt \
  --num_envs 256 \
  --steps 2000 \
  --headless
```

Depending on how RSL-RL numbers resumed checkpoints in your local install, the
final model may be `model_7500.pt` or a model near the end of the new run. Use the
latest checkpoint in the v1.4.1 rescue run if needed.

## Diagnostic pass targets

For the +1000 diagnostic, the main pass/fail questions are:

```text
stable standing all conditions       > 90%
posture-conforming standing          improves versus model_6500 or remains acceptable
global hard torque occupancy         < 5–6%
right/left knee hard torque          no return to 40–50% standing occupancy
raw action excess                    < 12–15%
zero-command drift                   < 0.06 m/s
loaded envelope active fraction      selective, not runaway
policy-to-drive lag                  not sharply worse than v1.4 Stand/Hardware 6500
```

If the rescue run stays healthy, the next clean experiment is a fresh v1.4.1 Hardware
run from the v1.4.0 Stand `model_5000.pt`.
