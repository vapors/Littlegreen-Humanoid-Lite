# Berkeley Humanoid Lite v1.4.3 — Move-Now Hardware Curriculum

v1.4.3 is an additive Hardware task intended to answer the specific failure observed in v1.4.1/v1.4.2: the policy stayed upright but appeared reluctant to lift a foot, often leaning/crouching/bracing under commanded motion instead of stepping.

## New task

```text
Velocity-Lilgreen-Hardware-ST3215-Loaded-v3
```

The following remain unchanged from the v1.4 Stand baseline and v1.4 loaded actuator family:

```text
45-D observation
12-D action
action contract v3
q_target = q_default + 0.20 * bounded_action
physical joint limits
q_default
nominal COM height
loaded ST3215 data source
no command-bin curriculum
```

## Main changes

### 1. Anti-bracing gait pressure

New movement-only rewards activate when commanded horizontal speed is clearly nonzero:

```text
moving_velocity_along_command
moving_no_progress_l1
moving_single_support_time
moving_double_support_penalty
swing_foot_clearance
swing_foot_velocity_along_command
```

These are intended to reward actual translation, single-support phases, and swing-foot lift while discouraging double-support bracing under nonzero movement commands.

### 2. Faster training actuator response

The ONNX/action contract is unchanged, but v1.4.3 uses a less conservative training response proxy:

```text
response_delay_scale: 0.60
tau proxy scale:      0.78
Stage-A velocity curve scale: 1.12
velocity randomization:        1.04–1.18
loaded velocity scale:          1.04–1.16
```

This is a training experiment to test whether the prior response model was conservative enough to make the policy prefer bracing/shuffling over stepping.

### 3. Continuous move-now curriculum

The curriculum still samples continuous velocity commands; it does not split into command bins.

| Stage | Approx. Hardware iterations | Standing envs | vx | vy | yaw | Push |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0–500 | 45% | ±0.25 | ±0.08 | ±0.25 | 0.00 |
| 1 | 500–2000 | 35% | ±0.45 | ±0.16 | ±0.50 | 0.00 |
| 2 | 2000–4000 | 28% | ±0.65 | ±0.26 | ±0.75 | 0.02 |
| 3 | 4000+ | 22% | ±0.85 | ±0.35 | ±1.00 | 0.06 |

Reset disturbance and mass randomization are kept gentler than the v1.4.2 locomotion-pressure run so the new gait rewards, not random disturbance survival, drive the experiment.

## Recommended run from the v1.4 Stand checkpoint

```bash
python scripts/rsl_rl/train_eval.py \
  --task Velocity-Lilgreen-Hardware-ST3215-Loaded-v3 \
  --num_envs 4096 \
  --max_iterations 5000 \
  --seed 42 \
  --run_name hardware_st3215_loaded_v143_from_stand5000_seed42 \
  --resume \
  --load_run 2026-07-10_19-48-48_stand_st3215_loaded_v140_fresh_seed42 \
  --checkpoint model_5000.pt \
  --headless
```

## Forced-command analysis with leg-lift metrics

Use the new analyzer for forced-command quantitative checks:

```bash
python scripts/rsl_rl/analyze_policy_v1_4_3.py \
  --task Velocity-Lilgreen-Hardware-ST3215-Loaded-v3 \
  --load_run <v143-run-folder> \
  --checkpoint model_5500.pt \
  --num_envs 256 \
  --steps 2000 \
  --eval_force_command 0.25 0.0 0.0 \
  --headless
```

Important new output keys include:

```text
moving_command_fraction
single_support_fraction
moving_single_support_fraction
moving_double_support_fraction
left_foot_air_fraction
right_foot_air_fraction
moving_max_air_time_mean_s
moving_swing_clearance_mean_m
moving_swing_clearance_p95_m
moving_swing_clearance_max_m
moving_swing_foot_forward_velocity_mean_mps
foot_lift_count_total
foot_lift_count_per_env_second
```

## Suggested forced videos

```bash
# forward moderate
--eval_force_command 0.25 0.0 0.0

# forward stronger
--eval_force_command 0.40 0.0 0.0

# reverse
--eval_force_command -0.20 0.0 0.0

# turn in place
--eval_force_command 0.0 0.0 0.35

# left/right strafe
--eval_force_command 0.0 0.12 0.0
--eval_force_command 0.0 -0.12 0.0
```

## First checkpoints to inspect

```text
model_5500.pt
model_6000.pt
model_6500.pt
model_7500.pt
```

The first go/no-go question is no longer only stable standing. The key question is whether forced nonzero commands produce measurable single support, foot air time, swing clearance, and actual command-aligned translation.
