# Berkeley Humanoid Lite v1.4.4 Alternating-Step Hardware Curriculum

New task:

```bash
Velocity-Lilgreen-Hardware-ST3215-Loaded-v4
```

This branch keeps the v1.4 policy contract unchanged: 45-D observation, 12-D
action, action contract v3, q_default + 0.20 rad residual, physical joint limits,
and the loaded ST3215 actuator model data source. It is additive; v1.4.0 through
v1.4.3 tasks remain registered.

## Purpose

v1.4.3 proved that the robot can lift a foot, but the easiest policy exploit was
to hold one foot up and brace on the other side. v1.4.4 shifts the reward target
from “lift a foot” to “alternate, place, and move with the command.”

## Main changes

- Adds local alternating single-support reward without introducing command bins.
- Adds command-aligned progress and swing-foot motion emphasis.
- Adds penalties for long one-foot holds, left/right air-time imbalance, and
  stance-foot sliding during single support.
- Adds a relaxed moving-height reward at nominal COM height minus 0.040 m, so the
  robot can bend its knees while moving. Standing height is also slightly relaxed
  but still close to the v1.4 nominal pose.
- Keeps the quicker actuator training response from v1.4.3, but backs it off
  slightly to reduce one-foot bracing exploits.

## Curriculum

| Stage | Hardware iterations | Standing envs | vx | vy | yaw | Push |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0-500 | 40% | +/-0.28 | +/-0.08 | +/-0.22 | 0.00 |
| 1 | 500-2000 | 30% | +/-0.50 | +/-0.16 | +/-0.45 | 0.00 |
| 2 | 2000-4000 | 25% | +/-0.70 | +/-0.26 | +/-0.70 | 0.02 |
| 3 | 4000+ | 20% | +/-0.90 | +/-0.36 | +/-0.95 | 0.05 |

## Run from v1.4 Stand model_5000

```bash
python scripts/rsl_rl/train_eval.py \
  --task Velocity-Lilgreen-Hardware-ST3215-Loaded-v4 \
  --num_envs 4096 \
  --max_iterations 5000 \
  --seed 42 \
  --run_name hardware_st3215_loaded_v144_from_stand5000_seed42 \
  --resume \
  --load_run 2026-07-10_19-48-48_stand_st3215_loaded_v140_fresh_seed42 \
  --checkpoint model_5000.pt \
  --headless
```

## Forced-command analysis

```bash
python scripts/rsl_rl/analyze_policy_v1_4_4.py \
  --task Velocity-Lilgreen-Hardware-ST3215-Loaded-v4 \
  --load_run <v144-run-folder> \
  --checkpoint model_5500.pt \
  --num_envs 256 \
  --steps 2000 \
  --eval_force_command 0.25 0.0 0.0 \
  --headless
```

Important new metrics:

- `left_lift_count_total`, `right_lift_count_total`
- `lift_balance_abs_fraction`
- `alternation_count_total`
- `same_side_repeat_count_total`
- `alternation_fraction_of_lift_transitions`
- `moving_air_time_imbalance_mean_s`
- `moving_long_air_time_fraction`

