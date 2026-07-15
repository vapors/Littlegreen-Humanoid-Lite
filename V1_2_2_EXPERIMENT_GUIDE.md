# v1.2.2 Stand-v0 experiment guide

## 1. Smoke test

```bash
python scripts/rsl_rl/train_eval.py \
  --task Velocity-Lilgreen-Stand-v0 \
  --num_envs 64 \
  --max_iterations 5 \
  --run_name stand_v122_smoke \
  --headless
```

Confirm the reward table includes:
- `soft_torque_utilization`
- `stand_base_height`

Confirm Curriculum Manager contains:
- `policy_diagnostics`

Check the first diagnostics block for:
- `base_com_height_mean_m`
- `standing_base_com_height_mean_m`
- `standing_base_com_height_error_mean_m`

The first-pass target is 0.656 m. A small error is expected; if the standing COM height is materially different, adjust `desired_height` before the long run.

## 2. Main Stand run

Recommended initial qualification run:

```bash
python scripts/rsl_rl/train_eval.py \
  --task Velocity-Lilgreen-Stand-v0 \
  --num_envs 4096 \
  --max_iterations 1500 \
  --seed 42 \
  --run_name stand_v122_seed42 \
  --headless
```

1000 iterations is acceptable for a faster first comparison. 1500 is preferred if the run remains stable.

## 3. What to watch during training

Primary action health:
- `raw_action_excess_fraction`
- `bounded_saturation_fraction`
- `target_limit_fraction`

Primary torque health:
- `torque_over_soft_fraction`
- `torque_limit_fraction`
- per-joint torque soft/limit fractions

Standing quality:
- `standing_all_conditions_fraction`
- individual standing gate fractions
- `zero_command_xy_drift_mean_mps`
- `standing_foot_slip_mean_mps`
- base-COM height and error

The individual gate fractions are intended to explain why strict 10 s / 20 s continuous standing success may remain at zero.

## 4. Offline analysis

```bash
python scripts/rsl_rl/analyze_policy_v1_2.py \
  --task Velocity-Lilgreen-Stand-v0 \
  --load_run <stand_v122_run_directory> \
  --checkpoint <checkpoint.pt> \
  --num_envs 256 \
  --steps 1500 \
  --headless
```

Compare against the v1.2 baseline:

- raw action excess: 29.5%
- target saturation: 29.6%
- torque limit fraction: 8.9%
- zero-command drift: 0.0489 m/s
- standing foot slip: 0.0263 m/s
- linear tracking RMSE: 0.0709 m/s
- yaw tracking RMSE: 0.1959 rad/s

Desired direction after v1.2.2:
- raw excess: below 15%, then toward 10%
- torque limit fraction: below 5%
- drift: at or below 0.05 m/s
- foot slip: at or below 0.03 m/s
- tracking error: no material regression
- timeout completions increasingly dominate falls

Do not transfer to Hardware-v0 solely because mean reward rises. Evaluate policy action and torque health, standing gates, drift, slip, and deterministic rollout statistics together.
