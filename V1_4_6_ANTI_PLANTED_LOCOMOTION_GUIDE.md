# Berkeley Humanoid Lite v1.4.6 Anti-Planted Locomotion

This branch keeps the v1.4.5s3 standing profile/contract and changes only the Hardware transition objective.

## New task

```text
Velocity-Lilgreen-Hardware-ST3215-Loaded-v6
```

No new Stand task is added. Use the best v5s3 Stand checkpoint, currently `model_3500.pt`, as the seed.

## Goal

The v5s3 Hardware run stayed planted under a forced forward command and shifted load onto the right knee/ankle. v1.4.6 makes that local optimum a losing strategy:

1. Policy-only warm start from Stand-v5s3: load actor/action std and observation normalization, reset critic and optimizer.
2. Moving no-progress timeout: under clear nonzero horizontal commands, episodes that fail to make command-aligned progress terminate after a short grace period.
3. Long double-support / planted-no-progress penalties: brief double-support transitions remain allowed, but standing planted under a movement command becomes expensive.

The command distribution remains continuous; no command bins are introduced. Non-standing horizontal commands are floored to 0.22 m/s.

## Recommended run

```bash
python scripts/rsl_rl/train_eval.py \
  --task Velocity-Lilgreen-Hardware-ST3215-Loaded-v6 \
  --num_envs 4096 \
  --max_iterations 5000 \
  --seed 42 \
  --run_name hardware_st3215_loaded_v146_antiplanted_from_v145s3_stand3500_seed42 \
  --resume \
  --resume_log_root logs/rsl_rl/lilgreen_v1_4_5_st3215_stabilized_forward_s3 \
  --load_run <stand-v145s3-run-folder> \
  --checkpoint model_3500.pt \
  --policy_only_warm_start \
  --headless
```

## First checkpoints to inspect

Analyze and render forced-command clips at:

```text
model_500   (+500 Hardware, policy-only checkpoint numbering restarts)
model_1000  (+1000 Hardware)
model_1500  (+1500 Hardware)
model_2000  (+2000 Hardware)
model_3000  (+3000 Hardware)
```

Start with forced forward `0.25`, `0.40`, and `0.50 m/s`. The key success signal is not perfect tracking yet; it is that double-support planted time falls and left/right stepping attempts replace right-leg bracing.

## Keep unchanged

- action contract v4 / vector residual profile
- v5s3 stabilized athletic q_default
- 0.460 m standing height target
- ST3215 loaded actuator model
- grounded/no-flight protections
