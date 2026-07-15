# v1.4.5 Training Guide

Train a new standing checkpoint first because v1.4.5 changes q_default and uses a
per-joint residual vector.

## 1. Athletic stand baseline

```bash
python scripts/rsl_rl/train_eval.py \
  --task Velocity-Lilgreen-Stand-ST3215-Loaded-v5 \
  --num_envs 4096 \
  --max_iterations 5000 \
  --seed 42 \
  --run_name stand_st3215_loaded_v145_athletic_seed42 \
  --headless
```

Analyze:

```bash
python scripts/rsl_rl/analyze_policy_v1_4_5.py \
  --task Velocity-Lilgreen-Stand-ST3215-Loaded-v5 \
  --load_run <stand-v145-run-folder> \
  --checkpoint model_5000.pt \
  --num_envs 256 \
  --steps 2000 \
  --headless
```

## 2. Hardware continuation from v1.4.5 Stand

```bash
python scripts/rsl_rl/train_eval.py \
  --task Velocity-Lilgreen-Hardware-ST3215-Loaded-v5 \
  --num_envs 4096 \
  --max_iterations 5000 \
  --seed 42 \
  --run_name hardware_st3215_loaded_v145_from_stand5000_seed42 \
  --resume \
  --load_run <stand-v145-run-folder> \
  --checkpoint model_5000.pt \
  --headless
```

## 3. Forced-command gait checks

```bash
python scripts/rsl_rl/analyze_policy_v1_4_5.py \
  --task Velocity-Lilgreen-Hardware-ST3215-Loaded-v5 \
  --load_run <hardware-v145-run-folder> \
  --checkpoint model_5500.pt \
  --num_envs 256 \
  --steps 2000 \
  --eval_force_command 0.25 0.0 0.0 \
  --headless
```

Use forced commands such as:

```text
0.25  0.00  0.00   forward
0.40  0.00  0.00   stronger forward
-0.20 0.00  0.00   reverse
0.00  0.12  0.00   strafe
0.00  0.00  0.35   turn
```
