# Berkeley Humanoid Lite v1.4.7 Phase-Guided Alternating Gait

Purpose: move beyond v1.4.6 anti-planted locomotion, which forced motion but produced one-sided hopping/falling rather than grounded alternating steps.

## New task

- `Velocity-Lilgreen-Hardware-ST3215-Loaded-v7`

## Contract notes

This is an intentional observation-contract change for training/deployment:

- Action contract remains v4 / vector residual from v1.4.5s3.
- Policy action output remains 12-D.
- Policy observation changes from 45-D to 47-D by appending gait phase `[sin(phase), cos(phase)]`.
- Deployment must not run v1.4.7 policies on a runtime that assumes the 45-D v4 observation.

## Main changes

- Adds a simple per-episode gait clock, period 0.72 s.
- Adds phase-conditioned contact targets:
  - half-cycle A: left stance / right swing;
  - half-cycle B: right stance / left swing;
  - brief double support allowed near transitions.
- Adds phase-guided swing clearance, swing velocity, and foot placement rewards.
- Adds long-air-hold penalty and persistent no-support termination to prevent the v1.4.6 hopping/falling exploit.
- Softens v1.4.6 no-progress termination so the phase scaffold has time to organize.
- Keeps continuous command distributions; no command bins were added.
- Keeps non-standing command floor at 0.22 m/s.

## Recommended run

```bash
python scripts/rsl_rl/train_eval.py \
  --task Velocity-Lilgreen-Hardware-ST3215-Loaded-v7 \
  --num_envs 4096 \
  --max_iterations 8000 \
  --seed 42 \
  --run_name hardware_st3215_loaded_v147_phase_guided_from_v145s3_stand3500_seed42 \
  --resume \
  --resume_log_root logs/rsl_rl/lilgreen_v1_4_5_st3215_stabilized_forward_s3 \
  --load_run 2026-07-13_12-10-53_stand_st3215_loaded_v145s3_forward_com_height460_seed42 \
  --checkpoint model_3500.pt \
  --policy_only_warm_start \
  --headless
```

Because v1.4.7 expands the observation input from 45 to 47 dims, `train_eval.py` now partially expands the actor input layer during policy-only warm start. It copies the old 45-D columns and leaves the two new gait-phase columns freshly initialized.

## First analysis checkpoints

- `model_500`
- `model_1000`
- `model_1500`
- `model_2000`
- `model_3000`
- `model_5000`
- `model_8000`

Use forced commands around `vx=0.25`, `vx=0.40`, and `vx=0.50`.

## Desired behavior

- Double support should be transitional, not continuously planted.
- No-support fraction should stay very low.
- Left/right foot-air fractions should alternate and remain balanced over time.
- Right knee/right ankle hard saturation should no longer become the dominant solution.
- Actual forward velocity should become nonzero under forced `vx` commands.
