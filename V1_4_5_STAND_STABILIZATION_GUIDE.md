# Berkeley Humanoid Lite v1.4.5 Stand Stabilization

This branch keeps the v1.4.5 contract-v4/vector-residual work, but adds a safer
Stand-v5 stabilization path after the first athletic Stand run learned a visibly
right-leaning, right-knee/right-ankle braced crouch.

## New tasks

```text
Velocity-Lilgreen-Stand-ST3215-Loaded-v5s
Velocity-Lilgreen-Hardware-ST3215-Loaded-v5s
```

The original v5 tasks remain registered and reproducible.

## What changed

The stabilized profile keeps the larger sagittal residual authority from v1.4.5,
but moderates the default stance:

```text
hip_pitch    -0.24 rad
knee_pitch    0.62 rad
ankle_pitch  -0.22 rad
```

Full action-order q_default:

```text
[0.00, 0.00, -0.24, 0.62, -0.22, 0.00,
 0.00, 0.00, -0.24, 0.62, -0.22, 0.00]
```

The residual vector is unchanged from v1.4.5 so Hardware gait training still has
extra hip/knee/ankle pitch range:

```text
[0.24, 0.16, 0.42, 0.58, 0.48, 0.26,
 0.24, 0.16, 0.42, 0.58, 0.48, 0.26]
```

## Height targets

The stabilized Stand target is set directly into the visually identified sweet spot:

```text
standing COM target: 0.435 m
moving COM target:   0.415 m
```

The standing height reward uses a broad enough standard deviation to behave like a
safe band, not a hard squat target.

## Stand-only stabilization terms

The new Stand-v5s task adds stand-only terms to reduce the right-leg brace without
training a gait-time knee-symmetry habit:

- standing sagittal-chain soft-torque utilization penalty
- standing foot contact force balance penalty
- standing COM-over-feet penalty
- moderate q_default posture retention
- gentle interval base-velocity perturbations

These are intentionally tied to the Stand-v5s task. Hardware-v5s can still learn
asymmetrical gait phases.

## Recommended run

```bash
python scripts/rsl_rl/train_eval.py \
  --task Velocity-Lilgreen-Stand-ST3215-Loaded-v5s \
  --num_envs 4096 \
  --max_iterations 5000 \
  --seed 42 \
  --run_name stand_st3215_loaded_v145s_stabilized_seed42 \
  --headless
```

Analyze at:

```text
model_500.pt
model_1000.pt
model_2500.pt
model_4999.pt or model_5000.pt
```

Use the existing v1.4.5 analyzer with the v5s task:

```bash
python scripts/rsl_rl/analyze_policy_v1_4_5.py \
  --task Velocity-Lilgreen-Stand-ST3215-Loaded-v5s \
  --load_run <stand-v145s-run-folder> \
  --checkpoint model_2500.pt \
  --num_envs 256 \
  --steps 2000 \
  --headless
```

## Go / no-go for Hardware-v5s

Move to Hardware-v5s only if the stand checkpoint is visibly lower than old v1.4,
not deeply squatted, and no longer leaning/bracing on one side. The desired result
is roughly:

```text
COM height: 0.43-0.44 m
knees slightly bent, not locked and not deep crouched
hard torque preferably below ~10% global
no single sagittal joint near continuous hard torque
no visible right-leg lean
```

Then continue Hardware:

```bash
python scripts/rsl_rl/train_eval.py \
  --task Velocity-Lilgreen-Hardware-ST3215-Loaded-v5s \
  --num_envs 4096 \
  --max_iterations 5000 \
  --seed 42 \
  --run_name hardware_st3215_loaded_v145s_from_stand5000_seed42 \
  --resume \
  --load_run <stand-v145s-run-folder> \
  --checkpoint model_5000.pt \
  --headless
```
