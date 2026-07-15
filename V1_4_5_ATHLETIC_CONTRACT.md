# Berkeley Humanoid Lite v1.4.5 Athletic Contract

v1.4.5 is a deployment-impacting Track 1 experiment.  It keeps the neural-network
interface shape unchanged, but changes the pose/action profile used by training and
exported policy metadata.

## New tasks

- `Velocity-Lilgreen-Stand-ST3215-Loaded-v5`
- `Velocity-Lilgreen-Hardware-ST3215-Loaded-v5`

Older v1.4.0-v1.4.4 tasks remain registered for reproducibility.

## Policy interface

- Observation dimension: 45
- Action dimension: 12
- Canonical joint order: unchanged
- Residual equation: `q_target = clip(q_default + clip(a_raw,-1,1) * residual_scale_rad, physical_lower, physical_upper)`

## Deployment-impacting changes

The v1.4.5 profile changes both `q_default` and `action_residual_scale_rad`:

```text
q_default athletic pitch chain:
hip_pitch   -0.30 rad
knee_pitch   0.78 rad
ankle_pitch -0.30 rad
```

Per-joint residual scales, in canonical action order:

```text
[0.24, 0.16, 0.42, 0.58, 0.48, 0.26,
 0.24, 0.16, 0.42, 0.58, 0.48, 0.26]
```

This gives substantially more walking authority to hip pitch, knee pitch, and
ankle pitch without widening hip yaw/roll as aggressively.

Exported v1.4.5 policies set:

```text
action_contract_version: 4
deployment_contract_profile: v1_4_5_athletic_vector_residual
```

Do not deploy these policies with a runtime that assumes the old scalar ±0.20 rad
v1.4.0-v1.4.4 residual profile.

## Height targets

The old nominal geometric COM height was about 0.4899 m. v1.4.5 intentionally
uses a lower athletic stance:

```text
stand target  = nominal - 0.045 m
moving target = nominal - 0.080 m
```

The goal is to encourage knee bend and reduce ankle-dominant tall bracing.
