# Track 1 Handoff — 12-Joint ST3215 Actuator Identification Dataset v2

## Scope

This package contains 12 nominal v2.4.2 direct-path ST3215 identification runs and 120 step trials. The final corrected left-knee run (`20260710T022812Z...`) replaces the original mechanically impaired knee run. All nominal runs use `max_envelope_fixed_0_0`, 50 Hz target updates, and suspended unloaded support.

## Revision-critical finding

Removing the interfering long left-knee servo-horn screw materially changed the knee response. The final left knee now tracks the right knee closely at large amplitude:

- +0.20 rad peak velocity: left 0.913 rad/s vs right 0.890 rad/s.
- +0.20 rad static gain: left 0.951 vs right 0.943.
- The previous extreme load/current anomaly is not used as nominal training variation. Historical impaired runs remain in the comparison CSV for fault/stress studies only.

The final left knee still shows direction-dependent overshoot and some long settling tails. Preserve that transient behavior as real actuator variation, but do not train against the old mechanically obstructed under-response.

## Global timing envelope (0.10–0.20 rad trials)

| Metric | p05 | median | p95 |
|---|---:|---:|---:|
| ROS publish → driver receipt | 0.240 ms | 0.344 ms | 0.540 ms |
| Driver receipt → SyncWrite start | 0.81 ms | 10.28 ms | 19.47 ms |
| Write → first encoder motion | 43.4 ms | 62.6 ms | 69.0 ms |
| Write → sustained motion | 59.4 ms | 65.3 ms | 85.0 ms |
| Equivalent first-order τ proxy | 53.4 ms | 71.1 ms | 82.7 ms |

## Joint-family training summary

| Family | v_peak @0.20 mean | sustained onset median | τ_eq median | 1-count settle median | large static gain | max overshoot | bilateral v asym median |
|---|---:|---:|---:|---:|---:|---:|---:|
| ankle_pitch | 0.905 rad/s | 66.3 ms | 68.1 ms | 319 ms | 0.964 | 0.0% | 1.1% |
| ankle_roll | 0.915 rad/s | 63.6 ms | 71.0 ms | 331 ms | 0.973 | 0.0% | 2.4% |
| hip_pitch | 0.902 rad/s | 70.3 ms | 73.7 ms | 351 ms | 0.966 | 15.0% | 1.9% |
| hip_roll | 0.916 rad/s | 63.6 ms | 75.6 ms | 328 ms | 0.978 | 0.0% | 2.6% |
| hip_yaw | 0.878 rad/s | 65.0 ms | 73.8 ms | 808 ms | 0.963 | 16.6% | 3.3% |
| knee_pitch | 0.915 rad/s | 65.8 ms | 64.9 ms | 306 ms | 0.959 | 15.0% | 3.3% |

## Training use

1. Keep transport and actuator dynamics separate: near-zero ROS delivery, 0–20 ms bus phase, then physical onset and response dynamics.
2. Use per-family/per-joint `tau_eq` as a response proxy, not physical damping.
3. Use amplitude→peak-velocity curves; hard saturation was not reached at 0.20 rad.
4. Model several-count residual error and center hysteresis as small-signal nonlinearity.
5. Preserve hip-yaw overshoot and long settling.
6. Use the final corrected left-knee run as nominal. Keep the original and intermediate knee runs only as optional fault/stress cases.
7. Do not invent N·m/rad stiffness or N·m·s/rad damping from these unloaded tests. Run known-load hold and known-inertia transient tests for those physical quantities.

## Files

- `trial_metrics.csv`: all 120 nominal trials.
- `joint_summary.csv`: 12 nominal joint summaries.
- `joint_family_summary.csv`: pooled family metrics.
- `bilateral_asymmetry.csv`: left/right comparison by family and amplitude.
- `track1_actuator_model.yaml` / `.json`: machine-readable training handoff.
- `quality_report.csv`: acquisition integrity and timing.
- `source_manifest.csv`: nominal source provenance.
- `excluded_runs_manifest.csv`: mechanically impaired knee runs excluded from nominal data.
- `left_knee_revision_comparison.csv`: original → partial fix → final fix → right reference comparison.
