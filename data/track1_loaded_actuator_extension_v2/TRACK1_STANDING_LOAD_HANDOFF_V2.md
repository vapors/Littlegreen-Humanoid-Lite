# Track 1 Handoff — Updated Loaded Standing Speed Sweep v2

## Executive conclusion

The updated standing-load campaign is materially more useful for Track 1 than the earlier 1 s floor-limited sweep.

The completed runs provide four distinct requested-speed conditions using the updated pose library:

- 1.0 rad/s
- 2.0 rad/s
- 3.0 rad/s
- 4.5 rad/s

The supplied archive does **not** contain a completed 4.0 rad/s run. One earlier 1.0 rad/s run aborted with stale joint feedback and used a different pose-library hash; it is excluded from all v2 fitted targets and summaries.

The strongest Track 1 result is now clearer:

1. **knee pitch** has the strongest loaded speed saturation and the most persistent stand-return effort asymmetry;
2. **hip pitch** shows strong speed-dependent tracking degradation and velocity saturation, but relatively little direction asymmetry;
3. **ankle pitch** is speed-sensitive, but the broad stand-return penalty seen in the earlier synthesis is not persistent across the new sweep;
4. roll families are comparatively mild;
5. the updated medium-crouch pose changes the transition geometry enough that the old and new dynamic sweeps should not be pooled blindly.

This v2 synthesis should supersede the v1 dynamic speed-response interpretation. The suspended single-joint model remains the nominal actuator model.

## Dataset quality

Completed runs:

| run_id                         |   requested_speed_rad_s |   cycles |   read_ok_fraction |   write_ok_fraction |   telemetry_drop_max |   cycle_work_us_p99 |   feedback_sweep_us_p99 | imu_available   |
|:-------------------------------|------------------------:|---------:|-------------------:|--------------------:|---------------------:|--------------------:|------------------------:|:----------------|
| 20260711T001844Z_standing_load |                  1.0000 |     3251 |             1.0000 |              1.0000 |                    0 |           8039.5370 |               7834.2065 | False           |
| 20260711T002012Z_standing_load |                  2.0000 |     2596 |             1.0000 |              1.0000 |                    0 |           7908.7704 |               7691.1602 | False           |
| 20260711T002123Z_standing_load |                  3.0000 |     2559 |             1.0000 |              1.0000 |                    0 |           8523.7953 |               8352.2976 | False           |
| 20260711T002435Z_standing_load |                  4.5000 |     2544 |             1.0000 |              1.0000 |                    0 |           8082.6610 |               7881.1369 | False           |

All completed runs had:

- zero telemetry drops;
- 100% aggregate `read_ok`;
- 100% `write_ok`;
- the same updated pose-library SHA-256;
- no usable IMU samples.

The IMU limitation remains: balance disturbance and tether engagement cannot be quantified from this archive.

## Transition-floor interpretation

With the 0.25 s floor:

- 1.0 rad/s: no dynamic ladder segments hit the floor;
- 2.0 rad/s: no dynamic ladder segments hit the floor;
- 3.0 rad/s: 4 of 6 dynamic segments hit the floor;
- 4.5 rad/s: 4 of 6 dynamic segments hit the floor.

Track 1 should use the logged `qref_peak_rad_s` for model fitting, not the CLI speed label by itself.

## Loaded speed-response result

Median family response across both movement directions:

| Family | qref peak 1.0 | qdot 1.0 | qref peak 4.5 | qdot 4.5 | gain 1.0→4.5 | RMS error growth |
|---|---:|---:|---:|---:|---:|---:|
| ankle_pitch | 0.266 | 0.222 | 0.504 | 0.373 | 0.952→0.618 | 2.86× |
| ankle_roll | 0.194 | 0.134 | 0.498 | 0.333 | 0.861→0.644 | 2.63× |
| hip_pitch | 0.898 | 0.802 | 2.074 | 0.910 | 0.892→0.429 | 2.15× |
| hip_roll | 0.233 | 0.188 | 0.543 | 0.393 | 0.885→0.723 | 2.37× |
| hip_yaw | 0.078 | 0.099 | 0.116 | 0.093 | 0.978→0.501 | 0.91× |
| knee_pitch | 1.015 | 0.769 | 2.667 | 0.959 | 0.865→0.434 | 3.34× |

The most important patterns are:

- **knee pitch:** median reference peak rises 2.63× from the 1.0 to 4.5 run, but measured peak qdot rises only 1.25×; tracking RMS grows 3.34× and median speed gain falls from 0.865 to 0.434.
- **hip pitch:** median reference peak rises 2.31×, measured qdot only 1.14×; tracking RMS grows 2.15× and speed gain falls from 0.892 to 0.429.
- **ankle pitch:** speed gain falls from 0.952 to 0.618 while tracking RMS grows 2.86×.
- **hip roll and ankle roll:** degradation exists but is less severe.
- **hip yaw:** use cautiously because commanded movement amplitudes are smaller and less uniformly informative.

These are strong validation targets for the v1.3.0 actuator-aware task family.

## Empirical loaded velocity envelopes

A simple piecewise fit was calculated:

`v_meas = min(gain * v_ref, vmax)`

| Family | Crouch gain / vmax | Return gain / vmax |
|---|---:|---:|
| ankle_pitch | 0.836 / 0.770 | 0.816 / 0.700 |
| ankle_roll | 0.707 / 0.722 | 0.710 / 0.758 |
| hip_pitch | 0.556 / 1.701 | 0.569 / 1.730 |
| hip_roll | 0.853 / 0.406 | 0.772 / 0.395 |
| hip_yaw | 0.774 / 0.644 | 0.708 / 0.494 |
| knee_pitch | 0.709 / 1.178 | 0.715 / 1.163 |

These values are **empirical loaded trajectory envelopes**, not intrinsic ST3215 free-speed limits or torque-speed constants. The full fit diagnostics and RMSE values are in `empirical_velocity_saturation_fit.csv`.

Recommended use: calibrate or validate family response limits and randomization, rather than replacing the suspended actuator model directly.

## Direction asymmetry update

### Knee pitch remains the strongest direction-conditioned effect

| Requested speed | Return/crouch RMS | Return/crouch current | Return/crouch load |
|---:|---:|---:|---:|
| 1.0 | 1.70× | 6.50× | 2.66× |
| 2.0 | 1.27× | 8.17× | 1.89× |
| 3.0 | 1.15× | 8.00× | 1.58× |
| 4.5 | 1.14× | 7.86× | 1.58× |

The tracking-error ratio narrows at high speed because both directions become strongly rate-limited, but the current and load proxies remain strongly asymmetric. This supports a modest loaded stand-return modifier for the knee family.

### Hip pitch is mostly speed-conditioned, not direction-conditioned

Across the sweep, crouch and return tracking RMS are close compared with the large speed effect. Track 1 should primarily tune hip-pitch loaded velocity response and lag rather than impose a strong direction penalty.

### Ankle pitch conclusion is revised

The earlier v1 synthesis suggested a broad stand-return penalty. In the updated-pose sweep:

- at 1.0 rad/s, return tracking RMS is about 1.33× crouch and current about 1.67×;
- from 2.0 to 4.5 rad/s, the return/crouch tracking ratio is about 0.88–0.92.

Therefore the stronger conclusion is now: **ankle pitch is speed-sensitive and saturating, but its direction penalty is not robust across the updated sweep**.

## Updated medium-crouch pose matters

The medium-crouch target changed substantially relative to the v1 synthesis. The largest target changes are:

- `leg_right_hip_pitch_joint`: -0.604 rad
- `leg_left_hip_pitch_joint`: -0.512 rad
- `leg_left_hip_roll_joint`: -0.272 rad
- `leg_left_hip_yaw_joint`: -0.207 rad
- `leg_right_knee_pitch_joint`: +0.121 rad

The maximum absolute joint-target change is 0.604 rad.

Because this changes the shallow→medium and medium→deep transition geometry, the old and new dynamic sweeps should not be pooled into one speed-response fit without explicitly accounting for pose-path differences.

Use `medium_pose_delta_vs_v1.csv` for the full comparison.

## Static loaded hold update

Updated median approximate pose power:

| pose_name      |   base_com_height_mean_m |   power_median_w |   power_min_w |   power_max_w |   power_p95_median_w |   min_joint_voltage_v |
|:---------------|-------------------------:|-----------------:|--------------:|--------------:|---------------------:|----------------------:|
| deep_crouch    |                   0.3100 |           0.8525 |        0.7754 |        0.9288 |               1.0459 |               11.8000 |
| medium_crouch  |                   0.3800 |           2.4468 |        2.3147 |        2.6943 |               3.9608 |               11.7000 |
| normal_stand   |                   0.4600 |           1.1594 |        1.0062 |        1.6643 |               1.5128 |               11.8000 |
| shallow_crouch |                   0.4200 |           1.6044 |        1.1622 |        1.7381 |               2.3156 |               11.8000 |

As before, this is a relative telemetry-derived power proxy, not calibrated total robot electrical power.

The new medium pose now carries the largest median static power proxy in this dataset. The deep-crouch value remains non-monotonic with depth, so static power should not be converted directly into a motor torque parameter.

## Recommended Track 1 propagation

### 1. Preserve the suspended actuator model

Keep the existing separation of:

- transport timing;
- bus phase wait;
- actuator onset;
- amplitude-dependent velocity response;
- equivalent lag;
- small-signal residual error and hysteresis.

### 2. Use this v2 dataset as the loaded speed-response calibration layer

Replay the exact updated pose ladder:

`normal → shallow → medium → deep → medium → shallow → normal`

Then compare simulated and real:

- actual reference peak qdot;
- measured peak qdot;
- tracking RMS and p95;
- lag proxy;
- direction-conditioned knee response;
- static pose bias.

### 3. First modeling priorities

**Knee pitch**
- strong speed saturation;
- strongest tracking-error growth;
- persistent stand-return effort asymmetry;
- highest-priority loaded correction family.

**Hip pitch**
- strong speed saturation and tracking degradation;
- modest direction asymmetry;
- prioritize family response envelope and lag.

**Ankle pitch**
- declining speed gain and increasing tracking error with speed;
- avoid hard-coding a strong stand-return penalty based on the old sweep alone.

**Roll families**
- use conservative family randomization and validation.

**Hip yaw**
- keep cautious treatment because the available pose path excites it less cleanly.

### 4. Do not overfit yet

There is one completed run per speed condition and no completed 4.0 rad/s run in this archive. The fitted envelopes are suitable as first-pass validation targets and conservative randomization centers, not precision motor constants.

## Files

- `TRACK1_STANDING_LOAD_HANDOFF_V2.md`
- `track1_loaded_actuator_extension_v2.yaml`
- `track1_loaded_actuator_extension_v2.json`
- `source_run_manifest.csv`
- `data_quality_summary.csv`
- `transition_joint_augmented.csv`
- `transition_family_speed_summary.csv`
- `transition_family_speed_direction_summary.csv`
- `direction_asymmetry_by_speed.csv`
- `speed_response_growth_summary.csv`
- `empirical_velocity_saturation_fit.csv`
- `captured_pose_vectors_and_hold_bias.csv`
- `medium_pose_delta_vs_v1.csv`
- `static_pose_power_summary.csv`
- `static_joint_summary.csv`
- `static_error_load_proxy_fit.csv`
