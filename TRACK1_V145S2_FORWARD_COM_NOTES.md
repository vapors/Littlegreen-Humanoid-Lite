# Berkeley Humanoid Lite v1.4.5s2 — Stand Forward-COM Stabilization

This branch is a small follow-up to v1.4.5s. It keeps the v1.4.5/v5s deployment-impacting contract family unchanged:

- action_contract_version: 4
- vector residual scale
- stabilized athletic q_default from v5s
- same ST3215 loaded actuator model
- no knee-symmetry reward

The change is focused on the Stand seed quality before Hardware-v5s:

- standing COM height target: 0.450 m
- moving COM height target: 0.420 m
- standing COM-over-feet forward target: 0.055 m
- forward COM band half-width: 0.010 m
- weak projected-gravity-x forward-lean target: 0.052, roughly a 3 degree cue in the current convention

New tasks:

- Velocity-Lilgreen-Stand-ST3215-Loaded-v5s2
- Velocity-Lilgreen-Hardware-ST3215-Loaded-v5s2

Recommended first run:

```bash
python scripts/rsl_rl/train_eval.py \
  --task Velocity-Lilgreen-Stand-ST3215-Loaded-v5s2 \
  --num_envs 4096 \
  --max_iterations 4000 \
  --seed 42 \
  --run_name stand_st3215_loaded_v145s2_forward_com_seed42 \
  --headless
```

Inspect 1000, 2500, 3500, and 4000. Earlier v5s runs peaked near 2500 and degraded after 5000, so do not assume later is better.

Analyzer additions in `analyze_policy_v1_4_5.py`:

- standing_com_forward_offset_mean_m
- standing_com_forward_offset_p05_m
- standing_com_forward_offset_median_m
- standing_com_forward_offset_p95_m
- standing_projected_gravity_x_mean
- standing_projected_gravity_x_p05
- standing_projected_gravity_x_p95

Move to Hardware-v5s2 only if the selected Stand checkpoint has stable posture, low global hard torque, no visible rearward COM bias, and no strong right-leg brace.
