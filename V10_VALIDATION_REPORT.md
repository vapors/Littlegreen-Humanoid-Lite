# Littlegreen-Humanoid-Lite v2.0.0 v10 Validation Report

## Completed offline validation

### Source and syntax

- Python `compileall` over `scripts/` and `source/`: **PASS**
- Static test suite: **65 passed**
- New task registration found: **PASS**
- Existing task IDs removed: **none**
- Unique registered task IDs: **25 → 26**, with only `Velocity-Lilgreen-Hardware-ST3215-Loaded-v10` added

### Contract preservation

The following files are byte-for-byte unchanged from the v9 bundle:

- `hardware_contract.py`
- `hardware_actions.py`
- `st3215_actions.py`
- `st3215_actuator_model.py`
- `st3215_loaded_actuator_model.py`
- `train_eval.py`

The included v5s3 Stand `model_3500.pt` checkpoint is byte-for-byte unchanged.

### v10 structure

- Standalone reward configuration: **24 explicit terms**
- No inherited v5→v9 reward stack: **PASS**
- No no-progress termination in v10: **PASS**
- Reset-safe no-support termination retained: **PASS**
- 47-D observation shape retained: **PASS by static configuration inspection**
- 12-D action contract retained: **PASS by static contract lock**
- Canonical joint order and physical-limit assertions retained: **PASS**

### Phase and reward math checks

- Movement onset resets the phase boundary: **PASS by source/static test**
- First swing side balances by environment parity and alternates per movement onset: **PASS by source/static test**
- Phase state advances at most once per simulator step: **PASS by cache/static test**
- Episode reset detection includes episode-length rollback: **PASS by source/static test**
- Stage-A scheduled double support: `16%`, 72 ms per transfer window
- Stage-D scheduled double support: `10%`, 39 ms per transfer window
- Swing-clearance reward at zero lift: **exactly 0.0** in numerical formula check
- Swing-clearance reward at mid-swing target: **1.0** in numerical formula check
- Placement reward at zero step: **exactly 0.0**
- Placement reward below clearance gate: **exactly 0.0**

### Curriculum

- Default PPO run length: **5,000 iterations**
- Stage boundaries at 64 steps/iteration: approximately **1500 / 3000 / 4200**
- Final command envelope: `vx 0.25–0.55`, `vy ±0.08`, `yaw ±0.22`
- Moving COM-height target: `0.455 m`
- Moving forward COM target: `0.080–0.085 m`
- External pushes disabled: **PASS**

### Export metadata

The v10 exporter marks the revised 47-D phase semantics and sets:

```yaml
deployment_requires_command_synchronized_phase_v10: true
```

This prevents treating the unchanged observation dimension as semantic compatibility with v7–v9 deployment phase generation.

## Not run in this environment

The following require the Isaac Lab training host:

- Isaac Lab extension installation/import
- Gym environment construction
- PhysX contact-sensor behavior
- command resampling and phase-onset behavior in simulation
- RSL-RL 45→47 warm-start execution
- PPO smoke/full training
- checkpoint rollout analysis
- ONNX/JIT export

The following require the real-robot Track 2 stack:

- ROS 2 observation-contract implementation
- command-synchronized phase reproduction
- policy shadow mode
- guarded live deployment

No claim of Isaac/PhysX or hardware runtime validation is made.

## Release artifact validation

- v9→v10 Git patch apply check: **PASS**
- Patched-file comparison against the release tree: **PASS**
- Release ZIP integrity (`unzip -t`): **PASS**
- Generated caches and `.pyc` files excluded from the release ZIP: **PASS**
