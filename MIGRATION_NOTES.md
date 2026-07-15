# Migrating from Berkeley-Humanoid-Lite v1.4.8 to Littlegreen-Humanoid-Lite v2.0.0

## Scope

v2.0.0 is a source-tree and package-identity refactor. It does not intentionally change task behavior, robot contracts, rewards, actuator models, or export math.

## Repository and package names

```text
Berkeley-Humanoid-Lite_v1_4_8
  -> Littlegreen-Humanoid-Lite_v2_0_0

berkeley_humanoid_lite
  -> littlegreen_humanoid_lite

berkeley_humanoid_lite_assets
  -> littlegreen_humanoid_lite_assets

berkeley_humanoid_lite_lowlevel
  -> littlegreen_humanoid_lite_lowlevel
```

Imports must use the new package names. No old-name alias packages are included.

## Recommended clean installation

Do not install v2.0.0 over an active editable v1.4.8 checkout. Use a clean environment or uninstall the old editable distributions first.

From the v2.0.0 repository root:

```bash
python -m pip uninstall -y   berkeley_humanoid_lite   berkeley_humanoid_lite_assets   berkeley_humanoid_lite_lowlevel

python -m pip install -e source/littlegreen_humanoid_lite_assets
python -m pip install -e source/littlegreen_humanoid_lite_lowlevel
python -m pip install -e source/littlegreen_humanoid_lite
```

Use the project’s normal Isaac Lab environment rather than the generic system Python.

## Task names

Task IDs are intentionally unchanged. Existing commands continue to select names such as:

```bash
--task Velocity-Lilgreen-Stand-v0
--task Velocity-Lilgreen-Stand-ST3215-Loaded-v5s3
--task Velocity-Lilgreen-Hardware-ST3215-Loaded-v7
--task Velocity-Lilgreen-Hardware-ST3215-Loaded-v8
```

The two legacy `Velocity-Berkeley-Humanoid-Lite-*` IDs are also retained.

## Script changes

Canonical exporter:

```bash
python scripts/rsl_rl/export_policy_littlegreen.py ...
```

Stable wrapper retained for command compatibility:

```bash
python scripts/rsl_rl/export_policy.py ...
```

Removed duplicate script names:

```text
export_policy_bhl.py
export_policy_lgh.py
```

The new default ROS 2 copy destination is:

```text
~/littlegreen_ros2_ws/src/littlegreen_biped_pkg/src/configs
```

Explicit CLI output/copy paths still take precedence.

## Scene files

```text
bhl_biped_scene.xml -> littlegreen_biped_scene.xml
bhl_scene.xml       -> littlegreen_scene.xml
```

References in active MuJoCo code were updated. Other Berkeley-named asset layers are retained as upstream/legacy compatibility assets, especially binary USD files that should not be rewritten as text.

## Bundled policy

The bundled `configs/policy_latest.yaml` and `configs/policy.onnx` come from the supplied latest policy archive:

```text
task: Velocity-Lilgreen-Hardware-ST3215-Loaded-v7
observations: 47
actions: 12
action contract: 4
ONNX SHA-256: 3e663fbde931d8056fe8461a19a6d7bc759329d9c36b3418e4bb3a95196bf627
```

The current newest task registration remains v8, but the supplied policy is v7 and remains labelled v7.

## Historical checkpoints and logs

`logs/`, `checkpoints/`, `data/`, and binary USD files were preserved byte-for-byte.

Important compatibility limitation: Python pickles often store their defining module names. A pickle written with `berkeley_humanoid_lite...` may not load after the rename because no compatibility alias is provided. Preserve originals, and perform any required pickle remapping only on copies with a reviewed migration tool. Do not modify historical artifacts in place.

For `.pt` checkpoints, test loading on the Isaac training host. State-dict-only checkpoints are usually less namespace-sensitive than pickled full objects, but this release does not claim universal checkpoint compatibility.

## Applying the patch artifact

The supplied patch represents changes inside the repository tree. A Git patch cannot rename the enclosing archive directory itself, so first copy/extract v1.4.8 to the v2.0.0 root name, then apply:

```bash
cp -a Berkeley-Humanoid-Lite_v1_4_8 Littlegreen-Humanoid-Lite_v2_0_0
cd Littlegreen-Humanoid-Lite_v2_0_0
git init
git add -A
git commit -m "Berkeley-Humanoid-Lite v1.4.8 baseline"
git apply --binary ../Berkeley-Humanoid-Lite_v1_4_8_to_Littlegreen_v2_0_0.patch
```

Using the complete release ZIP is preferred because it also guarantees the expected top-level directory and removes generated metadata cleanly.

## Post-install checks

Before resuming work:

```bash
python scripts/list_envs.py
python -m pytest -q source/littlegreen_humanoid_lite/test
```

Then run the Isaac-host acceptance sequence documented in `VALIDATION_REPORT.md`.

## Rollback

The original `Berkeley-Humanoid-Lite_v1_4_8.zip` was not modified. Keep it alongside v2.0.0 until the renamed workspace has passed training-host and deployment-host acceptance tests.
