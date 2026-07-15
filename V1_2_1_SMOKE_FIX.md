# Berkeley Humanoid Lite v1.2.1 Smoke-Test Fix

## Fixed

The v1.2 bounded hardware action term failed during the initial environment reset with:

```text
IndexError: shape mismatch: indexing tensors could not be broadcast together with shapes [64], [12]
```

The problem was in `BoundedDefaultCenteredJointPositionAction.reset()`:

```python
joint_pos[env_ids, joint_ids]
```

When both `env_ids` and `joint_ids` are 1-D advanced-index tensors, PyTorch attempts to broadcast them element-wise. During the 64-environment smoke test those shapes were `[64]` and `[12]`, so the reset failed before the first PPO step.

v1.2.1 selects the 12 action joints first and then selects the requested environments:

```python
current_action_joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
self._processed_actions[env_ids] = current_action_joint_pos[env_ids]
```

This produces the intended `[num_reset_envs, 12]` tensor and also works for partial environment resets later in training.

No action semantics, reward weights, curriculum settings, PPO parameters, task registrations, or export contract were changed.
