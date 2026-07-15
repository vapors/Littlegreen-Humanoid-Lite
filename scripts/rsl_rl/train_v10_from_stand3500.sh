#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-5000}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-hardware_st3215_loaded_v10_transfer_lift_place_from_stand3500_seed42}"
SOURCE_EXPERIMENT="logs/rsl_rl/lilgreen_v1_4_5_st3215_stabilized_forward_s3"
SOURCE_RUN="2026-07-13_12-10-53_stand_st3215_loaded_v145s3_forward_com_height460_seed42"
SOURCE_CHECKPOINT="model_3500.pt"

CHECKPOINT_PATH="${PROJECT_ROOT}/${SOURCE_EXPERIMENT}/${SOURCE_RUN}/${SOURCE_CHECKPOINT}"
if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: stand3500 checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 2
fi

python scripts/rsl_rl/train_eval.py \
  --task Velocity-Lilgreen-Hardware-ST3215-Loaded-v10 \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${MAX_ITERATIONS}" \
  --seed "${SEED}" \
  --run_name "${RUN_NAME}" \
  --resume \
  --policy_only_warm_start \
  --resume_log_root "${SOURCE_EXPERIMENT}" \
  --load_run "${SOURCE_RUN}" \
  --checkpoint "${SOURCE_CHECKPOINT}" \
  --headless
