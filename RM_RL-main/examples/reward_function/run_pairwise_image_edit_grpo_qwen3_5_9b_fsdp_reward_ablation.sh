#!/usr/bin/env bash
# Reward-ablation wrapper for pairwise image-edit GRPO.
#
# This keeps the base launch script unchanged and only appends reward-specific
# Hydra overrides.

set -xeuo pipefail

BASE_SCRIPT="/inspire/qb-ilm/project/deepgen/wangdianyi-240107110022/BXH/RL_train/verl/examples/reward_function/run_pairwise_image_edit_grpo_qwen3_5_9b_fsdp.sh"

PAIRWISE_REWARD_TYPE=${PAIRWISE_REWARD_TYPE:-bt_sigmoid_format}
PAIRWISE_ACC_MARGIN=${PAIRWISE_ACC_MARGIN:-0}
PAIRWISE_THRESHOLD_MARGIN=${PAIRWISE_THRESHOLD_MARGIN:-0}
# Threshold rewards: the base script already forwards ``tau``/``margin`` from the
# environment, but ``tau_raw`` (tau in raw final_score points) and the
# noise-aware dynamic threshold only exist as Hydra keys, so forward them here.
PAIRWISE_TAU_RAW=${PAIRWISE_TAU_RAW:-}
PAIRWISE_THRESHOLD_TYPE=${PAIRWISE_THRESHOLD_TYPE:-}
PAIRWISE_USE_DYNAMIC_THRESHOLD=${PAIRWISE_USE_DYNAMIC_THRESHOLD:-}
PAIRWISE_DYNAMIC_SCALE_MULT=${PAIRWISE_DYNAMIC_SCALE_MULT:-}

EXTRA=()
[ -n "${PAIRWISE_TAU_RAW}" ] && EXTRA+=(+reward.pairwise_image_edit.tau_raw="${PAIRWISE_TAU_RAW}")
[ -n "${PAIRWISE_THRESHOLD_TYPE}" ] && EXTRA+=(+reward.pairwise_image_edit.threshold_type="${PAIRWISE_THRESHOLD_TYPE}")
[ -n "${PAIRWISE_USE_DYNAMIC_THRESHOLD}" ] && EXTRA+=(+reward.pairwise_image_edit.use_dynamic_threshold="${PAIRWISE_USE_DYNAMIC_THRESHOLD}")
[ -n "${PAIRWISE_DYNAMIC_SCALE_MULT}" ] && EXTRA+=(+reward.pairwise_image_edit.dynamic_scale_mult="${PAIRWISE_DYNAMIC_SCALE_MULT}")

bash "${BASE_SCRIPT}" \
    +reward.pairwise_image_edit.reward_type="${PAIRWISE_REWARD_TYPE}" \
    +reward.pairwise_image_edit.acc_margin="${PAIRWISE_ACC_MARGIN}" \
    +reward.pairwise_image_edit.threshold_margin="${PAIRWISE_THRESHOLD_MARGIN}" \
    ${EXTRA[@]+"${EXTRA[@]}"} \
    "$@"
