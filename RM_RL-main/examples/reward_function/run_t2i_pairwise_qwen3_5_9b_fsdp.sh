#!/bin/bash
# T2I pairwise judge RL (GRPO) wrapper for the URM stage6 T2I judge.
#
# Reuses the (domain-agnostic) launcher run_pairwise_image_edit_grpo_qwen3_5_9b_fsdp.sh
# but points it at the T2I 3-dim v2 checklist prompts and overrides the reward
# manager's required dimension_summary keys from the image-edit dims to the
# T2I dims.  Everything else is identical to the image-edit recipe.
#
# Required env (fill in): MODEL_PATH, TRAIN_FILE, TEST_FILE
#   MODEL_PATH : a MERGED T2I judge checkpoint (merge stage6 LoRA into its base
#                with ms-swift: swift merge-lora --model <base> --adapters
#                <stage6 best ckpt> --output_dir <merged> --merge_lora true)
#   TRAIN_FILE : verl two-row jsonl from convert_urm_t2i_pairwise_jsonl.py
#   TEST_FILE  : a disjoint held-out pair set (same format)
#
# Example:
#   PROJECT_NAME=URM_T2I_RL EXPERIMENT_NAME=stage6_pairwise_hard4k \
#   MODEL_PATH=/efs/.../checkpoint-250-merged \
#   TRAIN_FILE=/efs/.../rl_pairwise_4k_train.verl.jsonl \
#   TEST_FILE=/efs/.../rl_pairwise_val.verl.jsonl \
#   bash examples/reward_function/run_t2i_pairwise_qwen3_5_9b_fsdp.sh
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HERE=$ROOT/examples/reward_function

export SYSTEM_PROMPT_FILE=${SYSTEM_PROMPT_FILE:-$HERE/t2i_3dim_v2_system_prompt.txt}
export USER_PROMPT_TEMPLATE_FILE=${USER_PROMPT_TEMPLATE_FILE:-$HERE/t2i_3dim_v2_user_prompt_template.txt}

# Reward manager dimension_summary keys for the T2I judge (default).
DIMS=${DIMENSIONS:-'"Prompt Alignment","Aesthetics","Technical Quality"'}
DIMS_ARG="+reward.pairwise_image_edit.dimensions=[$DIMS]"

for v in MODEL_PATH TRAIN_FILE TEST_FILE; do
  : "${!v:?set $v to run (see header)}"
done

bash "$HERE/run_pairwise_image_edit_grpo_qwen3_5_9b_fsdp.sh" "$DIMS_ARG" "$@"
