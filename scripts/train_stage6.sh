#!/bin/bash
# Stage-6 training -- SINGLE-STAGE (stage4-style), fine-relabeled final_score.
#
# Data comes from run_stage6_pipeline.sh steps 1-3:
#   stage1/t2i_v4_20k_final_prompt_3dim_v2_stage6_{train,val}.jsonl
# (rows identical to stage4's except final_score is the Gemini+stage0 average).
#
# Reuses scripts/train_skyjm_gen_9b_stage4.sh verbatim (template qwen3_5, LoRA,
# bf16, zero2, IMAGE_MAX_TOKEN_NUM=1024, batch2/accum4). Default MODEL = stage0
# base, the same anchor stage-4 was trained from (clean A/B vs stage4 ck500).
#
# To warm-start from stage4/stage5's best instead, merge and pass MODEL=...:
#   swift merge-lora --model <base> --adapters <best ckpt> --merge_lora true \
#       --output_dir <merged> --exist_ok true
#   MODEL=<merged> bash scripts/train_stage6.sh
#
# Usage:
#   bash scripts/train_stage6.sh
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

MODEL=${MODEL:-"$ROOT/output/skyjm_gen_9b_pointwise_stage0/v3-20260818-224412/checkpoint-1500-merged"}
DATASET=${DATASET:-"$ROOT/stage1/t2i_v4_20k_final_prompt_3dim_v2_stage6_train.jsonl"}
VAL_DATASET=${VAL_DATASET:-"$ROOT/stage1/t2i_v4_20k_final_prompt_3dim_v2_stage6_val.jsonl"}
OUT=${OUT:-"$ROOT/output/skyjm_gen_9b_stage6"}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-2}

for p in "$MODEL" "$DATASET" "$VAL_DATASET"; do
  if [[ ! -e "$p" ]]; then echo "ERROR: missing $p" >&2; exit 1; fi
done

echo "[stage6] $(date '+%F %T') model=$MODEL"
echo "[stage6] dataset=$DATASET"
echo "[stage6] val=$VAL_DATASET"
echo "[stage6] out=$OUT epochs=$NUM_TRAIN_EPOCHS"

MODEL="$MODEL" \
DATASET="$DATASET" \
VAL_DATASET="$VAL_DATASET" \
OUT="$OUT" \
NUM_TRAIN_EPOCHS="$NUM_TRAIN_EPOCHS" \
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}" \
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}" \
IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}" \
  bash scripts/train_skyjm_gen_9b_stage4.sh
