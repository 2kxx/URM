#!/bin/bash
# Train the stage-4 URM T2I checklist model (expanded stage-1 data, 3-dim v2
# system prompt) with ms-swift LoRA SFT on 8xH20.
#
# This mirrors scripts/train_skyjm_gen_9b_stage3.sh:
#   - MODEL: the SAME stage0 best checkpoint used for stage 3
#     (output/skyjm_gen_9b_pointwise_stage0/v3-20260818-224412/checkpoint-1500-merged)
#   - Hyper-parameters (LoRA rank 32 / lr 1e-5 / cosine / warmup 0.1 / 2 epochs)
#     are kept identical to the stage-3 run.
#   - The only changes: dataset -> the newly built stage-1 (20k expanded) train/val
#     jsonl built with the latest 3-dimension system prompt, output dir -> stage4.
#
# Prereqs (see stage1/README.md / scripts/setup_env.sh):
#   bash scripts/download_urm_t2i.sh
#   python scripts/build_stage1_dataset.py
#   python scripts/split_stage1_dataset.py
#   # stage0 checkpoint must exist at $MODEL (bring it over from the previous server
#   # or re-run the stage0 training on this machine)
#
# Usage:
#   bash scripts/train_skyjm_gen_9b_stage4.sh
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

# Same starting checkpoint as the previous stage-3 run (stage0 merged best ckpt).
MODEL=${MODEL:-"$ROOT/output/skyjm_gen_9b_pointwise_stage0/v3-20260818-224412/checkpoint-1500-merged"}
DATASET=${DATASET:-"$ROOT/stage1/t2i_v4_20k_final_prompt_3dim_v2_train.jsonl"}
VAL_DATASET=${VAL_DATASET:-"$ROOT/stage1/t2i_v4_20k_final_prompt_3dim_v2_val.jsonl"}
OUT=${OUT:-"$ROOT/output/skyjm_gen_9b_stage4"}

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-2}

# Qwen3.5 uses token budgets from qwen_vl_utils:
# image pixels ~= IMAGE_MAX_TOKEN_NUM * 32 * 32.
IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM:-1024}
VIDEO_MAX_TOKEN_NUM=${VIDEO_MAX_TOKEN_NUM:-128}
FPS_MAX_FRAMES=${FPS_MAX_FRAMES:-12}
MAX_LENGTH=${MAX_LENGTH:-8192}
MAX_STEPS=${MAX_STEPS:--1}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-2}
SAVE_STEPS=${SAVE_STEPS:-250}
EVAL_STEPS=${EVAL_STEPS:-250}

WANDB_DIR=${WANDB_DIR:-"$ROOT/wandb"}
SWIFT_BIN=${SWIFT_BIN:-"$ROOT/.venv/bin/swift"}

if [[ ! -x "$SWIFT_BIN" ]]; then
  SWIFT_BIN=$(command -v swift || true)
fi
if [[ -z "$SWIFT_BIN" || ! -x "$SWIFT_BIN" ]]; then
  echo "Cannot find the swift executable. Activate the project .venv first." >&2
  exit 127
fi

if [[ ! -d "$MODEL" ]]; then
  echo "ERROR: starting checkpoint not found: $MODEL" >&2
  echo "Bring it over from the previous server or override MODEL." >&2
  exit 1
fi
for f in "$DATASET" "$VAL_DATASET"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: dataset not found: $f" >&2
    echo "Run scripts/build_stage1_dataset.py + scripts/split_stage1_dataset.py first." >&2
    exit 1
  fi
done

VAL_ARGS=()
if [[ -n "$VAL_DATASET" ]]; then
  VAL_ARGS=(--val_dataset "$VAL_DATASET")
else
  VAL_ARGS=(--split_dataset_ratio 0.00)
fi

mkdir -p "$OUT" "$WANDB_DIR" "$ROOT/logs"

export NPROC_PER_NODE CUDA_VISIBLE_DEVICES
export IMAGE_MAX_TOKEN_NUM VIDEO_MAX_TOKEN_NUM FPS_MAX_FRAMES
export WANDB_DIR
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

echo "MODEL=$MODEL"
echo "DATASET=$DATASET"
echo "VAL_DATASET=$VAL_DATASET"
echo "OUT=$OUT"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "GLOBAL_BATCH_SIZE=$((NPROC_PER_NODE * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "NUM_TRAIN_EPOCHS=$NUM_TRAIN_EPOCHS"
echo "MAX_STEPS=$MAX_STEPS"
echo "SWIFT_BIN=$SWIFT_BIN"

# Print the label distribution once before distributed training starts. This
# makes accidental dataset/split changes immediately visible in the train log.
PRINT_FINAL_SCORE_DISTRIBUTION=${PRINT_FINAL_SCORE_DISTRIBUTION:-1}
if [[ "$PRINT_FINAL_SCORE_DISTRIBUTION" == "1" ]]; then
  SCORE_DATASETS=("$DATASET")
  if [[ -n "$VAL_DATASET" ]]; then
    SCORE_DATASETS+=("$VAL_DATASET")
  fi
  python "$ROOT/scripts/show_final_score_distribution.py" "${SCORE_DATASETS[@]}"
fi

"$SWIFT_BIN" sft \
  --model "$MODEL" \
  --dataset "$DATASET" \
  "${VAL_ARGS[@]}" \
  --template qwen3_5 \
  --tuner_type lora \
  --lora_rank 32 \
  --lora_alpha 64 \
  --target_modules all-linear \
  --add_non_thinking_prefix true \
  --loss_scale ignore_empty_think \
  --torch_dtype bfloat16 \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --gradient_checkpointing true \
  --learning_rate 1e-5 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --max_length "$MAX_LENGTH" \
  --group_by_length true \
  --load_from_cache_file true \
  --dataset_num_proc 16 \
  --dataloader_num_workers 4 \
  --save_steps "$SAVE_STEPS" \
  --eval_steps "$EVAL_STEPS" \
  --save_total_limit 10 \
  --logging_steps 1 \
  --max_steps "$MAX_STEPS" \
  --output_dir "$OUT" \
  --save_only_model false \
  --report_to tensorboard \
  --deepspeed zero2 \
  "$@"
