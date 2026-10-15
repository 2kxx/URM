#!/bin/bash
# Stage-6 one-shot pipeline:
#   1) annotate the 20k Gemini corpus with stage0 (vLLM, 8 shards)   -> stage1/stage0_annotations.jsonl
#   2) align stage0 0-4 -> 0-10 and average with Gemini (only final_score) -> stage1/..._stage6_fine.json
#   3) rebuild SINGLE-STAGE (stage4-style) SFT rows from the fine labels -> stage1/t2i_v4_20k_..._v2_stage6_{train,val}.jsonl
#   4) train stage6 (single-stage, stage0 base)                     -> output/skyjm_gen_9b_stage6
#
# Every step skips itself if its output already exists, so a rerun just continues.
#
# Run once on /efs:
#   nohup bash scripts/run_stage6_pipeline.sh > logs/stage6_pipeline.log 2>&1 &
#
# Env overrides (all optional):
#   SOURCE / ANN / FINE / STAGE6_OUT / STAGE6_MODEL
#   ENGINE (vllm|transformers)  TP  GPU_MEM_UTIL  NUM_SHARDS
#   WEIGHT_STAGE0   NUM_TRAIN_EPOCHS
#   SKIP_ANNOTATE / SKIP_FINE / SKIP_DATA / SKIP_TRAIN = 1   (force-skip a step)
#   AUTO_EVAL=1   also run the multi-checkpoint eval on genai_bench+mmrb2 afterwards
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

log() { echo "[$(date '+%F %T')] $*"; }

# ---- paths (overridable) ---------------------------------------------------
SOURCE=${SOURCE:-"$ROOT/stage1/t2i_v4_20k_checklist_gemini_scored_clean.json"}
STAGE0_CKPT=${STAGE0_CKPT:-"$ROOT/output/skyjm_gen_9b_pointwise_stage0/v3-20260818-224412/checkpoint-1500-merged"}
ANN=${ANN:-"$ROOT/stage1/stage0_annotations.jsonl"}
FINE=${FINE:-"$ROOT/stage1/t2i_v4_20k_checklist_gemini_scored_stage6_fine.json"}
S6_OUT_DIR=${S6_OUT_DIR:-"$ROOT/stage1"}
S6_FULL=${S6_FULL:-"$S6_OUT_DIR/t2i_v4_20k_final_prompt_3dim_v2_stage6.jsonl"}
S6_DATASET=${S6_DATASET:-"$S6_OUT_DIR/t2i_v4_20k_final_prompt_3dim_v2_stage6_train.jsonl"}
S6_VAL=${S6_VAL:-"$S6_OUT_DIR/t2i_v4_20k_final_prompt_3dim_v2_stage6_val.jsonl"}
STAGE6_OUT=${STAGE6_OUT:-"$ROOT/output/skyjm_gen_9b_stage6"}

# ---- knobs -----------------------------------------------------------------
ENGINE=${ENGINE:-vllm}
TP=${TP:-1}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}
NUM_SHARDS=${NUM_SHARDS:-8}
WEIGHT_STAGE0=${WEIGHT_STAGE0:-0.5}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-2}
PY=${PY:-"$ROOT/.venv/bin/python"}
if [[ ! -x "$PY" ]]; then PY=$(command -v python3 || true); fi

for p in "$SOURCE" "$STAGE0_CKPT"; do
  [[ -e "$p" ]] || { log "ERROR: missing $p" >&2; exit 1; }
done

# ---- 1) stage0 annotate -----------------------------------------------------
if [[ "${SKIP_ANNOTATE:-0}" == "1" || -f "$ANN" ]]; then
  log "stage0 annotation exists/skipped: $ANN"
else
  log "annotating stage0 with $ENGINE on $NUM_SHARDS GPUs -> $ANN"
  PIDS=()
  for i in $(seq 0 $((NUM_SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES=$i "$PY" scripts/annotate_stage0_t2i.py \
        --source "$SOURCE" --ckpt "$STAGE0_CKPT" --engine "$ENGINE" \
        --tp "$TP" --gpu-mem-util "$GPU_MEM_UTIL" \
        --out "$ANN" --shard "$i" --num-shards "$NUM_SHARDS" \
        > "logs/stage0_annotate_shard_$i.log" 2>&1 &
    PIDS+=("$!")
  done
  for pid in "${PIDS[@]}"; do wait "$pid" || { log "FAILED: stage0 annotate shard"; exit 1; }; done
  "$PY" scripts/annotate_stage0_t2i.py --merge --out "$ANN" \
      || { log "FAILED: stage0 annotate merge"; exit 1; }
fi
[[ -s "$ANN" ]] || { log "ERROR: annotation empty: $ANN" >&2; exit 1; }

# ---- 2) fine labels (only final_score changes) ------------------------------
if [[ "${SKIP_FINE:-0}" == "1" || -f "$FINE" ]]; then
  log "fine labels exist/skipped: $FINE"
else
  log "building fine labels -> $FINE (weight_stage0=$WEIGHT_STAGE0)"
  "$PY" scripts/build_stage6_fine_labels.py \
      --gemini "$SOURCE" --stage0 "$ANN" --out "$FINE" \
      --weight-stage0 "$WEIGHT_STAGE0" || { log "FAILED: fine labels"; exit 1; }
fi
[[ -f "$FINE" ]] || { log "ERROR: fine labels missing: $FINE" >&2; exit 1; }

# ---- 3) rebuild stage6 e2e data ---------------------------------------------
if [[ "${SKIP_DATA:-0}" == "1" || -f "$S6_DATASET" ]]; then
  log "stage6 data exists/skipped: $S6_DATASET"
else
  log "building SINGLE-STAGE stage6 data (3-dim v2, one image per row) -> $S6_FULL"
  "$PY" scripts/build_stage1_dataset.py \
      --source "$FINE" --output "$S6_FULL" \
      || { log "FAILED: stage6 build"; exit 1; }
  "$PY" scripts/split_stage1_dataset.py \
      --input "$S6_FULL" \
      --indices "$S6_FULL.indices.json" \
      --train-out "$S6_DATASET" --val-out "$S6_VAL" \
      || { log "FAILED: stage6 split"; exit 1; }
fi
for p in "$S6_DATASET" "$S6_VAL"; do
  [[ -f "$p" ]] || { log "ERROR: missing $p" >&2; exit 1; }
done

# ---- 4) train stage6 --------------------------------------------------------
if [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
  log "SKIP_TRAIN=1, stopping before training"
  log "stage6 pipeline data steps done. Train with: bash scripts/train_stage6.sh"
elif ls "$STAGE6_OUT"/v*/checkpoint-* >/dev/null 2>&1; then
  log "stage6 checkpoints already exist under $STAGE6_OUT, skip training"
else
  log "training stage6 -> $STAGE6_OUT"
  MODEL="${STAGE6_MODEL:-}" NUM_TRAIN_EPOCHS="$NUM_TRAIN_EPOCHS" \
  DATASET="$S6_DATASET" VAL_DATASET="$S6_VAL" OUT="$STAGE6_OUT" \
    bash scripts/train_stage6.sh || { log "FAILED: stage6 training"; exit 1; }
  log "stage6 training finished"
fi

# ---- 5) optional eval -------------------------------------------------------
if [[ "${AUTO_EVAL:-0}" == "1" ]]; then
  log "AUTO_EVAL=1 -> eval on genai_bench + mmrb2"
  STAGE3_OUT="$STAGE6_OUT" OUT_ROOT="$ROOT/output/benchmarks_stage6/stage3" \
  ENGINE="${ENGINE}" \
    bash scripts/eval_twostage_multi_ckpt.sh || { log "FAILED: stage6 eval"; exit 1; }
fi

log "stage6 pipeline all done."
