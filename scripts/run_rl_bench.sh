#!/bin/bash
# Merge RL checkpoints (verl) into full HF models and evaluate them on
# genai_bench + mmrb2 with vLLM (the same 3-dim-v2 protocol as stage6).
#
# Usage:
#   cd /efs/jacuzzitang/projects/URM
#   nohup bash scripts/run_rl_bench.sh > logs/rl_bench_all.log 2>&1 &
#
# Env knobs:
#   STEPS   = space list of global_step to test (default "340 360 380 400 420 500")
#   RUN_DIR = verl run ckpt dir  (default .../stage6_pairwise_rl_hard4k_v2)
#   BASE    = full HF model used as tokenizer source (default stage6 ck250-merged)
#   GPU     = which gpu to use for each eval (default 0)
#   EVAL_TEMPLATE = system/user template jsonl for the scorer. MUST be the
#          de-primed 3-dim-v2 template (default stage3/t2i_final_prompt_3dim_v2.jsonl).
#          Do NOT fall back to stage3/t2i_final_prompt.jsonl: it still contains
#          "Physical Plausibility / World Knowledge" and has no
#          "labels must be exactly these three" constraint, so it re-primes the
#          RL model into the old 4-dim taxonomy at eval time.
#   MAX_NEW_TOKENS / EVAL_TEMP        -> passed to the scorer (default 8192 / 0.0)
#   EVAL_FREQUENCY_PENALTY / EVAL_PRESENCE_PENALTY /
#   EVAL_REPETITION_PENALTY / EVAL_NO_REPEAT_NGRAM
#          -> repetition-control knobs for breaking the degenerate loops RL
#             judges fall into (vLLM; forwarded to ms-swift RequestConfig).
#   EVAL_TASKS = optional space list of subtasks to score (default: all tasks
#          of each benchmark).  For a pure T2I table pass
#          EVAL_TASKS="image_generation t2i" (base stage6 was scored on these).
#   BENCH_SUFFIX = optional label appended to the output dir, so a repetition-
#          controlled rerun does NOT clobber the baseline:
#          output/bench_rl_t2i<SUFFIX>/step$S   (default "" -> output/bench_rl_t2i)
#   MERGE_PREFIX = optional label between "merged" and the step in the merged
#          output dir, so a v3b rerun does NOT overwrite v2's merged models:
#          output/rl_t2i/merged<PREFIX>_step$S   (default "" -> merged_step$S)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY=${PY:-"$(cd "$ROOT/.." && pwd)/verl_venv/bin/python"}
EVALPY=${EVALPY:-/usr/local/bin/python3}
RUN_DIR=${RUN_DIR:-$ROOT/checkpoints/URM_T2I_RL/stage6_pairwise_rl_hard4k_v2}
BASE=${BASE:-$ROOT/output/skyjm_gen_9b_stage6/ck250-merged}
EVAL_TEMPLATE=${EVAL_TEMPLATE:-$ROOT/stage3/t2i_final_prompt_3dim_v2.jsonl}
STEPS=${STEPS:-"340 360 380 400 420 500"}
GPU=${GPU:-0}
SUFFIX=${BENCH_SUFFIX:-}
MP=${MERGE_PREFIX:-}
mkdir -p output/rl_t2i "output/bench_rl_t2i${SUFFIX}" logs

echo "[rl-bench] steps = $STEPS"
for S in $STEPS; do
  CK="$RUN_DIR/global_step_$S/actor"
  M="$ROOT/output/rl_t2i/merged${MP}_step$S"
  B="$ROOT/output/bench_rl_t2i${SUFFIX}/step$S"
  LOG="logs/rl_bench_step$S.log"
  if [ ! -d "$CK" ]; then
    echo "[rl-bench] skip step $S: no ckpt at $CK"
    continue
  fi

  # ---- 1) merge verl FSDP checkpoint -> full HF model ----
  needs_merge=0
  if [ ! -d "$M" ]; then
    needs_merge=1
  elif [ ! -e "$M/model.safetensors" ] && [ ! -e "$M/model.safetensors.index.json" ] \
       && [ ! -e "$M/pytorch_model.bin" ] && [ ! -e "$M/pytorch_model.bin.index.json" ]; then
    echo "[rl-bench] removing incomplete merge $M"; rm -rf "$M"; needs_merge=1
  fi
  if [ "$needs_merge" = "1" ]; then
    echo "[rl-bench] merging step $S -> $M"
    "$PY" scripts/merge_verl_lora.py \
        --actor "$CK" \
        --base "$BASE" \
        --out "$M" > "$LOG" 2>&1
    if [ $? -ne 0 ]; then
      echo "[rl-bench] MERGE FAILED step $S (see $LOG)"; tail -n 8 "$LOG"
      continue
    fi
    # copy tokenizer / processor files if the merger did not emit them
    for f in "$BASE"/tokenizer* "$BASE"/*.tiktoken "$BASE"/vocab* \
             "$BASE"/merges.txt "$BASE"/special_tokens_map* \
             "$BASE"/preprocessor_config* "$BASE"/chat_template*; do
      [ -e "$f" ] || continue
      b=$(basename "$f")
      [ -e "$M/$b" ] || cp -n "$f" "$M/" 2>/dev/null
    done
  fi

  # merged dirs need the FULL base args.json (template/processor settings); a
  # minimal stub makes ms-swift load with wrong defaults -> garbage/empty output
  cp "$BASE/args.json" "$M/args.json"
  "$PY" - "$M/args.json" "$M" <<'PY'
import json, sys
p, m = sys.argv[1], sys.argv[2]
d = json.load(open(p)); d["model"] = str(m)
json.dump(d, open(p, "w"), ensure_ascii=False, indent=2)
PY

  # ---- 2) evaluate with vLLM (full model, no adapter) ----
  echo "[rl-bench] eval step $S (gpu $GPU)"
  SAMPLING_ARGS=()
  [ -n "${EVAL_FREQUENCY_PENALTY:-}" ] && SAMPLING_ARGS+=(--frequency-penalty "$EVAL_FREQUENCY_PENALTY")
  [ -n "${EVAL_PRESENCE_PENALTY:-}" ] && SAMPLING_ARGS+=(--presence-penalty "$EVAL_PRESENCE_PENALTY")
  [ -n "${EVAL_REPETITION_PENALTY:-}" ] && SAMPLING_ARGS+=(--repetition-penalty "$EVAL_REPETITION_PENALTY")
  [ -n "${EVAL_NO_REPEAT_NGRAM:-}" ] && SAMPLING_ARGS+=(--no-repeat-ngram-size "$EVAL_NO_REPEAT_NGRAM")
  TASKS_ARGS=()
  # shellcheck disable=SC2206
  [ -n "${EVAL_TASKS:-}" ] && TASKS_ARGS=(--tasks ${EVAL_TASKS})
  echo "[rl-bench] sampling extras: ${SAMPLING_ARGS[*]:-none}"
  CUDA_VISIBLE_DEVICES="$GPU" "$EVALPY" scripts/eval_benchmarks_stage3.py \
      --benchmark genai_bench mmrb2 \
      "${TASKS_ARGS[@]}" \
      --template "$EVAL_TEMPLATE" \
      --ckpt "$M" \
      --engine vllm \
      --full-model \
      --max-new-tokens ${MAX_NEW_TOKENS:-8192} \
      --temperature ${EVAL_TEMP:-0.0} \
      "${SAMPLING_ARGS[@]}" \
      --no-resume \
      --output-dir "$B" >> "$LOG" 2>&1
  echo "[rl-bench] ----- step $S -----"
  tail -n 6 "$LOG"
done
echo "[rl-bench] ALL DONE -> output/bench_rl_t2i/step*/"
