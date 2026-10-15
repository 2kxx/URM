#!/bin/bash
# Run the RL checkpoint evals across all GPUs in parallel (merge + vLLM bench).
#
# It fans the candidate global_steps out over NDEVICES GPUs (max one eval per
# GPU), runs waves until done, and skips steps that already have reports or are
# missing a checkpoint.  Each step goes through scripts/run_rl_bench.sh
# (merge -> add args.json -> eval_benchmarks_stage3.py --full-model on genai+mmrb2).
#
# Usage:
#   cd /efs/jacuzzitang/projects/URM
#   nohup bash scripts/run_rl_bench_all.sh > logs/rl_bench_all.log 2>&1 &
#
# Env:
#   STEPS    space list (default "60 80 200 340 360 380 400 420 500")
#   NDEVICES GPUs to use (default 8)
#   GPU_START first gpu index (default 0)
#   BENCH_SUFFIX optional output-tree label (default "" -> output/bench_rl_t2i).
#                For a repetition-controlled rerun pass e.g. BENCH_SUFFIX=_rp
#                plus EVAL_FREQUENCY_PENALTY=1.2 (forwarded to each step).
#   EVAL_TASKS optional space list of tasks (default: all tasks of each bench;
#                pass "image_generation t2i" for a pure T2I-only run).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STEPS=${STEPS:-"60 80 200 340 360 380 400 420 500"}
NDEV=${NDEVICES:-8}
G0=${GPU_START:-0}
RUN_DIR=${RUN_DIR:-$ROOT/checkpoints/URM_T2I_RL/stage6_pairwise_rl_hard4k_v2}
SUFFIX=${BENCH_SUFFIX:-}
BENCH_ROOT=$ROOT/output/bench_rl_t2i${SUFFIX}
mkdir -p logs

echo "[rl-bench-all] $(date '+%F %T') steps=$STEPS ndev=$NDEV"

# ---- decide which steps still need work ----
TODO=""
for S in $STEPS; do
  CK="$RUN_DIR/global_step_$S/actor"
  if [ ! -d "$CK" ]; then
    echo "[rl-bench-all] skip step $S (no ckpt at $CK)"
    continue
  fi
  if find "$BENCH_ROOT/step$S" -maxdepth 3 -name "*_report.json" 2>/dev/null | grep -q .; then
    echo "[rl-bench-all] skip step $S (already has eval reports)"
    continue
  fi
  TODO="$TODO $S"
done
echo "[rl-bench-all] todo =${TODO}"
if [ -z "$TODO" ]; then
  echo "[rl-bench-all] nothing to do"
  touch logs/rl_bench_ALLDONE
  exit 0
fi

# ---- fan out over GPUs in waves ----
set -- $TODO
while [ $# -gt 0 ]; do
  launched=0
  for gpu in $(seq "$G0" $((G0 + NDEV - 1))); do
    [ $# -gt 0 ] || break
    S=$1; shift
    echo "[rl-bench-all] launch step $S on gpu $gpu"
    GPU=$gpu STEPS="$S" nohup bash scripts/run_rl_bench.sh > "logs/rl_bench_$S.log" 2>&1 &
    launched=$((launched + 1))
  done
  wait
  echo "[rl-bench-all] wave done (launched=$launched), remaining=$#"
done

touch logs/rl_bench_ALLDONE
echo "[rl-bench-all] $(date '+%F %T') ALL DONE -> $BENCH_ROOT"
