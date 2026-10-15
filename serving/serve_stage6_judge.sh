#!/usr/bin/env bash
# Serve the URM stage-6 SFT judge (SkyJM-Gen-9B, Qwen3.5-VL) as an
# OpenAI-compatible endpoint for the Flow-Factory reward client.
#
#   # 2 GPUs for the judge, 6 left for SD3.5 (see NUM_PROCESSES=6 in render_config.sh)
#   CUDA_VISIBLE_DEVICES=6,7 TENSOR_PARALLEL_SIZE=2 \
#     nohup bash sd35_rl/scripts/serve_stage6_judge.sh > sd35_rl/logs/judge_server.log 2>&1 &
#
#   # or the whole box dedicated to the judge (a separate machine / a second run)
#   CUDA_VISIBLE_DEVICES=0,1,2,3 TENSOR_PARALLEL_SIZE=4 bash sd35_rl/scripts/serve_stage6_judge.sh
#
# The reward client (JUDGE_MAX_TOKENS, default 4096) decides the completion cap,
# so --max-model-len must cover: system prompt (~1.2k) + user text + image tokens
# + the completion cap. Use sd35_rl/scripts/token_budget.py to check the numbers.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SD35_ROOT="$(cd "${HERE}/.." && pwd)"
URM_ROOT="${URM_ROOT:-$(cd "${SD35_ROOT}/.." && pwd)}"

LOG_DIR="${SD35_ROOT}/logs"
mkdir -p "${LOG_DIR}"

# --------------------------------------------------------------------------
# Model / endpoint
# --------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-${STAGE6_CKPT:-${URM_ROOT}/output/skyjm_gen_9b_stage6/ck250-merged}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-stage6-judge}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
VLLM_BIN="${VLLM_BIN:-vllm}"
# Alternative when the env has no `vllm` console script (only the module):
#   VLLM_PYTHON=/path/to/judge_env/bin/python  ->  <python> -m vllm.entrypoints.cli.main serve
VLLM_PYTHON="${VLLM_PYTHON:-}"

# --------------------------------------------------------------------------
# Parallelism (defaults follow CUDA_VISIBLE_DEVICES)
# --------------------------------------------------------------------------
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  VISIBLE_COUNT="$(awk -F',' '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")"
else
  VISIBLE_COUNT=1
fi
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-${VISIBLE_COUNT}}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

# --------------------------------------------------------------------------
# Context / output-token budget
# --------------------------------------------------------------------------
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-4096}"   # keep in sync with the YAML
PROMPT_BUDGET="${PROMPT_BUDGET:-2048}"         # system prompt + user text headroom
IMAGE_TOKEN_BUDGET="${IMAGE_TOKEN_BUDGET:-1024}"  # 1024^2 image at patch16/merge2
MIN_MODEL_LEN=$(( JUDGE_MAX_TOKENS + PROMPT_BUDGET + IMAGE_TOKEN_BUDGET ))
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"        # >= MIN_MODEL_LEN, rounded up
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"             # caps KV-cache concurrency
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-1048576}"

# vLLM's HF-processor kwargs. Set MM_PROCESSOR_KWARGS=none if your vLLM build
# rejects the flag (the reward client already downscales images).
MM_PROCESSOR_KWARGS="${MM_PROCESSOR_KWARGS:-{\"max_pixels\":${IMAGE_MAX_PIXELS},\"min_pixels\":262144\}}"

# ENGINE=vllm  -> `vllm serve` (needs a vLLM whose openai SDK import works)
# ENGINE=swift -> `swift deploy --infer_backend vllm` (ms-swift's own OpenAI server;
#                 no `vllm.entrypoints.openai.api_server` import, so an outdated
#                 `openai` package in that env does not matter)
ENGINE="${ENGINE:-vllm}"

EXTRA_ARGS=()
SWIFT_EXTRA_ARGS=()
if [[ "${ENGINE}" == "swift" ]]; then
  [[ "${ENFORCE_EAGER:-0}" == "1" ]] && SWIFT_EXTRA_ARGS+=(--vllm_enforce_eager true)
  if [[ "${MM_PROCESSOR_KWARGS}" != "none" ]]; then
    SWIFT_EXTRA_ARGS+=(--vllm_engine_kwargs "{\"mm_processor_kwargs\": ${MM_PROCESSOR_KWARGS}}")
  fi
  if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
    echo "[judge] NOTE: VLLM_EXTRA_ARGS is ignored with ENGINE=swift; use --vllm_engine_kwargs / SWIFT_EXTRA_ARGS" >&2
  fi
  if [[ -n "${SWIFT_EXTRA_ARGS_ENV:-}" ]]; then
    # shellcheck disable=SC2206
    SWIFT_EXTRA_ARGS+=(${SWIFT_EXTRA_ARGS_ENV})
  fi
else
  [[ "${ENFORCE_EAGER:-0}" == "1" ]] && EXTRA_ARGS+=(--enforce-eager)
  [[ "${TRUST_REMOTE_CODE:-1}" == "1" ]] && EXTRA_ARGS+=(--trust-remote-code)
  if [[ "${MM_PROCESSOR_KWARGS}" != "none" ]]; then
    EXTRA_ARGS+=(--mm-processor-kwargs "${MM_PROCESSOR_KWARGS}")
  fi
  if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS+=(${VLLM_EXTRA_ARGS})
  fi
fi

if [[ ! -e "${MODEL_PATH}" ]]; then
  echo "ERROR: judge checkpoint not found: ${MODEL_PATH}" >&2
  echo "       pass MODEL_PATH=/path/to/stage6/ck250-merged (merged full HF model, not a LoRA adapter)" >&2
  exit 1
fi

cat <<EOF
[judge] $(date '+%F %T') serving ${MODEL_PATH}
[judge] engine          : ${ENGINE}
[judge] endpoint        : http://${HOST}:${PORT}/v1  (model id: ${SERVED_MODEL_NAME})
[judge] gpus            : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all} tp=${TENSOR_PARALLEL_SIZE} dp=${DATA_PARALLEL_SIZE} mem=${GPU_MEMORY_UTILIZATION}
[judge] max-model-len   : ${MAX_MODEL_LEN}  (completion ${JUDGE_MAX_TOKENS} + prompt ${PROMPT_BUDGET} + image ${IMAGE_TOKEN_BUDGET} = ${MIN_MODEL_LEN} minimum)
[judge] max-num-seqs    : ${MAX_NUM_SEQS}
[judge] image budget    : ${MM_PROCESSOR_KWARGS}
EOF

if (( MAX_MODEL_LEN < MIN_MODEL_LEN )); then
  echo "[judge] WARNING: MAX_MODEL_LEN=${MAX_MODEL_LEN} < ${MIN_MODEL_LEN}; replies will be truncated mid-JSON." >&2
fi

if [[ "${ENGINE}" == "swift" ]]; then
  SWIFT_PYTHON="${SWIFT_PYTHON:-$(command -v python3 || true)}"
  if [[ -z "${SWIFT_PYTHON}" ]] || ! "${SWIFT_PYTHON}" -c "import swift" >/dev/null 2>&1; then
    echo "ERROR: ENGINE=swift needs an env with ms-swift; set SWIFT_PYTHON=/path/to/python" >&2
    exit 1
  fi
  echo "[judge] launcher        : ${SWIFT_PYTHON} -m swift.cli.deploy --infer_backend vllm"
  exec "${SWIFT_PYTHON}" -m swift.cli.deploy \
    --model "${MODEL_PATH}" \
    --infer_backend vllm \
    --served_model_name "${SERVED_MODEL_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --vllm_tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
    --vllm_gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --vllm_max_model_len "${MAX_MODEL_LEN}" \
    --vllm_max_num_seqs "${MAX_NUM_SEQS}" \
    --vllm_limit_mm_per_prompt '{"image": 1}' \
    "${SWIFT_EXTRA_ARGS[@]}" \
    "$@"
fi

if [[ -n "${VLLM_PYTHON}" ]]; then
  VLLM_CMD=("${VLLM_PYTHON}" -m vllm.entrypoints.cli.main serve)
  echo "[judge] launcher        : ${VLLM_PYTHON} -m vllm.entrypoints.cli.main serve"
else
  VLLM_CMD=("${VLLM_BIN}" serve)
  echo "[judge] launcher        : ${VLLM_BIN} serve"
fi

exec "${VLLM_CMD[@]}" "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --data-parallel-size "${DATA_PARALLEL_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --limit-mm-per-prompt '{"image": 1}' \
  --disable-log-requests \
  "${EXTRA_ARGS[@]}" \
  "$@"
