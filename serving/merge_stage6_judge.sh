#!/usr/bin/env bash
# Merge the stage-6 judge LoRA adapter into a full HF checkpoint, so vLLM can
# serve it (serve_stage6_judge.sh needs a complete model dir, not an adapter).
#
#   bash sd35_rl/scripts/merge_stage6_judge.sh
#   BASE_MODEL=/path/to/stage0-merged ADAPTER=/path/to/checkpoint-250 \
#   OUT=/path/to/ck250-merged bash sd35_rl/scripts/merge_stage6_judge.sh
#
# Defaults match the checkpoints in this repo:
#   base    = output/skyjm_gen_9b_pointwise_stage0/v3-20260818-224412/checkpoint-1500-merged  (18G, full model + tokenizer)
#   adapter = output/skyjm_gen_9b_stage6/v0-20260902-172103/checkpoint-250                   (LoRA r32, 173M)
#   out     = output/skyjm_gen_9b_stage6/ck250-merged                                        (~19G)
#
# Only `adapter_model.safetensors` + `adapter_config.json` are needed from the
# adapter checkpoint -- do NOT ship its `global_step*/` ZeRO optimizer states.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SD35_ROOT="$(cd "${HERE}/.." && pwd)"
URM_ROOT="${URM_ROOT:-$(cd "${SD35_ROOT}/.." && pwd)}"

BASE_MODEL="${BASE_MODEL:-${URM_ROOT}/output/skyjm_gen_9b_pointwise_stage0/v3-20260818-224412/checkpoint-1500-merged}"
ADAPTER="${ADAPTER:-${URM_ROOT}/output/skyjm_gen_9b_stage6/v0-20260902-172103/checkpoint-250}"
OUT="${OUT:-${URM_ROOT}/output/skyjm_gen_9b_stage6/ck250-merged}"
EXIST_OK="${EXIST_OK:-true}"

SWIFT_BIN="${SWIFT_BIN:-}"
if [[ -z "${SWIFT_BIN}" ]]; then
  if [[ -x "${URM_ROOT}/.venv/bin/swift" ]]; then
    SWIFT_BIN="${URM_ROOT}/.venv/bin/swift"
  else
    SWIFT_BIN="$(command -v swift || true)"
  fi
fi

for path in "${BASE_MODEL}" "${ADAPTER}"; do
  [[ -e "${path}" ]] || { echo "ERROR: missing ${path}" >&2; exit 1; }
done
[[ -f "${ADAPTER}/adapter_model.safetensors" ]] || {
  echo "ERROR: ${ADAPTER} is not a LoRA adapter (no adapter_model.safetensors)." >&2
  echo "       If it is already a merged model, just point MODEL_PATH at it." >&2
  exit 1
}
if [[ -z "${SWIFT_BIN}" ]]; then
  echo "ERROR: ms-swift CLI not found. Activate the swift env, or set" >&2
  echo "       SWIFT_BIN=/path/to/ms-swift-env/bin/swift" >&2
  exit 1
fi

echo "[merge] swift   : ${SWIFT_BIN}"
echo "[merge] base    : ${BASE_MODEL}"
echo "[merge] adapter : ${ADAPTER}"
echo "[merge] out     : ${OUT}"

if [[ -f "${OUT}/config.json" && "${EXIST_OK}" == "true" ]]; then
  echo "[merge] output already looks complete -> skipping (EXIST_OK=true)"
else
  mkdir -p "$(dirname "${OUT}")"
  "${SWIFT_BIN}" merge-lora \
    --model "${BASE_MODEL}" \
    --adapters "${ADAPTER}" \
    --merge_lora true \
    --output_dir "${OUT}" \
    --exist_ok true \
    "$@"
fi

# vLLM needs the tokenizer/processor/chat template alongside the weights; copy
# anything the merger did not carry over from the base model.
for pattern in tokenizer* vocab* merges.txt special_tokens_map* preprocessor_config* \
               processor_config* chat_template* generation_config* video_preprocessor*; do
  for src in "${BASE_MODEL}"/${pattern}; do
    [[ -e "${src}" ]] || continue
    dst="${OUT}/$(basename "${src}")"
    [[ -e "${dst}" ]] || cp -n "${src}" "${dst}"
  done
done

echo "[merge] weights:"
ls -la "${OUT}" | grep -E "safetensors|config.json" | head || true
echo "[merge] done -> MODEL_PATH=${OUT}"
echo "[merge] next  : CUDA_VISIBLE_DEVICES=6,7 TENSOR_PARALLEL_SIZE=2 MODEL_PATH=${OUT} \\"
echo "[merge]           bash sd35_rl/scripts/serve_stage6_judge.sh"
