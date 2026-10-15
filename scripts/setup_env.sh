#!/usr/bin/env bash
# Bootstrap the URM project environment from scratch on a fresh GPU server.
#
# What it does:
#   1. create .venv (reuse an existing conda torch-base env via
#      --system-site-packages when available, otherwise a clean venv + torch)
#   2. install requirements (transformers / qwen-vl-utils / ms-swift deps)
#   3. install ms-swift editable from third_party/ms-swift
#   4. best-effort install of the Qwen3.5 acceleration deps
#      (flash-linear-attention / causal-conv1d / flash-attn)
#   5. verify the `swift` entry point works
#
# Usage:
#   bash scripts/setup_env.sh
#
# Env overrides:
#   USE_SYSTEM_PY  use a specific python that already has a torch build
#                  (e.g. /opt/conda/envs/torch-base/bin/python3)
#   FORCE_CLEAN    rm -rf .venv first (default 0)
#   INSTALL_TORCH  install torch into a clean venv (default 0; set 1 when no
#                  system torch is available, CUDA_VISIBLE_DEVICES/GPU must be present)
#   TORCH_CUDA     cuda version for the torch index url, e.g. cu128 (default cu128)
#   SKIP_ACCEL     skip the flash-* acceleration deps (default 0)
#   HTTPS_PROXY    export it first when the server needs a proxy for pip/git
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

if [[ "${FORCE_CLEAN:-0}" == "1" ]]; then
  rm -rf .venv
fi

# --------------------------------------------------------------------------- #
# pick a base python
# --------------------------------------------------------------------------- #
BASE_PY=""
for cand in \
  "${USE_SYSTEM_PY:-}" \
  /opt/conda/envs/torch-base/bin/python3 \
  "$(command -v python3 || true)"; do
  if [[ -n "$cand" && -x "$cand" ]]; then
    if "$cand" -c "import torch" >/dev/null 2>&1; then
      BASE_PY="$cand"
      break
    fi
  fi
done

if [[ -n "$BASE_PY" ]]; then
  echo "[setup] using system python WITH torch: $BASE_PY"
  "$BASE_PY" -c "import torch; print('[setup]   torch', torch.__version__, 'cuda', torch.version.cuda)"
  python3 -m venv --system-site-packages .venv
else
  if [[ "${INSTALL_TORCH:-0}" != "1" ]]; then
    echo "[setup] WARNING: no python with torch found; install torch first or set INSTALL_TORCH=1" >&2
  fi
  echo "[setup] creating clean venv"
  python3 -m venv .venv
  if [[ "${INSTALL_TORCH:-0}" == "1" ]]; then
    echo "[setup] installing torch ($TORCH_CUDA) from the official index"
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install "torch>=2.7" --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
  fi
fi

source .venv/bin/activate
python -m pip install --upgrade pip

# --------------------------------------------------------------------------- #
# project requirements (torch expected from the system env)
# --------------------------------------------------------------------------- #
echo "[setup] installing requirements (env/requirements.txt)"
pip install -r env/requirements.txt

echo "[setup] installing ms-swift (editable) from third_party/ms-swift"
pip install -e third_party/ms-swift

# --------------------------------------------------------------------------- #
# optional Qwen3.5 acceleration deps (best-effort; skip on failure is allowed
# for a smoke run -- transformers fallback still works)
# --------------------------------------------------------------------------- #
if [[ "${SKIP_ACCEL:-0}" != "1" ]]; then
  echo "[setup] installing flash-linear-attention"
  pip install 'flash-linear-attention>=0.4.2' --no-build-isolation || echo "[setup] WARN flash-linear-attention failed (OK for smoke)"
  echo "[setup] installing causal-conv1d"
  pip install 'git+https://github.com/Dao-AILab/causal-conv1d' --no-build-isolation || echo "[setup] WARN causal-conv1d failed (OK for smoke)"
  echo "[setup] installing flash-attn"
  pip install 'flash-attn==2.8.3' --no-build-isolation || echo "[setup] WARN flash-attn failed (OK for smoke)"
fi

# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #
echo "[setup] verifying imports"
python -c "import torch, transformers, swift; print('[setup] OK: torch', torch.__version__, '| transformers', transformers.__version__, '| swift', swift.__file__)"
command -v swift >/dev/null && echo "[setup] OK: swift CLI at $(command -v swift)"

echo
echo "[setup] DONE. Next steps:"
echo "  1) log in to Hugging Face (the data repo is gated):"
echo "       export HTTPS_PROXY=http://star-proxy.oa.com:3128   # if needed"
echo "       hf auth login                                      # or: export HF_TOKEN=hf_xxx"
echo "  2) bash scripts/download_urm_t2i.sh"
echo "  3) bash scripts/build_stage1_dataset.sh"
echo "  4) bash scripts/train_skyjm_gen_9b_stage4.sh"
echo "  5) bash scripts/eval_stage4.sh && bash scripts/eval_benchmarks_stage4.sh"
