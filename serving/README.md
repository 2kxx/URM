# Serving the stage-6 judge

OpenAI-compatible HTTP server for the released judge (`2kxx/URM` on
HuggingFace).  The request/response contract, token budget and failure modes are
in [`../reward_protocol/README.md`](../reward_protocol/README.md).

| file | purpose |
|---|---|
| `serve_stage6_judge.sh` | start the server (`ENGINE=swift` via ms-swift, or `ENGINE=vllm`) |
| `check_judge_server.py` | one-shot smoke test + prints real token counts |
| `merge_stage6_judge.sh` | merge a LoRA adapter into a servable full model |

```bash
CUDA_VISIBLE_DEVICES=0,1 TENSOR_PARALLEL_SIZE=2 \
ENGINE=swift SWIFT_PYTHON=python3 MODEL_PATH=2kxx/URM \
  bash serve_stage6_judge.sh

python check_judge_server.py --api-base-url http://127.0.0.1:8000/v1 \
  --model stage6-judge --image /path/to/a/generated.png
```
