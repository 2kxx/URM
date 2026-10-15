# Pointwise 3-dim checklist protocol (stage-6 judge)

This is the exact request/response contract the stage-6 judge was trained and
served with. Any OpenAI-compatible client can reproduce it; the Flow-Factory
adapter in `../overlay/` is a reference implementation.

## Input

One image + one generation prompt (pointwise: no second candidate).

* `system` = `system_prompt.txt` (byte-identical to the training/eval prompt)
* `user`   = image part + text part, where the text is `user_prompt_template.txt`
  with `{instruction}` replaced by the generation prompt:

```json
[
  {"role": "system", "content": "<system_prompt.txt>"},
  {"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
      {"type": "text", "text": "The image above is the generated image.\nText prompt: <prompt>\n\nGenerate checklist-style evaluation points ..."}
  ]}
]
```

## Output

The model emits **only JSON** (no markdown), with this schema:

```json
{
  "eval_points": [{"question": "...", "dimension": "Prompt Alignment", "score": 0}],
  "dimension_summary": {
    "Prompt Alignment": "...",
    "Aesthetics": "...",
    "Technical Quality": "..."
  },
  "score_reason": "...",
  "final_score": 3.5
}
```

| Field | Meaning |
|---|---|
| `eval_points[].question` | one atomic, visually verifiable check |
| `eval_points[].dimension` | **must be exactly one of** `Prompt Alignment`, `Aesthetics`, `Technical Quality` |
| `eval_points[].score` | `0` or `1` (1 = satisfied); uncertain -> 0 |
| `final_score` | overall score on the **0–10** scale, decimals allowed |

Checklist size the model was trained to produce: **8–14 points** (Prompt Alignment
4–8, Aesthetics 2–3, Technical Quality 2–3).

## Reward mapping

```
reward = clamp(final_score / 10, 0, 1)
```

Unparseable / truncated replies fall back to a configured constant
(`fallback_reward`, default `0.0`). Truncated replies are rescued by a regex on
`"final_score": x` when that token is present.

## Serving

```bash
# ms-swift (works with qwen3_5; recommended)
CUDA_VISIBLE_DEVICES=0,1 python -m swift.cli.deploy \
  --model <this-repo> --infer_backend vllm --served_model_name stage6-judge \
  --host 0.0.0.0 --port 8000 \
  --vllm_tensor_parallel_size 2 --vllm_gpu_memory_utilization 0.85 \
  --vllm_max_model_len 16384 --vllm_max_num_seqs 64 \
  --vllm_limit_mm_per_prompt '{"image": 1}'
```

## Token budget

| Part | Tokens |
|---|---|
| system prompt | ~1200 |
| user text | ~80 |
| image (512², patch 16 / merge 2) | 256 |
| **completion cap** | **4096** |
| required `--max-model-len` | >= 5846 (use 16384 for headroom) |

Terminating replies are <= ~2.5k tokens and always carry `final_score`.

## Two failure modes worth knowing

1. **Do not use a frequency/presence penalty.** A non-zero `frequency_penalty`
   also penalises the JSON *syntax* tokens (`"`, `,`, `:`); the model then aborts
   mid-JSON with `finish_reason="stop"` and no `final_score`, which silently
   collapses the reward to the fallback. Observed at 1.2.
2. **Self-repeat loops** (the same checklist question hundreds of times, never
   emitting `final_score`) appear only in judges that were themselves
   RL-degraded. This checkpoint terminates on 99.9% of inputs; guard the tail
   with a checklist-size cap instead of a repetition penalty.

## Scoring a set of images (reference)

`../scripts/check_judge_server.py` sends one real request through this exact
contract and prints `prompt_tokens` / `completion_tokens` / `finish_reason` /
`eval_points` / `final_score`.

## Pairwise use

The judge is pointwise. For preference benchmarks (GenAI-Bench, MMRB2) score the
two candidates separately and compare `final_score`, reporting ties within a
margin. Pairwise accuracy and tie rate both depend on that margin; state it.
