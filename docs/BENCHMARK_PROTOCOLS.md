# GenAI-Bench / Qwen3.5 Pairwise Protocols

## Core distinction

GenAI-Bench itself does **not** define a judge system prompt. The benchmark
provides image pairs, text prompts, and human preference votes. The system
prompt and output format belong to the judge/RM protocol that consumes the
benchmark.

Therefore "GenAI's system prompt" is not a meaningful object. There are only
the prompts of the individual judges:

| Protocol | Prompt source | Output | Scoring |
|---|---|---|---|
| URM / GenAI-aligned | `stage3/t2i_final_prompt_3dim_v2.jsonl` | per-image checklist + `final_score` | compare `final_score`; raw / no-tie / tie=0.5 |
| SKYLENAGE-JUDGER | `SKYLENAGE-JUDGER/judges/skyjm_rm/templates.py:GENTEMPLATE` | 3-5 weighted dimensions + `\boxed{A}` / `\boxed{B}` | parse boxed preference |
| RationalRewards baseline | `benchmarks_rr_src/modular_rm_eval/prompts.py:GEN_PROMPT_TEMPLATE` | per-aspect `Score A/B` + `Winner` | numeric-score ranking |

These prompts are not interchangeable:

- URM is a **single-image pointwise** judge. It scores image A and image B
  separately with the same 3-dim checklist prompt, then compares
  `final_score`.
- SKYLENAGE is a **two-image pairwise** judge. It asks for weighted dimensions
  and a direct `\boxed{A/B}` choice.
- RationalRewards is a **two-image pairwise** judge. It asks for per-aspect
  numeric scores and a `Winner`.

## Comparison rules

### Cross-model comparison (Qwen3.5 vs other Qwen judges vs URM judge)

Use one common protocol for every model. For the GenAI/URM target, use the
pointwise 3-dim protocol:

```bash
cd genai_bench_eval_pkg
TEMPLATE=stage3/t2i_final_prompt_3dim_v2.jsonl \
CKPT=/path/to/model \
FULL_MODEL=1 \
ENGINE=vllm \
TASKS=image_generation \
NDEVICES=8 \
bash run_genai_bench.sh
```

For both GenAI-Bench t2i and MMRB2 t2i:

```bash
TEMPLATE=stage3/t2i_final_prompt_3dim_v2.jsonl \
CKPT=/path/to/model \
FULL_MODEL=1 \
ENGINE=vllm \
BENCHMARKS="genai_bench mmrb2" \
TASKS="image_generation t2i" \
NDEVICES=8 \
bash run_genai_bench.sh
```

Apply this identical command to every model being compared. Metrics are:

```text
raw       correct / N                 (ties count as wrong)
no_tie    correct / (N - ties)
tie=0.5   (correct + 0.5 * ties) / N
```

This is the only apples-to-apples comparison with the URM judge numbers.

### Native-protocol reproduction

Use these only to reproduce the RM's own published/deployment protocol:

```bash
# SKYLENAGE-JUDGER \boxed{A/B}
cd qwen35_pairwise_eval_pkg
MODEL_PATH=/path/to/Qwen3.5-9B NDEVICES=8 bash run_qwen35_pairwise.sh

# RationalRewards numeric-score baseline
cd qwen35_rr_baseline_eval_pkg
MODEL_PATH=/path/to/Qwen3.5-9B bash run_qwen35_rr_baseline.sh
```

Results from these two native protocols are not directly comparable to each
other or to the URM pointwise results, because both the prompts and the
scoring functions differ.

## Current baseline note

`output/benchmarks_qwen35_baseline` was generated with the SKYLENAGE-JUDGER
`GEN_TEMPLATE` prompt and `\boxed{A/B}` parsing. Its
`qwen35_9b_pairwise.scores.json` (64.96%) is therefore a SKYLENAGE-protocol
number, not a GenAI/URM 3-dim result.

For Qwen3.5 versus the URM judge on GenAI-Bench, rerun the pointwise package
with `CKPT=/path/to/Qwen3.5-9B` and the `t2i_final_prompt_3dim_v2.jsonl`
template.
