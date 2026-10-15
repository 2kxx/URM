# URM - T2I checklist reward model (training code)

Training code for **URM**, the pointwise **checklist judge** for text-to-image
reward modeling: a Qwen3.5-9B vision-language model that scores one generated
image against its prompt and returns a JSON rubric whose `final_score` (0-10)
is the reward (`final_score / 10`).

```
input : prompt + 1 image
output: {"eval_points": [...], "dimension_summary": {...},
         "score_reason": "...", "final_score": 3.5}
reward: final_score / 10
```

This repo keeps only the two settings that produced the released checkpoints:

| stage | recipe | released ckpt |
|---|---|---|
| **SFT** (stage-6) | single-stage LoRA SFT on ~18.6k 3-dim-v2 rows with **fine-relabeled `final_score`** | `checkpoint-250` |
| **RL** (mid+anchor 4k) | GRPO pairwise on 4,000 mid + easy-anchor pairs, hard `mean_threshold_format` reward | `global_step_100` |

## Layout

```
scripts/                      training + data-build scripts
  train_stage6.sh             SFT entry point (stage-6)
  train_skyjm_gen_9b_stage4.sh  the LoRA SFT recipe it reuses
  run_stage6_pipeline.sh      annotate -> fine labels -> build -> train
  annotate_stage0_t2i.py      stage0 (0-4) annotation used to build fine labels
  build_stage6_fine_labels.py build_stage1_dataset.py split_stage1_dataset.py
  run_rl_mid_anchor_4k.sh     RL entry point (GRPO, mid+anchor 4k)
  build_rl_middle_anchor.py   builds the 4k mid+easy-anchor split
  build_rl_pairwise_mixed.py build_rl_tie_val.py audit_rl_pairwise_data.py
  merge_verl_lora.py          merge a verl LoRA ckpt into a full HF model
  run_rl_bench.sh             merge + eval RL checkpoints (genai_bench / mmrb2)
RM_RL-main/                   verl fork used for RL (v0.9.0.dev) + reward manager
  examples/reward_function/pairwise_image_edit_reward_manager.py
  examples/reward_function/t2i_3dim_v2_system_prompt.txt
  examples/reward_function/t2i_3dim_v2_user_prompt_template.txt
stage3/                       the 3-dim v2 eval prompt/template
reward_protocol/              serving-time request/response contract
docs/                         runbooks, audits and the results summary
env/                          pinned requirements
```

The training **data** and the released **weights** live on HuggingFace, not in
this repo (see below).

## Data and weights

| artifact | where |
|---|---|
| SFT rows (stage-6 train/val) + RL 4k split | HF dataset `2kxx/URM_Train_Data` |
| raw T2I corpus (prompts + images) | HF dataset `2kxx/URM_Data_T2I` |
| released merged judge | HF model `2kxx/URM` |

## SFT (stage-6)

Base = the stage0 pointwise scorer merged into Qwen3.5-9B
(`output/skyjm_gen_9b_pointwise_stage0/.../checkpoint-1500-merged`).

```bash
# 1) environment (needs a torch+CUDA host; qwen3_5 needs transformers==5.15.0)
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
pip install -r env/requirements.txt
pip install -e third_party/ms-swift           # pinned upstream checkout

# 2) rebuild the fine-relabeled data (GPU: vLLM/transformers annotation)
bash scripts/run_stage6_pipeline.sh           # writes stage1/..._stage6_{train,val}.jsonl

# 3) train (8 GPUs; LoRA r32/alpha64 all-linear, lr 1e-5, cosine, 2 epochs)
bash scripts/train_stage6.sh
```

Why stage-6 (fine labels): the Gemini `final_score` is coarse (25 values, 0.5
grid, 13% saturated at 10 -> the source of the pairwise ties).  Stage-6 replaces
it with `fine = 0.5*Gemini + 0.5*aligned(stage0)` (a global affine map of the
stage0 0-4 score onto the 0-10 scale, `y = 1.656*x + 1.851`).  Same rows, only
`final_score` changes; this halves the tie rate and lifts raw pairwise accuracy
by 5-8 points (see `docs/RESULTS_SUMMARY.md`).

## RL (mid + easy-anchor 4k)

Base = the merged SFT best ckpt (`ck250-merged`).

```bash
# 1) build the split from the audited judge-wrong/easy pools
python scripts/build_rl_middle_anchor.py \
  --analyzed output/rl_hard_sel/analyzed.jsonl \
  --out-dir  output/rl_mid_anchor_4k/rl_data      # 4000 pairs / 8000 rows

# 2) run GRPO pairwise (hard mean_threshold_format reward, margin 0.05, n=8)
bash scripts/run_rl_mid_anchor_4k.sh
```

The pairwise reward forces the two sides of a pair (`side=A/B`, written
adjacently by the builder) into the same group, so `data.shuffle=False`.  The
`mid` bucket (`0 < gap <= 1.0`) carries the fine discrimination; the `easy`
anchors (`gap > 1.5`) keep the groups non-degenerate.

## Docs

- `docs/RESULTS_SUMMARY.md` - stage-3/4/5/6 + two-stage comparison, why stage-6.
- `docs/RL_V4_RUNBOOK.md` - the RL recipe, reward math and read-out protocol.
- `docs/RL_AUDIT_20260908.md`, `docs/RL_MEANREWARD_AUDIT_20260914.md` - audits.
- `docs/BENCHMARK_PROTOCOLS.md` - genai_bench / mmrb2 / hpdv3 evaluation.

## Serving

The OpenAI-compatible serving contract (system prompt, request/response schema,
token budget) is in `reward_protocol/`.
