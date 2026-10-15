# Pairwise Image-Edit GRPO Data Format

Each preference pair is stored as two rows. The preferred edited image is
`side=A`; the worse edited image is `side=B`. Both rows must share
`extra_info.pair_id`.

If your source file is one row per pair with fields such as `source_image`,
`winner_image`, `loser_image`, and `instruction`, convert it first:

```bash
python3 examples/reward_function/convert_pairwise_image_edit_jsonl.py \
  --input /path/to/rl_pairwise_non_adjacent_20k.jsonl \
  --output /path/to/rl_pairwise_non_adjacent_20k.verl.jsonl
```

By default, the converter does not repeat shared prompt text in every JSONL
row. The launch scripts pass the shared prompts once through:

```bash
data.system_prompt_file=examples/reward_function/pairwise_image_edit_stage3_system_prompt.txt
data.user_prompt_template_file=examples/reward_function/pairwise_image_edit_stage3_user_prompt_template.txt
```

The dataset only needs to keep the variable fields, especially `instruction`.
Use `--embed_system_prompt --embed_user_prompt` only if you need a
self-contained JSONL file.

```jsonl
{"data_source":"pairwise_image_edit","instruction":"Replace the sky with a sunset while preserving the building.","images":["/abs/path/source.png","/abs/path/edit_A.png"],"reward_model":{"ground_truth":"A>B"},"extra_info":{"pair_id":"sample-0001","side":"A","index":"sample-0001"}}
{"data_source":"pairwise_image_edit","instruction":"Replace the sky with a sunset while preserving the building.","images":["/abs/path/source.png","/abs/path/edit_B.png"],"reward_model":{"ground_truth":"A>B"},"extra_info":{"pair_id":"sample-0001","side":"B","index":"sample-0001"}}
```

Expected rollout JSON shape:

```json
{
  "eval_points": [
    {"question": "The sky is changed to a sunset.", "dimension": "Instruction Following", "score": 1},
    {"question": "The building structure is preserved.", "dimension": "Visual Consistency", "score": 1}
  ],
  "dimension_summary": {
    "Instruction Following": "...",
    "Visual Consistency": "...",
    "Visual Quality": "..."
  },
  "score_reason": "...",
  "final_score": 8.0
}
```

Scores in `[0, 1]` are used directly. Scores in `(1, score_max]` are divided by
`score_max`, which defaults to `10`.

The reward manager reads:

- `extra_info.pair_id`: pair identifier.
- `extra_info.side`: `A` or `B`.
- `instruction`: edit instruction substituted into the shared user prompt template.
- `images`: multimodal images consumed by the verl RL dataset.
- `data.system_prompt_file`: shared system prompt prepended by `RLHFDataset`.
- `data.user_prompt_template_file`: shared user prompt template used when the row
  has no `prompt` field.

If your dataset uses another prompt field name, pass it through the launch
script, for example `PROMPT_KEY=messages` or `PROMPT_KEY=conversations`.

For each training batch, keep both rows of a pair in the same batch. A simple
way is to store rows as adjacent `A, B` pairs and avoid shuffling at the row
level before batching.
