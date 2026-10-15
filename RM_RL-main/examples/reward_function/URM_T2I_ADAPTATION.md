# Adapting RM_RL (pairwise_image_edit) to the URM T2I judge

## What the recipe actually does

`run_pairwise_image_edit_grpo_*.sh` + `pairwise_image_edit_reward_manager.py`
implement **pairwise judge RL**:

1. Each preference pair is split into **two prompt rows** sharing
   `extra_info.pair_id`; the winner row has `extra_info.side=A`, loser `side=B`.
2. The judge (Qwen3.5-VL) is rolled out on each row alone and must output the
   usual JSON checklist:
   `eval_points` / `dimension_summary` / `score_reason` / `final_score`.
3. GRPO groups the A/B rows by `pair_id`. The reward manager turns the model's
   `final_score` ordering into a Bradley-Terry reward
   (`sigmoid((sA - sB - margin)/tau)` for A, mirrored for B) plus a small
   `format_bonus` for well-formed JSON.
4. Training therefore directly pushes the judge to **separate winner from loser
   (fewer ties / fewer wrong orderings)** — exactly the "hard pair" objective.

No code in the reward path is edit-specific except one spot (see below), so the
recipe maps 1:1 onto a T2I preference judge.

## What "my generation" must look like (the contracts)

### 1) Rollout JSON (what the model generates per row)
Keep the exact stage6 schema — it is already correct:
```json
{"eval_points":[{"question":"...","dimension":"Prompt Alignment","score":0|1}, ...],
 "dimension_summary":{"Prompt Alignment":"...","Aesthetics":"...","Technical Quality":"..."},
 "score_reason":"...",
 "final_score": 7.5}
```
Two gotchas:
- `final_score` is normalized to `[0,1]` by `score_max` (default 10) before the
  BT term — that is fine.
- **`dimension_summary` keys are checked against a fixed list.** The image-edit
  version required `("Instruction Following","Visual Consistency","Visual
  Quality")`. For T2I you must override to your three dimensions, otherwise the
  `format_bonus` is always 0:
  ```
  +reward.pairwise_image_edit.dimensions="Prompt Alignment,Aesthetics,Technical Quality"
  ```
  (the reward manager now accepts this as list or comma/pipe-separated string;
  default stays the image-edit dims).

### 2) Prompt given to the judge (per row = ONE candidate image)
`examples/reward_function/t2i_3dim_v2_user_prompt_template.txt`
```
<image>
The image above is the generated image.
Text prompt: {instruction}

Generate checklist-style evaluation points for this text-to-image case, score each point as 0 or 1, assign a final score from 0 to 10, and output the JSON result.
```
plus the shared T2I system prompt
`examples/reward_function/t2i_3dim_v2_system_prompt.txt` (extracted verbatim
from `stage3/t2i_final_prompt_3dim_v2.jsonl`, i.e. the exact prompt the stage6
judge was trained/evaled with). One `<image>` token per row matches the
`images=[<that image>]` field (image-edit used two images per row; T2I uses one).

### 3) Training data JSONL (two rows per pair)
```jsonl
{"data_source":"urm_t2i_pairwise","instruction":"a red car in rain",
 "images":["/abs/winner.png"],
 "reward_model":{"ground_truth":"A>B"},
 "extra_info":{"pair_id":"gen-x-000123","side":"A","index":"gen-x-000123"}}
{"data_source":"urm_t2i_pairwise","instruction":"a red car in rain",
 "images":["/abs/loser.png"],
 "extra_info":{"pair_id":"gen-x-000123","side":"B","index":"gen-x-000123"}}
```
Build it from the `rl_hard_top4000.jsonl` / `analyzed.jsonl` produced by
`scripts/score_pairs_for_rl.py`:
```bash
python3 examples/reward_function/convert_urm_t2i_pairwise_jsonl.py \
    --input  /efs/jacuzzitang/projects/URM/output/rl_hard_sel/rl_hard_top4000.jsonl \
    --output /efs/jacuzzitang/projects/URM/rl_data/rl_pairwise_4k_train.verl.jsonl
```
Keep both rows of every pair adjacent and set `data.shuffle=False` (as the
recipe does) so GRPO always sees A and B together. Hold out a disjoint subset
(e.g. 200-400 pairs) as `TEST_FILE` for `val_before_train`/`test_freq`.

## Launch (reuses the image-edit launcher with env overrides)
```bash
MODEL_PATH=/efs/.../stage6-merged \
TRAIN_FILE=/efs/.../rl_pairwise_4k_train.verl.jsonl \
TEST_FILE=/efs/.../rl_pairwise_val.verl.jsonl \
bash examples/reward_function/run_t2i_pairwise_qwen3_5_9b_fsdp.sh \
    ...same knobs as run.sh (PAIR_BATCH_SIZE, ACTOR_LR, KL_LOSS_COEF, ...)
```
`MODEL_PATH` must be a **merged** full-weight judge checkpoint — merge the best
stage6 LoRA into its base first with ms-swift `merge-lora`.

## Things to decide when you actually train
- Base for RL init: stage6 best raw (ck250) or combined (ck500) merged.
- Reward only uses `final_score`. If you also want checklist sum to matter in
  the reward signal, add `pts_sum` into the manager's score (not required;
  training `final_score` alone usually also fixes the tie that combined was
  patching).
- `MODEL_PATH` image budget: keep `+data.mm_processor_kwargs.max_pixels`
  matching the judge's training (1024x1024 ~ 1048576 px as in run.sh).
