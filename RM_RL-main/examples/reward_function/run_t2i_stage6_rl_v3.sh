#!/bin/bash
# ---------------------------------------------------------------------------
# URM T2I judge RL (GRPO) v3 -- "improved recipe" small-step validation run.
#
# Base = stage6 ck250 (same as v2). v3 fixes the three v2 failure modes:
#   B) 4-dim taxonomy drift        -> eval_points dimension whitelist
#      (allowed_dimensions) + strict dimension_summary + dimension_penalty
#   A) repetition-loop / runaway    -> invalid_rollout_penalty (explicitly
#      negative for non-terminating/truncated rollouts) + max_checkpoint_count
#      cap so a bloated-but-parseable checklist also fails format.
#   #2) linear_attn got LoRA'd      -> exclude_modules also drops gdn /
#      linear_attn so the LoRA can later be loaded as a vLLM adapter too.
# System prompt / eval template already de-primed (10:39 edit).
#
# Usage (run on the /efs node that has the verl venv + checkpoints):
#   cd /efs/jacuzzitang/projects/URM
#   nohup bash RM_RL-main/examples/reward_function/run_t2i_stage6_rl_v3.sh \
#        > logs/rl_t2i_v3.log 2>&1 &
#
# Small-step validation: checkpoints save every SAVE_FREQ=20; stop when you
# have the step you want (ctrl-c after e.g. step 120) or, if your verl supports
# it, set VALIDATION_STEPS=120 to stop the run automatically.
#
# Env knobs (all optional):
#   MODEL_PATH   merged stage6 ck250 (default /efs/.../ck250-merged)
#   TRAIN_FILE   hard4k all-wrong train rows (default v2 path)
#   TEST_FILE    held-out val rows      (default v2 path)
#   EXPERIMENT_NAME (default stage6_pairwise_rl_hard4k_v3)
#   TOTAL_EPOCHS / SAVE_FREQ / TEST_FREQ / ROLLOUT_N / ... pass-through
#   FORMAT_BONUS (0.15) MAX_CHECKPOINTS (16) DIM_PENALTY (0.1) INVALID_PENALTY (0.2)
#   ROLLOUT_TEMPERATURE (0.7)  train rollout temp. v2 used verl default 1.0,
#      which almost never samples the greedy-only repetition attractor, so the
#      reward penalty never reaches it. Lowering to ~0.7 brings the rollouts
#      into the near-greedy regime where the loop/format pathologies actually
#      show up in training and get penalized.
#   VAL_TEMPERATURE (0.0)      val/rollout eval temp used for the format/val
#      metric; keep greedy so val format matches the real benchmark eval.
#   KL_LOSS_COEF (0.02)        v2 value; raise to ~0.03-0.05 if you see the
#      policy drifting far from the clean stage6 base (4-dim relapse, bloat).
#   VALIDATION_STEPS (unset) -> if set, append trainer.max_steps
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HERE=$ROOT/RM_RL-main/examples/reward_function

EFS=${EFS:-/efs/jacuzzitang/projects/URM}
export PYTHON=${PYTHON:-"$(dirname "$EFS")/verl_venv/bin/python"}
export MODEL_PATH=${MODEL_PATH:-$EFS/output/skyjm_gen_9b_stage6/ck250-merged}
export TRAIN_FILE=${TRAIN_FILE:-$EFS/output/rl_hard_sel/rl_data/rl_pairwise_train_allwrong.verl.jsonl}
export TEST_FILE=${TEST_FILE:-$EFS/output/rl_hard_sel/rl_data/rl_pairwise_val.verl.jsonl}

export PROJECT_NAME=${PROJECT_NAME:-URM_T2I_RL}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-stage6_pairwise_rl_hard4k_v3}
export FORMAT_BONUS=${FORMAT_BONUS:-0.15}
export MAX_CHECKPOINTS=${MAX_CHECKPOINTS:-16}
export DIM_PENALTY=${DIM_PENALTY:-0.1}
export INVALID_PENALTY=${INVALID_PENALTY:-0.2}
export PAIRWISE_TAU=${PAIRWISE_TAU:-0.2}
export ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.7}
export VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.0}

# canonical dims (system prompt requires exactly these)
DIMS='["Prompt Alignment","Aesthetics","Technical Quality"]'

EXTRA_ARGS=(
  # ---- reproduce v2 rollout/model overrides (final config that v2 used) ----
  actor_rollout_ref.model.use_fused_kernels=False
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa
  actor_rollout_ref.actor.use_torch_compile=False
  actor_rollout_ref.ref.use_torch_compile=False
  actor_rollout_ref.rollout.enable_chunked_prefill=True
  actor_rollout_ref.rollout.enable_prefix_caching=False
  actor_rollout_ref.rollout.max_num_seqs=512
  actor_rollout_ref.rollout.load_format=safetensors
  actor_rollout_ref.rollout.layered_summon=True
  actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE}
  actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  +data.apply_chat_template_kwargs.enable_thinking=False
  +data.mm_processor_kwargs.max_pixels=1048576
  data.filter_overlong_prompts_workers=8
  trainer.resume_mode=disable

  # ---- LoRA: exclude the linear-attention (gdn) tower that vLLM LoRA cannot
  #      serve; keep dense attention + MLP trainable. Verify after merge that
  #      no linear_attn/gdn deltas remain (scripts/verify_merge.py).
  actor_rollout_ref.model.lora_rank=${LORA_RANK:-32}
  actor_rollout_ref.model.lora_alpha=${LORA_ALPHA:-64}
  actor_rollout_ref.model.target_modules=all-linear
  'actor_rollout_ref.model.exclude_modules=.*visual.*|.*gdn.*|.*linear_attn.*'
  +actor_rollout_ref.actor.checkpoint.save_lora_only=True
  'actor_rollout_ref.actor.checkpoint.save_contents=["model"]'
  'actor_rollout_ref.actor.checkpoint.load_contents=["model"]'

  # ---- v3 reward shaping (mean_threshold like v2 + format/schema guards) ----
  +reward.pairwise_image_edit.reward_type=mean_threshold_format
  +reward.pairwise_image_edit.threshold_margin=0.0
  +reward.pairwise_image_edit.dimensions=${DIMS}
  +reward.pairwise_image_edit.allowed_dimensions=${DIMS}
  +reward.pairwise_image_edit.strict_summary=True
  +reward.pairwise_image_edit.max_checkpoint_count=${MAX_CHECKPOINTS}
  +reward.pairwise_image_edit.dimension_penalty=${DIM_PENALTY}
  +reward.pairwise_image_edit.invalid_rollout_penalty=${INVALID_PENALTY}
)

if [ -n "${VALIDATION_STEPS:-}" ]; then
  EXTRA_ARGS+=(+trainer.max_steps="${VALIDATION_STEPS}")
fi

echo "[t2i-v3] $(date '+%F %T') MODEL=$MODEL_PATH"
echo "[t2i-v3] train=$TRAIN_FILE"
echo "[t2i-v3] test =$TEST_FILE"
echo "[t2i-v3] exp  =$PROJECT_NAME/$EXPERIMENT_NAME"
echo "[t2i-v3] fmt_bonus=$FORMAT_BONUS max_ckpts=$MAX_CHECKPOINTS dim_pen=$DIM_PENALTY invalid_pen=$INVALID_PENALTY"
echo "[t2i-v3] rollout_temp=$ROLLOUT_TEMPERATURE val_temp=$VAL_TEMPERATURE"

cd "$ROOT"
bash "$HERE/run_t2i_pairwise_qwen3_5_9b_fsdp.sh" "${EXTRA_ARGS[@]}"
