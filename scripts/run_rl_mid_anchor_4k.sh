#!/bin/bash
# RL (GRPO, pairwise) -- the released "mid + easy-anchor 4k" setting.
#
# Base = the merged stage-6 SFT best checkpoint (``ck250-merged``).
#
# DATA: ``rl_mid_anchor_4k`` -- 4,000 pairs (mid 3,600 + easy anchor 400) built
#   by ``scripts/build_rl_middle_anchor.py``.  Selection on the signed raw gap
#   ``score(winner) - score(loser)``: ``mid`` = ``0 < gap <= 1.0``,
#   ``easy anchor`` = ``gap > 1.5``; judge-wrong pairs and exact ties are
#   deliberately excluded (under the hard reward below those buckets are mostly
#   dead groups).  Held-out val = the 300 judge-wrong pairs
#   (``rl_hard_sel/.../rl_pairwise_val.verl.jsonl``) plus the tie val set.
#
# REWARD: ``mean_threshold_format`` -- the hard 0/1 step
#   ``R = 1[score > counterpart_threshold + margin]`` with
#   ``THRESHOLD_MARGIN=0.05`` (0.5 raw points on the 0-10 scale) and
#   ``threshold_type=mean``.  On this pool |gap| is p50=0.75 / p75=1.0 raw
#   points, so a 0.5-point margin lands in the informative band; the easy
#   anchors give the pair groups gradient that the mid pairs alone would not.
#
# Everything else (model, LoRA rank 32, rollout n=8, KL 0.02, the v3 schema
# guards) matches the v4/v5 recipe.  ``data.shuffle`` stays False on purpose:
# the pairwise reward needs the two rows of a pair (side A / side B, which the
# builder writes adjacently) inside the same training batch.  The builder
# already shuffles pair order, so no extra shuffling is needed.
#
# Chosen checkpoint: ``global_step_100`` (best genai_bench no-tie accuracy).
#
# Sweep helper (same data, three taus, arm at a time):
#   bash scripts/run_rl_tau_sweep.sh               # prints the 3 commands
#   RUN=1 bash scripts/run_rl_tau_sweep.sh         # actually launches them
#
# Usage (on the /efs node):
#   cd /efs/jacuzzitang/projects/URM
#   nohup bash scripts/run_rl_mid_anchor_4k.sh > logs/rl_t2i_midanchor4k.log 2>&1 &
#
# Small-step validation: SAVE_FREQ=20; stop after ~120-200 steps and merge/eval:
#   RUN_DIR=.../stage6_pairwise_rl_midanchor4k_v5_r2 MERGE_PREFIX=_mida \
#   BENCH_SUFFIX=_midanchor4k_v5_r2 STEPS="100" EVAL_TASKS="image_generation t2i" \
#   bash scripts/run_rl_bench.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EFS=${EFS:-/efs/jacuzzitang/projects/URM}
PYTHON=${PYTHON:-"$(dirname "$EFS")/verl_venv/bin/python"}
[ -x "$PYTHON" ] || { echo "ERROR: python not executable: $PYTHON"; exit 1; }
# The runner must import the fork in RM_RL-main, not a stale site-packages
# install from another checkout.
export PYTHONPATH="$ROOT/RM_RL-main${VERL_EXTRA_PYTHONPATH:+:$VERL_EXTRA_PYTHONPATH}"
# The platform's transformer_engine import needs an explicit CUDA include dir;
# without it PEFT's optional TE probe can abort on some hosts.
export NVTE_CUDA_INCLUDE_DIR=${NVTE_CUDA_INCLUDE_DIR:-/usr/local/cuda/include}
MODEL_PATH=${MODEL_PATH:-$EFS/output/skyjm_gen_9b_stage6/ck250-merged}
# unchanged hard4k data (judge-wrong pairs incl. the ties); swap in
# output/rl_label_hard_sel/rl_data/rl_pairwise_{train,val}_hard.verl.jsonl to
# try the all-hard-by-label-margin split instead.
TRAIN_FILE=${TRAIN_FILE:-$EFS/output/rl_mid_anchor_4k/rl_data/rl_pairwise_train_mixed.verl.jsonl}
TEST_FILE=${TEST_FILE:-$EFS/output/rl_hard_sel/rl_data/rl_pairwise_val.verl.jsonl}
# Optional SECOND val set made of the pairs the judge currently TIES
# (build_rl_tie_val.py; own data_source -> val-aux/urm_t2i_pairwise_tie/...).
# OFF by default: the normal val (the 300 judge-wrong pairs) is the metric we
# compare against v2/v3b, and it is the "之前做不对的" set.  The tie breakdown is
# checked post-hoc by re-scoring the tie pool with the merged checkpoint.
TIE_VAL_FILE=${TIE_VAL_FILE:-$EFS/output/rl_hard_sel/rl_data/rl_pairwise_val_tie.verl.jsonl}
USE_TIE_VAL=${USE_TIE_VAL:-1}
SYSTEM_PROMPT_FILE=${SYSTEM_PROMPT_FILE:-$EFS/RM_RL-main/examples/reward_function/t2i_3dim_v2_system_prompt.txt}
USER_PROMPT_TEMPLATE_FILE=${USER_PROMPT_TEMPLATE_FILE:-$EFS/RM_RL-main/examples/reward_function/t2i_3dim_v2_user_prompt_template.txt}
PAIRWISE_REWARD_MANAGER=${PAIRWISE_REWARD_MANAGER:-$EFS/RM_RL-main/examples/reward_function/pairwise_image_edit_reward_manager.py}

PROJECT_NAME=${PROJECT_NAME:-URM_T2I_RL}

# ---- reward calibration (from scripts/audit_rl_pairwise_data.py) -----------
# tau_raw is in raw final_score points on the 0-10 scale; margin 0 keeps the
# boundary band (judge gap ~0.5-1.0 raw points on this pool) informative.
REWARD_TYPE=${REWARD_TYPE:-mean_threshold_format}
PAIRWISE_MARGIN=${PAIRWISE_MARGIN:-0.0}
PAIRWISE_TAU_RAW=${PAIRWISE_TAU_RAW:-0.5}
THRESHOLD_MARGIN=${THRESHOLD_MARGIN:-0.05}
PAIRWISE_THRESHOLD_TYPE=${PAIRWISE_THRESHOLD_TYPE:-mean}
PAIRWISE_USE_DYNAMIC_THRESHOLD=${PAIRWISE_USE_DYNAMIC_THRESHOLD:-False}
PAIRWISE_DYNAMIC_SCALE_MULT=${PAIRWISE_DYNAMIC_SCALE_MULT:-1.0}
FORMAT_BONUS=${FORMAT_BONUS:-0.1}
# only used by REWARD_TYPE=mean_threshold_tiered:
#   r = 1[gap > THRESHOLD_MARGIN] + TIER_WEIGHT * 1[gap > TIER_MARGIN]
TIER_MARGIN=${TIER_MARGIN:-0.05}
TIER_WEIGHT=${TIER_WEIGHT:-1.0}
# experiment name carries the tau so a sweep does not collide in wandb / ckpt dir
EXPERIMENT_NAME=${EXPERIMENT_NAME:-stage6_pairwise_rl_midanchor4k_v5_r2}

# ---- v3 schema / degeneracy guards (unchanged) -----------------------------
MAX_CHECKPOINTS=${MAX_CHECKPOINTS:-16}
DIM_PENALTY=${DIM_PENALTY:-0.1}
INVALID_PENALTY=${INVALID_PENALTY:-0.2}

# ---- optimisation / rollout (unchanged from v3b) ---------------------------
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-5}
# 4000 pairs -> 250 steps/epoch at train_batch_size=32 rows (16 pairs).
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
LR=${LR:-2e-6}
KL=${KL:-0.02}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.7}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.0}
# rollout 数量：训练每个 prompt 采样 ROLLOUT_N 条（pair 组 = 2*ROLLOUT_N 条）。
# VAL_N 保持 1：验证就是单点判定，对齐部署口径，指标名是 .../mean@1，
# 与 v2/v3b/v4 的历史曲线直接可比（>1 时才会出现 std@N 等诊断键）。
ROLLOUT_N=${ROLLOUT_N:-8}
VAL_N=${VAL_N:-1}
FILTER_OVERLONG_PROMPTS=${FILTER_OVERLONG_PROMPTS:-True}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-512}

for f in "$MODEL_PATH" "$TRAIN_FILE" "$TEST_FILE" "$SYSTEM_PROMPT_FILE" "$USER_PROMPT_TEMPLATE_FILE" "$PAIRWISE_REWARD_MANAGER"; do
  [ -e "$f" ] || { echo "missing $f"; exit 1; }
done

VAL_FILES="${TEST_FILE}"
if [ "$USE_TIE_VAL" = "1" ] && [ -f "$TIE_VAL_FILE" ]; then
  VAL_FILES="[${TEST_FILE},${TIE_VAL_FILE}]"
fi

echo "[v4] $(date '+%F %T') exp=$PROJECT_NAME/$EXPERIMENT_NAME"
echo "[v4] python=$PYTHON"
echo "[v4] PYTHONPATH=$PYTHONPATH"
echo "[v4] NVTE_CUDA_INCLUDE_DIR=$NVTE_CUDA_INCLUDE_DIR"
"$PYTHON" -c 'import importlib.metadata as m; print("[v4] vllm=" + m.version("vllm"))' || {
  echo "ERROR: vllm is missing in $PYTHON"
  echo "       set PYTHON=$(dirname "$EFS")/verl_venv/bin/python, or rebuild that venv"
  exit 1
}
"$PYTHON" -c 'import verl; print("[v4] verl=" + verl.__file__)' || {
  echo "ERROR: verl is not importable with PYTHONPATH=$PYTHONPATH"
  exit 1
}
"$PYTHON" -c 'import inspect, peft; from peft import LoraConfig; assert "target_parameters" in inspect.signature(LoraConfig.__init__).parameters; print("[v4] peft=" + peft.__version__)' || {
  echo "ERROR: peft in $PYTHON does not support LoraConfig(target_parameters=...)"
  echo "       install a newer peft into the venv (peft>=0.19.1/0.20.0) and retry"
  exit 1
}
echo "[v4] train=$TRAIN_FILE"
echo "[v4] test =$VAL_FILES"
echo "[v4] reward=$REWARD_TYPE margin=$PAIRWISE_MARGIN tau_raw=$PAIRWISE_TAU_RAW thr_margin=$THRESHOLD_MARGIN"
echo "[v4] dynamic_threshold=$PAIRWISE_USE_DYNAMIC_THRESHOLD (mult=$PAIRWISE_DYNAMIC_SCALE_MULT) fmt_bonus=$FORMAT_BONUS tier_margin=${TIER_MARGIN:-n/a} tier_w=$TIER_WEIGHT"
echo "[v4] rollout_n=$ROLLOUT_N (pair group=$((2*ROLLOUT_N)))  val_n=$VAL_N  val_temp=$VAL_TEMPERATURE"

MAX_STEPS_ARG=()
if [ -n "${MAX_STEPS:-}" ]; then
  MAX_STEPS_ARG=(+trainer.max_steps="${MAX_STEPS}")
fi

"$PYTHON" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False +algorithm.pairwise_group_key=pair_id \
  data.train_files="${TRAIN_FILE}" data.val_files=${VAL_FILES} \
  data.prompt_key=prompt data.image_key=images \
  data.system_prompt_file="${SYSTEM_PROMPT_FILE}" data.user_prompt_template_file="${USER_PROMPT_TEMPLATE_FILE}" \
  data.train_batch_size=32 data.max_prompt_length=32768 data.max_response_length=4096 \
  data.filter_overlong_prompts=${FILTER_OVERLONG_PROMPTS} data.truncation=error data.shuffle=False \
  actor_rollout_ref.model.path="${MODEL_PATH}" actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp2 actor_rollout_ref.actor.optim.lr="${LR}" \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
  actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef="${KL}" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.n=${ROLLOUT_N} actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
  actor_rollout_ref.rollout.val_kwargs.n=${VAL_N} \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reward.num_workers=1 reward.reward_manager.source=importlib reward.reward_manager.name=PairwiseImageEditRewardManager \
  reward.reward_manager.module.path="${PAIRWISE_REWARD_MANAGER}" \
  +reward.pairwise_image_edit.reward_type="${REWARD_TYPE}" \
  +reward.pairwise_image_edit.margin="${PAIRWISE_MARGIN}" \
  +reward.pairwise_image_edit.tau_raw="${PAIRWISE_TAU_RAW}" \
  +reward.pairwise_image_edit.threshold_margin="${THRESHOLD_MARGIN}" \
  +reward.pairwise_image_edit.threshold_type="${PAIRWISE_THRESHOLD_TYPE}" \
  +reward.pairwise_image_edit.tier_margin="${TIER_MARGIN}" \
  +reward.pairwise_image_edit.tier_weight="${TIER_WEIGHT}" \
  +reward.pairwise_image_edit.use_dynamic_threshold="${PAIRWISE_USE_DYNAMIC_THRESHOLD}" \
  +reward.pairwise_image_edit.dynamic_scale_mult="${PAIRWISE_DYNAMIC_SCALE_MULT}" \
  +reward.pairwise_image_edit.format_bonus="${FORMAT_BONUS}" \
  +reward.pairwise_image_edit.score_max=10 +reward.pairwise_image_edit.min_checkpoint_count=1 \
  +reward.pairwise_image_edit.max_checkpoint_count="${MAX_CHECKPOINTS}" \
  +reward.pairwise_image_edit.dimension_penalty="${DIM_PENALTY}" \
  +reward.pairwise_image_edit.invalid_rollout_penalty="${INVALID_PENALTY}" \
  '+reward.pairwise_image_edit.dimensions=["Prompt Alignment","Aesthetics","Technical Quality"]' \
  '+reward.pairwise_image_edit.allowed_dimensions=["Prompt Alignment","Aesthetics","Technical Quality"]' \
  +reward.pairwise_image_edit.strict_summary=True \
  +reward.pairwise_image_edit.pair_id_key=pair_id +reward.pairwise_image_edit.side_key=side \
  trainer.balance_batch=True 'trainer.logger=["console","wandb"]' \
  trainer.project_name="${PROJECT_NAME}" trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node=8 trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ}" trainer.test_freq="${TEST_FREQ}" trainer.total_epochs="${TOTAL_EPOCHS}" \
  actor_rollout_ref.model.use_fused_kernels=False +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.actor.use_torch_compile=False actor_rollout_ref.ref.use_torch_compile=False \
  actor_rollout_ref.rollout.enable_chunked_prefill=True actor_rollout_ref.rollout.enable_prefix_caching=False \
  actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_SEQS} actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.model.lora_rank=32 actor_rollout_ref.model.lora_alpha=64 \
  actor_rollout_ref.model.target_modules=all-linear \
  'actor_rollout_ref.model.exclude_modules=.*visual.*|.*gdn.*|.*linear_attn.*' \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  +actor_rollout_ref.actor.checkpoint.save_lora_only=True \
  'actor_rollout_ref.actor.checkpoint.save_contents=["model"]' \
  'actor_rollout_ref.actor.checkpoint.load_contents=["model"]' \
  +data.apply_chat_template_kwargs.enable_thinking=False +data.mm_processor_kwargs.max_pixels=1048576 \
  data.filter_overlong_prompts_workers=8 trainer.resume_mode=disable \
  trainer.val_before_train=True \
  "${MAX_STEPS_ARG[@]}"
