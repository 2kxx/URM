#!/usr/bin/env bash
# Pairwise image-edit reward-model RL | GRPO | Qwen2.5-VL-7B | FSDP
#
# Each preference pair is represented by two prompt rows:
#   - side=A: the preferred edited image
#   - side=B: the worse edited image
# Both rows must share the same extra_info.pair_id.

set -xeuo pipefail

INFER_BACKEND=${INFER_BACKEND:-vllm}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}
TRAIN_FILES=${TRAIN_FILES:?Set TRAIN_FILES to a JSONL/JSON/parquet path}
VAL_FILES=${VAL_FILES:-$TRAIN_FILES}
PROMPT_KEY=${PROMPT_KEY:-prompt}
IMAGE_KEY=${IMAGE_KEY:-images}
SYSTEM_PROMPT_FILE=${SYSTEM_PROMPT_FILE:-examples/reward_function/pairwise_image_edit_stage3_system_prompt.txt}
USER_PROMPT_TEMPLATE_FILE=${USER_PROMPT_TEMPLATE_FILE:-examples/reward_function/pairwise_image_edit_stage3_user_prompt_template.txt}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# This is the number of A/B pairs per train batch. verl's data.train_batch_size
# counts prompt rows, so the actual prompt batch is 2 * PAIR_BATCH_SIZE.
PAIR_BATCH_SIZE=${PAIR_BATCH_SIZE:-32}
train_batch_size=$((PAIR_BATCH_SIZE * 2))
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-128}

rollout_n=${ROLLOUT_N:-4}
rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.6}

max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-2048}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}

actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.01}

total_epochs=${TOTAL_EPOCHS:-3}
save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:-5}

project_name=${PROJECT_NAME:-verl_pairwise_image_edit_grpo}
experiment_name=${EXPERIMENT_NAME:-qwen2_5_vl_7b_pairwise_image_edit_${INFER_BACKEND}_fsdp}

PAIRWISE_REWARD_MANAGER=${PAIRWISE_REWARD_MANAGER:-examples/reward_function/pairwise_image_edit_reward_manager.py}

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    +algorithm.pairwise_group_key=pair_id
    data.train_files="$TRAIN_FILES"
    data.val_files="$VAL_FILES"
    data.prompt_key="$PROMPT_KEY"
    data.image_key="$IMAGE_KEY"
    data.system_prompt_file="$SYSTEM_PROMPT_FILE"
    data.user_prompt_template_file="$USER_PROMPT_TEMPLATE_FILE"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation=error
    data.shuffle=False
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

REWARD=(
    reward.num_workers=1
    reward.reward_manager.source=importlib
    reward.reward_manager.name=PairwiseImageEditRewardManager
    reward.reward_manager.module.path="$PAIRWISE_REWARD_MANAGER"
    +reward.pairwise_image_edit.margin=${PAIRWISE_MARGIN:-0.05}
    +reward.pairwise_image_edit.tau=${PAIRWISE_TAU:-0.1}
    +reward.pairwise_image_edit.format_bonus=${FORMAT_BONUS:-0.1}
    +reward.pairwise_image_edit.score_max=${SCORE_MAX:-10}
    +reward.pairwise_image_edit.min_checkpoint_count=${MIN_CHECKPOINT_COUNT:-1}
    +reward.pairwise_image_edit.pair_id_key=pair_id
    +reward.pairwise_image_edit.side_key=side
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
)

EXTRA=(
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.model.use_fused_kernels=True
    actor_rollout_ref.rollout.multi_stage_wake_up=True
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
)

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
