#!/usr/bin/env bash
# Pairwise image-edit reward-model RL | GRPO | Qwen3.5-9B | FSDP
#
# Required:
#   MODEL_PATH: local path or HF id of the Qwen3.5-9B SFT checkpoint/model
#   TRAIN_FILE or TRAIN_FILES: pairwise image-edit training data
#
# Each preference pair is represented by two prompt rows:
#   - side=A: the preferred edited image
#   - side=B: the worse edited image
# Both rows must share the same extra_info.pair_id.

set -xeuo pipefail

INFER_BACKEND=${INFER_BACKEND:-vllm}
PYTHON=${PYTHON:-python3}
DEVICE=${DEVICE:-$($PYTHON -c 'import torch_npu' 2>/dev/null && echo npu || echo gpu)}

MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to your Qwen3.5-9B model/checkpoint path, not the data jsonl}
TRAIN_FILE=${TRAIN_FILE:-${TRAIN_FILES:?Set TRAIN_FILE or TRAIN_FILES to a JSONL/JSON/parquet path}}
TEST_FILE=${TEST_FILE:-${VAL_FILES:-$TRAIN_FILE}}
PROMPT_KEY=${PROMPT_KEY:-prompt}
IMAGE_KEY=${IMAGE_KEY:-images}
SYSTEM_PROMPT_FILE=${SYSTEM_PROMPT_FILE:-examples/reward_function/pairwise_image_edit_stage3_system_prompt.txt}
USER_PROMPT_TEMPLATE_FILE=${USER_PROMPT_TEMPLATE_FILE:-examples/reward_function/pairwise_image_edit_stage3_user_prompt_template.txt}

NNODES=${NNODES:-1}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-}

# verl's data.train_batch_size counts prompt rows. Since every pair has A/B rows,
# the actual prompt batch size is 2 * PAIR_BATCH_SIZE.
PAIR_BATCH_SIZE=${PAIR_BATCH_SIZE:-32}
TRAIN_BATCH_SIZE=$((PAIR_BATCH_SIZE * 2))
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}

ROLLOUT_N=${ROLLOUT_N:-4}
ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.6}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-32768}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-49152}

ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.01}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-5}

PROJECT_NAME=${PROJECT_NAME:-verl_pairwise_image_edit_grpo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_5_9b_pairwise_image_edit_${INFER_BACKEND}_fsdp}
PAIRWISE_REWARD_MANAGER=${PAIRWISE_REWARD_MANAGER:-examples/reward_function/pairwise_image_edit_reward_manager.py}

n_devices_per_node=${NDEVICES_PER_NODE:-8}
case "${DEVICE}" in
    gpu)
        ;;
    npu)
        export HCCL_CONNECT_TIMEOUT=1500
        export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
        export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
        export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
        n_devices_per_node=${NDEVICES_PER_NODE:-16}
        ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.5}
        ;;
    *)
        echo "Unsupported DEVICE=${DEVICE}. Expected 'gpu' or 'npu'." >&2
        exit 1
        ;;
esac

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    +algorithm.pairwise_group_key=pair_id
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.prompt_key="${PROMPT_KEY}"
    data.image_key="${IMAGE_KEY}"
    data.system_prompt_file="${SYSTEM_PROMPT_FILE}"
    data.user_prompt_template_file="${USER_PROMPT_TEMPLATE_FILE}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation=error
    data.shuffle=False
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

REWARD=(
    reward.num_workers=1
    reward.reward_manager.source=importlib
    reward.reward_manager.name=PairwiseImageEditRewardManager
    reward.reward_manager.module.path="${PAIRWISE_REWARD_MANAGER}"
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
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${n_devices_per_node}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
)

EXTRA=(
    actor_rollout_ref.model.use_fused_kernels=True
    actor_rollout_ref.rollout.multi_stage_wake_up=True
)

if [ "${DEVICE}" = npu ]; then
    EXTRA+=(
        actor_rollout_ref.model.use_fused_kernels=False
        actor_rollout_ref.rollout.multi_stage_wake_up=False
        actor_rollout_ref.rollout.free_cache_engine=False
        +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    )
fi

"$PYTHON" -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
