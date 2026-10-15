#!/usr/bin/env bash
# Pairwise image-edit reward-model RL | GRPO | Qwen3.5-35B-A3B | FSDP
#
# Each preference pair is represented by two prompt rows:
#   - side=A: the preferred edited image
#   - side=B: the worse edited image
# Both rows must share the same extra_info.pair_id.

set -xeuo pipefail

########################### user-adjustable ###########################
# DEVICE is auto-detected by probing torch_npu; override only for special cases.
DEVICE=${DEVICE:-$(python3 -c 'import torch_npu' 2>/dev/null && echo npu || echo gpu)}
INFER_BACKEND=${INFER_BACKEND:-vllm}

PROJECT_NAME=${PROJECT_NAME:-verl_pairwise_image_edit_grpo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_5_35b_pairwise_image_edit_${INFER_BACKEND}_fsdp}

NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-}
NNODES=${NNODES:-1}

GEN_TP=${GEN_TP:-4}
SP_SIZE=${SP_SIZE:-1}
FSDP_SIZE=${FSDP_SIZE:-}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.6}
ROLLOUT_N=${ROLLOUT_N:-4}

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3.5-35B-A3B"}
TRAIN_FILE=${TRAIN_FILE:-${TRAIN_FILES:?Set TRAIN_FILES or TRAIN_FILE to a JSONL/JSON/parquet path}}
TEST_FILE=${TEST_FILE:-${VAL_FILES:-$TRAIN_FILE}}
PROMPT_KEY=${PROMPT_KEY:-prompt}
IMAGE_KEY=${IMAGE_KEY:-images}
SYSTEM_PROMPT_FILE=${SYSTEM_PROMPT_FILE:-examples/reward_function/pairwise_image_edit_stage3_system_prompt.txt}
USER_PROMPT_TEMPLATE_FILE=${USER_PROMPT_TEMPLATE_FILE:-examples/reward_function/pairwise_image_edit_stage3_user_prompt_template.txt}

# verl's data.train_batch_size counts prompt rows. Since every pair has A/B rows,
# the actual prompt batch size is 2 * PAIR_BATCH_SIZE.
PAIR_BATCH_SIZE=${PAIR_BATCH_SIZE:-16}
TRAIN_BATCH_SIZE=$((PAIR_BATCH_SIZE * 2))
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}

ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.01}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-5}

PAIRWISE_REWARD_MANAGER=${PAIRWISE_REWARD_MANAGER:-examples/reward_function/pairwise_image_edit_reward_manager.py}
LOG_DIR=${LOG_DIR:-logs}
########################### end user-adjustable ###########################

########################### derived defaults ###########################
n_devices_per_node=${NDEVICES_PER_NODE:-8}
fsdp_size=${FSDP_SIZE:-8}

case "${DEVICE}" in
    gpu)
        ;;
    npu)
        export HCCL_CONNECT_TIMEOUT=1500
        export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
        export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
        export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
        n_devices_per_node=16
        fsdp_size=16
        ;;
    *)
        echo "Unsupported DEVICE=${DEVICE}. Expected 'gpu' or 'npu'." >&2
        exit 1
        ;;
esac

start_time=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p "${LOG_DIR}"

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    +algorithm.pairwise_group_key=pair_id
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation=error
    data.prompt_key="${PROMPT_KEY}"
    data.image_key="${IMAGE_KEY}"
    data.system_prompt_file="${SYSTEM_PROMPT_FILE}"
    data.user_prompt_template_file="${USER_PROMPT_TEMPLATE_FILE}"
    data.shuffle=False
)

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size}
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.fsdp_config.offload_policy=True
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE}
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

REF=(
    actor_rollout_ref.ref.strategy=fsdp2
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True
    actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE}
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.fsdp_config.offload_policy=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.ignore_eos=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=8192
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=6144
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
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node=${n_devices_per_node}
    trainer.nnodes=${NNODES}
    trainer.balance_batch=False
    trainer.val_before_train=True
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
)

case "${DEVICE}" in
    gpu)
        ;;
    npu)
        ROLLOUT+=(
            +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0
        )
        ;;
    *)
        echo "Unsupported DEVICE=${DEVICE}. Expected 'gpu' or 'npu'." >&2
        exit 1
        ;;
esac

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${ROLLOUT[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "$@" 2>&1 | tee "${LOG_DIR}/qwen3_5-35b-pairwise-image-edit-${start_time}.log"
