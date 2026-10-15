#!/usr/bin/env bash
set -euo pipefail

cd /inspire/qb-ilm/project/deepgen/wangdianyi-240107110022/BXH/RL_train/verl

CKPT=/inspire/qb-ilm/project/deepgen/wangdianyi-240107110022/BXH/RL_train/verl/checkpoints/CVPR_2027_Image_Editing_RL/qwen3_5_9b_pairwise_image_edit_mean_threshold_format_fix_data/global_step_150/actor

OUT=/inspire/qb-ilm/project/deepgen/wangdianyi-240107110022/BXH/RL_train/verl/checkpoints/CVPR_2027_Image_Editing_RL/qwen3_5_9b_pairwise_image_edit_mean_threshold_format_fix_data/global_step_150_merged

FULL_OUT=${OUT}_full_hf

python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$CKPT" \
  --target_dir "$OUT" \
  --trust-remote-code || true

