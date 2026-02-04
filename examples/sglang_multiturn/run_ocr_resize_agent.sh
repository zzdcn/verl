#!/bin/bash
# OCR Resize Agent Training Script
#
# This script trains an OCR agent with resize tool capability using GRPO algorithm.
# The agent learns to resize images appropriately to improve OCR accuracy.
#
# Requirements:
# - 8x H100 GPUs (adjust n_gpus_per_node for different setups)
# - Preprocessed OCR dataset (use examples/data_preprocess/ocr_resize_agent_loop.py)
# - Qwen2.5-VL-3B-Instruct model (or other VLM)
#
# Usage:
#   # Basic usage with default settings
#   bash run_ocr_resize_agent.sh
#
#   # Custom data path
#   bash run_ocr_resize_agent.sh data.train_files=/path/to/train.parquet
#
#   # Custom model
#   bash run_ocr_resize_agent.sh actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-7B-Instruct
#
#   # Adjust batch size for smaller GPUs
#   bash run_ocr_resize_agent.sh data.train_batch_size=64 actor_rollout_ref.rollout.n=8

set -x

# Increase file descriptor limit for large-scale training
ulimit -n 65536

# Project paths
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
TOOL_CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/ocr_resize_tool_config.yaml"

# Default data paths (override via command line)
DATA_DIR="$PROJECT_DIR/temp_resource/ocr_dummy/out"
TRAIN_DATA="${TRAIN_DATA:-$DATA_DIR/train.parquet}"
VAL_DATA="${VAL_DATA:-$DATA_DIR/test.parquet}"

# Default model path
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"

# GPU configuration (auto-detect if not provided)
if [ -z "${N_GPUS:-}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    N_GPUS="$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')"
  else
    N_GPUS=8
  fi
fi
N_NODES="${N_NODES:-1}"

# Dummy mode: safe defaults for tiny datasets / quick validation
DUMMY_MODE="${DUMMY_MODE:-1}"
if [ "$DUMMY_MODE" = "1" ]; then

  data_train_batch_size=${N_GPUS}
  data_gen_batch_size=${N_GPUS}
  data_val_batch_size=1
  rollout_n=8
  tensor_model_parallel_size=1
  ppo_micro_batch_size_per_gpu=1
  ppo_mini_batch_size=${N_GPUS}
  total_epochs=1
  save_freq=1000
  test_freq=2
  filter_overlong_prompts=False
  truncation=ignore
else
  data_train_batch_size=128
  data_gen_batch_size=""
  data_val_batch_size=""
  rollout_n=8
  tensor_model_parallel_size=2
  ppo_micro_batch_size_per_gpu=16
  ppo_mini_batch_size=128
  total_epochs=3
  save_freq=500
  test_freq=100
  filter_overlong_prompts=True
  truncation=error
fi

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='ocr_resize_grpo' \
    \
    `# Algorithm settings` \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    \
    `# Data settings` \
    data.train_batch_size=$data_train_batch_size \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=$filter_overlong_prompts \
    data.truncation=$truncation \
    data.return_raw_chat=True \
    data.return_multi_modal_inputs=True \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    \
    `# Model settings` \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    `# Actor settings` \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    `# Rollout settings` \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=$rollout_n \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    \
    `# Multi-turn and tool settings` \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=5 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG_PATH" \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    \
    `# Reference model settings` \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    `# Trainer settings` \
    trainer.project_name='ocr_resize_agent' \
    trainer.experiment_name='grpo_qwen2.5vl_3b_resize_tool' \
    trainer.logger='["console"]' \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes="$N_NODES" \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.total_epochs=$total_epochs \
    trainer.critic_warmup=0 \
    \
    `# Tracing settings (for debugging)` \
    actor_rollout_ref.rollout.trace.backend=tensorboard \
    actor_rollout_ref.rollout.trace.token2text=True \
    \
    "$@"
