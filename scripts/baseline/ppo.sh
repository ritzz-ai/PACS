#!/bin/bash
set -ux

METHOD=verl
ADV_ESTIMATOR=gae
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ROLLOUT_DATA_DIR="../outputs/rollout_data/${METHOD}_${ADV_ESTIMATOR}_${TIMESTAMP}"
PROJECT_NAME="${METHOD}_${ADV_ESTIMATOR}"
EXPERIMENT_NAME="Qwen2.5-7B_${TIMESTAMP}"
TENSORBOARD_DIR="../tensorboard_log/${PROJECT_NAME}/${EXPERIMENT_NAME}"

mkdir -p ${ROLLOUT_DATA_DIR}
mkdir -p ${TENSORBOARD_DIR}

cp $0 ${TENSORBOARD_DIR}/run.sh

python3 -m verl.trainer.main_ppo \
    --config-path="${PWD%/*}/configs" \
    --config-name=ppo_trainer \
    algorithm.adv_estimator=${ADV_ESTIMATOR} \
    data.train_files=datasets/deepscaler.parquet \
    data.val_files='["datasets/gsm8k.parquet", "datasets/math_500.parquet", "datasets/aime_2024.parquet", "datasets/aime_2025.parquet","datasets/amc23.parquet"]' \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="Qwen/Qwen2.5-7B" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path="Qwen/Qwen2.5-7B" \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_micro_batch_size_per_gpu=8 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "tensorboard"]' \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=8 \
    trainer.save_freq=20 \
    trainer.test_freq=-1 \
    trainer.total_epochs=20 \
    trainer.total_training_steps=300 \
    trainer.rollout_data_dir=${ROLLOUT_DATA_DIR}/rollout \
    trainer.validation_data_dir=${ROLLOUT_DATA_DIR}/validation \
    custom_reward_function.path=src/reward_function.py \
    custom_reward_function.name=compute_score \
    2>&1 | tee ../logs/${PROJECT_NAME}_${EXPERIMENT_NAME}.log
