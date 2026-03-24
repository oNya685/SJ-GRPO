# Discliamer: the model used in the script is only for academic purpose.
hf auth login

set -x

# Data preparation scripts are available in ``examples/data_preprocess``.
# Example usage:
#
#   python3 examples/data_preprocess/math_dataset.py --local_dir data/math

math_train_path=$HOME/data/math/train.parquet
math_test_path=$HOME/data/math/test.parquet

train_files="['$math_train_path']"
test_files="['$math_test_path']"

# prepare model ckpt
hf download Qwen/Qwen3-4B-Instruct --local-dir huggingface.co/Qwen/Qwen3-4B-Instruct &
wait

python3 -m recipe.sj-grpo.sj_grpo_main_ppo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path="huggingface.co/Qwen/Qwen3-4B-Instruct" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm  \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name='sj-grpo-vllm' \
    trainer.val_before_train=True \
    trainer.experiment_name='Qwen3-4B-Instruct_debug' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    trainer.total_epochs=30 $@