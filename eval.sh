# original models
cur_time=$(date "+%Y-%m-%d-%H-%M-%S")
python -m utils.eval /data/models/Qwen2.5-3B-Instruct > eval_3B_$cur_time.log 2>&1
python -m utils.eval /data/models/Qwen2.5-7B-Instruct > eval_7B_$cur_time.log 2>&1

# GRPO models
# /data/cuisijia/verl_kuhn_poker/grpo_qwen2.5_3b_instruct_v0819/global_step_25/actor
# python -m scripts.model_merger merge \
#   --backend fsdp \
#   --local_dir /data/cuisijia/verl_kuhn_poker/grpo_qwen2.5_3b_instruct_v0819/global_step_25/actor \
#   --target_dir /data/cuisijia/verl_kuhn_poker/grpo_qwen2.5_3b_instruct_v0819/merged_hf_actor_step_25

python -m utils.eval /data/cuisijia/verl_kuhn_poker/grpo_qwen2.5_3b_instruct_v0819/merged_hf_actor_step_25 > eval_grpo_3B_$cur_time.log 2>&1

# /data/cuisijia/verl_kuhn_poker/grpo_qwen2.5_3b_instruct_mixed_64k_v0821/global_step_25/actor
# python -m scripts.model_merger merge \
#   --backend fsdp \
#   --local_dir /data/cuisijia/verl_kuhn_poker/grpo_qwen2.5_3b_instruct_mixed_64k_v0821/global_step_25/actor \
#   --target_dir /data/cuisijia/verl_kuhn_poker/grpo_qwen2.5_3b_instruct_mixed_64k_v0821/merged_hf_actor_step_25

python -m utils.eval /data/cuisijia/verl_kuhn_poker/grpo_qwen2.5_3b_instruct_mixed_64k_v0821/merged_hf_actor_step_25 > eval_grpo_3B_mixed_64k_$cur_time.log 2>&1
