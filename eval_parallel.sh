cd /home/cuisijia/llm_opponent_modeling

cur_time=$(date "+%m-%d-%H-%M")

# 3B model
echo "eval /data/models/Qwen2.5-3B-Instruct at $(date "+%m-%d-%H-%M")"
python -m utils.eval_llm_parallel /data/models/Qwen2.5-3B-Instruct > eval_logs/${cur_time}_eval_parallel_3B.log 2>&1
echo "done /data/models/Qwen2.5-3B-Instruct at $(date "+%m-%d-%H-%M")"

# 3B grpo-64k
echo "eval /data/cuisijia/verl_kuhn_poker/qwen2.5_3b_64k_v0830/merged_actor_100 at $(date "+%m-%d-%H-%M")"
python -m utils.eval_llm_parallel /data/cuisijia/verl_kuhn_poker/qwen2.5_3b_64k_v0830/merged_actor_100 > eval_logs/${cur_time}_eval_parallel_3b_64k_v0830.log 2>&1
echo "done /data/cuisijia/verl_kuhn_poker/qwen2.5_3b_64k_v0830/merged_actor_100 at $(date "+%m-%d-%H-%M")"

# 3B mixed-64k
echo "eval /data/cuisijia/verl_kuhn_poker/qwen2.5_3b_mixed_64k_v0829/merged_actor_100 at $(date "+%m-%d-%H-%M")"
python -m utils.eval_llm_parallel /data/cuisijia/verl_kuhn_poker/qwen2.5_3b_mixed_64k_v0829/merged_actor_100 > eval_logs/${cur_time}_eval_parallel_3b_mixed_64k_v0829.log 2>&1
echo "done /data/cuisijia/verl_kuhn_poker/qwen2.5_3b_mixed_64k_v0829/merged_actor_100 at $(date "+%m-%d-%H-%M")"

# 3B mixedoppo-64k
echo "eval /data/cuisijia/verl_kuhn_poker/qwen2.5_3b_mixedoppo_64k_v0826/merged_actor_100 at $(date "+%m-%d-%H-%M")"
python -m utils.eval_llm_parallel /data/cuisijia/verl_kuhn_poker/qwen2.5_3b_mixedoppo_64k_v0826/merged_actor_100 > eval_logs/${cur_time}_eval_parallel_3b_mixedoppo_64k_v0826.log 2>&1
echo "done /data/cuisijia/verl_kuhn_poker/qwen2.5_3b_mixedoppo_64k_v0826/merged_actor_100 at $(date "+%m-%d-%H-%M")"

# 7B model
echo "eval /data/models/Qwen2.5-7B-Instruct at $(date "+%m-%d-%H-%M")"
python -m utils.eval_llm_parallel /data/models/Qwen2.5-7B-Instruct > eval_logs/${cur_time}_eval_parallel_7B.log 2>&1
echo "done /data/models/Qwen2.5-7B-Instruct at $(date "+%m-%d-%H-%M")"

# 7B grpo-64k
echo "eval /data/cuisijia/verl_kuhn_poker/qwen2.5_7b_v0815/merged_actor_100 at $(date "+%m-%d-%H-%M")"
python -m utils.eval_llm_parallel /data/cuisijia/verl_kuhn_poker/qwen2.5_7b_v0815/merged_actor_100 > eval_logs/${cur_time}_eval_parallel_7b_v0815.log 2>&1
echo "done /data/cuisijia/verl_kuhn_poker/qwen2.5_7b_v0815/merged_actor_100 at $(date "+%m-%d-%H-%M")"

# 7B mixed-64k
echo "eval /data/cuisijia/verl_kuhn_poker/qwen2.5_7b_mixed_64k_v0824/merged_actor_100 at $(date "+%m-%d-%H-%M")"
python -m utils.eval_llm_parallel /data/cuisijia/verl_kuhn_poker/qwen2.5_7b_mixed_64k_v0824/merged_actor_100 > eval_logs/${cur_time}_eval_parallel_7b_mixed_64k_v0824.log 2>&1
echo "done /data/cuisijia/verl_kuhn_poker/qwen2.5_7b_mixed_64k_v0824/merged_actor_100 at $(date "+%m-%d-%H-%M")"

# 7B mixedoppo-64k
echo "eval /data/cuisijia/verl_kuhn_poker/qwen2.5_7b_mixedoppo_64k_v0826/merged_actor_100 at $(date "+%m-%d-%H-%M")"
python -m utils.eval_llm_parallel /data/cuisijia/verl_kuhn_poker/qwen2.5_7b_mixedoppo_64k_v0826/merged_actor_100 > eval_logs/${cur_time}_eval_parallel_7b_mixedoppo_64k_v0826.log 2>&1
echo "done /data/cuisijia/verl_kuhn_poker/qwen2.5_7b_mixedoppo_64k_v0826/merged_actor_100 at $(date "+%m-%d-%H-%M")"

echo "all done at $(date "+%m-%d-%H-%M")"