cd /home/wangxinqi/llm-opponent-modeling

cur_time=$(date "+%m-%d-%H-%M")

# 你的3B训练模型
echo "eval /home/wangxinqi/llm-opponent-modeling/verl_leduc/qwen2.5_3b_mixed_64k_v0916_leduc/merged_actor_100 at $(date "+%m-%d-%H-%M")"
python -m utils.eval_leduc_parallel /home/wangxinqi/llm-opponent-modeling/verl_leduc/qwen2.5_3b_mixed_64k_v0916_leduc/merged_actor_100 > eval_logs/${cur_time}_eval_leduc_parallel_3b_leduc_v0924.log 2>&1
echo "done /home/wangxinqi/llm-opponent-modeling/verl_leduc/qwen2.5_3b_mixed_64k_v0916_leduc/merged_actor_100 at $(date "+%m-%d-%H-%M")"


# 3B 基础模型
echo "eval /data/models/Qwen2.5-3B-Instruct at $(date "+%m-%d-%H-%M")"
python -m utils.eval_leduc_parallel /data/models/Qwen2.5-3B-Instruct > eval_logs/${cur_time}_eval_leduc_parallel_3B_v0924.log 2>&1
echo "done /data/models/Qwen2.5-3B-Instruct at $(date "+%m-%d-%H-%M")"

# 7B 基础模型
#echo "eval /data/models/Qwen2.5-7B-Instruct at $(date "+%m-%d-%H-%M")"
#python -m utils.eval_leduc_parallel /data/models/Qwen2.5-7B-Instruct > eval_logs/${cur_time}_eval_leduc_parallel_7B.log 2>&1
#echo "done /data/models/Qwen2.5-7B-Instruct at $(date "+%m-%d-%H-%M")"


echo "all done at $(date "+%m-%d-%H-%M")"

# 3B model
#echo "eval /data/models/Qwen2.5-3B-Instruct at $(date "+%m-%d-%H-%M")"
#python -m utils.eval_leduc_parallel /data/models/Qwen2.5-3B-Instruct > eval_logs/${cur_time}_eval_leduc_parallel_3B.log 2>&1
#echo "done /data/models/Qwen2.5-3B-Instruct at $(date "+%m-%d-%H-%M")"

#cho "eval /home/wangxinqi/llm-opponent-modeling/verl_leduc/qwen2.5_3b_mixed_64k_v0916_leduc/merged_actor_100 at $(date "+%m-%d-%H-%M")"
#python -m utils.eval_leduc_parallel /home/wangxinqi/llm-opponent-modeling/verl_leduc/qwen2.5_3b_mixed_64k_v0916_leduc/merged_actor_100 > eval_logs/${cur_time}_eval_leduc_parallel_3b_mixed_64k_v0916.log 2>&1
#echo "done /home/wangxinqi/llm-opponent-modeling/verl_leduc/qwen2.5_3b_mixed_64k_v0916_leduc/merged_actor_100 at $(date "+%m-%d-%H-%M")"
