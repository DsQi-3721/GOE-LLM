
while kill -0 $1 2>/dev/null; do
    echo "Process $1 is still running. Waiting for it to finish..."
    sleep 60  # Wait for 60 seconds before checking again
done

echo "Process $1 has finished. Starting evaluation..."

for i in {1..3}
do
    cur_time=$(date "+%Y-%m-%d-%H-%M-%S")
    
    echo "Evaluation round $i, time: $cur_time"

    python -m utils.eval /data/models/Qwen2.5-3B-Instruct > eval_3B_$cur_time.log 2>&1
    
    python -m utils.eval /data/models/Qwen2.5-7B-Instruct > eval_7B_$cur_time.log 2>&1
    
    python -m utils.eval /data/cuisijia/verl_kuhn_poker/grpo_qwen2.5_3b_instruct_v0819/merged_hf_actor_step_25 > eval_grpo_3B_$cur_time.log 2>&1
    
    python -m utils.eval /data/cuisijia/verl_kuhn_poker/grpo_qwen2.5_3b_instruct_mixed_64k_v0821/merged_hf_actor_step_25 > eval_grpo_3B_mixed_64k_$cur_time.log 2>&1

done
echo "Evaluation completed."
