# merge models

## 3B models
# /data/cuisijia/verl_kuhn_poker/qwen2.5_3b_64k_v0830/global_step_100/actor
python -m scripts.model_merger merge \
  --backend fsdp \
  --local_dir /data/cuisijia/verl_kuhn_poker/qwen2.5_3b_64k_v0830/global_step_100/actor \
  --target_dir /data/cuisijia/verl_kuhn_poker/qwen2.5_3b_64k_v0830/merged_actor_100

# /data/cuisijia/verl_kuhn_poker/qwen2.5_3b_mixed_64k_v0829/global_step_100/actor
python -m scripts.model_merger merge \
  --backend fsdp \
  --local_dir /data/cuisijia/verl_kuhn_poker/qwen2.5_3b_mixed_64k_v0829/global_step_100/actor \
  --target_dir /data/cuisijia/verl_kuhn_poker/qwen2.5_3b_mixed_64k_v0829/merged_actor_100

# /data/cuisijia/verl_kuhn_poker/qwen2.5_3b_mixedoppo_64k_v0826/global_step_100/actor
python -m scripts.model_merger merge \
  --backend fsdp \
  --local_dir /data/cuisijia/verl_kuhn_poker/qwen2.5_3b_mixedoppo_64k_v0826/global_step_100/actor \
  --target_dir /data/cuisijia/verl_kuhn_poker/qwen2.5_3b_mixedoppo_64k_v0826/merged_actor_100

## 7B models
# /data/cuisijia/verl_kuhn_poker/qwen2.5_7b_v0815/global_step_125/actor
python -m scripts.model_merger merge \
  --backend fsdp \
  --local_dir /data/cuisijia/verl_kuhn_poker/qwen2.5_7b_v0815/global_step_125/actor \
  --target_dir /data/cuisijia/verl_kuhn_poker/qwen2.5_7b_v0815/merged_actor_100

# /data/cuisijia/verl_kuhn_poker/qwen2.5_7b_v0815/global_step_250/actor
python -m scripts.model_merger merge \
  --backend fsdp \
  --local_dir /data/cuisijia/verl_kuhn_poker/qwen2.5_7b_v0815/global_step_250/actor \
  --target_dir /data/cuisijia/verl_kuhn_poker/qwen2.5_7b_v0815/merged_actor_final

# /data/cuisijia/verl_kuhn_poker/qwen2.5_7b_mixed_64k_v0824/global_step_100/actor
python -m scripts.model_merger merge \
  --backend fsdp \
  --local_dir /data/cuisijia/verl_kuhn_poker/qwen2.5_7b_mixed_64k_v0824/global_step_100/actor \
  --target_dir /data/cuisijia/verl_kuhn_poker/qwen2.5_7b_mixed_64k_v0824/merged_actor_100

# /data/cuisijia/verl_kuhn_poker/qwen2.5_7b_mixedoppo_64k_v0826/global_step_100/actor
python -m scripts.model_merger merge \
  --backend fsdp \
  --local_dir /data/cuisijia/verl_kuhn_poker/qwen2.5_7b_mixedoppo_64k_v0826/global_step_100/actor \
  --target_dir /data/cuisijia/verl_kuhn_poker/qwen2.5_7b_mixedoppo_64k_v0826/merged_actor_100
