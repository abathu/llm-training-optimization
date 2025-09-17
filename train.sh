#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="/home/ubuntu/llm-training-optimization/training_$(date +%Y%m%d_%H%M%S).log"
CFG="/home/ubuntu/llm-training-optimization/deepspeed_config_z3_qlora.yaml"

echo "训练日志将保存到: $LOG_FILE"

args=(
  --config_file "$CFG"
  train.py
  --model_name_or_path Qwen/Qwen2.5-0.5B
  --dataset_name china-ai-law-challenge/cail2018
  --splits first_stage_train,first_stage_test
  --chat_template_format chatml
  --max_seq_length 1024
  --packing True
  --group_by_length True
  --bf16 True
  --tf32 True
  --use_peft_lora True
  --use_4bit_quantization True
  --bnb_4bit_compute_dtype bfloat16
  --use_flash_attn True
  --per_device_train_batch_size 16
  --per_device_eval_batch_size 16
  --gradient_accumulation_steps 2
  --gradient_checkpointing True
  --use_reentrant True
  --optim paged_adamw_8bit
  --dataloader_num_workers 8
  --lora_r 8
  --lora_alpha 16
  --lora_dropout 0.1
  --num_train_epochs 3
  --learning_rate 5e-5
  --warmup_ratio 0.1
  --logging_steps 10
  --save_steps 500
  --output_dir /home/ubuntu/llm-training-optimization/qwen-cail-sft
)

# old 2e-4

# 用 nohup 后台跑
nohup accelerate launch "${args[@]}" > "$LOG_FILE" 2>&1 &

echo "训练进程ID: $!"
echo "训练已在后台启动，查看日志: tail -f \"$LOG_FILE\""