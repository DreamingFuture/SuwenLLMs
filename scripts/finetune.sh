#!/bin/bash
export WANDB_MODE=disabled # 禁用wandb

# 使用chinese-alpaca-plus-7b-merged模型在law_data.json数据集上finetune
experiment_name="train-clm-llama-7B-full"

# 单卡或者模型并行
    # 我的理解是从这个检查点开始
    # 报错 ValueError: loaded state dict contains a parameter group that doesn't match the size of optimizer's group
    # 如果这个写"" ，就可以微调
CUDA_VISIBLE_DEVICES=3,4 python finetune.py \
    --base_model "/data2/yuxiang/data/LLMs/Linly-Chinese-LLaMA-7b-hf" \
    --data_path "./resources/event4_instruction_tune.json" \
    --output_dir "./tuned/chinese-alpaca-plus-7b-merged-train_info_and_report-event4" \
    --batch_size 64 \
    --micro_batch_size 8 \
    --num_epochs 2 \
    --learning_rate 3e-4 \
    --cutoff_len 256 \
    --val_set_size 0 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules "[q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj]" \
    --train_on_inputs True \
    --add_eos_token True \
    --group_by_length False \
    --wandb_project "" \
    --wandb_run_name "" \
    --wandb_watch "" \
    --wandb_log_model "" \
    --resume_from_checkpoint "./outputs/chinese-alpaca-plus-7b-merged-train_info_and_report/checkpoint-3000" \
    --prompt_template_name "alpaca" \



# 多卡数据并行
# WORLD_SIZE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=1234 finetune.py \
#     --base_model "minlik/chinese-alpaca-plus-7b-merged" \
#     --data_path "./data/finetune_law_data.json" \
#     --output_dir "./outputs/"${experiment_name} \
#     --batch_size 64 \
#     --micro_batch_size 8 \
#     --num_epochs 20 \
#     --learning_rate 3e-4 \
#     --cutoff_len 256 \
#     --val_set_size 0 \
#     --lora_r 8 \
#     --lora_alpha 16 \
#     --lora_dropout 0.05 \
#     --lora_target_modules "[q_proj,v_proj]" \
#     --train_on_inputs True \
#     --add_eos_token True \
#     --group_by_length False \
#     --wandb_project \
#     --wandb_run_name \
#     --wandb_watch \
#     --wandb_log_model \
#     --resume_from_checkpoint "./outputs/"${experiment_name} \
#     --prompt_template_name "alpaca" \