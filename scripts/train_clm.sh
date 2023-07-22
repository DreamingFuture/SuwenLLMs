#!/bin/bash
#SBATCG --gpus=4
module load compilers/gcc/12.2.0
# module load 
# /home/bingxing2/home/scx6503/.conda/envs/llama_etuning/bin/python -m bitsandbytes
# conda activate llama_etuning
    # --base_model '/home/bingxing2/home/scx6503/yx/Linly-Chinese-LLaMA-7B' \

# /home/bingxing2/home/scx6503/.conda/envs/llama_etuning/bin/python train_clm.py \
# WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=1235 train_clm.py \
# WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 /home/bingxing2/home/scx6503/.conda/envs/llama_etuning/bin/python torch.distributed.launch --nproc_per_node=2 --master_port=1235 train_clm.py \
# WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=1235 train_clm.py \
# WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 /home/bingxing2/home/scx6503/.conda/envs/llama_etuning/bin/torchrun --nproc_per_node=2 --master_port=1235 train_clm.py \
# /home/bingxing2/home/scx6503/.conda/envs/llama_etuning/bin/python train_clm.py \
WORLD_SIZE=4 /home/bingxing2/home/scx6503/.conda/envs/llama_etuning/bin/torchrun --nproc_per_node=4 --master_port=1235 train_clm.py \
    --base_model '/home/bingxing2/home/scx6503/yx/Linly-Chinese-LLaMA-7B' \
    --data_path '/home/bingxing2/home/scx6503/yx/LaWGPT/data/Sample.json' \
    --output_dir './outputs/train-clm-llama-7B' \
    --batch_size 32 \
    --micro_batch_size 2 \
    --num_epochs 800 \
    --learning_rate 0.0003 \
    --cutoff_len 400 \
    --val_set_size 0 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj, v_proj, k_proj, o_proj]' \
    --train_on_inputs True \
    --add_eos_token True \
    --group_by_length True \
    --fp16