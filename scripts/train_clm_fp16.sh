#!/bin/bash
module load compilers/gcc/12.2.0
export OMP_NUM_THREADS=8
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
# WORLD_SIZE=4 /home/bingxing2/home/scx6503/.conda/envs/llama_etuning/bin/torchrun --nproc_per_node=4 --master_port=1235 train_clm.py \
# /home/bingxing2/home/scx6503/.conda/envs/py39/bin/python train_clm.py \
# WORLD_SIZE=4 /home/bingxing2/home/scx6503/.conda/envs/py39/bin/torchrun --nproc_per_node=4 --master_port=1235 train_clm.py \
# WORLD_SIZE=2 /home/bingxing2/home/scx6503/.conda/envs/llama_etuning/bin/torchrun --nproc_per_node=2 --master_port=22110 train_clm.py \
# /home/bingxing2/home/scx6503/.conda/envs/py39/bin/python train_clm.py \
# WORLD_SIZE=2 /home/bingxing2/home/scx6503/.conda/envs/py39/bin/torchrun --nproc_per_node=2 --master_port=22110 train_clm.py \
# WORLD_SIZE=3 /home/bingxing2/home/scx6503/.conda/envs/py39/bin/torchrun --nproc_per_node=3 --master_port=22111 train_clm.py \
# WORLD_SIZE=2 /home/bingxing2/home/scx6503/.conda/envs/py39/bin/torchrun --nproc_per_node=2 --master_port=22110 train_clm.py \
# /home/bingxing2/home/scx6503/.conda/envs/llama_etuning/bin/python train_clm.py \
WORLD_SIZE=2 /home/bingxing2/home/scx6503/.conda/envs/llama_etuning/bin/torchrun --nproc_per_node=2 --master_port=22150 train_clm.py \
    --base_model '/home/bingxing2/home/scx6503/yx/chinese-llama-plus-13b-hf' \
    --data_path '/home/bingxing2/home/scx6503/yx/LaWGPT/data/train_info_and_repot.json' \
    --output_dir './outputs/true-chinese-llama-plus-13b-hf-train_info_and_report' \
    --batch_size  4 \
    --micro_batch_size 1 \
    --num_epochs 2 \
    --learning_rate 0.0001 \
    --cutoff_len 496 \
    --val_set_size 0 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj, v_proj, k_proj, o_proj, gate_proj, down_proj, up_proj]' \
    --train_on_inputs True \
    --add_eos_token True \
    --group_by_length True 

