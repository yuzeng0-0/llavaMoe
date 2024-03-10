from llava.train.train import train
import torch
import pdb

# 起命令一般格式
# srun -p llm3 --gres gpu:8 --quotatype reserved bash xxx.sh

# srun -p mllm --gres gpu:8 --quotatype reserved bash scripts/v1_5/slurm_pretrain.sh
# srun -p mllm --gres gpu:8 --quotatype reserved bash scripts/v1_5/slurm_finetune.sh

if __name__ == "__main__":
    # pdb.set_trace()
    train(attn_implementation="flash_attention_2")
