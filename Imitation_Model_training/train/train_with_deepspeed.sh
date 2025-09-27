#!/bin/bash

# DeepSpeed分布式训练脚本
# 使用DeepSpeed ZeRO优化大幅减少GPU内存需求

export WANDB_DISABLED=true

# 配置文件
MODEL_CONFIG="config/llama2.json"
DEEPSPEED_CONFIG="config/deepspeed_config.json"

# GPU数量
NUM_GPUS=4

echo "开始DeepSpeed训练..."
echo "GPU数量: $NUM_GPUS"
echo "模型配置: $MODEL_CONFIG"
echo "DeepSpeed配置: $DEEPSPEED_CONFIG"

# 使用deepspeed启动训练
deepspeed --num_gpus=$NUM_GPUS finetune_imitation_model_my.py \
    --model_config_file $MODEL_CONFIG \
    --deepspeed $DEEPSPEED_CONFIG

echo "DeepSpeed训练完成！"