#!/bin/bash
# =============================================================================
# DeepSeek-V3.2 HuggingFace -> Megatron Checkpoint 转换
# =============================================================================
#
# 将 HuggingFace checkpoint 转换为 Megatron 格式
# 需要在 **2 节点 16 GPU** 上运行 (每节点 8 GPU)
#
# 节点0: NNODES=2 NODE_RANK=0 MASTER_ADDR=<node0_ip> bash convert_hf_to_megatron.sh
# 节点1: NNODES=2 NODE_RANK=1 MASTER_ADDR=<node0_ip> bash convert_hf_to_megatron.sh
#
# =============================================================================

set -e

# 环境变量
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=1
export PYTHONWARNINGS=ignore
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=INFO

# 路径配置
MEGATRON_PATH="/workspace/Megatron-LM"
HF_MODEL_PATH="/data/models/DeepSeek-V3.2"
CONVERTED_OUTPUT="/data/models/DeepSeek-V3.2-megatron-tp16"

# 分布式配置
GPUS_PER_NODE=8
NNODES=${NNODES:-2}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}

# 并行策略 (需要和训练脚本一致)
TP=16
PP=1
EP=1

WORLD_SIZE=$((GPUS_PER_NODE * NNODES))

echo "==================================================================="
echo "转换 DeepSeek-V3.2 HuggingFace -> Megatron 格式"
echo "==================================================================="
echo "节点: $NODE_RANK / $NNODES"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "World Size: $WORLD_SIZE"
echo "HuggingFace 模型路径: $HF_MODEL_PATH"
echo "输出路径: $CONVERTED_OUTPUT"
echo "TP: $TP, PP: $PP, EP: $EP"
echo "==================================================================="

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# 模型参数 (DeepSeek-V3.2)
MODEL_ARGS="
    --num-layers 61 \
    --hidden-size 7168 \
    --ffn-hidden-size 18432 \
    --num-attention-heads 128 \
    --seq-length 2048 \
    --max-position-embeddings 4096 \
    --position-embedding-type rope \
    --rotary-base 10000 \
    --rotary-scaling-factor 40 \
    --mscale 1.0 \
    --mscale-all-dim 1.0 \
    --no-rope-fusion \
    --make-vocab-size-divisible-by 8080 \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0
"

# MLA 参数
MLA_ARGS="
    --multi-latent-attention \
    --q-lora-rank 1536 \
    --kv-lora-rank 512 \
    --qk-head-dim 128 \
    --qk-pos-emb-head-dim 64 \
    --v-head-dim 128 \
    --qk-layernorm
"

# MoE 参数
MOE_ARGS="
    --num-experts 256 \
    --moe-layer-freq ([0]*3+[1]*58) \
    --moe-ffn-hidden-size 2048 \
    --moe-shared-expert-intermediate-size 2048 \
    --moe-router-topk 8 \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-aux-loss-coeff 0.0 \
    --moe-router-group-topk 4 \
    --moe-router-num-groups 8 \
    --moe-router-pre-softmax \
    --moe-router-topk-scaling-factor 2.5 \
    --moe-router-score-function sigmoid \
    --moe-router-enable-expert-bias \
    --moe-router-dtype fp32
"

# MTP 参数
MTP_ARGS="
    --mtp-num-layers 1 \
    --mtp-loss-scaling-factor 1.0
"

# 并行参数
PARALLEL_ARGS="
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --expert-model-parallel-size $EP \
    --sequence-parallel \
    --use-distributed-optimizer
"

# 转换参数
CONVERT_ARGS="
    --pretrained-model-path $HF_MODEL_PATH \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model $HF_MODEL_PATH \
    --save $CONVERTED_OUTPUT \
    --save-interval 1 \
    --use-mcore-models \
    --bf16 \
    --init-model-with-meta-device \
    --no-load-optim \
    --no-load-rng \
    --micro-batch-size 1 \
    --global-batch-size 1
"

mkdir -p $CONVERTED_OUTPUT

cd $MEGATRON_PATH

# 运行转换
torchrun $DISTRIBUTED_ARGS \
    examples/post_training/modelopt/convert_model.py \
    $MODEL_ARGS \
    $MLA_ARGS \
    $MOE_ARGS \
    $MTP_ARGS \
    $PARALLEL_ARGS \
    $CONVERT_ARGS

echo "==================================================================="
echo "转换完成！"
echo "Megatron checkpoint 保存在: $CONVERTED_OUTPUT"
echo "==================================================================="
