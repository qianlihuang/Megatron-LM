#!/bin/bash
# =============================================================================
# DeepSeek-V3.2 MTP 训练 - 简单版 (TP=16, 2节点)
# =============================================================================
#
# 使用 TP=16 在 2 个节点上训练，每节点 8 GPU
# 通过 ModelOpt 直接加载 HuggingFace checkpoint
#
# 节点0: NNODES=2 NODE_RANK=0 MASTER_ADDR=<node0_ip> bash train_simple_tp16.sh
# 节点1: NNODES=2 NODE_RANK=1 MASTER_ADDR=<node0_ip> bash train_simple_tp16.sh
#
# =============================================================================

set -e

# 环境变量
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=1
export PYTHONWARNINGS=ignore
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# 路径配置
MEGATRON_PATH="/workspace/Megatron-LM"
HF_MODEL_PATH="/data/models/DeepSeek-V3.2"
OUTPUT_DIR="/data/finetuned_mtp_v32"
DATA_PATH="/workspace/deepseek_v32_mtp_megatron/data/processed/train"

# 分布式配置
GPUS_PER_NODE=8
NNODES=${NNODES:-2}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}

# 并行策略
TP=16
PP=1
EP=1

WORLD_SIZE=$((GPUS_PER_NODE * NNODES))

echo "==================================================================="
echo "DeepSeek-V3.2 MTP 训练 (TP=$TP, PP=$PP, EP=$EP)"
echo "==================================================================="
echo "节点: $NODE_RANK / $NNODES"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "World Size: $WORLD_SIZE"
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
    --make-vocab-size-divisible-by 3232 \
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
    --moe-grouped-gemm \
    --moe-aux-loss-coeff 0.0 \
    --moe-router-group-topk 4 \
    --moe-router-num-groups 8 \
    --moe-router-pre-softmax \
    --moe-router-topk-scaling-factor 2.5 \
    --moe-router-score-function sigmoid \
    --moe-router-enable-expert-bias \
    --moe-router-dtype fp32
"

# MTP 参数 - 冻结基础模型只训练MTP层
MTP_ARGS="
    --mtp-num-layers 1 \
    --mtp-loss-scaling-factor 1.0 \
    --freeze-base-model
"

# 并行参数
PARALLEL_ARGS="
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --expert-model-parallel-size $EP \
    --sequence-parallel \
    --use-distributed-optimizer
"

# 训练参数
TRAINING_ARGS="
    --micro-batch-size 1 \
    --global-batch-size 16 \
    --train-iters 100 \
    --lr 1e-5 \
    --min-lr 1e-6 \
    --lr-decay-style cosine \
    --lr-warmup-iters 10 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --bf16
"

# 数据参数 - 使用 mock data 测试
DATA_ARGS="
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model $HF_MODEL_PATH \
    --mock-data
"

# Checkpoint 参数 - 使用 mock data 先测试，不加载模型
CHECKPOINT_ARGS="
    --save $OUTPUT_DIR \
    --save-interval 50 \
    --no-load-optim \
    --no-load-rng
"

# 日志参数
LOGGING_ARGS="
    --log-interval 1 \
    --log-throughput \
    --eval-interval 100000 \
    --eval-iters 0
"

# 其他参数
OTHER_ARGS="
    --use-mcore-models \
    --no-create-attention-mask-in-dataloader \
    --recompute-granularity selective \
    --recompute-modules moe_act mlp \
    --manual-gc \
    --manual-gc-interval 10
"

mkdir -p $OUTPUT_DIR

cd $MEGATRON_PATH

# 运行训练
torchrun $DISTRIBUTED_ARGS \
    pretrain_gpt.py \
    $MODEL_ARGS \
    $MLA_ARGS \
    $MOE_ARGS \
    $MTP_ARGS \
    $PARALLEL_ARGS \
    $TRAINING_ARGS \
    $DATA_ARGS \
    $CHECKPOINT_ARGS \
    $LOGGING_ARGS \
    $OTHER_ARGS

echo "==================================================================="
echo "训练完成!"
echo "==================================================================="
