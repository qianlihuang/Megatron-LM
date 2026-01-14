# DeepSeek-V3.2 MTP Layer 微调指南

## 概述

本指南说明如何使用 Megatron-LM 微调 DeepSeek-V3.2 的 MTP (Multi-Token Prediction) 层，以提高 vLLM-magik 中 Eagle speculative decoding 的接受率。

## 关于 MTP 和 Eagle Speculative Decoding

DeepSeek-V3.2 的 MTP 层（layer 61）是一个额外的 transformer decoder 层，用于预测下一个 token。它包含：

- `enorm`, `hnorm`: Layer normalization
- `eh_proj`: 将 embedding 和 hidden states 投影合并
- 完整的 transformer decoder block (包含 MLA attention 和 MoE)
- `shared_head`: 共享的 output head

通过微调 MTP 层，可以让它更好地预测下一个 token，从而提高 Eagle speculative decoding 的接受率。

## 前提条件

1. DeepSeek-V3.2 模型位于 `/data/models/DeepSeek-V3.2`
2. 2 个节点，每节点 8 GPU (TP=16)
3. 训练数据已处理完成在 `/workspace/Megatron-LM/MTPfinetuning-tmp/data/processed/train`

## 步骤 1: 准备训练数据

训练数据应该是 Megatron 格式的 `.bin` 和 `.idx` 文件。当前数据位于：
```
/workspace/Megatron-LM/MTPfinetuning-tmp/data/processed/train.bin
/workspace/Megatron-LM/MTPfinetuning-tmp/data/processed/train.idx
```

如需处理新数据，使用：
```bash
python /workspace/Megatron-LM/tools/preprocess_data.py \
    --input /path/to/your/data.jsonl \
    --output-prefix /workspace/Megatron-LM/MTPfinetuning-tmp/data/processed/train \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /data/models/DeepSeek-V3.2 \
    --workers 8
```

## 步骤 2: 运行训练

在两个节点上运行：

**节点 0 (Master):**
```bash
cd /workspace/Megatron-LM/MTPfinetuning-tmp/scripts
GLOO_SOCKET_IFNAME=bond0.2007 \
NCCL_SOCKET_IFNAME=bond0.2007 \
NNODES=2 \
NODE_RANK=0 \
MASTER_ADDR=xingqiao-125 \
bash tp16.sh
```

**节点 1:**
```bash
cd /workspace/Megatron-LM/MTPfinetuning-tmp/scripts
GLOO_SOCKET_IFNAME=bond0.2007 \
NCCL_SOCKET_IFNAME=bond0.2007 \
NNODES=2 \
NODE_RANK=1 \
MASTER_ADDR=xingqiao-125 \
bash tp16.sh
```

训练参数说明：
- `--freeze-base-model`: 冻结基础模型，只训练 MTP 层
- `--mtp-num-layers 1`: MTP 层数量为 1
- `--mtp-loss-scaling-factor 1.0`: MTP loss 权重
- `--train-iters 100`: 训练迭代次数
- `--lr 1e-5`: 学习率

## 步骤 3: 转换 Checkpoint 格式

### 方法 A: 直接提取基础 MTP（不含微调权重）

如果只想测试 vLLM 加载 MTP 层：
```bash
cd /workspace/Megatron-LM/MTPfinetuning-tmp/scripts
python convert_mtp_to_hf.py \
    --base-model /data/models/DeepSeek-V3.2 \
    --output-dir /data/finetuned_layer61 \
    --extract-only
```

### 方法 B: 转换微调后的 Checkpoint

Megatron 使用分布式 checkpoint 格式 (`.distcp`)，需要先合并再转换：

1. **合并分布式 checkpoint:**
```bash
cd /workspace/Megatron-LM

# 将 TP=16 的 checkpoint 合并为 TP=1
python tools/checkpoint/convert.py \
    --model-type GPT \
    --loader mcore \
    --saver mcore \
    --load-dir /data/finetuned_mtp_v32/iter_0000100 \
    --save-dir /data/finetuned_mtp_v32/merged \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1
```

2. **提取 MTP 权重并转换为 HuggingFace 格式:**
```bash
python /workspace/Megatron-LM/MTPfinetuning-tmp/scripts/convert_mtp_to_hf.py \
    --megatron-ckpt /data/finetuned_mtp_v32/merged \
    --base-model /data/models/DeepSeek-V3.2 \
    --output-dir /data/finetuned_layer61
```

## 步骤 4: 在 vLLM-magik 中部署

使用 Eagle speculative decoding 启动 vLLM：

```bash
vllm serve /data/models/DeepSeek-V3.2 \
  --tokenizer-mode deepseek_v32 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --data-parallel-start-rank 0 \
  --data-parallel-size-local 8 \
  --data-parallel-address ${DATA_PARALLEL_ADDRESS} \
  --data-parallel-rpc-port 55555 \
  --data-parallel-backend mp \
  --enable-expert-parallel \
  --max-model-len 32768 \
  --max-num-batched-tokens 16 \
  --max-num-seqs 16 \
  --reasoning-parser deepseek_v3 \
  --compilation-config '{"pass_config": {"fuse_norm_quant": "True", "fuse_act_quant": "True", "fuse_attn_quant": "False", "eliminate_noops": "True", "enable_sp": "False", "fuse_gemm_comms": "True", "fuse_allreduce_rms": "False"}}' \
  --speculative_config '{"method":"eagle","model":"/data/finetuned_layer61", "num_speculative_tokens": 3, "max_model_len": 32768}' \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
  --gpu-memory-utilization 0.85 \
  --distributed-executor-backend mp \
  --host 0.0.0.0 \
  --port 8080
```

## 预期效果

微调 MTP 层后，Eagle speculative decoding 的接受率应该会提升，因为：

1. MTP 层会学习到更好的 token 预测分布
2. 连续预测多个 token (num_speculative_tokens=3) 时，更准确的预测会带来更高的接受率
3. 更高的接受率意味着更少的回退，从而提高推理吞吐量

## 训练参数调优建议

1. **学习率**: 从 `1e-5` 开始，可尝试 `5e-6` 到 `2e-5`
2. **训练迭代**: 根据数据量调整，建议监控 loss 曲线
3. **Batch size**: 当前 global batch size 为 16，可根据 GPU 显存调整
4. **损失权重**: `--mtp-loss-scaling-factor` 控制 MTP loss 的权重

## 监控训练

查看训练日志中的 `mtp_1 loss`，这是 MTP 层的 loss。Loss 下降表示 MTP 层正在学习更好的预测。

## 故障排除

### 问题: Checkpoint 加载失败
确保模型路径正确，且有足够的 GPU 显存加载完整模型。

### 问题: vLLM 无法加载微调后的 MTP
检查 `/data/finetuned_layer61` 目录是否包含：
- `config.json`
- `model.safetensors`
- tokenizer 文件

### 问题: 接受率没有提升
- 检查训练是否收敛 (loss 是否下降)
- 尝试更多训练数据
- 调整学习率和训练迭代次数
