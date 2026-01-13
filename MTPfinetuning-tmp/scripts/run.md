```
cd /workspace/deepseek_v32_mtp_megatron/scripts
GLOO_SOCKET_IFNAME=bond0.2007 NCCL_SOCKET_IFNAME=bond0.2007 NNODES=2 NODE_RANK=0 MASTER_ADDR=xingqiao-125 bash train_simple_tp16.sh

cd /workspace/deepseek_v32_mtp_megatron/scripts
GLOO_SOCKET_IFNAME=bond0.2007 NCCL_SOCKET_IFNAME=bond0.2007 NNODES=2 NODE_RANK=1 MASTER_ADDR=xingqiao-125 bash train_simple_tp16.sh
```