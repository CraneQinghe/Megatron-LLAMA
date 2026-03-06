#!/bin/bash

# 设置数据集路径 (沿用 Qwen 脚本中的路径，或者根据实际情况修改)
DATASET="/data/haiqwa/zevin_nfs/code/Megatron-LLaMA/examples/LLaMA/dataset/dataset_text_document"

# 设置分布式训练参数
MASTER_ADDR=172.20.$1.2
NODE_RANK=$2
NNODES=1
TP_SIZE=1 # GPT2 Small 较小，单卡 TP1 即可，也可以设置 TP2
PP_SIZE=1
SEQ_LENGTH=1024 # 对应 config.json 中的 n_ctx/n_positions
WORLD_SIZE=2    # 假设使用 2 张卡
DP_SIZE=$(($WORLD_SIZE / $TP_SIZE / $PP_SIZE))

DTIME=`date +%m-%d`
MTIME=`date +%m-%d-%H-%M`
export LOG_PATH=/data/haiqwa/zevin_nfs/code/qinghe/Megatron-LLAMA/examples/LLaMA/logs/gpt2_small_seq${SEQ_LENGTH}/$DTIME
mkdir -p ${LOG_PATH}

# 设置日志文件路径
LOG_FILE="${LOG_PATH}/Megatron-gpt2_small_${SEQ_LENGTH}_DP${DP_SIZE}_TP${TP_SIZE}_PP${PP_SIZE}_${MTIME}_NODE_RANK${NODE_RANK}.log"

exec > >(tee -i $LOG_FILE) 2>&1
echo "DP_SIZE is: $DP_SIZE"
MICRO_BATCH_SIZE=8 # GPT2 Small 很小，Batch 可以开大点
GLOBAL_BATCH_SIZE=$(($DP_SIZE * $MICRO_BATCH_SIZE * 4)) # 这里的 4 是梯度累加步数

echo "GLOBAL_BATCH_SIZE is: $GLOBAL_BATCH_SIZE"

JOB_NAME="GPT2_Small_tp${TP_SIZE}_pp${PP_SIZE}_mbs${MICRO_BATCH_SIZE}_gpus${WORLD_SIZE}"

# 模型权重/词表路径
TOKENIZER_PATH="/data/haiqwa/zevin_nfs/code/qinghe/models/GPT2" 

TRAIN_ITERS=10
LOG_INTERVAL=1

export NCCL_SOCKET_IFNAME="eth0"
export GLOO_SOCKET_IFNAME="eth0"

# GPT2-Small Config 参数估算:
# vocab_size: 50257
# hidden_size (n_embd): 768
# num_layers (n_layer): 12
# num_attention_heads (n_head): 12
# max_position_embeddings: 1024
# 总参数量: ~124M (0.124B)

options=" \
    --sequence-parallel \
        --tensor-model-parallel-size ${TP_SIZE} \
        --pipeline-model-parallel-size ${PP_SIZE} \
    --num-layers 12 \
        --hidden-size 768 \
        --num-attention-heads 12 \
        --seq-length ${SEQ_LENGTH} \
        --max-position-embeddings 1024 \
        --openai-gelu \
        --layernorm-epsilon 1e-5 \
    --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path $TOKENIZER_PATH \
        --make-vocab-size-divisible-by 1 \
    --init-method-std 0.02 \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --train-iters ${TRAIN_ITERS} \
    --log-interval ${LOG_INTERVAL} \
    --eval-iters 0 \
    --lr 6.0e-4 \
        --lr-decay-style cosine \
    --adam-beta1 0.9 \
        --adam-beta2 0.95 \
    --use-distributed-optimizer \
    --dataloader-type cyclic \
    --data-path ${DATASET} \
    --fp16 \
    --use-flash-attn"

# 注意：GPT2 使用 pretrain_gpt.py 而非 pretrain_llama.py 以匹配其 Architecture (Absolute Pos Emb, Bias, etc.)
torchrun --master_addr=$MASTER_ADDR --node_rank=$NODE_RANK --nnodes=${NNODES} --nproc_per_node=2 --master_port=29601 /data/haiqwa/zevin_nfs/code/qinghe/Megatron-LLAMA/pretrain_gpt.py ${options}
