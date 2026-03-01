#!/bin/bash

# 设置数据集路径
DATASET="/data/haiqwa/zevin_nfs/code/Megatron-LLaMA/examples/LLaMA/dataset/dataset_text_document"

# 设置分布式训练参数
MASTER_ADDR=172.20.$1.2
NODE_RANK=$2
NNODES=2
TP_SIZE=8

PP_SIZE=1
SEQ_LENGTH=4096
WORLD_SIZE=16
DP_SIZE=$(($WORLD_SIZE / $TP_SIZE / $PP_SIZE))

DTIME=`date +%m-%d`
MTIME=`date +%m-%d-%H-%M`
export LOG_PATH=/data/haiqwa/zevin_nfs/code/qinghe/Megatron-LLAMA/examples/LLaMA/logs/qwen_seq${SEQ_LENGTH}/$DTIME
mkdir -p ${LOG_PATH}

# 设置日志文件路径
LOG_FILE="${LOG_PATH}/Megatron-qwen_${SEQ_LENGTH}_DP${DP_SIZE}_TP${TP_SIZE}_PP${PP_SIZE}_${MTIME}_NODE_RANK${NODE_RANK}.log"

exec > >(tee -i $LOG_FILE) 2>&1
echo "DP_SIZE is: $DP_SIZE"
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=$(($DP_SIZE * 4))

echo "GLOBAL_BATCH_SIZE is: $GLOBAL_BATCH_SIZE"

JOB_NAME="Qwen_7B_tp${TP_SIZE}_pp${PP_SIZE}_mbs${MICRO_BATCH_SIZE}_gpus${WORLD_SIZE}"

TOKENIZER_PATH="/data/haiqwa/zevin_nfs/model/DeepSeek-R1-Distill-Qwen-7B" # 请确保路径存在或指向正确的 Qwen Tokenizer
TRAIN_ITERS=10
EVAL_ITERS=0
LOG_INTERVAL=1

export NCCL_SOCKET_IFNAME="eth0"
export GLOO_SOCKET_IFNAME="eth0"

# Qwen-7B Config based on DeepSeek Distill
# hidden_size: 3584
# intermediate_size: 18944
# num_attention_heads: 28
# vocab_size: 152064
# tie_word_embeddings: false

options=" \
    --finetune \
    --sequence-parallel \
        --tensor-model-parallel-size ${TP_SIZE} \
        --pipeline-model-parallel-size ${PP_SIZE} \
    --num-layers 4 \
        --hidden-size 3584 \
        --ffn-hidden-size 18944 \
        --num-attention-heads 28 \
        --seq-length ${SEQ_LENGTH} \
        --max-position-embeddings 131072 \
        --no-position-embedding \
        --use-rotary-position-embeddings \
        --swiglu \
        --disable-bias-linear \
        --RMSNorm \
        --layernorm-epsilon 1e-6 \
        --causal-lm \
        --untie-embeddings-and-output-weights \
    --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path $TOKENIZER_PATH \
        --make-vocab-size-divisible-by 1 \
        
    --init-method-std 0.01 \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --train-iters ${TRAIN_ITERS} \
    --lr 6.0e-5 \
        --lr-decay-style cosine \
    --adam-beta1 0.9 \
        --adam-beta2 0.95 \
    --use-distributed-optimizer \
    --dataloader-type cyclic \
    --data-path ${DATASET} \
    --bf16 \
    --recompute-activations \
        --recompute-granularity selective \
    --use-flash-attn"

# 执行训练命令
torchrun --master_addr=$MASTER_ADDR --node_rank=$NODE_RANK --nnodes=${NNODES} --nproc_per_node=8 --master_port=29600 /data/haiqwa/zevin_nfs/code/qinghe/Megatron-LLAMA/pretrain_llama.py ${options}
