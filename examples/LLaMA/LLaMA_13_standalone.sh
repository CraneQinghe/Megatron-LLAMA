#!/bin/bash

# 设置数据集路径
DATASET="/data/haiqwa/zevin_nfs/code/Megatron-LLaMA/examples/LLaMA/dataset/dataset_text_document"

# 设置分布式训练参数
MASTER_ADDR=172.20.$1.2
NODE_RANK=$2
NNODES=1
TP_SIZE=2

PP_SIZE=2
SEQ_LENGTH=4096
WORLD_SIZE=8
DP_SIZE=$(($WORLD_SIZE / $TP_SIZE / $PP_SIZE))
# DP_SIZE=$3

DTIME=`date +%m-%d`
MTIME=`date +%m-%d-%H-%M`
export LOG_PATH=/data/haiqwa/zevin_nfs/code/qinghe/Megatron-LLAMA/examples/LLaMA/logs/seq${SEQ_LENGTH}/$DTIME
mkdir -p ${LOG_PATH}

# 设置日志文件路径，包含TP_SIZE, PP_SIZE, DP_SIZE
LOG_FILE="${LOG_PATH}/Megatron-llama_${SEQ_LENGTH}_DP${DP_SIZE}_TP${TP_SIZE}_PP${PP_SIZE}_${MTIME}_NODE_RANK${NODE_RANK}.log"


# 重定向所有输出到日志文件，同时也显示在终端上
exec > >(tee -i $LOG_FILE) 2>&1
echo "DP_SIZE is: $DP_SIZE"
MICRO_BATCH_SIZE=1
# GLOBAL_BATCH_SIZE=$((($WORLD_SIZE / $TP_SIZE) / $PP_SIZE * 4))
GLOBAL_BATCH_SIZE=$(($DP_SIZE * 4))

echo "GLOBAL_BATCH_SIZE is: $GLOBAL_BATCH_SIZE"

JOB_NAME="LLaMA_tp${TP_SIZE}_pp${PP_SIZE}_mbs${MICRO_BATCH_SIZE}_gpus${WORLD_SIZE}"

LOAD_CHECKPOINT_PATH="/root/zevin_nfs/code/Megatron-LLaMA/examples/LLaMA/cp"
SAVE_CHECKPOINT_PATH="/root/zevin_nfs/code/Megatron-LLaMA/examples/LLaMA/cp"
# TOKENIZER_PATH="/data/haiqwa/gpt/jcz/Meta-Llama-3.1-8B-Instruct"
TOKENIZER_PATH="/data/haiqwa/zevin_nfs/andy/Auto-Parallelization/nnscaler_group1/llama-model-config/Meta-Llama-3.1-8B"
TRAIN_ITERS=50
EVAL_ITERS=0
EVAL_INTERVAL=1000
SAVE_INTERVAL=100
LOG_INTERVAL=1

export NCCL_SOCKET_IFNAME="eth0"
export GLOO_SOCKET_IFNAME="eth0"

# 设置 --tensorboard-queue-size 为 1 会显著减慢训练速度
options=" \
    --finetune \
    --sequence-parallel \
        --tensor-model-parallel-size ${TP_SIZE} \
        --pipeline-model-parallel-size ${PP_SIZE} \
    --num-layers 32 \
        --hidden-size 4096 \
        --num-attention-heads 32 \
        --seq-length ${SEQ_LENGTH} \
        --max-position-embeddings 131072 \
        --no-position-embedding \
        --use-rotary-position-embeddings \
        --swiglu \
        --ffn-hidden-size 14336\
        --disable-bias-linear \
        --RMSNorm \
        --layernorm-epsilon 1e-6 \
        --causal-lm \
    --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path $TOKENIZER_PATH \
        --make-vocab-size-divisible-by 1 \
        
    --init-method-std 0.01 \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --train-iters ${TRAIN_ITERS} \
    --lr 6.0e-5 \
        --lr-decay-iters 10 \
        --lr-warmup-iters 5 \
        --min-lr 6.0e-6 \
        --override-opt_param-scheduler \
        --lr-decay-style cosine \
    --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --clip-grad 1.0 \
        --weight-decay 0.1 \
        --use-distributed-optimizer \
        --reduce-bucket-size=2e8 \
        --no-gradient-accumulation-fusion \
    --dataloader-type cyclic \
        --data-impl mmap \
        --data-path ${DATASET} \
        --split 98,2,0 \
    --eval-interval ${EVAL_INTERVAL} \
        --eval-iters ${EVAL_ITERS} \
    --log-interval ${LOG_INTERVAL} \
        --tensorboard-queue-size 1000 \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
    --job-name ${JOB_NAME} \
    --bf16 \
    --recompute-activations \
        --recompute-granularity selective \
    --use-flash-attn"

# 执行训练命令
torchrun --master_addr=$MASTER_ADDR --node_rank=$NODE_RANK --nnodes=${NNODES} --nproc_per_node=${WORLD_SIZE} --master_port=29600 /data/haiqwa/zevin_nfs/code/qinghe/Megatron-LLAMA/pretrain_llama.py ${options}