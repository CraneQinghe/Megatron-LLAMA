#!/bin/bash

# 使用方法: 
# NNODES=1: bash sweep_parallel.sh 127.0.0.1 0 1
# NNODES=2: 
#   机器0: bash sweep_parallel.sh <MASTER_IP> 0 2
#   机器1: bash sweep_parallel.sh <MASTER_IP> 1 2

# 获取 Master 节点的中间 ID (例如 172.20.3.2 则传入 3)
MASTER_NODE_ID=${1:-"3"}
export MASTER_ADDR="172.20.${MASTER_NODE_ID}.2"
export NODE_RANK=${2:-0}
export NNODES=1

export WORLD_SIZE=$((NNODES * 8))


echo "--------------------------------------------------------"
echo "Starting Sweep: NNODES=$NNODES, WORLD_SIZE=$WORLD_SIZE"
echo "MASTER_ADDR=$MASTER_ADDR, NODE_RANK=$NODE_RANK"
echo "--------------------------------------------------------"

# 遍历各种并行配置
# TP/PP/DP 可选值: 1, 2, 4, 8
# NNODES 固定为 1 (单机压测)

for tp in 1 2 4; do
    for pp in 1 2 4 8; do
        for dp in 1 2; do
            # 计算当前配置需要的总卡数
            current_gpus=$((tp * pp * dp))
            
            # 条件过滤:
            # 1. 总卡数不能超过节点能提供的卡数 (NNODES * 8)
            # 2. 如果不是纯 TP 配置 (即 PP>1 或 DP>1)，则强制要求跑满最大卡数
            #    纯 TP 配置允许少于最大卡数的规模
            
            is_pure_tp=0
            # if [ $pp -eq 1 ] && [ $dp -eq 1 ]; then
            #     is_pure_tp=1
            # fi

            max_gpus=$((NNODES * 8))

            if [ $current_gpus -le $max_gpus ]; then
                if [ $is_pure_tp -eq 1 ] || [ $current_gpus -ge $max_gpus ]; then
                    export WORLD_SIZE=$current_gpus
                    export TP_SIZE=$tp
                    export PP_SIZE=$pp
                    
                    echo "========================================================"
                    echo "Running Config: DP=$dp, TP=$tp, PP=$pp (WORLD=$WORLD_SIZE)"
                    echo "========================================================"
                    
                    # 调用模型脚本
                    # bash LLaMA_13_standalone.sh
                    bash Qwen_7B_standalone.sh
                    
                    echo "Config DP=$dp, TP=$tp, PP=$pp finished."
                    echo "Waiting 5 seconds before next config..."
                    sleep 5
                fi
            fi
        done
    done
done

echo "Sweep completed!"
