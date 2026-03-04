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

WORLD_SIZE=$((NNODES * 8))

echo "--------------------------------------------------------"
echo "Starting Sweep: NNODES=$NNODES, WORLD_SIZE=$WORLD_SIZE"
echo "MASTER_ADDR=$MASTER_ADDR, NODE_RANK=$NODE_RANK"
echo "--------------------------------------------------------"

# 遍历各种并行配置
# TP: 1, 2, 4, 8
# PP: 1, 2, 4
# DP: 1, 2

for tp in 1 2 4 8; do
    for pp in 1 2 4; do
        for dp in 1 2 4 8 16; do
            # 检查配置是否合法 (DP * TP * PP == WORLD_SIZE)
            if [ $((dp * tp * pp)) -eq $WORLD_SIZE ]; then
                echo "========================================================"
                echo "Running Config: DP=$dp, TP=$tp, PP=$pp"
                echo "========================================================"
                
                export TP_SIZE=$tp
                export PP_SIZE=$pp
                
                # 调用原始脚本
                bash LLaMA_13_standalone.sh
                # bash Qwen_7B_standalone.sh
                
                echo "Config DP=$dp, TP=$tp, PP=$pp finished."
                echo "Waiting 5 seconds before next config..."
                sleep 5
            fi
        done
    done
done

echo "Sweep completed!"
