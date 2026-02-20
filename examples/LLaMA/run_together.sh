#!/bin/bash
set -e # 如果有任何命令失败，立即退出

# 配置变量
LOCAL_SCRIPT_NAME="/data/haiqwa/zevin_nfs/code/Megatron-LLaMA/examples/LLaMA/LLaMA_13_standalone.sh"
REMOTE_HOST="172.20.7.2"
REMOTE_PATH="/data/haiqwa/zevin_nfs/code/Megatron-LLaMA/examples/LLaMA"
REMOTE_STATUS_FILE="/tmp/remote_command_status_$$.txt"

# 修改脚本中的变量值（如果需要）
for DATASIZE in 4; do
    SEQL=$((DATASIZE * 1024))
    for DP_NUM in 1 2 4 8 16; do
        for TP_NUM in 1 2 4 8 16; do
            if (( DP_NUM * TP_NUM <= 16 )); then
                PP_NUM=$(( 16 / (DP_NUM * TP_NUM) ))

                sed -i 's/^TP_SIZE=[0-9]\+/TP_SIZE='"${TP_NUM}"'/' "$LOCAL_SCRIPT_NAME"
                sed -i 's/^PP_SIZE=[0-9]\+/PP_SIZE='"${PP_NUM}"'/' "$LOCAL_SCRIPT_NAME"
                sed -i 's/^SEQ_LENGTH=[0-9]\+/SEQ_LENGTH='"${SEQL}"'/' "$LOCAL_SCRIPT_NAME"

                echo "Modified $LOCAL_SCRIPT_NAME with: DATASIZE=$DATASIZE, DP_NUM=$DP_NUM, TP_NUM=$TP_NUM, PP_NUM=$PP_NUM, SEQ_LENGTH=$SEQL"

                # 在本机执行脚本
                echo "Starting local script in background..."
                bash "$LOCAL_SCRIPT_NAME" 1 0 &
                local_pid=$!

                # 立即通过 SSH 启动远程脚本并在后台运行
                echo "Starting remote script in background..."
                ssh "${REMOTE_HOST}" "nohup bash ${REMOTE_PATH}/$(basename "$LOCAL_SCRIPT_NAME") 1 1 > /dev/null 2>&1 & echo \$! > $REMOTE_STATUS_FILE" &
                remote_pid=$!

                # 等待本地任务完成
                wait $local_pid
                echo "Local script finished."

                # 等待远程任务完成
                while ! ssh "${REMOTE_HOST}" "[ -f $REMOTE_STATUS_FILE ] && kill -0 \$(cat $REMOTE_STATUS_FILE)"; do
                    sleep 1
                done
                ssh "${REMOTE_HOST}" "kill \$(cat $REMOTE_STATUS_FILE); rm -f $REMOTE_STATUS_FILE"
                echo "Remote script finished."
            fi
        done
    done
done

echo "All tasks initiated. Waiting for all tasks to complete..."

# 等待所有后台任务完成
wait

echo "All tasks completed."