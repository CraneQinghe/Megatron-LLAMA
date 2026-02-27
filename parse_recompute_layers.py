import json
with open("/Users/wangqinghe/Desktop/Hops论文/draw/hops_profiling_results_dp2_tp8_pp1_rank0.json") as f:
    d = json.load(f)

for i in range(1, 33):
    fw_key = f"Layer_{i}_Total_Forward_NoSync"
    bw_key = f"Layer_{i}_Total_Backward_NoSync"
print("\nLook closely at FW+BW")
for i in range(1, 4):
    print(f"L{i} FW: {d.get(f'Layer_{i}_Total_Forward_NoSync', {}).get('avg_time_ms', 0):.2f}")
    print(f"L{i} BW: {d.get(f'Layer_{i}_Total_Backward_NoSync', {}).get('avg_time_ms', 0):.2f}")
