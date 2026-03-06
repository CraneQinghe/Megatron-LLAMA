import json
import os

def get_layer_time(path):
    if not os.path.exists(path):
        return f"File not found: {path}"
    with open(path) as f:
        data = json.load(f)
        for k, v in data.items():
            if "_Total_Forward_Detailed_unprofiled" in k:
                return v.get("avg_time_ms")
    return "Metric not found"

p1b = "/Users/wangqinghe/Desktop/Hops论文/draw/hops_profiling_results_llama_1B_dp1_tp2_pp1_rank0.json"
p8b_dp2 = "/Users/wangqinghe/Desktop/Hops论文/draw/hops_profiling_results_llama_8B_dp2_tp2_pp2_rank0.json"

print(f"1B (TP2) Layer Avg: {get_layer_time(p1b)}")
print(f"8B (TP2) Layer Avg: {get_layer_time(p8b_dp2)}")
