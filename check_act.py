import json
import os

def check_json(path):
    if not os.path.exists(path): return "Not found"
    with open(path) as f:
        data = json.load(f)
        peak = data.get("System_Peak_Memory_Allocated_MB_True", 0)
        curr = data.get("System_Current_Memory_Allocated_MB", 0)
        act = peak - curr
        
        # Look for layer activations
        layer1_act = 0
        for k, v in data.items():
            if "Layer_1_Total_Forward" in k and isinstance(v, dict):
                layer1_act = v.get("avg_mem_mb", 0)
                break
        
        return f"Peak: {peak/1024:.2f} GB, Curr: {curr/1024:.2f} GB, Act: {act/1024:.2f} GB, Layer1_Act: {layer1_act:.2f} MB"

p8b_tp8 = "/Users/wangqinghe/Desktop/Hops论文/draw/hops_profiling_results_llama_8B_dp1_tp8_pp1_rank0.json"
p8b_tp4 = "/Users/wangqinghe/Desktop/Hops论文/draw/hops_profiling_results_llama_8B_dp2_tp4_pp1_rank0.json"
p1b_tp2 = "/Users/wangqinghe/Desktop/Hops论文/draw/hops_profiling_results_llama_1B_dp1_tp2_pp2_rank0.json"

print("Llama-8B TP8:", check_json(p8b_tp8))
print("Llama-8B TP4:", check_json(p8b_tp4))
print("Llama-1B TP2 Baseline:", check_json(p1b_tp2))
