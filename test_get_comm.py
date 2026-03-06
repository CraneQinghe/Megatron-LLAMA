import os
import json
import sys

# Replace this with the actual path
COMM_RESULTS_DIR = "/Users/wangqinghe/Desktop/Hops论文/code/hops/src/metrics/comm_profiler/results"

nccl_cache = {}
if os.path.exists(COMM_RESULTS_DIR):
    for f in os.listdir(COMM_RESULTS_DIR):
        if f.startswith("nccl_base_g") and f.endswith(".json"):
            try:
                parts = f.replace(".json", "").split("_")
                g = int([p for p in parts if p.startswith('g')][0][1:])
                s = 1
                for p in parts:
                    if p.startswith('s') and len(p) > 1 and p[1:].isdigit():
                        s = int(p[1:])
                idx = parts.index('mesh')
                ma = int(parts[idx+1])
                mb = int(parts[idx+2])
                with open(os.path.join(COMM_RESULTS_DIR, f), 'r') as f_in:
                    data = json.load(f_in)
                    nccl_cache[(g, s, ma, mb)] = data.get('results', [])
            except Exception as e:
                continue

available_keys = list(nccl_cache.keys())
print("Available keys:", available_keys)

def get_best(group_num, stride, mesh_a, mesh_b):
    best_k = min(available_keys, key=lambda x: (
        abs(x[2] - mesh_a), 
        abs(x[3] - mesh_b), 
        abs(x[1] - stride), 
        abs(x[0] - group_num)
    ))
    return best_k

print("For DP2-TP4-PP2 Emb sync (g=2, s=8, ma=2, mb=1):", get_best(2, 8, 2, 1))
print("For DP2-TP4-PP2 with g=4 -> (g=4, s=8, ma=2, mb=1):", get_best(4, 8, 2, 1))

