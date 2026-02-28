import json
import os

filepath = "/Users/wangqinghe/Desktop/Hops论文/draw/hops_profiling_dynamic_window_dp2_tp8_pp1_rank0.json"
if not os.path.exists(filepath):
    print("File not found:", filepath)
    exit()

with open(filepath) as f:
    d = json.load(f)

def print_stat(k):
    if k in d:
        v = d[k]
        print(f"--- {k} ---")
        print(f"  count: {v['count']}")
        print(f"  avg_time_ms: {v['avg_time_ms']:.2f} ms")
        if "all_ms" in v:
            print(f"  all_ms: [{', '.join(f'{x:.2f}' for x in v['all_ms'])}]")
    else:
        print(f"--- {k} : NOT FOUND ---")

keys_to_check = [
    "Iteration_Unprofiled",
    "Iteration_Profiled",
    "Iteration_Warmup",
    "Forward_Backward_Pipeline",
    "Microbatch_Forward_Step",
    "Microbatch_Forward_Step_Last",
    "Microbatch_Data_Loader",
    "Microbatch_Model_Forward",
    "Layer_1_Total_Forward_NoSync"
]

for k in keys_to_check:
    print_stat(k)

# Layer sum check
layer_fwd_sum_ms = 0
layer_count = 0
for i in range(1, 33):
    layer_key_nosync = f"Layer_{i}_Total_Forward_NoSync"
    layer_key_detailed = f"Layer_{i}_Total_Forward_Detailed"
    if layer_key_nosync in d:
        layer_fwd_sum_ms += d[layer_key_nosync].get("avg_time_ms", 0)
        layer_count += 1
    elif layer_key_detailed in d:
        layer_fwd_sum_ms += d[layer_key_detailed].get("avg_time_ms", 0)
        layer_count += 1

print(f"\nSum of {layer_count} Layers Forward: {layer_fwd_sum_ms:.2f} ms")
if "Microbatch_Model_Forward" in d:
    print(f"Microbatch_Model_Forward: {d['Microbatch_Model_Forward']['avg_time_ms']:.2f} ms")
    print(f"Difference (Hidden / Bubble / Other): {d['Microbatch_Model_Forward']['avg_time_ms'] - layer_fwd_sum_ms:.2f} ms")

# Sp communications
sp_comm_keys = ["SP_AllGather_Forward", "SP_ReduceScatter_Forward"]
sp_ag_avg = d.get("SP_AllGather_Forward", {}).get("avg_time_ms", 0)
sp_rs_avg = d.get("SP_ReduceScatter_Forward", {}).get("avg_time_ms", 0)

print(f"\nSP_AG_FWD avg: {sp_ag_avg:.2f} ms, SP_RS_FWD avg: {sp_rs_avg:.2f} ms")
