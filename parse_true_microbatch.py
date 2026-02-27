import json

with open("/Users/wangqinghe/Desktop/Hops论文/draw/hops_profiling_results_dp2_tp8_pp1_rank0.json") as f:
    d = json.load(f)

# Use DP_ReduceScatter_Wait as the gold standard for "number of tracked iterations"
# Since it happens exactly once per iteration, its count is the true window size for steady state 
# (because Iteration might track warmup too)
tracked_iters = d["DP_ReduceScatter_Wait"]["count"]  # should be 20

print(f"Total Iteration (Raw Avg): {d['Iteration']['avg_time_ms']:.2f} ms")
print(f"Number of tracked steady-state iterations: {tracked_iters}\n")

print("--- Normalized Times per Iteration (ms) ---")
for k in [
    "Iteration",
    "Forward_Backward_Pipeline",
    "Microbatch_Forward_Step",
    "Microbatch_Backward_Step",
    "Microbatch_Forward_Step_Last",
    "Microbatch_Backward_Step_Last",
    "DP_ReduceScatter_Wait",
    "DP_AllGather_Wait",
    "Main_ReduceScatter_Wait",
    "Main_AllGather_Wait"
]:
    if k in d:
        v = d[k]
        # Real average execution time of the block
        avg_time = v["avg_time_ms"]
        # How many times this block runs in ONE iteration
        calls_per_iter = v["count"] / tracked_iters
        
        if k == "Iteration":
            print(f"{k}: {avg_time:.2f}")
        else:
            print(f"{k}: {avg_time:.2f} x {calls_per_iter} = {avg_time * calls_per_iter:.2f}")
