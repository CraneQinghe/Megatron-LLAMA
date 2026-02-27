import json
with open("/Users/wangqinghe/Desktop/Hops论文/draw/hops_profiling_results_dp2_tp8_pp1_rank0.json") as f:
    d = json.load(f)

def v(k):
    return d.get(k, {}).get("avg_time_ms", 0) * d.get(k, {}).get("count", 0) / d["Iteration"]["count"]

iter_avg = d["Iteration"]["avg_time_ms"]
print(f"Total Iter: {iter_avg:.2f}")

# The things outside pipeline:
out = v("Optimizer_Zero_Grad") + v("Backward_Tail_Wait_Sync") + v("DP_Global_Sync_Barrier") + v("DP_ReduceScatter_Wait") + v("Optimizer_Step") + v("DP_AllGather_Sync_Barrier") + v("DP_AllGather_Wait") + v("Cancel_Gradients_Vision")

print(f"Total Outside: {out:.2f}")
print(f"Forward_Backward_Pipeline: {v('Forward_Backward_Pipeline'):.2f}")
diff = iter_avg - out - v('Forward_Backward_Pipeline')
print(f"Unaccounted Iteration Difference: {diff:.2f}")

