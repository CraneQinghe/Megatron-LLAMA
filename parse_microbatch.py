import json

with open("/Users/wangqinghe/Desktop/Hops论文/draw/hops_profiling_results_dp2_tp8_pp1_rank0.json") as f:
    d = json.load(f)

iter_count = d["Iteration"]["count"]
print(f"Total Iteration: {d['Iteration']['avg_time_ms']:.2f} ms")

for k in [
    "Forward_Backward_Pipeline",
    "Microbatch_Forward_Step",
    "Microbatch_Backward_Step",
    "Optimizer_Backward_Epilogue_NoSync",
    "Microbatch_Forward_Step_Last",
    "Microbatch_Backward_Step_Last",
    "Optimizer_Backward_Epilogue_Sync",
    "DP_ReduceScatter_Wait",
    "DP_AllGather_Wait",
    "Main_ReduceScatter_Wait",
    "Main_AllGather_Wait"
]:
    if k in d:
        v = d[k]
        total_time_per_iter = v["avg_time_ms"] * v["count"] / iter_count
        print(f"{k}: count={v['count']}, iter_count={iter_count}, avg_time={v['avg_time_ms']:.2f} ms, iter_normed={total_time_per_iter:.2f} ms")
    else:
        print(f"{k}: NOT FOUND")
