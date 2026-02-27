import json
with open("/Users/wangqinghe/Desktop/Hops论文/draw/hops_profiling_results_dp2_tp8_pp1_rank0.json") as f:
    d = json.load(f)

iter_count = d["Iteration"]["count"]
def disp(k):
    if k in d:
        tot = d[k]["avg_time_ms"] * d[k]["count"] / iter_count
        print(f"{k}: count={d[k]['count']}, time_per_iter={tot:.2f} ms")

disp("Forward_Backward_Pipeline")
disp("DP_ReduceScatter_Wait")
disp("DP_AllGather_Wait")
