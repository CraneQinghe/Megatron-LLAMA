import json
with open("/Users/wangqinghe/Desktop/Hops论文/draw/hops_profiling_results_dp2_tp8_pp1_rank0.json") as f:
    d = json.load(f)

for k, v in d.items():
    if type(v) == dict and "count" in v:
        # Check if the total time could account for the 4s difference
        total = v["avg_time_ms"] * v["count"] / d["Iteration"]["count"]
        if total > 50 and "Forward_Detailed" not in k and "Backward_Detailed" not in k and "Detailed" not in k and "NoSync" not in k:
             print(f"{k}: Iter_Normed_Total_Time = {total:.2f} ms， count={v['count']}")

