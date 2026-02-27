import json
with open("/Users/wangqinghe/Desktop/Hops论文/draw/hops_profiling_results_dp2_tp8_pp1_rank0.json") as f:
    d = json.load(f)

print(f"Total Iteration avg: {d['Iteration']['avg_time_ms']:.2f} ms")
