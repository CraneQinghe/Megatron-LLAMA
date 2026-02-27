import json
with open("/Users/wangqinghe/Desktop/Hops论文/draw/hops_profiling_results_dp2_tp8_pp1_rank0.json") as f:
    d = json.load(f)

for k, v in d.items():
    if type(v) == dict and "count" in v:
        if "Recompute" in k or "recompute" in k:
            print(f"{k}: count={v['count']}, iter_count={d['Iteration']['count']}, ratio={v['count']/d['Iteration']['count']}, avg_time={v['avg_time_ms']} ms")
