import json

with open("/Users/wangqinghe/Desktop/Hops论文/draw/hops_profiling_results_dp2_tp8_pp1_rank0.json") as f:
    d = json.load(f)

for k, v in d.items():
    if type(v) == dict and "count" in v:
        ratio = v["count"] / d["Iteration"]["count"]
        # Look for things that only happen 1x per iteration and are huge
        if ratio == 1.0 and "Detailed" not in k and "NoSync" not in k and "Detailed" not in k:
             print(f"{k}: iter_time = {v['avg_time_ms'] * ratio:.2f}")

