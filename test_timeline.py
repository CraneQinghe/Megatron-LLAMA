import json
with open("/Users/wangqinghe/Desktop/Hops论文/draw/hops_profiling_results_dp2_tp8_pp1_rank0.json") as f:
    d = json.load(f)

for k, v in d.items():
    if type(v) == dict and "count" in v:
        ratio = v["count"] / d["Iteration"]["count"]
        # Look for things that only happen 1x per iteration and are huge
        if 0.5 <= ratio <= 2 and v["avg_time_ms"] > 50:
             print(f"{k}: count={v['count']}, iter_count={d['Iteration']['count']}, ratio={ratio:.2f}, avg_time={v['avg_time_ms']:.2f} ms")
             
