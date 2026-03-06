import json

path = "/Users/wangqinghe/Desktop/Hops论文/draw/hops_profiling_results_qwen_7B_dp2_tp4_pp1_rank0.json"
with open(path) as f:
    data = json.load(f)

iter_count = data.get("Iteration_Unprofiled", {}).get("count", 4)
if iter_count == 0: iter_count = 1

metrics = []
for k, v in data.items():
    if isinstance(v, dict):
        # Prefer unprofiled for layers
        if "total_time_ms" in v:
            t = v["total_time_ms"] / iter_count
            metrics.append((k, t))
        elif "avg_time_ms" in v:
            c = v.get("count", 0)
            t = v["avg_time_ms"] * (c / iter_count)
            metrics.append((k, t))

print(f"{'Key':<50} | {'Time per Iter (ms)':<20}")
print("-" * 75)
for k, t in sorted(metrics, key=lambda x: x[1], reverse=True)[:40]:
    print(f"{k:<50} | {t:<20.2f}")
