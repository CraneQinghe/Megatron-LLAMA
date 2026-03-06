import json

path = "/Users/wangqinghe/Desktop/Hops论文/draw/hops_profiling_results_qwen_7B_dp2_tp4_pp1_rank0.json"
with open(path) as f:
    data = json.load(f)

print("Iteration Tags:")
for k in data:
    if "Iteration" in k:
        print(f"  {k}: {data[k].get('avg_time_ms')}")

print("\nTop 10 Largest Metrics (by avg_time_ms * count):")
metrics = []
for k, v in data.items():
    if isinstance(v, dict) and 'avg_time_ms' in v and 'count' in v:
        metrics.append((k, v['avg_time_ms'] * v.get('count', 1)))

for k, t in sorted(metrics, key=lambda x: x[1], reverse=True)[:20]:
    print(f"  {k}: {t:.2f} ms")
