import json

path = "/Users/wangqinghe/Desktop/Hops论文/draw/hops_profiling_results_qwen_7B_dp2_tp4_pp1_rank0.json"
with open(path) as f:
    data = json.load(f)

iter_count = 4

print("All significant metrics (> 100ms total):")
for k, v in sorted(data.items(), key=lambda x: x[1].get('total_time_ms', 0) if isinstance(x[1], dict) else 0, reverse=True):
    if isinstance(v, dict) and v.get('total_time_ms', 0) > 100:
         print(f"  {k}: total={v['total_time_ms']:.2f}, avg={v.get('avg_time_ms', 0):.2f}, count={v.get('count',0)}")
