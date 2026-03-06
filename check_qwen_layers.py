import json

path = "/Users/wangqinghe/Desktop/Hops论文/draw/hops_profiling_results_qwen_1B_dp1_tp2_pp1_rank0.json"
try:
    with open(path) as f:
        data = json.load(f)
    print(f"Total keys: {len(data)}")
    found = False
    for k in data:
        if "Forward_Detailed_unprofiled" in k:
            print(f"Key: {k}, Avg: {data[k].get('avg_time_ms')}")
            found = True
    if not found:
        print("No matching keys found")
except Exception as e:
    print(f"Error: {e}")
