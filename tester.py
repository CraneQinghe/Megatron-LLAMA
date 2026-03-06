import json

path1 = "/Users/wangqinghe/Desktop/Hops论文/draw/hops_profiling_results_llama_1B_dp1_tp4_pp2_rank0.json"
path2 = "/Users/wangqinghe/Desktop/Hops论文/draw/hops_profiling_results_llama_1B_dp1_tp4_pp2_rank4.json"

try:
    with open(path1) as f:
        data1 = json.load(f)
        layer1_bwd = None
        for k in data1:
            if "_Total_Backward" in k and "unprofiled" in k:
                layer1_bwd = data1[k].get("all_ms", [])
                break

    with open(path2) as f:
        data2 = json.load(f)
        layer2_bwd = None
        layer2_avg = None
        for k in data2:
            if "_Total_Backward" in k and "unprofiled" in k:
                layer2_bwd = data2[k].get("all_ms", [])
                layer2_avg = data2[k].get("avg_time_ms")
                break
                
    print("Rank 0 Bwd layer count:", len(layer1_bwd) if layer1_bwd else None)
    print("Rank 4 Bwd layer count:", len(layer2_bwd) if layer2_bwd else None)
    print("Rank 4 avg:", layer2_avg)
except Exception as e:
    print(e)
