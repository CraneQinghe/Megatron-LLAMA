import json
with open("/Users/wangqinghe/Desktop/Hops论文/draw/hops_profiling_results_dp2_tp8_pp1_rank0.json") as f:
    d = json.load(f)

iter_avg = d["Iteration"]["avg_time_ms"]
if "Forward_Backward_Pipeline" in d:
    fw_bw_avg = d["Forward_Backward_Pipeline"]["avg_time_ms"] * d["Forward_Backward_Pipeline"]["count"] / d["Iteration"]["count"]
else:
    fw_bw_avg = 0

print(f"1. Total Iteration: {iter_avg:.2f} ms")
print(f"2. Inner FW-BW Pipeline: {fw_bw_avg:.2f} ms")
print(f"3. Difference (Iter - FW_BW): {iter_avg - fw_bw_avg:.2f} ms")

if fw_bw_avg > 0:
    # Notice: per layer has count 80, iter count=40, so 80/40 = 2 per iter
    layers_total = sum(d.get(f"Layer_{i}_Total_Forward_NoSync", {}).get("avg_time_ms", 0) * d.get(f"Layer_{i}_Total_Forward_NoSync", {}).get("count", 0) / d["Iteration"]["count"]  for i in range(1, 33)) + \
                   sum(d.get(f"Layer_{i}_Total_Backward_NoSync", {}).get("avg_time_ms", 0) * d.get(f"Layer_{i}_Total_Backward_NoSync", {}).get("count", 0) / d["Iteration"]["count"] for i in range(1, 33)) 
    sp_total = sum(d.get(k, {}).get("avg_time_ms", 0) * d.get(k, {}).get("count", 0) / d["Iteration"]["count"] 
                   for k in ["SP_AllGather_Forward", "SP_AllGather_Backward_Recompute", "SP_ReduceScatter_Forward", "SP_ReduceScatter_Backward", "SP_AllGather_Backward"])
                   
    other_fw_bw = sum(d.get(k, {}).get("avg_time_ms", 0) * d.get(k, {}).get("count", 0) / d["Iteration"]["count"] for k in ["Embedding_Forward", "Logits_SP_Gather_Forward", "Logits_Forward", "Loss_Softmax_Forward", "Logits_Backward", "Embedding_Backward"])
    
    print(f"\nInside FW-BW ({fw_bw_avg:.2f} ms), we measured:")
    print(f"  - Layers Compute: {layers_total:.2f} ms")
    print(f"  - SP Comm (FW+BW): {sp_total:.2f} ms")
    print(f"  - Other Nodes (Embed/Logits): {other_fw_bw:.2f} ms")
    print(f"  - Unaccounted FW-BW (Bubbles/CPU Wait): {fw_bw_avg - (layers_total + sp_total + other_fw_bw):.2f} ms")

print(f"\nOutside FW-BW ({iter_avg - fw_bw_avg:.2f} ms):")

def get_norm(k):
    return d.get(k, {}).get("avg_time_ms", 0) * d.get(k, {}).get("count", 0) / d["Iteration"]["count"]

print(f"  - DP RS Wait: {get_norm('DP_ReduceScatter_Wait'):.2f} ms")
print(f"  - DP AG Wait: {get_norm('DP_AllGather_Wait'):.2f} ms")
print(f"  - Optimizer Step: {get_norm('Optimizer_Step'):.2f} ms")
print(f"  - Syncs & Others: {get_norm('Backward_Tail_Wait_Sync') + get_norm('DP_Global_Sync_Barrier') + get_norm('DP_AllGather_Sync_Barrier') + get_norm('Optimizer_Zero_Grad') + get_norm('Main_ReduceScatter_Wait') + get_norm('Main_AllGather_Wait'):.2f} ms")
print(f"  - Main_ReduceScatter_Wait (included in above): {get_norm('Main_ReduceScatter_Wait'):.2f} ms")
print(f"  - Main_AllGather_Wait (included in above): {get_norm('Main_AllGather_Wait'):.2f} ms")

