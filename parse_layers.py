import json
with open("/Users/wangqinghe/Desktop/Hops论文/draw/hops_profiling_results_dp2_tp8_pp1_rank0.json") as f:
    d = json.load(f)

total_layer_fw = 0
total_layer_bw = 0
for i in range(1, 33):
    fw_key = f"Layer_{i}_Total_Forward_NoSync"
    bw_key = f"Layer_{i}_Total_Backward_NoSync"
    if fw_key in d:
        total_layer_fw += d[fw_key]['avg_time_ms']
    if bw_key in d:
        total_layer_bw += d[bw_key]['avg_time_ms']

print(f"Total Layer Forward: {total_layer_fw:.2f} ms")
print(f"Total Layer Backward: {total_layer_bw:.2f} ms")
print(f"Total Layer (FW+BW): {total_layer_fw + total_layer_bw:.2f} ms")

sp_ag_keys = [k for k in d.keys() if "SP_AllGather_Forward" in k or "SP_AllGather_Backward_Recompute" in k]
sp_rs_keys = [k for k in d.keys() if "SP_ReduceScatter" in k]
print(f"SP AllGather Count: {len(sp_ag_keys)}")
print(f"SP ReduceScatter Count: {len(sp_rs_keys)}")

total_ag = 0
total_rs = 0
if "SP_AllGather_Forward" in d:
    total_ag += d["SP_AllGather_Forward"]["avg_time_ms"] * d["SP_AllGather_Forward"]["count"] / d["Iteration"]["count"]
if "SP_AllGather_Backward_Recompute" in d:
    total_ag += d["SP_AllGather_Backward_Recompute"]["avg_time_ms"] * d["SP_AllGather_Backward_Recompute"]["count"] / d["Iteration"]["count"]
if "SP_ReduceScatter_Backward" in d:
    total_rs += d["SP_ReduceScatter_Backward"]["avg_time_ms"] * d["SP_ReduceScatter_Backward"]["count"] / d["Iteration"]["count"]
if "SP_ReduceScatter_Forward" in d:
    total_rs += d["SP_ReduceScatter_Forward"]["avg_time_ms"] * d["SP_ReduceScatter_Forward"]["count"] / d["Iteration"]["count"]

print(f"Est. Total SP Comm (FW+BW, per iter): {total_ag + total_rs:.2f} ms")
print(f"SP AG: {total_ag:.2f}, SP RS: {total_rs:.2f}")

other_fw = 0
if "Embedding_Forward" in d:
    other_fw += d["Embedding_Forward"]["avg_time_ms"]
if "Logits_SP_Gather_Forward" in d:
    other_fw += d["Logits_SP_Gather_Forward"]["avg_time_ms"]
if "Logits_Forward" in d:
    other_fw += d["Logits_Forward"]["avg_time_ms"]
if "Loss_Softmax_Forward" in d:
    other_fw += d["Loss_Softmax_Forward"]["avg_time_ms"]

print(f"Other FW (Embed+Logits+Loss): {other_fw:.2f} ms")

other_bw = 0
if "Logits_Backward" in d:
    other_bw += d["Logits_Backward"]["avg_time_ms"]
if "Embedding_Backward" in d:
    other_bw += d["Embedding_Backward"]["avg_time_ms"]

print(f"Other BW (Logits+Embed): {other_bw:.2f} ms")

