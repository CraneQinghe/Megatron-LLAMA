import json

def get_stats(path):
    with open(path) as f:
        return json.load(f)

d1 = get_stats("/Users/wangqinghe/Desktop/Hops论文/draw/hops_profiling_results_dp2_tp8_pp1_rank0.json")

def v(d, k):
    if type(d.get(k)) == dict and "count" in d.get(k):
        return d[k]["avg_time_ms"] * d[k]["count"] / d["Iteration"]["count"]
    return 0

print("Difference Calculation For DP2_TP8: ")

layers_fw = sum(v(d1, f'Layer_{i}_Total_Forward_NoSync') for i in range(1, 33))
layers_bw = sum(v(d1, f'Layer_{i}_Total_Backward_NoSync') for i in range(1, 33))

sp_time = v(d1, 'SP_AllGather_Forward') + v(d1, 'SP_AllGather_Backward_Recompute') + v(d1, 'SP_ReduceScatter_Forward') + v(d1, 'SP_ReduceScatter_Backward') + v(d1, 'SP_AllGather_Backward')
other_fw = v(d1, 'Embedding_Forward') + v(d1, 'Logits_SP_Gather_Forward') + v(d1, 'Logits_Forward') + v(d1, 'Loss_Softmax_Forward')
other_bw = v(d1, 'Logits_Backward') + v(d1, 'Embedding_Backward')

compute_comm = layers_fw + layers_bw + sp_time + other_fw + other_bw

out = v(d1, "Optimizer_Zero_Grad") + v(d1, "Backward_Tail_Wait_Sync") + v(d1, "DP_Global_Sync_Barrier") + v(d1, "DP_ReduceScatter_Wait") + v(d1, "Optimizer_Step") + v(d1, "DP_AllGather_Sync_Barrier") + v(d1, "DP_AllGather_Wait") + v(d1, "Cancel_Gradients_Vision")


print(f"Total Iteration: {d1['Iteration']['avg_time_ms']:.2f}")

print(f"Total Compute+SP: {compute_comm:.2f}")
print(f"Total OutOfPipeline: {out:.2f}")

print(f"Difference: {d1['Iteration']['avg_time_ms'] - compute_comm - out:.2f}")

print("-" * 50)
print("Pipeline vs detailed compute")
print(f"Forward_Backward_Pipeline: {v(d1, 'Forward_Backward_Pipeline'):.2f}")
print(f"Detailed compute+SP: {compute_comm:.2f}")
print(f"Difference: {v(d1, 'Forward_Backward_Pipeline') - compute_comm:.2f}")

