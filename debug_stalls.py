import json

path = "/Users/wangqinghe/Desktop/Hops论文/draw/hops_profiling_results_qwen_7B_dp2_tp4_pp1_rank0.json"
with open(path) as f:
    data = json.load(f)

iter_count = 1  # Standard for these results
total_captured = 0
print(f"{'Metric':<40} | {'Time (ms)':<15}")
print("-" * 60)

# The components we currently track in planner_predict.py
tracked_prefixes = ["Layer", "Embedding_Forward", "Logits_Forward", "Loss_Softmax_Forward", 
                    "Logits_Backward", "PP_Comm", "DP_Global_Sync", "Backward_Tail_Wait", "DP_ReduceScatter", 
                    "Main_ReduceScatter", "DP_AllGather", "Main_AllGather", "Optimizer_Step", "WordEmbedding_AllReduce"]

components = []

for k, v in data.items():
    if not isinstance(v, dict) or 'total_time_ms' not in v: continue
    t = v['total_time_ms']
    
    # Check if this is one of our primary components
    matched = False
    for p in tracked_prefixes:
        if p in k:
            matched = True
            break
    
    if t > 50: # Only look at significant things
        components.append((k, t))

# Sort by time
for k, t in sorted(components, key=lambda x: x[1], reverse=True):
    print(f"{k:<40} | {t:<15.2f}")

iter_total = data.get('Iteration_Unprofiled', {}).get('avg_time_ms', 0)
print("-" * 60)
print(f"Iteration_Unprofiled Total: {iter_total:.2f} ms")
