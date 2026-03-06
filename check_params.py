import json

def get_params(path):
    try:
        with open(path) as f:
            data = json.load(f)
            return data.get('Model_Grad_Params_Total_MB', 'Unknown'), data.get('Model_Grad_Params_Single_Layer_MB', 'Unknown')
    except:
        return "File Not Found", "N/A"

p1b = "/Users/wangqinghe/Desktop/Hops论文/draw/hops_profiling_results_llama_1B_dp1_tp4_pp1_rank0.json"
p8b = "/Users/wangqinghe/Desktop/Hops论文/draw/hops_profiling_results_llama_8B_dp2_tp4_pp1_rank0.json"

print("1B Baseline Params (Total MB, Single Layer MB):", get_params(p1b))
print("8B Actual Params   (Total MB, Single Layer MB):", get_params(p8b))
