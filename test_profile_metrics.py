import json
import sys
import numpy as np

def run_test(path):
    try:
        with open(path) as f:
            j = json.load(f)
            
        b_key = None
        for k in j.keys():
            if "_Total_Backward_Detailed" in k and "unprofiled" in k:
                b_key = k
                break
        
        f_key = None
        for k in j.keys():
            if "_Total_Forward_Detailed" in k and "unprofiled" in k:
                f_key = k
                break
                
        if not b_key or not f_key:
            print("Missing unprofiled keys")
            return
            
        print("Forward samples:", len(j[f_key]['all_ms']), "per iter:", j[f_key]['count'])
        print("Backward samples:", len(j[b_key]['all_ms']), "per iter:", j[b_key]['count'])
    except Exception as e:
        print("Error:", e)

run_test('/Users/wangqinghe/Desktop/Hops论文/draw/hops_profiling_results_llama_1B_dp1_tp2_pp1_rank0.json')
