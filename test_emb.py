import os
import json

def get_topology_info(dp_size, tp_size, pp_size, comm_type='dp', gpus_per_node=8):
    def get_rank(pp, dp, tp):
        return pp * (dp_size * tp_size) + dp * tp_size + tp
    
    ranks = [get_rank(p, 0, 0) for p in range(pp_size)]
    stride = (ranks[1] - ranks[0]) if len(ranks) > 1 else 1
    nodes = set(r // gpus_per_node for r in ranks)
    mesh_a = len(nodes)
    mesh_b = len(ranks) // mesh_a
    return stride, mesh_a, mesh_b, len(ranks)

print(get_topology_info(2, 4, 2, 'emb'))
