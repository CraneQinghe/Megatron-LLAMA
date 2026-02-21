import torch
import torch.distributed as dist
import json
import os
import atexit

class HopsProfiler:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HopsProfiler, cls).__new__(cls)
            cls._instance.init()
        return cls._instance

    def init(self):
        self.events = {}
        self.stats = {}
        self.enabled = True
        self.layer_fwds_started = 0
        self.detailed_profiling_enabled = True
        self.async_events_to_measure = []
        self.heartbeat_layer_name = None
        atexit.register(self.dump)

    def record_model_info(self, model, args):
        try:
            if dist.is_initialized() and dist.get_rank() != 0:
                return
        except:
            pass
        
        # We need to explicitly isolate the parameters to extrapolate full model DP cost
        vocab_emb_params = 0
        layer_params = 0
        other_params = 0
        total_params = 0
        param_bytes = 2 # default fp16/bf16 fallback

        for m in model:
            for name, p in m.named_parameters():
                if not p.requires_grad:
                    continue
                num_params = p.numel()
                total_params += num_params
                param_bytes = p.element_size() # Extract real byte size of the parameter
                
                # Check naming convention inside Megatron for word embeddings and the output layer
                if 'word_embeddings' in name or 'position_embeddings' in name or 'final_layernorm' in name or "lm_head" in name in name:
                    vocab_emb_params += num_params
                elif 'layers.0.' in name:
                    layer_params += num_params
                else: 
                    # other residual logic parameters outside transformer loop
                    other_params += num_params
                
        bucket_size_mb = args.reduce_bucket_size / (1024**2) if hasattr(args, 'reduce_bucket_size') and args.reduce_bucket_size else 0

        self.stats["Model_Grad_Params_Total"] = {"count": 1, "total_ms": total_params}
        self.stats["Model_Grad_Params_Total_MB"] = {"count": 1, "total_ms": total_params * param_bytes / (1024**2)}
        self.stats["Model_Grad_Params_Embedding_And_Head"] = {"count": 1, "total_ms": vocab_emb_params}
        self.stats["Model_Grad_Params_Embedding_And_Head_MB"] = {"count": 1, "total_ms": vocab_emb_params * param_bytes / (1024**2)}
        self.stats["Model_Grad_Params_Single_Layer"] = {"count": 1, "total_ms": layer_params}
        self.stats["Model_Grad_Params_Single_Layer_MB"] = {"count": 1, "total_ms": layer_params * param_bytes / (1024**2)}
        self.stats["Param_Size_Bytes"] = {"count": 1, "total_ms": param_bytes}
        self.stats["Reduce_Bucket_Size_MB"] = {"count": 1, "total_ms": bucket_size_mb}

    def start(self, name):
        if not self.enabled: return
        try:
            if dist.is_initialized() and dist.get_rank() != 0:
                return
        except:
            pass
            
        is_layer_total = "Layer_" in name and "_Total" in name

        if is_layer_total and "Forward" in name:
            if self.heartbeat_layer_name is None:
                self.heartbeat_layer_name = name
            
            if name == self.heartbeat_layer_name:
                self.layer_fwds_started += 1
                if self.layer_fwds_started > 200: # 50 iters * 4 microbatches
                    self.detailed_profiling_enabled = False

        if not self.detailed_profiling_enabled and not is_layer_total:
            return

        real_name = name
        if is_layer_total:
            real_name = name + ("_Detailed" if self.detailed_profiling_enabled else "_NoSync")

        start_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        self.events[real_name] = start_evt

    def stop(self, name):
        if not self.enabled: return
        try:
            if dist.is_initialized() and dist.get_rank() != 0:
                return
        except:
            pass

        is_layer_total = "Layer_" in name and "_Total" in name
        real_name = name
        if is_layer_total:
            if name + "_Detailed" in self.events:
                real_name = name + "_Detailed"
            elif name + "_NoSync" in self.events:
                real_name = name + "_NoSync"

        if real_name not in self.events:
            return
        
        start_evt = self.events.pop(real_name)
        end_evt = torch.cuda.Event(enable_timing=True)
        end_evt.record()

        if self.detailed_profiling_enabled:
            # Force serialization and calculate locally to simulate the sync overhead
            torch.cuda.synchronize()
            elapsed = start_evt.elapsed_time(end_evt)
            if real_name not in self.stats:
                self.stats[real_name] = {"count": 0, "total_ms": 0.0}
            self.stats[real_name]["count"] += 1
            self.stats[real_name]["total_ms"] += elapsed
        else:
            # Fully Non-blocking: just record events to not affect system dynamics
            self.async_events_to_measure.append((real_name, start_evt, end_evt))

    def dump(self):
        print(f"[Debug HopsProfiler] dump() called!", flush=True)
        try:
            if dist.is_initialized() and dist.get_rank() != 0:
                print(f"[Debug HopsProfiler] Skipping dump because rank != 0", flush=True)
                return
        except:
            pass

        if self.async_events_to_measure:
            torch.cuda.synchronize()
            for n, s_evt, e_evt in self.async_events_to_measure:
                elapsed = s_evt.elapsed_time(e_evt)
                if n not in self.stats:
                    self.stats[n] = {"count": 0, "total_ms": 0.0}
                self.stats[n]["count"] += 1
                self.stats[n]["total_ms"] += elapsed
            self.async_events_to_measure.clear() # Prevent dual counting if dump() called multiple times

        if not self.stats:
            print(f"[Debug HopsProfiler] dump() failed because self.stats is empty!", flush=True)
            return

        res = {}
        static_keys = [
            "Model_Grad_Params_Total", "Model_Grad_Params_Total_MB",
            "Model_Grad_Params_Embedding_And_Head", "Model_Grad_Params_Embedding_And_Head_MB", 
            "Model_Grad_Params_Single_Layer", "Model_Grad_Params_Single_Layer_MB",
            "Param_Size_Bytes", "Reduce_Bucket_Size_MB"
        ]
        for k, v in self.stats.items():
            if k in static_keys:
                res[k] = v["total_ms"]
            else:
                avg_time = v["total_ms"] / v["count"] if v["count"] else 0
                res[k] = {
                    "count": v["count"],
                    "total_time_ms": v["total_ms"],
                    "avg_time_ms": avg_time
                }
            
        out_file = os.path.join(os.getcwd(), "hops_profiling_results.json")
        try:
            with open(out_file, "w") as f:
                json.dump(res, f, indent=4)
            print(f"\n[HopsProfiler] Successfully exported profiling stats to {out_file}\n", flush=True)
        except Exception as e:
            print(f"[HopsProfiler] Failed to write profile: {e}", flush=True)

hops_profiler = HopsProfiler()

class ProfileRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, name, is_start, x):
        ctx.name = name
        ctx.is_start = is_start
        if is_start:
            hops_profiler.start(f"{name}_Forward")
        else:
            hops_profiler.stop(f"{name}_Forward")
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.is_start:
            # End of backward pass for this region
            hops_profiler.stop(f"{ctx.name}_Backward")
        else:
            # Start of backward pass for this region
            hops_profiler.start(f"{ctx.name}_Backward")
        return None, None, grad_output

def mark_region_start(name, tensor):
    return ProfileRegion.apply(name, True, tensor)

def mark_region_end(name, tensor):
    return ProfileRegion.apply(name, False, tensor)
