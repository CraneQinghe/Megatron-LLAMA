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
        self.iters_recorded = 0
        self.detailed_profiling_enabled = True
        self.async_events_to_measure = []
        self.heartbeat_layer_name = None
        atexit.register(self.dump)

    def record_model_info(self, model, args):
        # try:
        #     if dist.is_initialized() and dist.get_rank() != 0:
        #         return
        # except:
        #     pass
        
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

        # --- Dynamic Shape Tracing Hook Injection (PyTorch Native FX alternative) ---
        self.recorded_shape_keys = set()
        # --- Dynamic Sequential Execution Tracer Hook Injection ---
        self.Layer_Sequential_Shapes = []
        self._traced_module_names = set()
        self._current_trace_step = 0
        
        def create_sequential_hook(module_name):
            def hook(module, inputs, output):
                if module_name in self._traced_module_names: 
                    return
                
                try:
                    # Parse inputs (sometimes can be a tuple, so we get the first tensor)
                    x_shape = "None"
                    if isinstance(inputs, tuple) and len(inputs) > 0 and hasattr(inputs[0], 'shape'):
                        x_shape = str(list(inputs[0].shape))
                    elif hasattr(inputs, 'shape'):
                        x_shape = str(list(inputs.shape))
                        
                    # Parse outputs
                    y_shape = "None"
                    if isinstance(output, tuple) and len(output) > 0 and hasattr(output[0], 'shape'):
                        y_shape = str(list(output[0].shape))
                    elif hasattr(output, 'shape'):
                        y_shape = str(list(output.shape))
                        
                    w_shape = str(list(module.weight.shape)) if hasattr(module, 'weight') and module.weight is not None else "None"
                    
                    # Record the execution order with step index
                    step_info = {
                        "Step": self._current_trace_step,
                        "Module_Name": module_name,
                        "Type": module.__class__.__name__,
                        "Input_Shape": x_shape,
                        "Output_Shape": y_shape,
                        "Weight_Shape": w_shape
                    }
                    self.Layer_Sequential_Shapes.append(step_info)
                    self._traced_module_names.add(module_name)
                    
                    self._current_trace_step += 1
                    
                    # Store inside the dummy Shape_Tracer payload so JSON serialization catches it
                    self.stats["Shape_Tracer_Sequential_Flow"] = {
                        "count": 1,
                        "avg_time_ms": 0,
                        "Sequential_Flow": self.Layer_Sequential_Shapes
                    }
                        
                except Exception as e:
                    pass
            return hook

        for m in model:
            # We strictly bind hooks to the very first Transformer Layer instance to avoid clutter
            for name, module in m.named_modules():
                # We want to capture leaf modules (ops) inside the first transformer layer
                if "layers.0" in name and not list(module.children()):
                    module.register_forward_hook(create_sequential_hook(name))

    def start(self, name):
        if not self.enabled: return
        # try:
        #     if dist.is_initialized() and dist.get_rank() != 0:
        #         return
        # except:
        #     pass
            
        is_layer_total = "Layer_" in name and "_Total" in name

        if is_layer_total and "Forward" in name:
            if self.heartbeat_layer_name is None:
                self.heartbeat_layer_name = name
            
            if name == self.heartbeat_layer_name:
                self.layer_fwds_started += 1
                if self.layer_fwds_started > 120: # 10 warmup + 20 iters * 4 microbatches
                    self.detailed_profiling_enabled = False

        if not self.detailed_profiling_enabled and not is_layer_total and name != "Iteration":
            return

        if getattr(self, 'iters_recorded', 0) >= 6 and name != "Iteration":
            return
            
        real_name = name
        if is_layer_total:
            real_name = name + ("_Detailed" if self.detailed_profiling_enabled else "_NoSync")

        start_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        start_mem = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
        self.events[real_name] = (start_evt, start_mem)

    def stop(self, name):
        if not self.enabled: return
        # try:
        #     if dist.is_initialized() and dist.get_rank() != 0:
        #         return
        # except:
        #     pass

        is_layer_total = "Layer_" in name and "_Total" in name
        real_name = name
        if is_layer_total:
            if name + "_Detailed" in self.events:
                real_name = name + "_Detailed"
            elif name + "_NoSync" in self.events:
                real_name = name + "_NoSync"

        if real_name not in self.events:
            return
        
        start_evt, start_mem = self.events.pop(real_name)
        end_evt = torch.cuda.Event(enable_timing=True)
        end_evt.record()
        end_mem = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
        mem_diff = end_mem - start_mem

        is_warmup = self.iters_recorded < 2
        is_bypassed = self.iters_recorded >= 6
        
        final_name = real_name
        if real_name == "Iteration":
            if is_bypassed:
                final_name = "Iteration_Unprofiled"
            elif is_warmup:
                final_name = "Iteration_Warmup"
            else:
                final_name = "Iteration_Profiled"

        if self.detailed_profiling_enabled or real_name == "Iteration":
            # Force serialization and calculate locally to simulate the sync overhead
            torch.cuda.synchronize()
            elapsed = start_evt.elapsed_time(end_evt)
            
            # Record if it's an Iteration stat (always), or if it's a normal stat inside the profiled window
            if real_name == "Iteration" or real_name == "Optimizer_Step" or (not is_warmup and not is_bypassed):
                if final_name not in self.stats:
                    self.stats[final_name] = {"count": 0, "total_ms": 0.0, "all_ms": [], "all_mem_mb": []}
                self.stats[final_name]["count"] += 1
                self.stats[final_name]["total_ms"] += elapsed
                self.stats[final_name]["all_ms"].append(elapsed)
                self.stats[final_name]["all_mem_mb"].append(mem_diff)
        else:
            # Fully Non-blocking: just record events to not affect system dynamics
            if real_name == "Optimizer_Step" or (not is_warmup and not is_bypassed):
                self.async_events_to_measure.append((real_name, start_evt, end_evt, mem_diff))
                
                # IMPORTANT FIX: Periodically flush CUDA events to prevent Event Pool Exhaustion!
                # Holding tens of thousands of torch.cuda.Event objects will hit hardware limits 
                # and trigger massive implicit syncs / freezing (~98s hangs).
                if len(self.async_events_to_measure) > 5000:
                    torch.cuda.synchronize()
                    for n, s_evt, e_evt, m_diff in self.async_events_to_measure:
                        elapsed = s_evt.elapsed_time(e_evt)
                        if n not in self.stats:
                            self.stats[n] = {"count": 0, "total_ms": 0.0, "all_ms": [], "all_mem_mb": []}
                        self.stats[n]["count"] += 1
                        self.stats[n]["total_ms"] += elapsed
                        self.stats[n]["all_ms"].append(elapsed)
                        self.stats[n]["all_mem_mb"].append(m_diff)
                    self.async_events_to_measure.clear()
                    
        if real_name == "Iteration":
            self.iters_recorded += 1

    def dump(self):
        if hasattr(self, '_has_dumped') and self._has_dumped:
            return
        self._has_dumped = True
        print(f"[Debug HopsProfiler] dump() called!", flush=True)

        if self.async_events_to_measure:
            torch.cuda.synchronize()
            for n, s_evt, e_evt, m_diff in self.async_events_to_measure:
                elapsed = s_evt.elapsed_time(e_evt)
                if n not in self.stats:
                    self.stats[n] = {"count": 0, "total_ms": 0.0, "all_ms": [], "all_mem_mb": []}
                self.stats[n]["count"] += 1
                self.stats[n]["total_ms"] += elapsed
                self.stats[n]["all_ms"].append(elapsed)
                self.stats[n]["all_mem_mb"].append(m_diff)
            self.async_events_to_measure.clear() # Prevent dual counting if dump() called multiple times

        if not self.stats:
            print(f"[Debug HopsProfiler] dump() failed because self.stats is empty!", flush=True)
            return

        res = {}
        res_shapes = {}
        static_keys = [
            "Model_Grad_Params_Total", "Model_Grad_Params_Total_MB",
            "Model_Grad_Params_Embedding_And_Head", "Model_Grad_Params_Embedding_And_Head_MB", 
            "Model_Grad_Params_Single_Layer", "Model_Grad_Params_Single_Layer_MB",
            "Param_Size_Bytes", "Reduce_Bucket_Size_MB"
        ]
        for k, v in self.stats.items():
            if "Shape_Tracer_" in k:
                if "Sequential_Flow" in v:
                    res_shapes[k.replace("Shape_Tracer_", "")] = v["Sequential_Flow"]
                else:
                    shape_data = {
                        "Input_X_Shape": v.get("Input_X_Shape"),
                        "Weight_W_Shape": v.get("Weight_W_Shape")
                    }
                    res_shapes[k.replace("Shape_Tracer_", "")] = shape_data
            elif k in static_keys:
                res[k] = v["total_ms"]
            elif isinstance(v, dict) and "total_ms" in v and "count" in v:
                avg_time = v["total_ms"] / v["count"] if v["count"] else 0
                res[k] = {
                    "count": v["count"],
                    "total_time_ms": v["total_ms"],
                    "avg_time_ms": avg_time,
                    "avg_mem_mb": sum(v.get("all_mem_mb", [0])) / len(v.get("all_mem_mb", [1])) if v.get("all_mem_mb") else 0.0,
                    "all_ms": v.get("all_ms", []),
                    "all_mem_mb": v.get("all_mem_mb", [])
                }
            else:
                # Custom metadata/payload or unrecognized format
                res[k] = v
            
        # 尝试动态获取全局配置并加上 Rank 专属标志避免覆写
        topo_suffix = ""
        rank_suffix = ""
        rank = 0
        try:
            from megatron import get_args
            args = get_args()
            tp_size = getattr(args, 'tensor_model_parallel_size', 1)
            pp_size = getattr(args, 'pipeline_model_parallel_size', 1)
            dp_size = getattr(args, 'data_parallel_size', 1)
            
            if dist.is_initialized():
                world_size = dist.get_world_size()
                # Use standard calc just in case DP size attribute is missing
                dp_size = world_size // (tp_size * pp_size)
                rank = dist.get_rank()
                
            topo_suffix = f"_dp{dp_size}_tp{tp_size}_pp{pp_size}"
            
            if dist.is_initialized():
                rank_suffix = f"_rank{rank}"
        except Exception:
            try:
                if dist.is_initialized():
                    rank = dist.get_rank()
            except:
                pass
            
        file_name = f"/data/haiqwa/zevin_nfs/code/qinghe/Megatron-LLAMA/examples/LLaMA/hops_profiling_results{topo_suffix}{rank_suffix}.json"
        
        if rank == 0:
            try:
                os.makedirs(os.path.dirname(file_name), exist_ok=True)
            except: pass
            try:
                with open(file_name, "w") as f:
                    json.dump(res, f, indent=4)
                print(f"\n[HopsProfiler] Successfully exported profiling stats to {file_name}", flush=True)
            except Exception as e:
                # Fallback to current directory
                fallback_name = f"hops_profiling_results{topo_suffix}{rank_suffix}.json"
                try:
                    with open(os.path.join(os.getcwd(), fallback_name), "w") as f2:
                        json.dump(res, f2, indent=4)
                    print(f"\n[HopsProfiler] Successfully exported profiling stats to fallback: {fallback_name}", flush=True)
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
