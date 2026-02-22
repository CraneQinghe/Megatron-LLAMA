import torch
import time
import argparse

def benchmark_linear(input_shape, weight_shape, max_iters=200, warmup=50):
    """
    模拟 PyTorch 底层的 nn.Linear(K, N) 以及 F.linear(x, weight)
    x 的形状是 [M, K]
    weight 的形状是 [N, K] 
    输出形状是 [M, N]
    """
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # 初始化 Tensor，使用 FP16 (如果在 CPU/MPS 上可能需要 fallback 到 FP32，这里自动处理一下)
    dtype = torch.float16 if device == 'cuda' else torch.float32
    
    x = torch.randn(input_shape, device=device, dtype=dtype)
    w = torch.randn(weight_shape, device=device, dtype=dtype)
    
    # 预热预热
    for _ in range(warmup):
        y = torch.matmul(x, w.t())
        
    if device == 'cuda': torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(max_iters):
        y = torch.matmul(x, w.t())
        
    if device == 'cuda': torch.cuda.synchronize()
    end = time.time()
    
    avg_time_ms = (end - start) / max_iters * 1000.0
    
    # 计算理论 FLOPs: 2 * M * N * K
    M = input_shape[0] if len(input_shape) == 1 else input_shape[0] * (input_shape[1] if len(input_shape)>1 else 1)
    if len(input_shape) == 3: M = input_shape[0] * input_shape[1]
    K = weight_shape[1]
    N = weight_shape[0]
    
    flops = 2 * M * N * K
    tflops_per_sec = (flops / 1e12) / (avg_time_ms / 1000.0)
    
    return avg_time_ms, tflops_per_sec

def benchmark_sequential_layer(qkv_shapes, o_shapes, gu_shapes, down_shapes, max_iters=200, warmup=50):
    """
    完全模拟 LLaMA 单个 Layer 内部前向传播的算子串行执行顺序。
    QKV -> O -> GateUp -> Down
    """
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    dtype = torch.float16 if device == 'cuda' else torch.float32
    
    # 提前分配好所有的输入和权重，模拟显存驻留
    x_qkv = torch.randn(qkv_shapes[0], device=device, dtype=dtype)
    w_qkv = torch.randn(qkv_shapes[1], device=device, dtype=dtype)
    
    x_o = torch.randn(o_shapes[0], device=device, dtype=dtype)
    w_o = torch.randn(o_shapes[1], device=device, dtype=dtype)
    
    x_gu = torch.randn(gu_shapes[0], device=device, dtype=dtype)
    w_gu = torch.randn(gu_shapes[1], device=device, dtype=dtype)
    
    x_down = torch.randn(down_shapes[0], device=device, dtype=dtype)
    w_down = torch.randn(down_shapes[1], device=device, dtype=dtype)
    
    for _ in range(warmup):
        _ = torch.matmul(x_qkv, w_qkv.t())
        _ = torch.matmul(x_o, w_o.t())
        _ = torch.matmul(x_gu, w_gu.t())
        _ = torch.matmul(x_down, w_down.t())
        
    if device == 'cuda': torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(max_iters):
        _ = torch.matmul(x_qkv, w_qkv.t())
        _ = torch.matmul(x_o, w_o.t())
        _ = torch.matmul(x_gu, w_gu.t())
        _ = torch.matmul(x_down, w_down.t())
        
    if device == 'cuda': torch.cuda.synchronize()
    end = time.time()
    
    avg_total_time = (end - start) / max_iters * 1000.0
    return avg_total_time

def profile_llama_layer(S=4096, B=1, H=4096, FFN=11008, tp_sizes=[1, 2, 4, 8]):
    """
    提取 LLaMA 核心张量在各种 TP 策略切分下的宽和高：
    注：Megatron 序列并行(SP)在发生矩阵乘前，已经AllGather把Seq长度集齐到了 S
    所以实际进入矩阵乘法的行数 M = S * B 是铁定不变的！变的是横向切除的列宽 N 和 K！
    """
    print(f"🚀 Profiling LLaMA TensorCore Non-Linearity (B={B}, S={S}, Hidden={H}, FFN={FFN})")
    print("=" * 110)
    print(f"{'TP':<4} | {'Module':<12} | {'Time(ms)':<10} | {'TFLOPs/s':<10} | {'Input [M, K]':<18} x {'Weight [N, K]':<18}")
    print("-" * 110)
    
    for tp in tp_sizes:
        M = S * B
        
        # 1. Attention QKV 投影 (ColumnParallelLinear)
        # 权重原本是 [3*H, H]，被切分为 [3*H/TP, H]
        qkv_time, qkv_tflops = benchmark_linear((M, H), (3 * H // tp, H))
        
        # 2. Attention O 投影 (RowParallelLinear)
        # 权重原本是 [H, H]，由于输入特征也是切好的，权重被切分为 [H, H/TP]
        o_time, o_tflops = benchmark_linear((M, H // tp), (H, H // tp))
        
        # 3. MLP Gate + Up 投影 (使用 SwiGLU 常见的合并 ColumnParallelLinear)
        # 权重原本是 [2*FFN, H]，被切分为了 [2*FFN/TP, H]
        gate_up_time, gu_tflops = benchmark_linear((M, H), (2 * FFN // tp, H))
        
        # 4. MLP Down 投影 (RowParallelLinear)
        # 权重原本是 [H, FFN]，被切分为 [H, FFN/TP]
        down_time, d_tflops = benchmark_linear((M, FFN // tp), (H, FFN // tp))
        
        total_attn = qkv_time + o_time
        total_mlp = gate_up_time + down_time
        
        # 串行合并跑（测量更真实的 L2 Cache 竞争与上下文切换开销）
        simulated_layer_time = benchmark_sequential_layer(
            ((M, H), (3 * H // tp, H)),
            ((M, H // tp), (H, H // tp)),
            ((M, H), (2 * FFN // tp, H)),
            ((M, FFN // tp), (H, FFN // tp))
        )
        
        print(f"{tp:<4} | {'Attn QKV':<12} | {qkv_time:<10.3f} | {qkv_tflops:<10.1f} | [{M:>5}, {H:>5}]       x [{3*H//tp:>5}, {H:>5}]")
        print(f"{tp:<4} | {'Attn O':<12} | {o_time:<10.3f} | {o_tflops:<10.1f} | [{M:>5}, {H//tp:>5}]       x [{H:>5}, {H//tp:>5}]")
        print(f"{tp:<4} | {'[Attn Total]':<12} | {total_attn:<10.3f} |")
        print(f"{tp:<4} | {'MLP GateUp':<12} | {gate_up_time:<10.3f} | {gu_tflops:<10.1f} | [{M:>5}, {H:>5}]       x [{2*FFN//tp:>5}, {H:>5}]")
        print(f"{tp:<4} | {'MLP Down':<12} | {down_time:<10.3f} | {d_tflops:<10.1f} | [{M:>5}, {FFN//tp:>5}]       x [{H:>5}, {FFN//tp:>5}]")
        print(f"{tp:<4} | {'[MLP Total]':<12} | {total_mlp:<10.3f} |")
        print(f"{tp:<4} | {'[LAYER REAL]':<12} | {simulated_layer_time:<10.3f} | {' '*10} | (Sequential Pipeline Simulation)")
        print("-" * 110)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--S', type=int, default=4096, help="Sequence Length")
    parser.add_argument('--B', type=int, default=1, help="Micro Batch Size")
    parser.add_argument('--H', type=int, default=4096, help="Hidden Size")
    parser.add_argument('--FFN', type=int, default=11008, help="FFN Intermediate Size")
    args = parser.parse_args()
    
    device_name = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"[*] Detected hardware device: {device_name.upper()}")
    if device_name != 'cuda':
        print("[!] 警告: 本地运行仅测试代码逻辑。欲观测真实的 TensorCore 非线性折损，请将此脚本提交至 A40/A100 GPU 集群运行！\n")
        
    profile_llama_layer(S=args.S, B=args.B, H=args.H, FFN=args.FFN, tp_sizes=[1, 2, 4, 8])
