"""Exercise 7.1.1 (starter): roofline classifier for one Course-8B layer on an H100."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np

PEAK = {"bf16": 989e12, "fp8": 1979e12}      # dense tensor-core FLOP/s, H100 SXM
BW = 3.35e12                                   # HBM bytes/s
d, d_ff, h, h_kv, d_h = 4096, 14336, 32, 8, 128

def gemm(M, K, N, b):
    """FLOPs and HBM bytes of an (M x K) @ (K x N) matmul, each operand read/written once."""
    # TODO: return (flops, bytes)
    return 1, 1

def rmsnorm(M, b):
    return 4 * M * d, 2 * b * M * d             # ~4 FLOPs per element; read x, write y

def attn_decode(B, T, b):
    # TODO: FLOPs = 4 * d_h per (query head, cached token); bytes = K and V of every cached token
    return 1, 1

def ops(M, T_ctx, decode, b):
    out = {"rmsnorm": rmsnorm(M, b),
           "qkv_proj": gemm(M, d, d + 2 * h_kv * d_h, b),
           "o_proj": gemm(M, d, d, b),
           "ffn_up_gate": gemm(M, d, 2 * d_ff, b),
           "ffn_down": gemm(M, d_ff, d, b)}
    if decode:
        out["attn"] = attn_decode(M, T_ctx, b)
    return out

def classify(name, flops, byts, peak):
    # TODO: return (intensity, "compute" or "memory", roofline time in seconds)
    return 0.0, "?", 0.0

def report(label, M, T_ctx, decode, dtype):
    b = 2 if dtype == "bf16" else 1
    peak = PEAK[dtype]
    print(f"\n{label}  [{dtype}, ridge {peak / BW:.0f} FLOP/byte]")
    total = 0.0
    for name, (f, by) in ops(M, T_ctx, decode, b).items():
        I, bound, t = classify(name, f, by, peak)
        total += t
        print(f"  {name:12s} I = {I:8.1f} FLOP/B  {bound:7s}  {t*1e6:9.1f} us")
    print(f"  layer total {total*1e6:9.1f} us   x32 layers = {total*32*1e3:7.2f} ms")

if __name__ == "__main__":
    report("Decode, B=1, T=4096", 1, 4096, True, "bf16")
    report("Decode, B=64, T=4096", 64, 4096, True, "bf16")
    report("Prefill / training micro-batch, M=8192 tokens", 8192, 0, False, "bf16")
    report("Decode, B=64, T=4096", 64, 4096, True, "fp8")
