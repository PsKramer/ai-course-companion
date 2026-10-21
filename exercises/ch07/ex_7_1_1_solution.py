"""Exercise 7.1.1 (solution): roofline classifier for one Course-8B layer on an H100."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np

PEAK = {"bf16": 989e12, "fp8": 1979e12}      # dense tensor-core FLOP/s, H100 SXM
BW = 3.35e12                                   # HBM bytes/s
d, d_ff, h, h_kv, d_h = 4096, 14336, 32, 8, 128

def gemm(M, K, N, b):
    """FLOPs and HBM bytes of an (M x K) @ (K x N) matmul, each operand read/written once."""
    return 2 * M * K * N, b * (M * K + K * N + M * N)

def rmsnorm(M, b):
    return 4 * M * d, 2 * b * M * d             # ~4 FLOPs per element; read x, write y

def attn_decode(B, T, b):
    flops = B * h * T * 4 * d_h                 # q.k and p.v for every query head
    byts = B * T * 2 * h_kv * d_h * b           # read K and V once per sequence
    return flops, byts

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
    I = flops / byts
    t_c, t_m = flops / peak, byts / BW
    return I, ("compute" if t_c > t_m else "memory"), max(t_c, t_m)

def report(label, M, T_ctx, decode, dtype):
    b = 2 if dtype == "bf16" else 1
    peak = PEAK[dtype]
    ridge = peak / BW
    print(f"\n{label}  [{dtype}, ridge {ridge:.0f} FLOP/byte]")
    total = 0.0
    for name, (f, by) in ops(M, T_ctx, decode, b).items():
        I, bound, t = classify(name, f, by, peak)
        total += t
        print(f"  {name:12s} I = {I:8.1f} FLOP/B  {bound:7s}  {t*1e6:9.1f} us")
    print(f"  layer total {total*1e6:9.1f} us   x32 layers = {total*32*1e3:7.2f} ms")
    return total

if __name__ == "__main__":
    report("Decode, B=1, T=4096", 1, 4096, True, "bf16")
    report("Decode, B=64, T=4096", 64, 4096, True, "bf16")
    report("Prefill / training micro-batch, M=8192 tokens", 8192, 0, False, "bf16")
    report("Decode, B=64, T=4096", 64, 4096, True, "fp8")
    # break-even batch where the FFN up/gate GEMM crosses the ridge (bf16)
    for B in range(1, 2048):
        f, by = gemm(B, d, 2 * d_ff, 2)
        if f / by >= PEAK["bf16"] / BW:
            print(f"\nffn_up_gate turns compute-bound at B = {B} tokens (bf16)")
            break
