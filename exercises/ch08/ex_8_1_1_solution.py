"""Exercise 8.1.1 solution: best-of-n with a perfect verifier on Course-8B (context-extended), cost and latency."""
import math
N, KV_BYTES, BW, HBM = 8.03e9, 131_072, 3.35e12, 80e9   # params, KV bytes/token, H100 HBM bytes/s, HBM bytes
W_BYTES = 2 * N                                          # bf16 weights, 16.06 GB
USD_PER_GPU_HOUR = 2.50

def coverage(p, n):
    """Probability that at least one of n independent samples is correct."""
    return 1 - (1 - p) ** n

def min_n(p, target):
    return math.ceil(math.log(1 - target) / math.log(1 - p))

def decode_seconds(B, prompt, out):
    """Seconds to decode `out` tokens for B sequences in lockstep (memory-bound decode, Ch1 1.3 model)."""
    t = 0.0
    for i in range(out):
        t += (W_BYTES + B * (prompt + i) * KV_BYTES) / BW
    return t

def fits(B, prompt, out, reserve=8e9):
    return W_BYTES + B * (prompt + out) * KV_BYTES + reserve <= HBM

if __name__ == "__main__":
    p, prompt, out, target = 0.20, 1_000, 6_000, 0.90
    print(f"coverage n=1,4,16,64: " + ", ".join(f"{coverage(p, n):.3f}" for n in (1, 4, 16, 64)))
    n = min_n(p, target)
    print(f"min n for {target:.0%} coverage at p={p}: {n}  (coverage {coverage(p, n):.3f})")
    assert fits(n, prompt, out), "batch does not fit in HBM"
    t_par = decode_seconds(n, prompt, out)               # all n samples in one batch
    t_seq = n * decode_seconds(1, prompt, out)           # one after another
    gpu_h = t_par / 3600
    usd = gpu_h * USD_PER_GPU_HOUR
    print(f"parallel (B={n}): {t_par:.1f} s wall-clock;  sequential: {t_seq:.1f} s")
    print(f"tokens generated: {n * out:,};  GPU cost of the parallel run: USD {usd:.3f}")
    print(f"expected USD per solved problem: USD {usd / coverage(p, n):.3f}")
    one = decode_seconds(1, prompt, out) / 3600 * USD_PER_GPU_HOUR
    print(f"single sample: USD {one:.4f}, solves {p:.0%} -> USD {one / p:.4f} per solve (but only {p:.0%} of problems)")
