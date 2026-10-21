"""Exercise 7.8.1 (solution): cost per million tokens for Course-8B decode across chips and formats."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

N, L, D = 8.03e9, 32, 4096                 # Course-8B parameters, layers, width
KV_BF16 = 131072                           # bytes of KV cache per token in bf16 (1.3)
T = 4096                                   # context length per sequence
EFF = 0.7                                  # assumption: kernels reach 70% of the roofline time bound

CHIPS = {   # peak dense FLOP/s per format, HBM B/s, assumed rental price in USD per GPU-hour
    "H100": {"peak": {"bf16": 989e12, "fp8": 1979e12}, "bw": 3.35e12, "usd_h": 2.00, "mem": 80e9},
    "B200": {"peak": {"bf16": 2250e12, "fp8": 4500e12, "nvfp4": 9000e12}, "bw": 7.7e12, "usd_h": 3.50, "mem": 180e9},
}
BITS = {"bf16": 16, "fp8": 8, "nvfp4": 4.5}  # weight bits per value (NVFP4 includes block scales)

def memory(fmt, B):
    """Bytes for weights plus the KV cache of B sequences of length T."""
    return N * BITS[fmt] / 8 + B * T * (KV_BF16 if fmt == "bf16" else KV_BF16 / 2)

def step_time(chip, fmt, B):
    """Time of one decode step for B sequences: max(memory, compute) / EFF.
    Weights in `fmt`; KV cache in bf16, or FP8 when weights are FP8 or NVFP4."""
    c = CHIPS[chip]
    w_bytes = N * BITS[fmt] / 8
    kv_bytes = B * T * (KV_BF16 if fmt == "bf16" else KV_BF16 / 2)
    flops = 2 * N * B + 4 * B * T * D * L     # weight matmuls + attention (QK^T and PV)
    return max((w_bytes + kv_bytes) / c["bw"], flops / c["peak"][fmt]) / EFF

def usd_per_mtok(chip, fmt, B):
    tps = B / step_time(chip, fmt, B)
    return CHIPS[chip]["usd_h"] / (tps * 3600) * 1e6, tps

def best_batch(chip, fmt, slo_ms):
    """Largest batch whose step time meets the per-token latency target and whose weights + KV fit."""
    best = None
    for B in range(1, 1025):
        if step_time(chip, fmt, B) * 1e3 <= slo_ms and memory(fmt, B) <= CHIPS[chip]["mem"]:
            best = B
    return best

if __name__ == "__main__":
    print("USD per million output tokens (tokens/s per GPU), T = 4,096, 70% of roofline:")
    print("  chip/format      B=1              B=16             B=64             B=256")
    for chip, fmt in [("H100", "bf16"), ("H100", "fp8"), ("B200", "fp8"), ("B200", "nvfp4")]:
        row = f"  {chip} {fmt:6s}"
        for B in (1, 16, 64, 256):
            if memory(fmt, B) > CHIPS[chip]["mem"]:
                row += "   does not fit   "; continue
            usd, tps = usd_per_mtok(chip, fmt, B)
            row += f"   {usd:6.3f} ({tps:6,.0f})"
        print(row)
    print("\nLargest batch that fits in memory and meets a 30 ms per-token target:")
    for chip, fmt in [("H100", "bf16"), ("H100", "fp8"), ("B200", "fp8"), ("B200", "nvfp4")]:
        B = best_batch(chip, fmt, 30)
        usd, tps = usd_per_mtok(chip, fmt, B)
        print(f"  {chip} {fmt:6s} B = {B:4d}: {tps:7,.0f} tokens/s, USD {usd:.3f} per million tokens")
