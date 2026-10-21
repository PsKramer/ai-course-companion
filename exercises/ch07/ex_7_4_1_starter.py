"""Exercise 7.4.1 (starter): emulate FP8/FP4 formats, block scaling and limited-precision accumulation."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np

# (mantissa bits, minimum normal exponent, largest finite value)
FORMATS = {"BF16": (7, -126, 3.39e38), "E4M3": (3, -6, 448.0), "E5M2": (2, -14, 57344.0), "E2M1": (1, 0, 6.0)}

def quantize(x, fmt):
    """Round to nearest representable value of a small float format (with subnormals), saturating at max."""
    m, emin, vmax = FORMATS[fmt]
    # TODO: clip to +-vmax; exponent e = floor(log2|x|) but not below emin; spacing q = 2^(e-m)
    q = 1.0
    return np.clip(np.round(x / q) * q, -vmax, vmax)

def per_tensor_fp8(x):
    s = np.abs(x).max() / 448.0                # one fp32 scale for the whole tensor
    return quantize(x / s, "E4M3") * s

def mxfp4(x, block=32):
    """OCP MX: blocks of 32 E2M1 values sharing a power-of-two (E8M0) scale 2^(floor(log2 amax) - 2)."""
    xb = x.reshape(-1, block)
    amax = np.abs(xb).max(1, keepdims=True)
    s = np.ones_like(amax)                     # TODO: power-of-two scale 2^(floor(log2 amax) - 2)
    return (quantize(xb / s, "E2M1") * s).reshape(x.shape)

def nvfp4(x, block=16):
    """NVFP4: blocks of 16 E2M1 values, an E4M3 scale per block, and one fp32 scale per tensor."""
    xb = x.reshape(-1, block)
    s_t = np.abs(x).max() / (448.0 * 6.0)      # tensor scale keeps block scales inside E4M3 range
    amax = np.abs(xb).max(1, keepdims=True)
    s_b = amax / 6.0                           # TODO: round the block scale (in units of s_t) to E4M3
    s_b = np.where(s_b == 0, 1.0, s_b)
    return (quantize(xb / s_b, "E2M1") * s_b).reshape(x.shape)

def trunc_bits(x, m):
    """Keep m mantissa bits after the leading one, rounding toward zero (a narrow accumulator)."""
    e = np.floor(np.log2(np.maximum(np.abs(x), 1e-300)))
    q = 2.0 ** (e - m)
    return np.trunc(x / q) * q

def accumulate(prod, m_acc=13, group=4, promote=None):
    """Sum products along axis 1: groups of `group` are added exactly, then into an accumulator with
    m_acc mantissa bits (13 -> about 14 significant bits). If promote=N, the narrow accumulator is
    flushed into an fp64 total every N products (DeepSeek-V3-style promotion)."""
    T, K = prod.shape
    g = prod.reshape(T, K // group, group).sum(2)
    total = np.zeros(T); acc = np.zeros(T)
    per = (promote or K) // group
    for i in range(g.shape[1]):
        acc = acc + g[:, i]                    # TODO: truncate acc to m_acc bits; flush into total every `per` groups
    return total + acc

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("E2M1 magnitudes:", sorted(set(np.abs(quantize(np.linspace(-7, 7, 1401), "E2M1")).tolist())))
    x = rng.standard_normal(1 << 16)
    idx = rng.choice(x.size, 64, replace=False); x[idx] *= 20      # 0.1% outliers at ~20 sigma
    print("\nRelative RMS error on N(0,1) data with 0.1% outliers (x20):")
    for name, f, bits in [("bf16 (E8M7)", lambda v: quantize(v, "BF16"), 16),
                          ("FP8 E4M3, per-tensor scale", per_tensor_fp8, 8),
                          ("MXFP4 (block 32, E8M0)", mxfp4, 4.25),
                          ("NVFP4 (block 16, E4M3)", nvfp4, 4.5)]:
        err = np.sqrt(np.mean((f(x) - x) ** 2) / np.mean(x ** 2))
        print(f"  {name:28s} {bits:5} bits/value   rel RMS error {err:.4f}")

    print("\nDot products of uniform [0,1) vectors, relative error vs exact (200 trials):")
    print("     K   narrow acc (mean / max)    promote every 128 (mean)")
    for K in [512, 1024, 2048, 4096, 8192, 16384]:
        a = rng.random((200, K)); b = rng.random((200, K))
        prod = a * b; exact = prod.sum(1)
        e1 = np.abs(accumulate(prod) - exact) / exact
        e2 = np.abs(accumulate(prod, promote=128) - exact) / exact
        print(f"{K:6d}   {e1.mean():.4f} / {e1.max():.4f}           {e2.mean():.5f}")
