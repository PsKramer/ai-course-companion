import numpy as np
rng = np.random.default_rng(0)
d_in, d_out = 4096, 1024
W = rng.standard_normal((d_out, d_in)).astype(np.float32) * 0.02
W[:, rng.choice(d_in, 8, replace=False)] *= 25          # a few outlier input channels (common in LLM weights)
X = rng.standard_normal((64, d_in)).astype(np.float32)  # a batch of activations
Y = X @ W.T

def int4_quant(W, group):
    """Symmetric INT4 (levels -7..7) with one scale per `group` consecutive input weights; group=None = per output row."""
    # TODO: reshape into groups, scale = max|w|/7 per group, round and clip to -7..7, dequantize
    return np.zeros_like(W)

def int4_per_tensor(W):
    scale = np.abs(W).max() / 7
    return np.clip(np.round(W / scale), -7, 7) * scale

E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], np.float32)
def mxfp4_quant(W, block=32):
    """OCP MXFP4-style: FP4 E2M1 elements, one power-of-two (E8M0) scale per 32 elements."""
    # TODO: power-of-two scale per block so max|w|/scale <= 6, round each element to the nearest E2M1 value
    return np.zeros_like(W)

def to_e4m3(x):
    """Round positive scales to FP8 E4M3 (3 mantissa bits); range ignored for this toy."""
    e = np.floor(np.log2(x))
    return np.round(x / 2.0 ** e * 8) / 8 * 2.0 ** e

def nvfp4_quant(W, block=16):
    """NVFP4-style: FP4 E2M1 elements, one FP8 (E4M3) scale per 16 elements (per-tensor FP32 scale omitted)."""
    # TODO: scale = max|w|/6 per 16-element block rounded with to_e4m3, then round elements to E2M1
    return np.zeros_like(W)

def rel_err(Wq):
    return np.linalg.norm(X @ Wq.T - Y) / np.linalg.norm(Y)

rows = [("INT4 per-tensor", int4_per_tensor(W), 4 + 32 / W.size),
        ("INT4 per-channel", int4_quant(W, None), 4 + 16 / d_in),
        ("INT4 group 128", int4_quant(W, 128), 4 + 16 / 128),
        ("INT4 group 32", int4_quant(W, 32), 4 + 16 / 32),
        ("MXFP4 block 32", mxfp4_quant(W), 4 + 8 / 32),
        ("NVFP4 block 16", nvfp4_quant(W), 4 + 8 / 16)]
for name, Wq, bits in rows:
    print(f"{name:17s} bits/weight {bits:5.2f}   output rel. error {100 * rel_err(Wq):6.2f} %")
