import numpy as np
rng = np.random.default_rng(0)
g, d_h, T = 4, 128, 4096          # 4 query heads share one KV head (Course-8B GQA group)
q = rng.standard_normal((g, d_h)).astype(np.float32)
K = rng.standard_normal((T, d_h)).astype(np.float32)
V = rng.standard_normal((T, d_h)).astype(np.float32)
scale = 1.0 / np.sqrt(d_h)

def reference(q, K, V):
    s = (q @ K.T) * scale                       # [g, T]
    p = np.exp(s - s.max(axis=1, keepdims=True))
    return (p @ V) / p.sum(axis=1, keepdims=True)

def online_softmax(q, K, V, tile=128):
    """One pass over KV tiles keeping running max m, denominator l, unnormalized output o."""
    m = np.full((q.shape[0], 1), -np.inf, np.float32)
    l = np.zeros((q.shape[0], 1), np.float32)
    o = np.zeros((q.shape[0], V.shape[1]), np.float32)
    for t0 in range(0, K.shape[0], tile):
        s = (q @ K[t0:t0 + tile].T) * scale
        m_new = np.maximum(m, s.max(axis=1, keepdims=True))
        alpha = np.exp(m - m_new)                # rescale old partial results
        p = np.exp(s - m_new)
        l = alpha * l + p.sum(axis=1, keepdims=True)
        o = alpha * o + p @ V[t0:t0 + tile]
        m = m_new
    return o, m, l                               # final output = o / l

def split_kv(q, K, V, n_splits=16):
    """Flash-Decoding: each split runs online_softmax independently, then combine via log-sum-exp."""
    parts = [online_softmax(q, Ks, Vs) for Ks, Vs in zip(np.array_split(K, n_splits), np.array_split(V, n_splits))]
    M = np.max(np.stack([m for _, m, _ in parts]), axis=0)
    num = sum(np.exp(m - M) * o for o, m, _ in parts)
    den = sum(np.exp(m - M) * l for _, m, l in parts)
    return num / den

ref = reference(q, K, V)
o, m, l = online_softmax(q, K, V)
print(f"online softmax max abs err  {np.abs(o / l - ref).max():.2e}")
print(f"split-KV (16) max abs err   {np.abs(split_kv(q, K, V) - ref).max():.2e}")
# bytes vs FLOPs for this KV head (bf16 storage)
kv_bytes = 2 * T * d_h * 2
flops = g * T * (2 * d_h + 2 * d_h)
print(f"arithmetic intensity        {flops / kv_bytes:.1f} FLOP/byte (= g)")
