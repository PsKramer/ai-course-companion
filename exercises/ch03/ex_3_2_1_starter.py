"""Exercise 3.2.1 (starter): causal grouped-query attention (GQA) in numpy.
Checks: GQA with h_kv = h equals multi-head attention (MHA); a loop reference matches;
causality holds; KV-cache bytes for MHA / GQA / MQA at Course-8B shapes.
numpy only, CPU, < 5 s.  Run: python ex_3_2_1_starter.py
"""
import numpy as np

def softmax(s, axis=-1):
    s = s - s.max(axis, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis, keepdims=True)

def gqa(x, W_Q, W_K, W_V, W_O, h, h_kv):
    """x [B, T, d]; W_Q [d, h*d_h]; W_K, W_V [d, h_kv*d_h]; W_O [h*d_h, d] -> [B, T, d]."""
    # TODO: project to q [B,h,T,d_h] and k,v [B,h_kv,T,d_h]; repeat k,v h//h_kv times on the head axis; scaled scores, causal mask (-inf above the diagonal), softmax, weighted sum of v, merge heads, apply W_O.
    raise NotImplementedError

def gqa_loop(x, W_Q, W_K, W_V, W_O, h, h_kv):
    """Slow reference: one query position and one head at a time."""
    B, T, d = x.shape
    d_h = W_Q.shape[1] // h
    out = np.zeros((B, T, h * d_h))
    for b in range(B):
        for i in range(h):
            j = i // (h // h_kv)
            Wq = W_Q[:, i * d_h:(i + 1) * d_h]; Wk = W_K[:, j * d_h:(j + 1) * d_h]; Wv = W_V[:, j * d_h:(j + 1) * d_h]
            for t in range(T):
                q = x[b, t] @ Wq
                K = x[b, :t + 1] @ Wk; Vv = x[b, :t + 1] @ Wv
                a = softmax(K @ q / np.sqrt(d_h))
                out[b, t, i * d_h:(i + 1) * d_h] = a @ Vv
    return out @ W_O

def kv_bytes_per_token(L, h_kv, d_h, bytes_per=2):
    # TODO: K and V, every layer: 2 * L * h_kv * d_h * bytes_per.
    raise NotImplementedError

if __name__ == "__main__":
    try:
        rng = np.random.default_rng(0)
        B, T, d, h, d_h = 2, 12, 64, 8, 8
        x = rng.standard_normal((B, T, d))
        W_Q = rng.standard_normal((d, h * d_h)) / np.sqrt(d)
        W_O = rng.standard_normal((h * d_h, d)) / np.sqrt(d)
        for h_kv in (8, 2, 1):
            W_K = rng.standard_normal((d, h_kv * d_h)) / np.sqrt(d)
            W_V = rng.standard_normal((d, h_kv * d_h)) / np.sqrt(d)
            y = gqa(x, W_Q, W_K, W_V, W_O, h, h_kv)
            assert np.allclose(y, gqa_loop(x, W_Q, W_K, W_V, W_O, h, h_kv), atol=1e-10)
            x2 = x.copy(); x2[:, -1] += 5.0                                 # change only the last token
            y2 = gqa(x2, W_Q, W_K, W_V, W_O, h, h_kv)
            assert np.allclose(y[:, :-1], y2[:, :-1]), "causality violated"
            print(f"h_kv={h_kv}: matches loop reference, causal")
        # GQA with h_kv = h is exactly MHA (no sharing); with h_kv = 1 it is multi-query attention (MQA)
        for name, hk in (("MHA", 32), ("GQA-8", 8), ("MQA", 1)):
            kb = kv_bytes_per_token(32, hk, 128)
            print(f"{name:6s} KV {kb / 1024:6.0f} KiB/token   {kb * 8192 / 2**30:5.2f} GiB per 8,192-token sequence")
        assert kv_bytes_per_token(32, 8, 128) == 128 * 1024
        print("OK")
    except NotImplementedError:
        print("Fill in the TODOs in gqa(), kv_bytes_per_token().")
