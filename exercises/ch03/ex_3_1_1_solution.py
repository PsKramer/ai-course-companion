"""Exercise 3.1.1 (solution): RMSNorm, a pre-norm residual block, and Course-8B's parameter budget.
numpy only, CPU, < 5 s.  Run: python ex_3_1_1_solution.py
"""
import numpy as np

def rmsnorm(x, g, eps=1e-6):
    """x [..., d] -> x / sqrt(mean(x^2) + eps) * g   (no mean subtraction, no bias)."""
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * g

def layernorm(x, g, b, eps=1e-6):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b

def course8b_params(L=32, d=4096, h=32, h_kv=8, d_ff=14336, V=128256):
    d_h = d // h
    attn = 2 * d * d + 2 * d * h_kv * d_h          # W_Q, W_O (d x d) + W_K, W_V (d x h_kv*d_h)
    ffn = 3 * d * d_ff                              # W_gate, W_up, W_down (SwiGLU)
    norms = 2 * d                                   # two RMSNorm gains per block
    emb = 2 * V * d                                 # untied input embedding + output unembedding
    total = L * (attn + ffn + norms) + emb + d      # + final RMSNorm
    return dict(attn=L * attn, ffn=L * ffn, norms=L * norms + d, emb=emb, total=total)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d = 4096
    x = rng.standard_normal((4, d)).astype(np.float32) * 3.0 + 0.5
    g = np.ones(d, np.float32)
    y = rmsnorm(x, g)
    print("RMS after RMSNorm:", np.sqrt((y * y).mean(-1)).round(4))
    print("scale-invariant:", np.allclose(rmsnorm(10 * x, g), y, atol=1e-5))
    # LayerNorm also removes the mean; RMSNorm keeps it (cheaper: one reduction instead of two)
    print("mean after RMSNorm vs LayerNorm:", y.mean(-1).round(3)[:2], layernorm(x, g, 0 * g).mean(-1).round(3)[:2])

    # Pre-norm residual stream: x_{l+1} = x_l + F(norm(x_l)). The stream's RMS grows with depth.
    L = 32
    xs = rng.standard_normal((64, 256)).astype(np.float32)
    W = [rng.standard_normal((256, 256)).astype(np.float32) / np.sqrt(256) for _ in range(L)]
    rms = []
    for l in range(L):
        xs = xs + np.tanh(rmsnorm(xs, np.ones(256, np.float32)) @ W[l])   # toy sublayer F
        rms.append(float(np.sqrt((xs * xs).mean())))
    print("residual RMS at layers 1, 8, 16, 32:", [round(rms[i], 2) for i in (0, 7, 15, 31)])

    p = course8b_params()
    for k in ("attn", "ffn", "norms", "emb"):
        print(f"{k:6s} {p[k] / 1e9:6.3f} B params  {100 * p[k] / p['total']:5.1f}%")
    print(f"total  {p['total'] / 1e9:6.3f} B params")
    assert abs(p["total"] / 1e9 - 8.03) < 0.01
    print("OK")
