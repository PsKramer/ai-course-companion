"""Exercise 8.5.1 solution: train a tiny sparse autoencoder (SAE) and check it recovers planted features."""
import os; os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
rng = np.random.default_rng(0)
d, F_true, k_true, n = 64, 256, 3, 40_000          # residual width, planted features, active per sample, samples
D_true = rng.normal(size=(F_true, d)); D_true /= np.linalg.norm(D_true, axis=1, keepdims=True)
idx = np.array([rng.choice(F_true, k_true, replace=False) for _ in range(n)])
coef = rng.uniform(1.0, 3.0, size=(n, k_true))
X = np.zeros((n, d))
for j in range(k_true):
    X += coef[:, j:j+1] * D_true[idx[:, j]]          # x = sum of k sparse feature directions

import sys
F, lam, lr, steps, bs = 512, float(sys.argv[1]) if len(sys.argv) > 1 else 0.3, 2e-3, 6000, 512  # dictionary size, L1 weight, Adam lr, steps, batch
W_enc = rng.normal(scale=0.1, size=(d, F)); b_enc = np.zeros(F)
W_dec = W_enc.T.copy(); W_dec /= np.linalg.norm(W_dec, axis=1, keepdims=True); b_dec = np.zeros(d)
params = [W_enc, b_enc, W_dec, b_dec]; m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]

def sae(x):
    pre = (x - b_dec) @ W_enc + b_enc
    f = np.maximum(pre, 0.0)                          # sparse feature activations [bs, F]
    return pre, f, f @ W_dec + b_dec                  # reconstruction x_hat [bs, d]

for t in range(1, steps + 1):
    x = X[rng.integers(0, n, bs)]
    pre, f, xh = sae(x)
    err = xh - x
    g_xh = 2 * err / bs                               # d(mean squared error)/d x_hat
    g_Wdec = f.T @ g_xh; g_bdec = g_xh.sum(0)
    g_f = g_xh @ W_dec.T + lam / bs                   # L1 penalty: f >= 0, so d|f|/df = 1 where active
    g_pre = g_f * (pre > 0)                           # ReLU gate
    g_Wenc = (x - b_dec).T @ g_pre; g_benc = g_pre.sum(0)
    g_bdec -= (g_pre @ W_enc.T).sum(0)                # b_dec also enters through the encoder input
    grads = [g_Wenc, g_benc, g_Wdec, g_bdec]
    for i, (p, g) in enumerate(zip(params, grads)):
        m[i] = 0.9 * m[i] + 0.1 * g; v[i] = 0.999 * v[i] + 0.001 * g * g
        p -= lr * (m[i] / (1 - 0.9**t)) / (np.sqrt(v[i] / (1 - 0.999**t)) + 1e-8)
    W_dec /= np.linalg.norm(W_dec, axis=1, keepdims=True)  # unit-norm decoder rows

_, f, xh = sae(X[:5000])
fvu = ((xh - X[:5000])**2).sum() / ((X[:5000] - X[:5000].mean(0))**2).sum()
l0 = (f > 1e-6).sum(1).mean()
cos = np.abs(D_true @ W_dec.T)                        # [F_true, F] cosine (both unit-norm)
recovered = (cos.max(1) > 0.9).mean()
print(f"fraction of variance unexplained: {fvu:.3f}")
print(f"mean active features per sample (L0): {l0:.1f}  (planted: {k_true})")
print(f"planted features recovered (max cosine > 0.9): {recovered:.1%}")
