"""Exercise 3.7.1 (solution): linear attention two ways, plus the gated delta rule.
(1) Parallel (training) form: O = (mask * (Q K^T)) V, causal, no softmax.
(2) Recurrent (inference) form: S_t = S_{t-1} + k_t v_t^T ;  o_t = S_t^T q_t.
They must agree. Then the gated delta rule S_t = a_t (I - b_t k_t k_t^T) S_{t-1} + b_t k_t v_t^T, and
state size vs a key-value (KV) cache.  numpy only, CPU, < 5 s.  Run: python ex_3_7_1_solution.py
"""
import numpy as np

def linear_attn_parallel(Q, K, V):
    """Q, K [T, d_k], V [T, d_v] -> O [T, d_v]. Cost ~ T^2 (like attention, but no softmax)."""
    T = Q.shape[0]
    A = np.tril(Q @ K.T)                        # causal mask: keep j <= t
    return A @ V

def linear_attn_recurrent(Q, K, V):
    """Same output with a fixed-size state S [d_k, d_v]. Cost ~ T * d_k * d_v; memory O(1) in T."""
    S = np.zeros((Q.shape[1], V.shape[1]))
    out = []
    for q, k, v in zip(Q, K, V):
        S = S + np.outer(k, v)
        out.append(S.T @ q)
    return np.array(out)

def gated_delta_recurrent(Q, K, V, alpha, beta):
    """Gated DeltaNet update. alpha_t in (0,1) decays old memory; beta_t in (0,1) is the write strength;
    (I - beta k k^T) first erases what the state currently stores under key k (keys L2-normalized)."""
    d_k, d_v = Q.shape[1], V.shape[1]
    S = np.zeros((d_k, d_v)); out = []
    for q, k, v, a, b in zip(Q, K, V, alpha, beta):
        S = a * (S - b * np.outer(k, k @ S)) + b * np.outer(k, v)
        out.append(S.T @ q)
    return np.array(out), S

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T, d_k, d_v = 64, 16, 16
    Q, K, V = (rng.standard_normal((T, n)) for n in (d_k, d_k, d_v))
    P, R = linear_attn_parallel(Q, K, V), linear_attn_recurrent(Q, K, V)
    print("parallel == recurrent:", np.allclose(P, R))
    assert np.allclose(P, R)

    # Delta rule as associative memory: write (k1 -> v1), then overwrite k1 with v2; read back with q = k1.
    k1 = np.zeros(d_k); k1[0] = 1.0
    v1, v2 = rng.standard_normal(d_v), rng.standard_normal(d_v)
    Kd = np.stack([k1, k1]); Vd = np.stack([v1, v2]); Qd = np.stack([k1, k1])
    lin = linear_attn_recurrent(Qd, Kd, Vd)[-1]
    dlt, _ = gated_delta_recurrent(Qd, Kd, Vd, alpha=np.ones(2), beta=np.ones(2))
    print(f"read-back error vs v2: plain linear {np.linalg.norm(lin - v2):.3f}, delta rule {np.linalg.norm(dlt[-1] - v2):.3f}")
    assert np.allclose(dlt[-1], v2)

    # State size vs KV cache for one layer with 16 heads of 128 dims, bf16 (2 bytes)
    h, dh, by = 16, 128, 2
    state = h * dh * dh * by
    for T_ctx in (8_192, 131_072, 1_048_576):
        kv = 2 * h * dh * T_ctx * by
        print(f"T={T_ctx:>9,}: KV {kv / 2**20:8.1f} MiB vs recurrent state {state / 2**20:.2f} MiB per layer per sequence")
    print("OK")
