"""Exercise 7.7.1 (solution): a cycle-level weight-stationary systolic array."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np

def systolic_ws(X, W):
    """Weight-stationary P x P array computing Y = X @ W for X (M x P), W (P x P).
    PE (k, n) holds W[k, n]. Activations enter row k from the left, delayed by k cycles, and move
    one PE right per cycle; partial sums move one PE down per cycle and leave the bottom of column n.
    Returns Y, the number of cycles, and the average fraction of PEs doing a useful MAC."""
    M, P = X.shape
    a = np.zeros((P, P)); p = np.zeros((P, P))           # registers: activation, partial sum
    Y = np.zeros((M, P)); busy = 0; t = 0
    done = 0
    while done < M * P:
        a_new = np.zeros((P, P)); valid = np.zeros((P, P), bool)
        a_new[:, 1:] = a[:, :-1]                          # activations shift right
        for k in range(P):                                # skewed input at the left edge
            m = t - k
            a_new[k, 0] = X[m, k] if 0 <= m < M else 0.0
        for k in range(P):                                # which PEs hold a real activation
            for n in range(P):
                valid[k, n] = 0 <= t - k - n < M
        p_new = np.zeros((P, P))
        p_new[0, :] = a_new[0, :] * W[0, :]
        p_new[1:, :] = p[:-1, :] + a_new[1:, :] * W[1:, :]  # partial sums shift down and accumulate
        busy += valid.sum()
        for n in range(P):                                # bottom row emits finished outputs
            m = t - (P - 1) - n
            if 0 <= m < M:
                Y[m, n] = p_new[P - 1, n]; done += 1
        a, p = a_new, p_new
        t += 1
    return Y, t, busy / (t * P * P)

def tiled_cycles(M, K, N, P, load_overlap=True):
    """Cycles for an M x K by K x N matmul on one P x P array: one pass of M + 2P - 2 cycles per
    weight tile, plus P cycles to load each tile unless loading is double-buffered."""
    tiles = int(np.ceil(K / P) * np.ceil(N / P))
    per = M + 2 * P - 2 + (0 if load_overlap else P)
    cyc = tiles * per
    return cyc, (M * K * N) / (cyc * P * P)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for M, P in [(6, 4), (32, 8)]:
        X = rng.standard_normal((M, P)); W = rng.standard_normal((P, P))
        Y, cyc, util = systolic_ws(X, W)
        print(f"P={P}, M={M}: max err {np.abs(Y - X @ W).max():.1e}, cycles {cyc} "
              f"(M + 2P - 2 = {M + 2*P - 2}), utilization {util:.1%} (M/(M+2P-2) = {M/(M+2*P-2):.1%})")

    print("\nCourse-8B FFN up+gate (K = 4,096, N = 28,672) on one array, weights double-buffered:")
    print("    M tokens    P=128 util   P=256 util   time at 940 MHz, P=256")
    for M in (1, 64, 512, 4096):
        _, u128 = tiled_cycles(M, 4096, 28672, 128)
        c256, u256 = tiled_cycles(M, 4096, 28672, 256)
        print(f"   {M:8d}     {u128:7.1%}      {u256:7.1%}      {c256/940e6*1e3:8.3f} ms")
