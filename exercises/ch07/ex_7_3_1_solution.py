"""Exercise 7.3.1 (solution): GEMM tiling, L2 reuse and HBM traffic on an H100."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np

SMS, PEAK, BW, b = 132, 989e12, 3.35e12, 2      # H100, bf16

def tiled_matmul(A, B, BM=64, BN=64, BK=32):
    """Reference tiled matmul: each output tile accumulates over K in BK-wide slices."""
    M, K = A.shape; N = B.shape[1]
    C = np.zeros((M, N), np.float32)
    for i in range(0, M, BM):
        for j in range(0, N, BN):
            acc = np.zeros((min(BM, M - i), min(BN, N - j)), np.float32)
            for k in range(0, K, BK):
                acc += A[i:i+BM, k:k+BK] @ B[k:k+BK, j:j+BN]
            C[i:i+BM, j:j+BN] = acc
    return C

def traffic_no_reuse(M, N, K, BM, BN):
    """Every tile streams its A strip (BM x K) and B strip (K x BN) from HBM."""
    tiles = (M // BM) * (N // BN)
    return tiles * (BM + BN) * K * b + M * N * b

def traffic_waves(M, N, K, BM, BN, order):
    """CTAs run in waves of SMS tiles; within a wave each distinct A/B strip is read from HBM once
    (L2 serves the other tiles that share it). order = 'row' or ('group', G)."""
    TM, TN = M // BM, N // BN
    if order == "row":
        seq = [(i, j) for i in range(TM) for j in range(TN)]
    else:
        G = order[1]
        seq = [(i, j) for i0 in range(0, TM, G) for j in range(TN) for i in range(i0, min(i0 + G, TM))]
    total = 0
    for w in range(0, len(seq), SMS):
        wave = seq[w:w + SMS]
        rows = {i for i, _ in wave}; cols = {j for _, j in wave}
        total += (len(rows) * BM + len(cols) * BN) * K * b
    return total + M * N * b

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    A = rng.standard_normal((256, 512)).astype(np.float32)
    B = rng.standard_normal((512, 384)).astype(np.float32)
    err = np.abs(tiled_matmul(A, B) - A @ B).max()
    print(f"tiled matmul max abs err vs numpy: {err:.2e}")

    M = N = K = 8192; BM, BN = 128, 256
    flops = 2 * M * N * K
    ideal = (M * K + K * N + M * N) * b
    print(f"\nGEMM {M}x{K}x{N} bf16: {flops/1e12:.2f} TFLOP, compute floor {flops/PEAK*1e3:.2f} ms")
    for name, q in [("ideal (each operand once)", ideal),
                    ("no reuse across tiles", traffic_no_reuse(M, N, K, BM, BN)),
                    ("row-major waves + L2", traffic_waves(M, N, K, BM, BN, "row")),
                    ("grouped waves (G=12) + L2", traffic_waves(M, N, K, BM, BN, ("group", 12)))]:
        I = flops / q
        t = max(flops / PEAK, q / BW)
        print(f"{name:28s} HBM {q/1e9:6.2f} GB  I = {I:7.0f} FLOP/B  roofline {t*1e3:5.2f} ms  "
              f"{'compute' if I > PEAK/BW else 'memory'}-bound")
    tiles = (M // BM) * (N // BN)
    waves = tiles / SMS
    print(f"\n{tiles} tiles on {SMS} SMs = {waves:.2f} waves; last wave {waves%1:.0%} full "
          f"-> {1 - waves/np.ceil(waves):.1%} of SM-time idle")
