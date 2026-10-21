"""Exercise 7.5.1 (starter): ring all-reduce in NumPy and an alpha-beta cost model for collectives."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import math
import numpy as np

def ring_allreduce(bufs):
    """Simulate ring all-reduce over n ranks: n-1 reduce-scatter steps, then n-1 all-gather steps.
    Returns the per-rank results and the bytes each rank sent."""
    n = len(bufs)
    chunks = [np.array_split(b.astype(np.float64).copy(), n) for b in bufs]
    sent = [0] * n
    for s in range(n - 1):
        pass   # TODO reduce-scatter: every rank r sends chunk (r - s) % n to rank r+1, which adds it in
    for s in range(n - 1):
        pass   # TODO all-gather: every rank r forwards its finished chunk (r + 1 - s) % n to rank r+1
    return [np.concatenate(c) for c in chunks], sent

def t_ring(S, n, alpha, beta):
    return 0.0   # TODO: 2(n-1) steps, each alpha + (S/n)/beta

def t_tree(S, n, alpha, beta):
    """Pipelined (double binary) tree: latency grows with log2 n; bandwidth term about 2 S / beta."""
    return 0.0   # TODO

def t_nvls(S, n, alpha, beta):
    """In-switch reduction: each GPU sends S once and receives the result once."""
    return 0.0   # TODO

def t_hier(S, g, nodes, a_in, b_in, a_out, b_out):
    """Reduce-scatter inside the node, all-reduce of 1/g of the data across nodes (one NIC per GPU,
    all g rails in parallel), all-gather inside the node."""
    rs = 0.0     # TODO: reduce-scatter time inside the node
    return 2 * rs + t_ring(S / g, nodes, a_out, b_out)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, size = 8, 1 << 20
    bufs = [rng.standard_normal(size).astype(np.float32) for _ in range(n)]
    out, sent = ring_allreduce(bufs)
    ref = np.sum([b.astype(np.float64) for b in bufs], axis=0)
    S = size * 8
    print(f"ring all-reduce, n={n}: max err {max(np.abs(o - ref).max() for o in out):.1e}; "
          f"bytes sent per rank / S = {sent[0] / S:.4f} (2(n-1)/n = {2*(n-1)/n:.4f})")

    # Assumed link parameters (labeled assumptions): NVLink 4 450 GB/s per direction, alpha 5 us per step;
    # InfiniBand NDR 400 Gb/s = 50 GB/s per GPU, alpha 10 us per step.
    NV = (5e-6, 450e9); IB = (10e-6, 50e9)
    G = 16e9                                     # Course-8B bf16 gradients
    print(f"\nCourse-8B gradient all-reduce, S = {G/1e9:.0f} GB:")
    print(f"  8 GPUs, NVLink ring:            {t_ring(G, 8, *NV)*1e3:7.1f} ms")
    print(f"  8 GPUs, NVLink SHARP (NVLS):    {t_nvls(G, 8, *NV)*1e3:7.1f} ms")
    print(f"  512 GPUs, flat ring over IB:    {t_ring(G, 512, *IB)*1e3:7.1f} ms")
    print(f"  512 GPUs, tree over IB:         {t_tree(G, 512, *IB)*1e3:7.1f} ms")
    h = t_hier(G, 8, 64, *NV, *IB)
    print(f"  64 nodes x 8, hierarchical:     {h*1e3:7.1f} ms  (inter-node part {t_ring(G/8, 64, *IB)*1e3:.1f} ms)")

    print("\nMessage size where ring and tree cost the same: S* = n*beta*alpha*(n-1-log2 n)")
    for name, nn, (a, b) in [("8 GPUs NVLink", 8, NV), ("64 GPUs IB", 64, IB), ("512 GPUs IB", 512, IB)]:
        Sx = nn * b * a * (nn - 1 - math.log2(nn))   # from setting t_ring = t_tree
        print(f"  {name:14s} S* = {Sx/1e6:10.1f} MB;  1 MB: ring {t_ring(1e6, nn, a, b)*1e6:8.0f} us, "
              f"tree {t_tree(1e6, nn, a, b)*1e6:6.0f} us")
