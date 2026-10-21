"""Exercise 4.5.1 (solution): a first-order parallelism planner for dense transformers.
Per-GPU memory, per-GPU communication bytes per step, pipeline bubble, and rough times,
for a (DP, ZeRO/FSDP mode, TP, PP) layout. All formulas are the ones derived in 4.4-4.5."""
from dataclasses import dataclass

@dataclass
class Model:            # Course-8B (defined in 1.1)
    N: float = 8.03e9; L: int = 32; d: int = 4096; h_kv: int = 8; d_h: int = 128; d_ff: int = 14336; T: int = 8192

NVLINK = 450e9   # bytes/s per direction per H100 (900 GB/s bidirectional), intra-node
IB = 50e9        # bytes/s per GPU (one 400 Gb/s NIC per GPU), inter-node
PEAK, MFU = 989e12, 0.40

def act_bytes_per_token_layer(m, recompute):
    if recompute == "full": return 2 * m.d
    return 2 * (6 * m.d + 2 * m.h_kv * m.d_h + 3 * m.d_ff)          # from 4.4

def ring(bytes_, n):                      # bytes each rank sends in a ring all-reduce (4.5)
    return 2 * (n - 1) / n * bytes_

def plan(m, n_gpu, tp, pp, mode, global_seqs=512, mb=1, recompute="none", shard=8):
    dp = n_gpu // (tp * pp); assert dp * tp * pp == n_gpu
    P = m.N / (tp * pp)                                               # params held per model-parallel rank
    n_micro = global_seqs // (dp * mb)                                # micro-batches per DP rank per step
    # ---- memory (bytes) ----
    if mode == "ddp":   state = 16 * P
    elif mode == "zero1": state = 4 * P + 12 * P / dp
    elif mode == "zero2": state = 2 * P + 14 * P / dp
    elif mode == "fsdp":  state = 16 * P / dp
    elif mode == "hsdp":  state = 16 * P / shard                     # shard inside a node, replicate across nodes
    in_flight = min(pp, n_micro) if pp > 1 else 1                     # 1F1B: first stage holds pp micro-batches
    acts = act_bytes_per_token_layer(m, recompute) / tp * (m.L / pp) * m.T * mb * in_flight
    # ---- communication per GPU per step (bytes sent) ----
    g = 2 * P                                                         # bf16 gradient (or weight) bytes per rank
    if mode in ("ddp", "zero1", "zero2"): dp_intra, dp_inter = 0, ring(g, dp)
    elif mode == "fsdp": dp_intra, dp_inter = 0, 1.5 * ring(g, dp)   # all-gather x2 + reduce-scatter = 3 x (n-1)/n x g
    elif mode == "hsdp":
        dp_intra = 1.5 * ring(g, shard)                               # FSDP inside the node over NVLink
        dp_inter = ring(g / shard, dp // shard)                       # all-reduce of each 1/shard across nodes
    tp_bytes = (m.L / pp) * n_micro * 4 * ring(mb * m.T * m.d * 2, tp) if tp > 1 else 0
    pp_bytes = n_micro * 2 * (mb * m.T * m.d * 2 / tp) if pp > 1 else 0
    bubble = (pp - 1) / (n_micro + pp - 1)
    t_compute = 6 * m.N * global_seqs * m.T / (n_gpu * PEAK * MFU)
    t_intra = (dp_intra + tp_bytes) / NVLINK                          # assumes TP fits inside one 8-GPU node
    t_inter = (dp_inter + pp_bytes) / IB
    return dict(dp=dp, n_micro=n_micro, state_GB=state / 1e9, acts_GB=acts / 1e9, total_GB=(state + acts) / 1e9,
                nvlink_GB=(dp_intra + tp_bytes) / 1e9, ib_GB=(dp_inter + pp_bytes) / 1e9,
                bubble=bubble, t_compute=t_compute, t_nvlink=t_intra, t_ib=t_inter)

if __name__ == "__main__":
    m = Model()
    configs = [("A  DDP, no sharding",          dict(tp=1, pp=1, mode="ddp")),
               ("B  FSDP over all 512",          dict(tp=1, pp=1, mode="fsdp")),
               ("C  HSDP (shard 8, replicate 64)", dict(tp=1, pp=1, mode="hsdp")),
               ("D  TP=8 in node + ZeRO-1",      dict(tp=8, pp=1, mode="zero1")),
               ("E  TP=2, PP=4, ZeRO-1",         dict(tp=2, pp=4, mode="zero1"))]
    print(f"{'layout':34s} {'DP':>4s} {'micro':>5s} {'state':>7s} {'acts':>7s} {'total':>7s} {'NVLink':>8s} {'IB':>7s} {'bubble':>6s} {'t_comp':>6s} {'t_NVL':>6s} {'t_IB':>6s}")
    for name, kw in configs:
        r = plan(m, 512, **kw)
        print(f"{name:34s} {r['dp']:4d} {r['n_micro']:5d} {r['state_GB']:6.1f}G {r['acts_GB']:6.1f}G {r['total_GB']:6.1f}G "
              f"{r['nvlink_GB']:7.1f}G {r['ib_GB']:6.1f}G {r['bubble']:6.1%} {r['t_compute']:5.2f}s {r['t_nvlink']:5.2f}s {r['t_ib']:5.2f}s")
    r = plan(m, 512, 1, 1, "hsdp", recompute="full")
    print(f"C with full recompute: total {r['total_GB']:.1f} GB per GPU (compute grows ~33%, see 4.4)")
