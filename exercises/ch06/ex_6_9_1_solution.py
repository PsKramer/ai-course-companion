"""Exercise 6.9.1: the throughput-interactivity frontier and cost per million tokens (roofline model)."""
N, L, d = 8.03e9, 32, 4096
S_KV = 131072                                  # bytes per token of KV (bf16)
GPUS = {  # name: (HBM bandwidth B/s, dense bf16 FLOP/s, HBM bytes, assumed USD per GPU-hour)
    "H100": (3.35e12, 989e12, 80e9, 2.50),
    "B200": (7.7e12, 2250e12, 180e9, 5.00),
}
T_CTX = 4096                                   # mean context during decode (tokens)

def decode_point(gpu, B, weight_bytes_per_param=2.0, kv_scale=1.0, mfu=0.5):
    bw, peak, hbm, price = GPUS[gpu]
    w = N * weight_bytes_per_param
    kv = B * T_CTX * S_KV * kv_scale
    if w + kv > 0.9 * hbm:
        return None                            # does not fit
    t = max((w + kv) / bw, 2 * N * B / (mfu * peak))
    tok_s = B / t
    usd_per_m = price / 3600 / tok_s * 1e6
    return 1000 * t, 1 / t, tok_s, usd_per_m   # TPOT ms, per-user tok/s, GPU tok/s, USD per M output tokens

def prefill_usd_per_m(gpu, mfu=0.5):
    bw, peak, hbm, price = GPUS[gpu]
    tok_s = mfu * peak / (2 * N)
    return price / 3600 / tok_s * 1e6

for gpu in GPUS:
    print(f"--- {gpu}  (assumed USD {GPUS[gpu][3]:.2f}/GPU-hour)  input (prefill) cost USD {prefill_usd_per_m(gpu):.3f} per M tokens")
    print("    B   TPOT ms  user tok/s  GPU tok/s  USD per M out")
    for B in [1, 8, 32, 64, 96, 128, 256]:
        r = decode_point(gpu, B)
        if r is None:
            print(f"  {B:4d}   does not fit in HBM"); continue
        print(f"  {B:4d}  {r[0]:7.2f}  {r[1]:10.0f}  {r[2]:9.0f}  {r[3]:12.3f}")

def cheapest(tpot_max_ms):
    best = None
    for gpu in GPUS:
        for B in range(1, 513):
            r = decode_point(gpu, B)
            if r and r[0] <= tpot_max_ms and (best is None or r[3] < best[2]):
                best = (gpu, B, r[3])
    return best

for slo in [8, 15, 30]:
    g, B, c = cheapest(slo)
    print(f"TPOT <= {slo:2d} ms: cheapest = {g} at B={B}, USD {c:.3f} per M output tokens")
