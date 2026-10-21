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
    # TODO: return None if weights + KV exceed 90% of HBM
    # TODO: step time t = max(bytes / bw, 2*N*B / (mfu*peak))
    # TODO: return (TPOT ms, per-user tok/s, GPU tok/s, USD per million output tokens)
    return None

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
    # TODO: search GPUs and B = 1..512 for the lowest USD per million meeting the TPOT target
    return ("none", 0, float("nan"))

for slo in [8, 15, 30]:
    g, B, c = cheapest(slo)
    print(f"TPOT <= {slo:2d} ms: cheapest = {g} at B={B}, USD {c:.3f} per M output tokens")
