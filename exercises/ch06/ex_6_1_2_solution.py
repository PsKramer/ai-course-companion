N, L, d, h_kv, d_h = 8.03e9, 32, 4096, 8, 128
BW, PEAK, HBM = 3.35e12, 989e12, 80e9        # bytes/s, FLOP/s, bytes
S_KV = 2 * L * h_kv * d_h * 2                # bytes per token (bf16)

def prefill_ms(T_p, mfu=0.5):
    flops = 2 * N * T_p + 2 * L * d * T_p**2
    return 1e3 * max(flops / (mfu * PEAK), 2 * N / BW)

def decode_step_ms(B, T):
    t_mem = (2 * N + B * T * S_KV) / BW
    t_cmp = 2 * N * B / PEAK
    return 1e3 * max(t_mem, t_cmp)

def max_batch(T, tpot_ms, usable_frac=0.9):
    B = 0
    while True:
        nb = B + 1
        lat_ok = decode_step_ms(nb, T) <= tpot_ms
        mem_ok = 2 * N + nb * T * S_KV <= usable_frac * HBM
        if not (lat_ok and mem_ok):
            return B, ("latency" if not lat_ok else "memory")
        B = nb

if __name__ == "__main__":
    for T_p in [128, 512, 2048, 8192]:
        print(T_p, round(prefill_ms(T_p), 2))
    for B in [1, 8, 32, 64, 96]:
        t = decode_step_ms(B, 4096)
        print(B, round(t, 2), round(1000 / t), round(1000 * B / t))
    print(max_batch(4096, 20), max_batch(4096, 40), max_batch(16384, 40))
