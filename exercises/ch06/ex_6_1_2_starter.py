N, L, d, h_kv, d_h = 8.03e9, 32, 4096, 8, 128
BW, PEAK, HBM = 3.35e12, 989e12, 80e9        # bytes/s, FLOP/s, bytes
S_KV = 2 * L * h_kv * d_h * 2                # bytes per token (bf16)

def prefill_ms(T_p, mfu=0.5):
    # TODO: FLOPs = 2 N T_p + 2 L d T_p^2; memory floor = one read of 2N bytes
    # return the larger of compute time (at mfu * PEAK) and memory time, in ms
    pass

def decode_step_ms(B, T):
    # TODO: roofline step time from the key formula, in ms
    pass

def max_batch(T, tpot_ms, usable_frac=0.9):
    # TODO: largest B such that decode_step_ms(B, T) <= tpot_ms
    #       AND 2N + B*T*S_KV <= usable_frac * HBM
    # return (B, which_limit) where which_limit is "latency" or "memory"
    pass

if __name__ == "__main__":
    for T_p in [128, 512, 2048, 8192]:
        print(T_p, prefill_ms(T_p))
    for B in [1, 8, 32, 64, 96]:
        t = decode_step_ms(B, 4096)
        print(B, t)
    print(max_batch(4096, 20), max_batch(4096, 40), max_batch(16384, 40))
