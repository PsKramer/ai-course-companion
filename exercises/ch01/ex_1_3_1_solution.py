"""Exercise 1.3.1 — solution."""
N, L, h_kv, d_h = 8.03e9, 32, 8, 128
BW, PEAK, HBM = 3.35e12, 989e12, 80e9

def kv_bytes_per_token(L, h_kv, d_h, bytes_per=2):
    return 2 * L * h_kv * d_h * bytes_per            # 131072 B = 128 KiB

def decode_step(B, T):
    w_bytes = 2 * N
    kv = B * T * kv_bytes_per_token(L, h_kv, d_h)
    t_mem = (w_bytes + kv) / BW
    t_cmp = 2 * N * B / PEAK
    t = max(t_mem, t_cmp)
    return round(t * 1e3, 2), round(B / t), (w_bytes + kv) <= HBM

if __name__ == "__main__":
    print("KV bytes/token:", kv_bytes_per_token(L, h_kv, d_h))
    print("B, (ms/step, tok/s aggregate, fits in 80 GB)")
    for B in [1, 8, 32, 64, 256]:
        print(B, decode_step(B, T=8192))
    print("short context T=512:")
    for B in [1, 64, 256, 1024]:
        print(B, decode_step(B, T=512))
    print("ridge point (FLOP/byte):", round(PEAK / BW))
