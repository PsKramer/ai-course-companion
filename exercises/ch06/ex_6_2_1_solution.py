def kv_bytes_per_token(L, h_kv=None, d_h=None, b=2, mla=None, T=None, window=None, n_global=None):
    if mla is not None:
        d_c, d_R = mla
        return L * (d_c + d_R) * b
    per_layer = 2 * h_kv * d_h * b
    if window is None:
        return L * per_layer
    n_local = L - n_global
    total = per_layer * (n_global * T + n_local * min(T, window))
    return total / T

def tokens_in_flight(hbm_bytes, weight_bytes, s_kv, headroom=0.10):
    return int((hbm_bytes - weight_bytes - headroom * hbm_bytes) / s_kv)

if __name__ == "__main__":
    course8b = kv_bytes_per_token(32, 8, 128)
    print("Course-8B", course8b)
    print("DeepSeek-V3 MLA", kv_bytes_per_token(61, mla=(512, 64)))
    print("Course-8B, 8 global + 24 local (W=1024) at 128k",
          kv_bytes_per_token(32, 8, 128, T=131072, window=1024, n_global=8))
    print("H100 tokens, bf16 KV:", tokens_in_flight(80e9, 16.06e9, course8b))
    print("H100 tokens, FP8 KV:", tokens_in_flight(80e9, 16.06e9, kv_bytes_per_token(32, 8, 128, b=1)))
