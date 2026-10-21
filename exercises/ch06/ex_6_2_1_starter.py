def kv_bytes_per_token(L, h_kv=None, d_h=None, b=2, mla=None, T=None, window=None, n_global=None):
    """GQA/MHA/MQA: 2*L*h_kv*d_h*b. MLA: pass mla=(d_c, d_R) -> L*(d_c+d_R)*b.
    Sliding window: pass T, window, n_global (other layers local) -> average bytes per token at context T."""
    # TODO
    pass

def tokens_in_flight(hbm_bytes, weight_bytes, s_kv, headroom=0.10):
    # TODO: (HBM - weights - headroom*HBM) / s_kv
    pass

if __name__ == "__main__":
    course8b = kv_bytes_per_token(32, 8, 128)
    print("Course-8B", course8b)
    print("DeepSeek-V3 MLA", kv_bytes_per_token(61, mla=(512, 64)))
    print("Course-8B, 8 global + 24 local (W=1024) at 128k",
          kv_bytes_per_token(32, 8, 128, T=131072, window=1024, n_global=8))
    print("H100 tokens, bf16 KV:", tokens_in_flight(80e9, 16.06e9, course8b))
    print("H100 tokens, FP8 KV:", tokens_in_flight(80e9, 16.06e9, kv_bytes_per_token(32, 8, 128, b=1)))
