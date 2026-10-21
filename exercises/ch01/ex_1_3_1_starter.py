"""Exercise 1.3.1 — Decode roofline for Course-8B on one H100 (bf16 weights, bf16 KV cache)."""
import numpy as np
N, L, h_kv, d_h = 8.03e9, 32, 8, 128
BW, PEAK, HBM = 3.35e12, 989e12, 80e9      # bytes/s, FLOP/s, bytes

def kv_bytes_per_token(L, h_kv, d_h, bytes_per=2):
    # TODO: K and V, every layer, every KV head
    return None

def decode_step(B, T):
    """Return (seconds per step, aggregate tokens/s, fits_in_HBM) for B sequences at context T."""
    # TODO: bytes moved = all weights once + every sequence's KV cache; FLOPs = 2 N per sequence
    # TODO: step time = max(memory time, compute time)  (roofline, ignore overheads)
    return None

if __name__ == "__main__":
    print("KV bytes/token:", kv_bytes_per_token(L, h_kv, d_h))
    for B in [1, 8, 32, 64, 256]:
        print(B, decode_step(B, T=8192))
