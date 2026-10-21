"""Exercise 1.2.1 — solution."""
cfg = dict(L=32, d=4096, h=32, h_kv=8, d_ff=14336, V=128256, tied_embeddings=False)

def param_count(L, d, h, h_kv, d_ff, V, tied_embeddings):
    d_h = d // h
    attn = d * h * d_h + 2 * d * h_kv * d_h + h * d_h * d
    ffn = 3 * d * d_ff
    norms = 2 * d
    emb = V * d * (1 if tied_embeddings else 2) + d
    return L * (attn + ffn + norms) + emb

def step_cost(N, B, T, n_gpus, peak_flops=989e12, mfu=0.40):
    flops = 6 * N * B * T
    seconds = flops / (n_gpus * peak_flops * mfu)
    state_bytes = 16 * N   # bf16 weights (2) + bf16 grads (2) + fp32 master (4) + Adam m, v (4+4)
    return flops, seconds, state_bytes

if __name__ == "__main__":
    N = param_count(**cfg)
    print(f"N = {N/1e9:.3f} B")
    f, s, m = step_cost(N, B=512, T=8192, n_gpus=512)
    print(f"step FLOPs = {f:.2e}, step time = {s:.2f} s, train state = {m/1e9:.0f} GB")
    total = 6 * N * 15e12
    print(f"15T-token run: C = {total:.2e} FLOPs = {total/(989e12*0.4*3600)/1e6:.2f} M H100-hours at 40% MFU")
