"""Exercise 1.2.1 — Count Course-8B's parameters and the cost of one training step."""
cfg = dict(L=32, d=4096, h=32, h_kv=8, d_ff=14336, V=128256, tied_embeddings=False)

def param_count(L, d, h, h_kv, d_ff, V, tied_embeddings):
    d_h = d // h
    # TODO: attention weights per layer: W_Q [d, h*d_h], W_K, W_V [d, h_kv*d_h], W_O [h*d_h, d]
    attn = None
    # TODO: SwiGLU FFN per layer: W_gate, W_up [d, d_ff], W_down [d_ff, d]
    ffn = None
    norms = 2 * d                      # two RMSNorm gains per layer
    # TODO: embeddings (+ separate unembedding if not tied) + final norm
    emb = None
    return L * (attn + ffn + norms) + emb

def step_cost(N, B, T, n_gpus, peak_flops=989e12, mfu=0.40):
    # TODO: training FLOPs per step (6 N per token), time per step, bytes of training state at 16 B/param
    flops = None
    seconds = None
    state_bytes = None
    return flops, seconds, state_bytes

if __name__ == "__main__":
    N = param_count(**cfg)
    print(f"N = {N/1e9:.3f} B")
    f, s, m = step_cost(N, B=512, T=8192, n_gpus=512)
    print(f"step FLOPs = {f:.2e}, step time = {s:.2f} s, train state = {m/1e9:.0f} GB")
