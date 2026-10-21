"""Exercise 4.4.1 (solution): (a) why the master weights stay in fp32, (b) an activation-memory
calculator for Course-8B. PyTorch CPU, runs in seconds."""
import torch
torch.set_num_threads(1)

# ---- (a) small updates vanish when the weight itself is stored in bf16 ----
def accumulate(dtype, w0=1.0, update=1e-3, steps=1000):
    w = torch.tensor(w0, dtype=dtype)
    for _ in range(steps):
        w = w + torch.tensor(update, dtype=dtype)             # what an optimizer step does to one weight
    return w.item()

print(f"fp32 weight after 1000 updates of 1e-3: {accumulate(torch.float32):.4f} (exact 2.0)")
print(f"bf16 weight after 1000 updates of 1e-3: {accumulate(torch.bfloat16):.4f}")
print(f"bf16 spacing at 1.0 = {torch.finfo(torch.bfloat16).eps} -> an update below half of that rounds away")

# ---- (b) saved activations per token per layer for a Llama-style block (bf16, FlashAttention) ----
def act_bytes_per_token_layer(d, h_kv, d_h, d_ff, b=2, recompute="none"):
    if recompute == "full":                     # keep only the block input; redo the forward in backward
        return d * b
    elems = (d          # RMSNorm-1 input (residual stream)
             + d        # RMSNorm-1 output = input to W_Q, W_K, W_V
             + d + 2 * h_kv * d_h   # Q, K, V (FlashAttention re-reads them; no T x T scores stored)
             + d        # attention output = input to W_O
             + d        # RMSNorm-2 input
             + d        # RMSNorm-2 output = input to W_gate, W_up
             + 2 * d_ff # gate and up projections (SwiGLU needs both)
             + d_ff)    # SiLU(gate) * up = input to W_down
    if recompute == "selective":                 # recompute the cheap SwiGLU product instead of storing it
        elems -= d_ff
    return elems * b

cfg = dict(d=4096, h_kv=8, d_h=128, d_ff=14336)
L, T, V = 32, 8192, 128256
for mode in ["none", "selective", "full"]:
    per_tok = act_bytes_per_token_layer(**cfg, recompute=mode) * L
    print(f"recompute={mode:9s}: {per_tok/1024:7.0f} KiB/token -> {per_tok*T/1e9:5.1f} GB per 8,192-token sequence")
print(f"logits for one sequence: bf16 {T*V*2/1e9:.1f} GB, fp32 {T*V*4/1e9:.1f} GB (chunked loss avoids both)")
