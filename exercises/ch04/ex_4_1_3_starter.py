"""Exercise 4.1.3 (starter): a tiny byte-level pre-training run on CPU.
Measures loss in nats/token, converts to bits per byte, and counts training FLOPs
with the 6N + 6LTd (causal attention) accounting from 4.1. Runs in ~1 min on a laptop CPU."""
import math, time, inspect, argparse, json, textwrap
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0); torch.set_num_threads(1)  # small model: one thread avoids oversubscription
# ---- data: Python standard-library source as a byte stream (V = 256 byte "tokens") ----
text = "".join(inspect.getsource(m) for m in (argparse, json.decoder, json.encoder, textwrap))
data = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
n_val = len(data) // 10
train, val = data[:-n_val], data[-n_val:]
V, T, d, L, h, B = 256, 128, 128, 2, 4, 32

class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Embedding(T, d)
        layer = nn.TransformerEncoderLayer(d, h, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, L)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)
    def forward(self, x):                                   # x: [B, T] int
        t = x.shape[1]
        mask = nn.Transformer.generate_square_subsequent_mask(t)
        z = self.emb(x) + self.pos(torch.arange(t))          # [B, T, d]
        z = self.blocks(z, mask=mask, is_causal=True)
        return self.head(self.norm(z))                       # logits [B, T, V]

def batch(src):
    ix = torch.randint(len(src) - T - 1, (B,))
    seq = torch.stack([src[i:i + T + 1] for i in ix])        # [B, T+1]
    return seq[:, :-1], seq[:, 1:]                           # inputs, targets shifted by one

def next_token_loss(logits, targets):
    # mean cross-entropy in nats per token over all B*T positions
    # TODO: mean cross-entropy (nats/token) between logits [B, T, V] and targets [B, T]
    raise NotImplementedError("TODO: next_token_loss")

def bits_per_byte(loss_nats_per_token, bytes_per_token):
    # TODO: convert nats/token to bits per byte
    raise NotImplementedError("TODO: bits_per_byte")

def train_flops_per_token(N_nonemb, L, T, d):
    # 6N for the weight matmuls + 6LTd for causal attention scores/values (fwd 2LTd, bwd 4LTd)
    # TODO: 6N for weight matmuls plus the causal-attention term from 4.1
    raise NotImplementedError("TODO: train_flops_per_token")

model = TinyLM()
N = sum(p.numel() for p in model.parameters())
N_nonemb = N - model.emb.weight.numel() - model.pos.weight.numel()
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)
steps, warm = 400, 40
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))

@torch.no_grad()
def val_loss():
    model.eval(); tot = 0.0
    for _ in range(10):
        x, y = batch(val); tot += next_token_loss(model(x), y).item()
    model.train(); return tot / 10

print(f"params N = {N:,} (non-embedding {N_nonemb:,}); train bytes = {len(train):,}")
print(f"step 0: val loss {val_loss():.3f} nats/token (uniform would be ln 256 = {math.log(256):.3f})")
t0 = time.time()
for s in range(steps):
    x, y = batch(train)
    loss = next_token_loss(model(x), y)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if (s + 1) % 100 == 0:
        vl = val_loss()
        print(f"step {s+1}: train {loss.item():.3f}  val {vl:.3f} nats/token = {bits_per_byte(vl, 1.0):.3f} bits/byte")
dt = time.time() - t0
tokens = steps * B * T
fpt = train_flops_per_token(N_nonemb, L, T, d)
print(f"FLOPs/token = {fpt:,} (attention share {6*L*T*d/fpt:.1%}); total = {fpt*tokens:.2e} FLOPs")
print(f"throughput = {tokens/dt:,.0f} tokens/s, achieved = {fpt*tokens/dt/1e9:.1f} GFLOP/s on this CPU")
