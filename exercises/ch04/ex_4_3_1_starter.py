"""Exercise 4.3.1 (starter): AdamW by hand, Muon's Newton-Schulz orthogonalization,
and the two LR schedules (warmup+cosine, WSD). numpy only, runs in seconds."""
import math, numpy as np
rng = np.random.default_rng(0)

# ---- 1. AdamW step (decoupled weight decay, Loshchilov & Hutter 2017) ----
def adamw_step(w, g, m, v, t, lr, b1=0.9, b2=0.95, eps=1e-8, wd=0.1):
    # TODO: Adam moments, bias correction, decoupled weight decay
    raise NotImplementedError("TODO: adamw_step")

# ---- 2. Newton-Schulz quintic iteration (coefficients from Jordan, 2024) ----
def newton_schulz5(G, steps=5):
    # TODO: normalize, transpose if tall, 5 quintic iterations with (a, b, c)
    raise NotImplementedError("TODO: newton_schulz5")

def muon_update(G, M, beta=0.95):
    M = beta * M + G                                          # momentum on the raw gradient
    O = newton_schulz5(M)
    scale = 0.2 * math.sqrt(max(G.shape))                     # Moonshot's RMS-matching scale (Liu et al., 2025)
    return scale * O, M

# ---- 3. LR schedules: return the LR at step s ----
def warmup_cosine(s, peak, warmup, total, floor=0.1):
    if s < warmup: return peak * (s + 1) / warmup
    p = (s - warmup) / max(1, total - warmup)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p)))

def wsd(s, peak, warmup, total, decay_frac=0.2, floor=0.0):
    # TODO: warmup, constant stage, linear decay over the last decay_frac of steps
    raise NotImplementedError("TODO: wsd")

if __name__ == "__main__":
    # gradient with a skewed spectrum, like real LLM gradients: a few huge directions
    m, n = 256, 896
    U, _ = np.linalg.qr(rng.normal(size=(m, m))); V, _ = np.linalg.qr(rng.normal(size=(n, m)))
    s = np.logspace(0, -3, m)                                 # singular values from 1 down to 1e-3
    G = (U * s) @ V.T
    O = newton_schulz5(G)
    sv = np.linalg.svd(O, compute_uv=False)
    print(f"gradient singular values: max {s.max():.3g}, min {s.min():.3g}, ratio {s.max()/s.min():.0f}")
    print(f"after NS5: max {sv.max():.3f}, median {np.median(sv):.3f}, min {sv.min():.3f}")
    print(f"NS5 FLOPs for [{m}x{n}] = {5*(4*m*m*n + 2*m**3)/1e9:.2f} GFLOPs")
    # Muon overhead vs 6ND for Course-8B (Jordan's bound T*m/B)
    print(f"Muon FLOP overhead for Course-8B: 5*4096/4.19e6 = {5*4096/4.19e6:.2%}")
    # AdamW sanity: minimise 0.5*||w||^2 with g = w
    w = np.ones(4); mm = np.zeros(4); vv = np.zeros(4)
    for t in range(1, 201): w, mm, vv = adamw_step(w, w, mm, vv, t, lr=1e-2)
    print(f"AdamW after 200 steps on 0.5||w||^2: w = {w.round(3)}")
    total, warm, peak = 100_000, 2_000, 3e-4
    for st in [0, 1_999, 50_000, 79_999, 90_000, 99_999]:
        print(f"step {st:>6}: cosine {warmup_cosine(st, peak, warm, total):.2e}   wsd {wsd(st, peak, warm, total):.2e}")
    # area under the LR curve: a proxy for how far the weights can travel
    area_c = sum(warmup_cosine(s_, peak, warm, total) for s_ in range(total))
    area_w = sum(wsd(s_, peak, warm, total) for s_ in range(total))
    print(f"LR area: WSD/cosine = {area_w/area_c:.2f}")
