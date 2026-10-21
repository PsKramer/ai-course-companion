import numpy as np
rng = np.random.default_rng(0)
V, GAMMA, STEPS = 8, 4, 50_000
p = rng.dirichlet(np.ones(V))                 # target model's next-token distribution (toy, context-free)
q = 0.6 * p + 0.4 * rng.dirichlet(np.ones(V)) # draft model: similar but not identical
alpha = np.minimum(p, q).sum()                # expected acceptance rate per drafted token

def speculative_step():
    """Draft GAMMA tokens from q, verify against p; return the emitted tokens (1..GAMMA+1)."""
    out = []
    for _ in range(GAMMA):
        x = rng.choice(V, p=q)
        if rng.random() < min(1.0, p[x] / q[x]):      # accept with prob min(1, p/q)
            out.append(x)
        else:                                          # reject: resample from residual max(0, p - q)
            r = np.maximum(p - q, 0); out.append(rng.choice(V, p=r / r.sum()))
            return out
    out.append(rng.choice(V, p=p))                    # all accepted: bonus token from the target
    return out

emitted = [speculative_step() for _ in range(STEPS)]
first = np.bincount([e[0] for e in emitted], minlength=V) / STEPS
alltok = np.bincount(np.concatenate(emitted), minlength=V); alltok = alltok / alltok.sum()
print(f"acceptance rate alpha        {alpha:.3f}")
print(f"tokens per verify step       measured {np.mean([len(e) for e in emitted]):.3f}   "
      f"formula {(1 - alpha ** (GAMMA + 1)) / (1 - alpha):.3f}")
print(f"TV(first token, p)           {0.5 * np.abs(first - p).sum():.4f}")
print(f"TV(all tokens, p)            {0.5 * np.abs(alltok - p).sum():.4f}")
print(f"TV(q, p) for comparison      {0.5 * np.abs(q - p).sum():.4f}")
