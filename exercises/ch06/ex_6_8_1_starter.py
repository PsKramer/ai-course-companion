import re
import numpy as np
rng = np.random.default_rng(0)

# ---- Part A: sampling filters on a synthetic next-token distribution ----
V = 50_000
logits = (-1.1 * np.log(np.arange(1, V + 1)) + rng.normal(0, 0.3, V)).astype(np.float64)  # Zipf-like

def filtered_probs(logits, temperature=1.0, top_k=0, top_p=1.0, min_p=0.0):
    z = logits / temperature
    p = np.exp(z - z.max()); p /= p.sum()
    keep = np.ones(V, bool)
    # TODO: top_k  -> keep only the k most probable tokens
    # TODO: top_p  -> keep the smallest set of most probable tokens whose mass reaches top_p
    # TODO: min_p  -> keep tokens with p >= min_p * max(p)
    q = np.where(keep, p, 0.0)
    return q / q.sum()

for name, kw in [("T=1.0", {}), ("T=0.7", {"temperature": 0.7}), ("top-k 50", {"top_k": 50}),
                 ("top-p 0.9", {"top_p": 0.9}), ("min-p 0.05", {"min_p": 0.05}),
                 ("T=1.5 + min-p 0.05", {"temperature": 1.5, "min_p": 0.05})]:
    q = filtered_probs(logits, **kw)
    nz = q[q > 0]
    print(f"{name:20s} tokens kept {np.count_nonzero(q):6d}   entropy {-(nz * np.log2(nz)).sum():5.2f} bits")

# ---- Part B: constrained decoding with a finite-state mask ----
VOCAB = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "12", "345", "-", "-9", "a", "hello", " ", "555-"]
PATTERN = re.compile(r"\d{3}-\d{4}")
def prefix_ok(s):
    """Can s still be extended to a full match of PATTERN? (tiny hand-written FSM)"""
    shape = "ddd-dddd"
    return len(s) <= len(shape) and all((c.isdigit() if t == "d" else c == "-") for c, t in zip(s, shape))

def constrained_generate(rng):
    out = ""
    for _ in range(20):                              # safety cap on length
        if PATTERN.fullmatch(out) or not any(prefix_ok(out + t) for t in VOCAB):
            break
        mask = np.array([prefix_ok(out + t) for t in VOCAB])
        z = rng.normal(0, 2, len(VOCAB))            # stand-in for model logits
        # TODO: forbid tokens that break the pattern (set their logits to -inf)
        p = np.exp(z - z.max()); p /= p.sum()
        out += VOCAB[rng.choice(len(VOCAB), p=p)]
    return out

samples = [constrained_generate(rng) for _ in range(5)]
print("constrained samples:", samples, "all match:", all(PATTERN.fullmatch(s) for s in samples))

# ---- Part C: floating-point addition is not associative ----
x = rng.standard_normal(4096).astype(np.float32) * 100
a = np.float32(0)
for v in x: a += v                                   # sequential order
b = x.reshape(64, 64).sum(axis=1, dtype=np.float32).sum(dtype=np.float32)   # split into 64 chunks
print(f"sum sequential {a:.6f}  sum in 64 chunks {b:.6f}  difference {abs(float(a) - float(b)):.2e}")
