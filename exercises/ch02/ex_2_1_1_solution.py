"""Exercise 2.1.1: MinHash + LSH near-duplicate detection (solution). numpy only.
Run: python ex_2_1_1_solution.py
"""
import re, zlib, random
import numpy as np
from collections import defaultdict

P = (1 << 61) - 1  # Mersenne prime for universal hashing h(x) = (a*x + b) mod P

def shingles(text, n=5):
    w = re.findall(r"\w+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}

def jaccard(a, b):
    return len(a & b) / len(a | b)

class MinHashLSH:
    def __init__(self, bands=14, rows=8, seed=0):
        self.b, self.r = bands, rows
        rng = np.random.default_rng(seed)
        k = bands * rows
        self.a = rng.integers(1, P, size=k, dtype=np.uint64).astype(object)
        self.c = rng.integers(0, P, size=k, dtype=np.uint64).astype(object)
        self.buckets = [defaultdict(list) for _ in range(bands)]

    def signature(self, sh):
        x = np.array([zlib.crc32(s.encode()) for s in sh], dtype=object)  # base 32-bit hash
        # (k, |sh|) table of permuted hashes, min over shingles -> k-dim signature
        return np.array([((a * x + c) % P).min() for a, c in zip(self.a, self.c)], dtype=object)

    def insert_and_query(self, doc_id, sig):
        cands = set()
        for i in range(self.b):
            key = tuple(sig[i * self.r:(i + 1) * self.r])
            cands.update(self.buckets[i][key])
            self.buckets[i][key].append(doc_id)
        return cands

def p_candidate(s, b, r):
    return 1 - (1 - s ** r) ** b

def mutate(text, frac, rng):
    w = text.split()
    for i in rng.sample(range(len(w)), int(frac * len(w))):
        w[i] = rng.choice(["foo", "bar", "baz", "qux", "lorem"])
    return " ".join(w)

if __name__ == "__main__":
    rng = random.Random(0)
    vocab = [f"w{i}" for i in range(5000)]
    base = [" ".join(rng.choice(vocab) for _ in range(300)) for _ in range(40)]
    docs = []
    for i, d in enumerate(base):
        docs.append((f"orig{i}", d))
        docs.append((f"near{i}", mutate(d, 0.03, rng)))   # ~3% words edited -> high Jaccard
        docs.append((f"far{i}", mutate(d, 0.40, rng)))    # 40% edited -> low Jaccard
    lsh = MinHashLSH(14, 8)
    sh = {k: shingles(v) for k, v in docs}
    flagged = {}
    for k, _ in docs:
        c = lsh.insert_and_query(k, lsh.signature(sh[k]))
        if c:
            flagged[k] = c
    near_hit = sum(f"orig{i}" in flagged.get(f"near{i}", ()) for i in range(40))
    far_hit = sum(f"orig{i}" in flagged.get(f"far{i}", ()) for i in range(40))
    js_near = np.mean([jaccard(sh[f"orig{i}"], sh[f"near{i}"]) for i in range(40)])
    js_far = np.mean([jaccard(sh[f"orig{i}"], sh[f"far{i}"]) for i in range(40)])
    print(f"near-dups: mean J={js_near:.2f}, theory P={p_candidate(js_near,14,8):.2f}, caught {near_hit}/40")
    print(f"far edits: mean J={js_far:.3f}, theory P={p_candidate(js_far,14,8):.4f}, caught {far_hit}/40")
    for s in (0.7, 0.75, 0.8, 0.85):
        print(f"P(candidate | s={s}) = {p_candidate(s,14,8):.3f}")
    assert near_hit >= 24 and far_hit <= 2
    print("OK")
