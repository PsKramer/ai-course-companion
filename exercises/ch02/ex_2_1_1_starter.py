"""Exercise 2.1.1 (starter): MinHash + LSH near-duplicate detection. numpy only, CPU, < 1 min.
Fill in the three TODOs, then run: python ex_2_1_1_starter.py
"""
import re, zlib, random
import numpy as np
from collections import defaultdict

P = (1 << 61) - 1  # Mersenne prime for universal hashing h(x) = (a*x + c) mod P

def shingles(text, n=5):
    """Set of word 5-grams (FineWeb uses word 5-grams)."""
    w = re.findall(r"\w+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}

def jaccard(a, b):
    return len(a & b) / len(a | b)

class MinHashLSH:
    def __init__(self, bands=14, rows=8, seed=0):
        self.b, self.r = bands, rows
        rng = np.random.default_rng(seed)
        k = bands * rows                                    # 112 hash functions, as in FineWeb
        self.a = rng.integers(1, P, size=k, dtype=np.uint64).astype(object)
        self.c = rng.integers(0, P, size=k, dtype=np.uint64).astype(object)
        self.buckets = [defaultdict(list) for _ in range(bands)]

    def signature(self, sh):
        x = [zlib.crc32(s.encode()) for s in sh]            # base 32-bit hash of each shingle
        # TODO 1: for each of the k hash functions (a_i, c_i), compute min over shingles of (a_i*x + c_i) % P.
        #         Return a length-k array (the MinHash signature).
        raise NotImplementedError

    def insert_and_query(self, doc_id, sig):
        # TODO 2: split sig into self.b bands of self.r rows; for each band use the tuple of its r values as a
        #         bucket key in self.buckets[band]. Collect every doc already in any of those buckets
        #         (the candidates), then add doc_id to each bucket. Return the candidate set.
        raise NotImplementedError

def p_candidate(s, b, r):
    # TODO 3: probability that two docs with Jaccard similarity s share at least one identical band.
    raise NotImplementedError

def mutate(text, frac, rng):
    w = text.split()
    for i in rng.sample(range(len(w)), int(frac * len(w))):
        w[i] = rng.choice(["foo", "bar", "baz", "qux", "lorem"])
    return " ".join(w)

if __name__ == "__main__":
    try:
        rng = random.Random(0)
        vocab = [f"w{i}" for i in range(5000)]
        base = [" ".join(rng.choice(vocab) for _ in range(300)) for _ in range(40)]
        docs = []
        for i, d in enumerate(base):
            docs += [(f"orig{i}", d), (f"near{i}", mutate(d, 0.03, rng)), (f"far{i}", mutate(d, 0.40, rng))]
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
    except NotImplementedError:
        print("Fill in the TODOs in signature(), insert_and_query() and p_candidate().")
