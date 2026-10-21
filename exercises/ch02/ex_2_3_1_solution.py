"""Exercise 2.3.1: byte-level BPE training + encoding from scratch (solution).
Run: python ex_2_3_1_solution.py
"""
import re
from collections import Counter

# Simplified GPT-2-style pre-tokenizer: merges never cross these chunk boundaries.
PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]{1,3}| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+""")

def pretokenize(text):
    return PAT.findall(text)

def get_pair_counts(words):
    """words: dict[tuple[int,...], freq] -> Counter of adjacent pairs weighted by freq."""
    counts = Counter()
    for w, f in words.items():
        for a, b in zip(w, w[1:]):
            counts[(a, b)] += f
    return counts

def merge_word(w, pair, new_id):
    out, i = [], 0
    while i < len(w):
        if i + 1 < len(w) and w[i] == pair[0] and w[i + 1] == pair[1]:
            out.append(new_id); i += 2
        else:
            out.append(w[i]); i += 1
    return tuple(out)

def train_bpe(text, vocab_size):
    assert vocab_size >= 256
    # 1) chunk, 2) bytes, 3) dedupe chunks into a frequency table (huge speedup on real corpora)
    words = Counter(tuple(ch.encode("utf-8")) for ch in pretokenize(text))
    merges = {}                                   # (a,b) -> new_id, insertion order = merge rank
    vocab = {i: bytes([i]) for i in range(256)}
    for new_id in range(256, vocab_size):
        counts = get_pair_counts(words)
        if not counts:
            break
        # tie-break deterministically: highest count, then smallest pair
        pair = max(counts, key=lambda p: (counts[p], -p[0], -p[1]))
        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        words = Counter({merge_word(w, pair, new_id): f for w, f in words.items()})
    return merges, vocab

def encode(text, merges):
    ids = []
    for ch in pretokenize(text):
        w = list(ch.encode("utf-8"))
        while len(w) >= 2:
            # apply the earliest-learned (lowest-rank) merge present: this is exactly what tiktoken does
            pairs = set(zip(w, w[1:]))
            best = min(pairs, key=lambda p: merges.get(p, float("inf")))
            if best not in merges:
                break
            w = list(merge_word(w, best, merges[best]))
        ids.extend(w)
    return ids

def decode(ids, vocab):
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")

if __name__ == "__main__":
    corpus = ("the quick brown fox jumps over the lazy dog. " * 20 +
              "the theory of the thermal throughput is thorough. " * 15 +
              "tokenization turns text into tokens; tokens turn into embeddings. " * 10 +
              "Größe and naïve café — 東京 — emoji 🙂 survive byte-level BPE. " * 3)
    merges, vocab = train_bpe(corpus, 256 + 200)
    print("first 10 merges:", [vocab[i] for i in list(merges.values())[:10]])
    for s in ["the thermal tokens", "Größe 🙂 東京", "unseen zebra xylophone"]:
        ids = encode(s, merges)
        assert decode(ids, vocab) == s, "round-trip failed"
        print(f"{s!r}: {len(s.encode())} bytes -> {len(ids)} tokens")
    n_bytes = len(corpus.encode()); n_tok = len(encode(corpus, merges))
    print(f"corpus compression: {n_bytes/n_tok:.2f} bytes/token")
    print("OK")
