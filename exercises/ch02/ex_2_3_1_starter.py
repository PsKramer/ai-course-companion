"""Exercise 2.3.1 (starter): byte-level BPE training + encoding from scratch. Pure Python, CPU, < 10 s.
Fill in the three TODOs, then run: python ex_2_3_1_starter.py
"""
import re
from collections import Counter

# Simplified GPT-2-style pre-tokenizer: merges never cross these chunk boundaries.
PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]{1,3}| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+""")

def pretokenize(text):
    return PAT.findall(text)

def get_pair_counts(words):
    """words: dict[tuple[int,...], freq] -> Counter of adjacent pairs weighted by freq."""
    # TODO 1: count every adjacent pair (a, b) inside every word, weighted by the word's frequency f.
    raise NotImplementedError

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
        # TODO 2: pick the most frequent pair (tie-break: smallest pair), record merges[pair] = new_id,
        #         set vocab[new_id] to the concatenated bytes, and rewrite every word with merge_word().
        raise NotImplementedError
    return merges, vocab

def encode(text, merges):
    ids = []
    for ch in pretokenize(text):
        w = list(ch.encode("utf-8"))
        # TODO 3: repeatedly apply the lowest-rank (earliest-learned) merge present in w until none applies.
        raise NotImplementedError
        ids.extend(w)
    return ids

def decode(ids, vocab):
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")

if __name__ == "__main__":
  try:
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
  except NotImplementedError:
    print("Fill in the TODOs in get_pair_counts(), train_bpe() and encode().")
