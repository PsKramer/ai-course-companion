from collections import OrderedDict
import numpy as np
rng = np.random.default_rng(0)
P = 16                                   # tokens per KV block (vLLM default)
SYS = [list(rng.integers(0, 50_000, 1024)) for _ in range(4)]   # 4 system prompts, 1,024 tokens each

def make_requests(n=6000):
    """Multi-turn chats: each conversation = system prompt + growing history; turns arrive interleaved."""
    convs, reqs = [], []
    for _ in range(n):
        if not convs or rng.random() < 0.15:                     # 15%: start a new conversation
            convs.append(list(SYS[rng.integers(4)]))
        c = convs[rng.integers(max(0, len(convs) - 400), len(convs))]  # the 400 most recent conversations are active
        c += list(rng.integers(0, 50_000, rng.integers(50, 400)))     # new user turn
        reqs.append(list(c))
        c += list(rng.integers(0, 50_000, rng.integers(100, 600)))    # model reply joins the history
    return reqs

def block_hashes(tokens):
    """Chained hash per full block: hash(parent_hash, block tokens), as in vLLM prefix caching."""
    h, out = None, []
    for i in range(0, len(tokens) - len(tokens) % P, P):
        h = hash((h, tuple(tokens[i:i + P]))); out.append(h)
    return out

def simulate(reqs, capacity_blocks):
    cache, hit_tok, total_tok = OrderedDict(), 0, 0
    for toks in reqs:
        hs = block_hashes(toks); n_hit = 0
        for h in hs:                     # longest cached prefix: stop at the first miss
            if h in cache: n_hit += 1; cache.move_to_end(h)
            else: break
        hit_tok += n_hit * P; total_tok += len(toks)
        for h in hs[n_hit:]:             # newly computed blocks enter the cache (LRU eviction)
            cache[h] = True
            if len(cache) > capacity_blocks: cache.popitem(last=False)
    return hit_tok / total_tok

reqs = make_requests()
print(f"requests {len(reqs)}, mean prompt {np.mean([len(r) for r in reqs]):,.0f} tokens")
for cap_tokens in [50_000, 200_000, 427_000, 2_000_000]:
    print(f"cache {cap_tokens:>9,} tokens ({cap_tokens * 131072 / 1e9:6.1f} GB for Course-8B): "
          f"prefill tokens served from cache {100 * simulate(reqs, cap_tokens // P):5.1f} %")
