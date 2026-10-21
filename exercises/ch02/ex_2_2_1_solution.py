"""Exercise 2.2.1: n-gram decontamination of a training shard against a benchmark (solution).
Run: python ex_2_2_1_solution.py
"""
import re

def norm_tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def ngrams(toks, n):
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}

def build_index(bench_items, n):
    idx = set()
    for q in bench_items:
        idx |= ngrams(norm_tokens(q), n)
    return idx

def contamination(doc, idx, n):
    """fraction of the doc's n-grams that appear in the benchmark index"""
    g = ngrams(norm_tokens(doc), n)
    return (len(g & idx) / len(g)) if g else 0.0

def decontaminate(docs, bench_items, n=13, thresh=0.0):
    idx = build_index(bench_items, n)
    keep, drop = [], []
    for d in docs:
        (drop if contamination(d, idx, n) > thresh else keep).append(d)
    return keep, drop

if __name__ == "__main__":
    bench = ["A train leaves the station at 3 pm traveling at 60 miles per hour toward a city 180 miles away. "
             "At what time does it arrive if it makes one 20 minute stop?",
             "Which element has the highest electronegativity on the Pauling scale and why does it exceed oxygen?"]
    docs = [
        "Forum post: a train leaves the station at 3 PM, traveling at 60 miles per hour toward a city 180 miles away. "
        "At what time does it arrive if it makes one 20-minute stop? Answer: 6:20 PM.",           # leaked w/ answer
        "Trains in Europe are fast. Many leave the station on time and travel at high speed.",       # clean
        "Fluorine has the highest electronegativity. Chemistry notes for students.",                # topical, not leaked
    ]
    for n in (8, 13):
        keep, drop = decontaminate(docs, bench, n=n)
        print(f"n={n}: dropped {len(drop)} -> {[d[:30] for d in drop]}")
    keep, drop = decontaminate(docs, bench, n=13)
    assert len(drop) == 1 and drop[0].startswith("Forum")
    # paraphrase defeats n-gram matching -> motivates embedding/LLM-based decontam
    para = "If a locomotive departs at 15:00 doing 60 mph to a town 180 mi off, stopping once for 20 min, when does it get there?"
    print("paraphrase contamination @13:", contamination(para, build_index(bench, 13), 13))
    print("OK")
