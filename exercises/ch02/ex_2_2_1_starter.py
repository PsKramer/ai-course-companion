"""Exercise 2.2.1 (starter): n-gram decontamination of a training shard against a benchmark. CPU, < 1 s.
Fill in the two TODOs, then run: python ex_2_2_1_starter.py
"""
import re

def norm_tokens(text):
    """Lowercase, keep alphanumeric runs: punctuation and case differences must not hide a leak."""
    return re.findall(r"[a-z0-9]+", text.lower())

def ngrams(toks, n):
    # TODO 1: return the set of all length-n tuples of consecutive tokens.
    raise NotImplementedError

def build_index(bench_items, n):
    idx = set()
    for q in bench_items:
        idx |= ngrams(norm_tokens(q), n)
    return idx

def contamination(doc, idx, n):
    # TODO 2: fraction of the doc's n-grams that also appear in the benchmark index (0.0 if the doc is shorter than n).
    raise NotImplementedError

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
        "At what time does it arrive if it makes one 20-minute stop? Answer: 6:20 PM.",
        "Trains in Europe are fast. Many leave the station on time and travel at high speed.",
        "Fluorine has the highest electronegativity. Chemistry notes for students.",
    ]
    try:
        for n in (8, 13):
            keep, drop = decontaminate(docs, bench, n=n)
            print(f"n={n}: dropped {len(drop)} -> {[d[:30] for d in drop]}")
        para = ("If a locomotive departs at 15:00 doing 60 mph to a town 180 mi off, stopping once for 20 min, "
                "when does it get there?")
        print("paraphrase contamination @13:", contamination(para, build_index(bench, 13), 13))
    except NotImplementedError:
        print("Fill in the TODOs in ngrams() and contamination().")
