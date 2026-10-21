"""Exercise 5.6.2 (starter): pass@k, majority voting and a noisy verifier, from the same samples.
Run: python ex_5_6_2_solution.py   (numpy only, ~5 s)
Setup: 300 problems. On problem i the model answers correctly with probability p_i (spread from very hard
to easy); wrong answers fall on 4 distinct wrong values, one of which is a 'popular' mistake."""
import numpy as np
from math import comb

def pass_at_k(n, c, k):
    """Unbiased estimator of pass@k from n samples with c correct (Chen et al., 2021):
    1 - C(n-c, k) / C(n, k)."""
    # TODO: 1 - C(n-c, k)/C(n, k), and 1.0 when n - c < k
    raise NotImplementedError("TODO: pass_at_k")

def majority_vote(answers):
    """Most frequent answer; ties broken by the smallest value (deterministic)."""
    # TODO: np.unique with counts; return the value with the largest count
    raise NotImplementedError("TODO: majority_vote")

def verifier_pick(answers, rng, acc=0.9):
    """A noisy verifier labels each sample correct (answer 0) with probability acc, else flips the label.
    Return the first sample labeled correct, or the majority answer if none is."""
    truth = answers == 0
    label = np.where(rng.random(len(answers)) < acc, truth, ~truth)
    return answers[np.argmax(label)] if label.any() else majority_vote(answers)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_prob, n = 300, 64
    p = rng.beta(0.6, 0.8, n_prob)                          # per-problem accuracy, many near 0 or 1
    wrong_p = np.array([0.55, 0.15, 0.15, 0.15])            # the popular mistake takes 55% of wrong answers
    samples = []
    for pi in p:
        correct = rng.random(n) < pi
        wrong = rng.choice([1, 2, 3, 4], n, p=wrong_p)
        samples.append(np.where(correct, 0, wrong))         # answer 0 is the right one
    samples = np.array(samples)
    print("  k   pass@k   maj@k   verifier@k (90% accurate)")
    res = {}
    for k in [1, 4, 16, 64]:
        pk = np.mean([pass_at_k(n, int((s == 0).sum()), k) for s in samples])
        # majority and verifier on disjoint chunks of k samples, averaged
        chunks = [s[j:j + k] for s in samples for j in range(0, n, k)]
        mk = np.mean([majority_vote(c) == 0 for c in chunks])
        vk = np.mean([verifier_pick(c, rng) == 0 for c in chunks])
        res[k] = (pk, mk, vk); print(f"{k:3d}   {pk:.3f}    {mk:.3f}   {vk:.3f}")
    plural = np.mean(p > (1 - p) * wrong_p[0])
    print(f"problems where the right answer is the most likely single answer: {plural:.0%} "
          f"(majority voting converges to this)")
    assert res[64][0] >= res[16][0] >= res[1][0] and res[64][1] <= plural + 0.05
    print("OK")
