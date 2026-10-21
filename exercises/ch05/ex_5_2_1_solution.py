"""Exercise 5.2.1 (solution): fit a Bradley-Terry reward model, then watch best-of-n over-optimize it.
Run: python ex_5_2_1_solution.py   (numpy only, ~2 s)
Setup: a response is a feature vector x in R^6. x[0] is 'length' (standardized). The gold (true) reward
likes quality features and moderate length but penalizes very long answers. Labelers' comparisons come
from the gold reward through the Bradley-Terry model. The proxy reward model is linear, so it cannot
represent the length penalty and learns 'longer is better' from the data."""
import numpy as np

D = 6
W_GOLD = np.array([1.0, 0.5, 0.5, 0.0, 0.0, 0.0])    # x[3..5] are irrelevant features

def gold_reward(X):
    return X @ W_GOLD - 1.0 * np.maximum(X[:, 0] - 0.5, 0) ** 2   # longer helps, until it hurts

def sample_responses(rng, n):
    return rng.normal(size=(n, D))

def make_preferences(rng, n_pairs):
    """Pairs (x_chosen, x_rejected) labeled by sampling from Bradley-Terry on the gold reward."""
    A, B = sample_responses(rng, n_pairs), sample_responses(rng, n_pairs)
    p_a = 1 / (1 + np.exp(-(gold_reward(A) - gold_reward(B))))
    a_wins = rng.random(n_pairs) < p_a
    return np.where(a_wins[:, None], A, B), np.where(a_wins[:, None], B, A)

def fit_bt(Xc, Xr, lr=0.5, steps=500, l2=1e-3):
    """Linear reward r(x) = w.x fit by minimizing -log sigmoid(r(chosen) - r(rejected)) + l2 |w|^2."""
    w = np.zeros(D)
    for _ in range(steps):
        z = (Xc - Xr) @ w
        g = -((1 - 1 / (1 + np.exp(-z)))[:, None] * (Xc - Xr)).mean(0) + 2 * l2 * w
        w -= lr * g
    return w

def bt_accuracy(w, Xc, Xr):
    return float(((Xc - Xr) @ w > 0).mean())

def best_of_n(rng, w, n, trials=4000):
    """Draw n responses per trial, keep the one the proxy scores highest; return mean gold and proxy."""
    X = rng.normal(size=(trials, n, D))
    pick = np.argmax(X @ w, axis=1)
    Xb = X[np.arange(trials), pick]
    return gold_reward(Xb).mean(), (Xb @ w).mean()

def kl_bon(n):
    """Analytic KL(best-of-n || base) = log n - (n-1)/n (Stiennon et al., 2020)."""
    return np.log(n) - (n - 1) / n

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    Xc, Xr = make_preferences(rng, 20000)
    Tc, Tr = make_preferences(rng, 5000)
    w = fit_bt(Xc, Xr)
    print("proxy weights:", np.round(w, 2), " (gold linear part:", W_GOLD, ")")
    print(f"held-out pairwise accuracy: proxy {bt_accuracy(w, Tc, Tr):.3f}")
    # the best any reward can do on noisy BT labels: agree with the gold ordering
    g = gold_reward(Tc) - gold_reward(Tr)
    print(f"held-out pairwise accuracy: gold  {(g > 0).mean():.3f}")
    print(" n      KL(nats)  proxy   gold")
    rows = []
    for n in [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024]:
        gm, pm = best_of_n(rng, w, n, trials=4000 if n <= 256 else 1000)
        rows.append((n, gm)); print(f"{n:5d}  {kl_bon(n):7.2f}  {pm:6.2f}  {gm:6.2f}")
    best = max(rows, key=lambda r: r[1])
    print(f"gold reward peaks at n = {best[0]} and falls after: over-optimization")
    assert rows[-1][1] < best[1]
    print("OK")
