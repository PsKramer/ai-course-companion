"""Exercise 3.5.1 (starter): a top-k mixture-of-experts (MoE) router with
(a) a Switch-style auxiliary load-balancing loss and (b) auxiliary-loss-free bias balancing.
The router scores are skewed on purpose so some experts are 'popular'.
numpy only, CPU, < 5 s.  Run: python ex_3_5_1_starter.py
"""
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def route(x, W_r, k, bias=None):
    """x [N, d], W_r [d, E]. Sigmoid affinities s [N, E]. Selection uses s + bias;
    gate weights use s only, normalized over the k chosen experts. Returns idx [N, k], gates [N, k], s."""
    # TODO: sigmoid affinities s; pick top-k by s + bias (bias may be None); gates = chosen s renormalized to sum to 1. Return idx, gates, s.
    raise NotImplementedError

def load(idx, E):
    """Fraction of routed slots each expert receives; perfectly balanced = 1/E each."""
    return np.bincount(idx.ravel(), minlength=E) / idx.size

def aux_loss(idx, s, E, k):
    """Switch-style balance loss sum_i f_i * P_i, with f_i = E * load_i (1.0 when balanced) and
    P_i = mean normalized affinity of expert i. A perfectly balanced router scores 1.0; skew pushes it up.
    In training this is multiplied by a small coefficient alpha and added to the language-model loss."""
    f = load(idx, E) * E                  # 1.0 per expert when balanced
    P = (s / s.sum(1, keepdims=True)).mean(0)   # mean normalized affinity per expert
    return float((f * P).sum())

if __name__ == "__main__":
    try:
        rng = np.random.default_rng(0)
        N, d, E, k = 4096, 64, 32, 4
        x = rng.standard_normal((N, d))
        W_r = rng.standard_normal((d, E)) * 0.1
        skew = np.zeros(E); skew[:4] = 1.5                    # four experts are favoured a priori
        x_aug = np.hstack([x, np.ones((N, 1))]); W_aug = np.vstack([W_r, skew])

        idx, g, s = route(x_aug, W_aug, k)
        L0 = load(idx, E)
        print(f"no balancing:   max load {L0.max() * E:.2f}x mean, min {L0.min() * E:.2f}x, aux loss {aux_loss(idx, s, E, k):.3f}")
        assert np.allclose(g.sum(1), 1.0)

        # Auxiliary-loss-free balancing: after each batch, nudge a per-expert bias toward the mean load.
        bias, gamma = np.zeros(E), 0.01
        for step in range(300):
            xb = np.hstack([rng.standard_normal((512, d)), np.ones((512, 1))])
            ib, _, _ = route(xb, W_aug, k, bias)
            bias -= gamma * np.sign(load(ib, E) - 1 / E)      # overloaded -> lower bias, underloaded -> raise
        idx2, g2, s2 = route(x_aug, W_aug, k, bias)
        L1 = load(idx2, E)
        print(f"bias balancing: max load {L1.max() * E:.2f}x mean, min {L1.min() * E:.2f}x, aux loss {aux_loss(idx2, s2, E, k):.3f}")
        # The bias changes WHICH experts are picked; gate values still come from s alone (see route):
        same = (np.sort(idx2, 1) == np.sort(idx, 1)).all(1)
        print(f"tokens whose expert set is unchanged: {same.mean():.0%}")
        assert L1.max() < L0.max() and L1.max() * E < 1.5
        print("OK")
    except NotImplementedError:
        print("Fill in the TODOs in route().")
