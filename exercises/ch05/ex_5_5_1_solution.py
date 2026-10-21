"""Exercise 5.5.1 (solution): Group Relative Policy Optimization (GRPO) on a toy verifiable task.
Run: python ex_5_5_1_solution.py   (numpy only, ~10 s)
Setup: 512 problems, each with K = 8 candidate answers, exactly one correct (the verifier knows which).
The policy scores candidate k of problem i as theta . phi[i, k] + b[i, k]; theta (16 numbers) is learned and
shared, b is a fixed per-problem offset that sets how hard each problem starts. Correct answers carry a
faint feature signature the policy can learn, so learning on some problems transfers to others."""
import numpy as np

NP, K, D = 512, 8, 16

def make_task(rng):
    types = rng.integers(0, 4, NP)
    mu = rng.normal(0, 1, (4, D)) / np.sqrt(D) * 2.0
    phi = rng.normal(0, 1, (NP, K, D)) / np.sqrt(D)
    correct = rng.integers(0, K, NP)
    phi[np.arange(NP), correct] += mu[types]
    b = rng.normal(0, 1.5, (NP, K))
    return phi, b, correct

def policy_probs(theta, phi, b, idx):
    z = phi[idx] @ theta + b[idx]; z -= z.max(-1, keepdims=True); e = np.exp(z)
    return e / e.sum(-1, keepdims=True)

def group_advantages(rewards, scale_by_std=True, eps=1e-6):
    """rewards: [P, G]. GRPO: (r - mean_group) / (std_group + eps); Dr. GRPO: r - mean_group."""
    a = rewards - rewards.mean(1, keepdims=True)
    if scale_by_std:
        a = a / (rewards.std(1, keepdims=True) + eps)
    return a

def clipped_surrogate_grad(theta, phi, b, idx, actions, adv, old_p, eps_low=0.2, eps_high=0.2):
    """Gradient (for ascent) of mean_i,g min(rho*A, clip(rho, 1-eps_low, 1+eps_high)*A),
    rho = pi(a)/pi_old(a). Each answer here is one 'token', so token- and sequence-level coincide."""
    p = policy_probs(theta, phi, b, idx)                         # [P, K]
    P_, G = actions.shape
    pa = p[np.arange(P_)[:, None], actions]                      # [P, G]
    rho = pa / old_p
    unclipped = rho * adv
    clipped = np.clip(rho, 1 - eps_low, 1 + eps_high) * adv
    active = (unclipped <= clipped)                              # gradient flows only where min picks the unclipped term
    # d rho / d theta = rho * (phi[a] - E_p[phi])
    phi_sel = phi[idx][np.arange(P_)[:, None], actions]          # [P, G, D]
    phi_bar = np.einsum("pk,pkd->pd", p, phi[idx])[:, None, :]   # [P, 1, D]
    w = (active * rho * adv)[..., None]
    return (w * (phi_sel - phi_bar)).sum((0, 1)) / (P_ * G)

def train(rng, phi, b, correct, steps=200, P_=64, G=8, lr=2.0, inner=2, scale_by_std=True):
    theta = np.zeros(D); log = []
    for step in range(steps + 1):
        if step % 50 == 0:
            acc = policy_probs(theta, phi, b, np.arange(NP))[np.arange(NP), correct].mean()
            log.append((step, acc))
        if step == steps: break
        idx = rng.choice(NP, P_, replace=False)
        p = policy_probs(theta, phi, b, idx)
        actions = np.array([rng.choice(K, G, p=pi) for pi in p])
        rewards = (actions == correct[idx][:, None]).astype(float)   # the verifier: 1 if correct
        adv = group_advantages(rewards, scale_by_std)
        old_p = p[np.arange(P_)[:, None], actions]
        if step == 0: zero_first = np.mean(rewards.std(1) == 0)
        for _ in range(inner):
            theta += lr * clipped_surrogate_grad(theta, phi, b, idx, actions, adv, old_p)
    zero_last = np.mean(rewards.std(1) == 0)
    return theta, log, zero_first, zero_last

if __name__ == "__main__":
    phi, b, correct = make_task(np.random.default_rng(0))
    for name, s in [("GRPO (std-scaled)", True), ("Dr. GRPO (no std)", False)]:
        theta, log, z0, z1 = train(np.random.default_rng(1), phi, b, correct, scale_by_std=s,
                                   lr=0.3 if s else 0.6)
        print(f"{name}: pass@1 " + "  ".join(f"step {t}: {a:.3f}" for t, a in log)
              + f" | zero-advantage groups: first batch {z0:.0%}, last batch {z1:.0%}")
    print("std-scaling weight 1/std for a binary reward with group pass rate q:")
    for q in [0.0625, 0.25, 0.5, 0.75, 0.9375]:
        print(f"  q = {q:.4f}: std {np.sqrt(q * (1 - q)):.3f}  weight {1 / np.sqrt(q * (1 - q)):.2f}")
    assert log[-1][1] > log[0][1] + 0.2
    print("OK")
