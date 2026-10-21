"""Exercise 5.3.2 (solution): Direct Preference Optimization (DPO) on a tabular policy.
Run: python ex_5_3_2_solution.py   (numpy only, ~2 s)
Setup: 8 prompts, K = 6 candidate responses each. The policy is a table of logits theta[prompt, response].
Preference pairs are drawn from the reference policy and labeled with Bradley-Terry on a hidden reward r.
DPO should move the policy toward the KL-regularized optimum pi*(y) ~ pi_ref(y) exp(r(y) / beta)."""
import numpy as np

P, K, BETA = 8, 6, 0.5

def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)

def log_softmax(z):
    z = z - z.max(-1, keepdims=True); return z - np.log(np.exp(z).sum(-1, keepdims=True))

def make_data(rng, n_pairs, ref_logits, r, unseen):
    """Sample pairs (prompt, chosen, rejected) from pi_ref; responses in `unseen` never appear."""
    pi_ref = softmax(ref_logits).copy(); pi_ref[unseen] = 0; pi_ref /= pi_ref.sum(-1, keepdims=True)
    x = rng.integers(0, P, n_pairs)
    a = np.array([rng.choice(K, p=pi_ref[i]) for i in x]); b = np.array([rng.choice(K, p=pi_ref[i]) for i in x])
    keep = a != b; x, a, b = x[keep], a[keep], b[keep]
    a_wins = rng.random(len(x)) < 1 / (1 + np.exp(-(r[x, a] - r[x, b])))
    return x, np.where(a_wins, a, b), np.where(a_wins, b, a)

def dpo_loss_and_grad(theta, ref_logits, x, yw, yl, beta=BETA):
    """Mean DPO loss -log sigmoid(beta * [(log pi(yw) - log ref(yw)) - (log pi(yl) - log ref(yl))])
    and its gradient with respect to the logit table theta."""
    lp, lr = log_softmax(theta), log_softmax(ref_logits)
    h = beta * ((lp[x, yw] - lr[x, yw]) - (lp[x, yl] - lr[x, yl]))
    loss = np.mean(np.log1p(np.exp(-h)))
    coef = -beta * (1 - 1 / (1 + np.exp(-h))) / len(x)        # d loss / d (logratio_w - logratio_l)
    pi = softmax(theta); g = np.zeros_like(theta)
    # d log pi(y) / d theta[x, :] = onehot(y) - pi[x]; the -pi terms cancel between yw and yl
    np.add.at(g, (x, yw), coef); np.add.at(g, (x, yl), -coef)
    return loss, g

def optimum(ref_logits, r, beta=BETA):
    return softmax(log_softmax(ref_logits) + r / beta)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, (P, K)); r = rng.normal(0, 1, (P, K))
    unseen = (np.arange(P), np.argmin(r, axis=1))       # one bad response per prompt never gets sampled
    x, yw, yl = make_data(rng, 30000, ref, r, unseen)
    print(f"{len(x)} preference pairs")
    theta = ref.copy()
    for step in range(3001):
        loss, g = dpo_loss_and_grad(theta, ref, x, yw, yl)
        if step in (0, 300, 3000):
            pi = softmax(theta); tv = 0.5 * np.abs(pi - optimum(ref, r)).sum(-1).mean()
            print(f"step {step:5d}  loss {loss:.4f}  TV(pi, pi*) {tv:.3f}  "
                  f"E[r] pi {np.sum(pi * r, -1).mean():+.3f}  pi* {np.sum(optimum(ref, r) * r, -1).mean():+.3f}  "
                  f"ref {np.sum(softmax(ref) * r, -1).mean():+.3f}")
        theta -= 2.0 * g
    pi = softmax(theta)
    # implicit reward beta*log(pi/ref) recovers r up to a per-prompt constant on responses seen in the data
    imp = BETA * (log_softmax(theta) - log_softmax(ref))
    seen = np.ones((P, K), bool); seen[unseen] = False
    c = [np.corrcoef(imp[i, seen[i]], r[i, seen[i]])[0, 1] for i in range(P)]
    print(f"mean correlation(implicit reward, true r) on seen responses: {np.mean(c):.3f}")
    print(f"mass on never-seen worst responses: ref {softmax(ref)[unseen].mean():.4f}, "
          f"pi* {optimum(ref, r)[unseen].mean():.4f}, DPO {pi[unseen].mean():.4f}")
    assert np.mean(c) > 0.9
    print("OK")
