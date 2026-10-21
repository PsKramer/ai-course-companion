"""Exercise 4.2.2 (starter): fit L(N, D) = E + A/N^alpha + B/D^beta to isoFLOP runs and
read off the compute-optimal allocation. Synthetic 'runs' are drawn from the Epoch AI
replication fit (Besiroglu et al., 2024) plus noise, so we know the right answer."""
import math, numpy as np, torch
torch.manual_seed(0); torch.set_num_threads(1); rng = np.random.default_rng(0)
TRUE = dict(E=1.8172, A=482.01, B=2085.43, alpha=0.3478, beta=0.3658)

def law(N, D, E, A, B, alpha, beta):
    return E + A / N**alpha + B / D**beta

# ---- 1. synthetic isoFLOP sweep: 5 budgets x 9 model sizes, D = C / (6N) ----
runs = []
for C in [1e18, 1e19, 1e20, 1e21, 1e22]:
    N_opt_guess = math.sqrt(C / 120)                       # ~20 tokens/param, from 1.4
    for N in N_opt_guess * np.logspace(-1, 1, 9):
        D = C / (6 * N)
        L = law(N, D, **TRUE) * math.exp(rng.normal(0, 0.005))   # 0.5% multiplicative noise
        runs.append((N, D, L))
N, D, Lobs = (torch.tensor(c, dtype=torch.float64) for c in zip(*runs))

# ---- 2. fit in log space with a Huber loss (the Chinchilla/Epoch procedure) ----
def fit(init):
    # parameters: e = log E, a = log A, b = log B, alpha, beta
    # TODO: fit (log E, log A, log B, alpha, beta) by minimising a Huber loss on log L; return (final_loss, params)
    raise NotImplementedError("TODO: fit")

best = min((fit([e, a, b, al, be]) for e in (0.0, 0.5) for a in (4.0, 6.0) for b in (5.0, 8.0)
            for al in (0.2, 0.4) for be in (0.2, 0.4)), key=lambda t: t[0])
e, a, b, al, be = best[1].tolist()
E, A, B = math.exp(e), math.exp(a), math.exp(b)
print(f"fit: E={E:.3f} A={A:.1f} B={B:.1f} alpha={al:.3f} beta={be:.3f}")
print(f"true: E={TRUE['E']} A={TRUE['A']} B={TRUE['B']} alpha={TRUE['alpha']} beta={TRUE['beta']}")

# ---- 3. compute-optimal allocation from the fit (derived in 4.2) ----
def optimum(C, E, A, B, alpha, beta):
    # TODO: closed-form N*(C) and D*(C) derived in 4.2
    raise NotImplementedError("TODO: optimum")
for C in [7.2e23, 1e25]:
    Nf, Df = optimum(C, E, A, B, al, be); Nt, Dt = optimum(C, **TRUE)
    print(f"C={C:.1e}: fit N*={Nf/1e9:.0f}B D*={Df/1e12:.2f}T ({Df/Nf:.0f} tok/param) | true N*={Nt/1e9:.0f}B ({Dt/Nt:.0f} tok/param)")
print(f"N* grows as C^{be/(al+be):.3f} (true {TRUE['beta']/(TRUE['alpha']+TRUE['beta']):.3f})")
