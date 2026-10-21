"""Exercise 5.7.1 (starter): off-policy (forward KL) versus on-policy (reverse KL) distillation.
Run: python ex_5_7_1_solution.py   (numpy only, ~3 s)
Setup: sequences of 2 tokens over a vocabulary of 6. The teacher puts 45% on 'answer A' = (0, 1) and 45% on
'answer B' = (3, 4), and spreads 10% over everything else. The student is too small to model the dependence
between tokens: it picks token 1 and token 2 independently, q(x1, x2) = q1(x1) q2(x2)."""
import numpy as np

V = 6

def teacher():
    p = np.full((V, V), 0.10 / (V * V - 2))
    p[0, 1] = p[3, 4] = 0.45
    return p                                            # joint p(x1, x2), sums to 1

def softmax(z):
    e = np.exp(z - z.max()); return e / e.sum()

def student_joint(a, b):
    return np.outer(softmax(a), softmax(b))

def kl(p, q):
    m = p > 0; return float((p[m] * np.log(p[m] / q[m])).sum())

def forward_kl_fit(p):
    """Off-policy distillation = SFT on teacher samples = minimize KL(p || q). For a factorized
    student the optimum is the product of the teacher's marginals (log of marginals as logits)."""
    # TODO: the factorized student's forward-KL optimum is the teacher's marginals; return their logs as logits
    raise NotImplementedError("TODO: forward_kl_fit")

def on_policy_reverse_kl(p, rng, steps=3000, batch=64, lr=0.5):
    """Sample from the student; each sampled token's 'reward' is the per-token negative reverse KL,
    log p_teacher(x_t | x_<t) - log q(x_t | x_<t); update the student with REINFORCE (no critic)."""
    # TODO: loop: sample x1, x2 from the student; per-token rewards log p_teacher(x_t|x_<t) - log q(x_t); REINFORCE with a batch-mean baseline; token 1's return includes token 2's reward
    raise NotImplementedError("TODO: on_policy_reverse_kl")

def coherent(q):
    return q[0, 1] + q[3, 4]

if __name__ == "__main__":
    p = teacher()
    qf = student_joint(*forward_kl_fit(p))
    qr = student_joint(*on_policy_reverse_kl(p, np.random.default_rng(0)))
    print("student            P(coherent answer)  P(mixed 0,4 or 3,1)  KL(p||q)  KL(q||p)")
    for name, q in [("forward KL (SFT)", qf), ("reverse KL (on-policy)", qr)]:
        print(f"{name:22s} {coherent(q):10.3f} {q[0, 4] + q[3, 1]:18.3f} {kl(p, q):11.3f} {kl(q, p):9.3f}")
    print(f"reverse-KL student: P(A) = {qr[0, 1]:.3f}, P(B) = {qr[3, 4]:.3f} (it commits to one answer)")
    print(f"teacher                {coherent(p):10.3f} {p[0, 4] + p[3, 1]:18.3f}")
    assert coherent(qr) > 0.7 and qf[0, 4] + qf[3, 1] > 0.3
    print("OK")
