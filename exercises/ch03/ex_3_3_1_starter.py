"""Exercise 3.3.1 (starter): rotary position embedding (RoPE).
Checks: rotation preserves length; the score q.k depends only on the offset m - n;
wavelengths per frequency pair for theta = 10,000 vs 500,000; position interpolation (PI) keeps
angles inside the trained range.  numpy only, CPU, < 5 s.  Run: python ex_3_3_1_starter.py
"""
import numpy as np

def rope_freqs(d_h, theta):
    """omega_i = theta^(-2i/d_h), i = 0 .. d_h/2 - 1  (radians per position)."""
    # TODO: return theta ** (-(0, 2, 4, ..., d_h-2) / d_h).
    raise NotImplementedError

def apply_rope(x, pos, theta=500_000.0, scale=1.0):
    """x [..., d_h] at integer position pos. Pairs (x[2i], x[2i+1]) rotate by angle (pos/scale)*omega_i."""
    # TODO: angle = (pos/scale) * omega; rotate each pair (x[2i], x[2i+1]) by its angle: (x1 c - x2 s, x1 s + x2 c).
    raise NotImplementedError

if __name__ == "__main__":
    try:
        rng = np.random.default_rng(0)
        d_h = 128
        q, k = rng.standard_normal(d_h), rng.standard_normal(d_h)
        assert np.isclose(np.linalg.norm(apply_rope(q, 777)), np.linalg.norm(q))
        print("norm preserved under rotation")
        # Same offset (m - n = 5) at very different absolute positions -> same score
        s1 = apply_rope(q, 10) @ apply_rope(k, 5)
        s2 = apply_rope(q, 5005) @ apply_rope(k, 5000)
        s3 = apply_rope(q, 12) @ apply_rope(k, 5)
        print(f"score(10,5)={s1:.6f}  score(5005,5000)={s2:.6f}  score(12,5)={s3:.6f}")
        assert np.isclose(s1, s2) and not np.isclose(s1, s3)
        for theta in (10_000.0, 500_000.0):
            lam = 2 * np.pi / rope_freqs(d_h, theta)
            print(f"theta={theta:>9,.0f}: fastest pair wavelength {lam[0]:.2f} positions, slowest {lam[-1]:,.0f} positions")
        # Position interpolation: a model trained to T=8,192, run at 32,768 with scale 4
        om = rope_freqs(d_h, 500_000.0)
        trained_max = 8191 * om[-1]
        print(f"slowest-pair angle: trained max {trained_max:.4f} rad, at 32,767 raw {32767 * om[-1]:.4f} rad, "
              f"with PI scale 4 {32767 / 4 * om[-1]:.4f} rad")
        assert 32767 / 4 <= 8192   # scaled positions stay inside [0, 8192)
        print("OK")
    except NotImplementedError:
        print("Fill in the TODOs in rope_freqs(), apply_rope().")
