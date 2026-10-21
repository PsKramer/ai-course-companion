import numpy as np
N, L, d = 8.03e9, 32, 4096
BW, PEAK, S_KV = 3.35e12, 989e12, 131072

def decode_step_s(contexts):
    """One decode step for sequences with the given context lengths (roofline, seconds)."""
    B = len(contexts)
    return max((2 * N + S_KV * sum(contexts)) / BW, 2 * N * B / PEAK)

def prefill_s(prompt_lens, mfu=0.5):
    """Prefill a group of prompts in one pass (seconds)."""
    flops = sum(2 * N * t + 2 * L * d * t * t for t in prompt_lens)
    return max(flops / (mfu * PEAK), 2 * N / BW)

rng = np.random.default_rng(0)
R, SLOTS = 256, 32
prompts = rng.integers(128, 2049, R)
outputs = rng.integers(50, 1001, R)

def static_batching():
    """Groups of SLOTS requests: prefill the group, then decode until the LONGEST output finishes.
    Finished sequences keep their slot (and their KV is still read) until the group ends.
    Return (total_seconds, finish_time_per_request)."""
    t, finish = 0.0, np.zeros(R)
    # TODO
    return t, finish

def continuous_batching():
    """Iteration-level batching: whenever a slot frees, the next queued request joins.
    Policy: run one prefill step for all joiners, then one decode step for all active sequences.
    Return (total_seconds, finish_time_per_request)."""
    t, finish = 0.0, np.zeros(R)
    # TODO
    return t, finish

for name, fn in [("static", static_batching), ("continuous", continuous_batching)]:
    total, fin = fn()
    print(f"{name:10s} total {total:6.1f} s  throughput {outputs.sum() / total:6.0f} tok/s  "
          f"mean E2E {fin.mean():5.1f} s  p50 {np.median(fin):5.1f} s")
