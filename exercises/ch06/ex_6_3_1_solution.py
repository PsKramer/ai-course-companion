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
    t, finish = 0.0, np.zeros(R)
    for s in range(0, R, SLOTS):
        idx = list(range(s, min(s + SLOTS, R)))
        t += prefill_s(prompts[idx])
        n_steps = outputs[idx].max()
        for k in range(1, n_steps):          # first token came from prefill
            # every slot stays in the batch until the longest finishes (padding)
            t += decode_step_s([prompts[i] + k for i in idx])
            for i in idx:
                if k == outputs[i] - 1:
                    finish[i] = t
        for i in idx:
            if outputs[i] == 1:
                finish[i] = t
    return t, finish

def continuous_batching():
    t, finish = 0.0, np.zeros(R)
    queue = list(range(R))
    active = {}                               # request -> tokens generated so far
    while queue or active:
        joiners = []
        while queue and len(active) + len(joiners) < SLOTS:
            joiners.append(queue.pop(0))
        if joiners:                           # prefill-first policy: one prefill step for all joiners
            t += prefill_s(prompts[joiners])
            for i in joiners:
                active[i] = 1
        done = [i for i, g in active.items() if g >= outputs[i]]
        for i in done:
            finish[i] = t
            del active[i]
        if not active:
            continue
        t += decode_step_s([prompts[i] + g for i, g in active.items()])
        for i in active:
            active[i] += 1
    return t, finish

for name, fn in [("static", static_batching), ("continuous", continuous_batching)]:
    total, fin = fn()
    print(f"{name:10s} total {total:6.1f} s  throughput {outputs.sum() / total:6.0f} tok/s  "
          f"mean E2E {fin.mean():5.1f} s  p50 {np.median(fin):5.1f} s")
