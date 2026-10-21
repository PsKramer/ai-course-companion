"""Exercise 4.6.1 (solution): (a) a loss-spike detector with a PaLM-style rollback policy on a
synthetic training log, (b) the Young/Daly checkpoint interval. numpy only."""
import math, numpy as np
rng = np.random.default_rng(1)

# ---- synthetic log: power-law loss + noise, grad-norm, and three injected spikes ----
steps = 20_000
s = np.arange(1, steps + 1)
loss = 2.0 + 6.0 * s ** -0.35 + rng.normal(0, 0.01, steps)
gnorm = 0.3 + 2.0 * s ** -0.5 + np.abs(rng.normal(0, 0.03, steps))
for start, height, length in [(6_000, 0.8, 120), (12_500, 0.15, 40), (17_000, 2.5, 400)]:
    bump = height * np.exp(-np.arange(length) / (length / 4))
    loss[start:start + length] += bump
    gnorm[start - 20:start + length] *= 1 + 8 * np.exp(-np.abs(np.arange(-20, length)) / 15)  # grad norm leads the loss

def detect_spikes(x, window=200, k=6.0, min_len=5):
    """Flag steps where x exceeds rolling median + k * rolling MAD (robust to the slow trend)."""
    flags = np.zeros(len(x), bool)
    for i in range(window, len(x)):
        ref = x[i - window:i]
        med = np.median(ref); mad = np.median(np.abs(ref - med)) * 1.4826 + 1e-9
        flags[i] = x[i] > med + k * mad
    events, i = [], 0                                   # merge flagged steps into events
    while i < len(x):
        if flags[i]:
            j = i
            while j < len(x) and flags[j]: j += 1
            if j - i >= min_len: events.append((i, j))
            i = j
        else: i += 1
    return events

def policy(event, loss, ckpt_every=500, recover_tol=0.02):
    """Decide: ride it out if the loss returns to its pre-spike level quickly, else roll back."""
    a, b = event
    pre = np.median(loss[a - 200:a])
    peak = loss[a:b].max()
    recovered = np.any(np.abs(loss[b:b + 300] - pre) < recover_tol)
    if peak - pre < 0.3 and recovered:
        return f"ride out (peak +{peak - pre:.2f}, recovered)"
    ckpt = (a - 100) // ckpt_every * ckpt_every       # PaLM: restart ~100 steps before the spike
    return f"roll back to step {ckpt}, skip batches {ckpt}..{b + 100}, consider lower LR (peak +{peak - pre:.2f})"

loss_events = detect_spikes(loss)
gn_events = detect_spikes(gnorm)
print("loss spikes:", loss_events)
print("grad-norm alarms:", gn_events)
for ev in loss_events:
    lead = [g for g in gn_events if g[0] <= ev[0] <= g[1] + 50]
    print(f"spike at {ev[0]}: grad-norm alarm {ev[0] - lead[0][0] if lead else 'n/a'} steps earlier -> {policy(ev, loss)}")

# ---- (b) Young/Daly: optimal checkpoint interval tau = sqrt(2 * delta * M) ----
def young_daly(delta_s, mtbf_s, restart_s):
    tau = math.sqrt(2 * delta_s * mtbf_s)
    waste = delta_s / tau + tau / (2 * mtbf_s) + restart_s / mtbf_s
    return tau, waste
rate = 419 / (16_384 * 54)                  # unexpected interruptions per GPU-day (Llama 3 405B, 54-day window)
for name, n_gpu, delta in [("Course-8B, 512 GPUs", 512, 30), ("405B-class, 16,384 GPUs", 16_384, 60)]:
    mtbf = 86_400 / (rate * n_gpu)
    tau, w = young_daly(delta, mtbf, restart_s=600)
    print(f"{name}: job MTBF {mtbf/3600:.1f} h, checkpoint every {tau/60:.0f} min, lost time {w:.1%}")
