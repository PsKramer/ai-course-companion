"""Exercise 7.6.1 (solution): datacenter power, energy per token and failure-driven goodput."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import math

PUE = 1.2              # assumption: facility power / IT power for a modern liquid-cooled site
PRICE = 0.08           # assumption: USD 0.08 per kWh industrial electricity price

def cluster_power_mw(n_gpus, it_kw_per_gpu, pue=PUE):
    """Facility power in MW: IT power per GPU (GPU + its share of CPUs, NICs, switches) times PUE."""
    return n_gpus * it_kw_per_gpu * pue / 1e3

def energy_mwh(gpu_hours, it_kw_per_gpu, pue=PUE):
    return gpu_hours * it_kw_per_gpu * pue / 1e3

def joules_per_token(power_w, tokens_per_s):
    return power_w / tokens_per_s

def job_mtbf_h(n_gpus, per_gpu_rate_per_h):
    """Mean time between failures of a synchronous job: 1 / (N * lambda)."""
    return 1.0 / (n_gpus * per_gpu_rate_per_h)

def young_daly(delta_s, mtbf_s, restart_s=0.0):
    """Optimal checkpoint interval tau = sqrt(2 delta M) and the fraction of time lost:
    checkpoint overhead delta/tau + expected rework tau/(2M) + restart R/M."""
    tau = math.sqrt(2 * delta_s * mtbf_s)
    waste = delta_s / tau + tau / (2 * mtbf_s) + restart_s / mtbf_s
    return tau, min(waste, 1.0)

if __name__ == "__main__":
    # IT power per GPU from node maximums: DGX B200 14.3 kW and DGX H100 10.2 kW, each for 8 GPUs
    b200_kw, h100_kw = 14.3 / 8, 10.2 / 8
    for n in (16_384, 100_000):
        print(f"{n:>7,} B200 GPUs: {cluster_power_mw(n, b200_kw):6.1f} MW at PUE {PUE}")

    gh = 0.51e6                                  # Course-8B 15T-token pretrain, H100-hours at 40% MFU (1.4)
    e = energy_mwh(gh, h100_kw)
    print(f"\nCourse-8B pretraining: {e:,.0f} MWh, electricity USD {e*1e3*PRICE:,.0f} at USD {PRICE}/kWh")

    tps = 64 / 14.52e-3                          # Course-8B decode, B=64, T=4096 on one H100 (7.1)
    print(f"Decode energy: GPU only {joules_per_token(700, tps):.3f} J/token; "
          f"all-in {joules_per_token(h100_kw*1e3*PUE, tps):.3f} J/token ({tps:,.0f} tokens/s)")

    lam = 419 / (16_384 * 54 * 24)               # Llama 3: 419 unexpected interruptions, 54 days, 16,384 GPUs
    print(f"\nPer-GPU failure rate {lam:.2e}/h (per-GPU MTBF {1/lam:,.0f} h = {1/lam/8766:.1f} years)")
    print("   GPUs   job MTBF   ckpt 10 s: tau / waste    ckpt 60 s: tau / waste    ckpt 300 s: tau / waste")
    for n in (1_024, 16_384, 100_000, 300_000):
        m = job_mtbf_h(n, lam)
        row = f"{n:>7,} {m*60:8.0f} min "
        for d in (10, 60, 300):
            tau, w = young_daly(d, m * 3600, restart_s=600)     # assumption: 10 min to restart
            row += f"   {tau/60:6.1f} min / " + (f"{w:5.1%}" if w < 1 else "100%+") + "      "
        print(row)
