"""Exercise 7.2.1 (solution): occupancy and latency hiding on an H100 SM."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import math

# H100 (compute capability 9.0) per-SM limits
MAX_THREADS, MAX_BLOCKS, REGS, SMEM = 2048, 32, 65536, 228 * 1024
WARP, REG_GRAN = 32, 256          # registers are allocated per warp in chunks of 256
SMS, BW, LAT = 132, 3.35e12, 656 / 1.83e9   # HBM B/s; ~656-cycle load latency (measured on Hopper)

def occupancy(threads, regs_per_thread, smem_per_block):
    warps = math.ceil(threads / WARP)
    regs_per_warp = math.ceil(regs_per_thread * WARP / REG_GRAN) * REG_GRAN
    limits = {
        "threads": MAX_THREADS // threads,
        "blocks": MAX_BLOCKS,
        "registers": REGS // (regs_per_warp * warps),
        "shared memory": SMEM // smem_per_block if smem_per_block else MAX_BLOCKS,
    }
    blocks = min(limits.values())
    limiter = min(limits, key=limits.get)
    return blocks, blocks * warps, blocks * warps / (MAX_THREADS // WARP), limiter

def achieved_bw(threads_per_sm, bytes_in_flight_per_thread):
    """Little's law: bandwidth = bytes in flight / latency, capped at peak."""
    inflight = SMS * threads_per_sm * bytes_in_flight_per_thread
    return min(BW, inflight / LAT), inflight

if __name__ == "__main__":
    cases = [("elementwise, 256 thr, 32 reg", 256, 32, 0),
             ("GEMM-like, 256 thr, 128 reg", 256, 128, 0),
             ("attention-like, 384 thr, 168 reg, 200 KB", 384, 168, 200 * 1024),
             ("tile kernel, 128 thr, 64 reg, 48 KB", 128, 64, 48 * 1024)]
    for name, t, r, s in cases:
        b, w, occ, lim = occupancy(t, r, s)
        print(f"{name:42s} blocks/SM {b:2d}  warps {w:2d}  occupancy {occ:5.1%}  limited by {lim}")
    need = BW * LAT
    print(f"\nbytes in flight to saturate HBM: {need/1e6:.2f} MB total, {need/SMS/1e3:.1f} KB per SM")
    for thr, per in [(512, 8), (512, 32), (2048, 8)]:
        bw, inflight = achieved_bw(thr, per)
        print(f"{thr:4d} threads/SM x {per:2d} B in flight: {inflight/1e6:5.2f} MB -> {bw/1e12:.2f} TB/s ({bw/BW:.0%})")
