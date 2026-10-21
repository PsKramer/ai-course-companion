"""Exercise 8.2.1 starter (fill the two TODOs; solution in ex_8_2_1_solution.py): token accounting for a 40-turn agent episode on Course-8B-128k (one H100, B=1)."""
N, KV_BYTES, BW, PEAK = 8.03e9, 131_072, 3.35e12, 989e12
W_BYTES, MFU_PREFILL, CTX_LIMIT = 2 * N, 0.5, 131_072

def episode(turns=40, system=4_000, tool_out=1_700, model_out=300, cache=True):
    ctx, prefill_tok, decode_s, prefill_s = system, 0, 0.0, 0.0
    cached = 0                                   # tokens whose KV is already resident
    for t in range(turns):
        new = None  # TODO: tokens that must be prefilled this turn (with and without cache)
        prefill_tok += new
        prefill_s += 2 * N * new / (MFU_PREFILL * PEAK)
        for i in range(model_out):               # memory-bound decode of the model's reply
            decode_s += (W_BYTES + (ctx + i) * KV_BYTES) / BW
        ctx += model_out + tool_out              # reply + tool result join the context
        cached = None  # TODO: which tokens' KV is resident at the start of next turn?
        if ctx > CTX_LIMIT:
            return dict(turns_done=t + 1, overflow=True, ctx=ctx)
    return dict(turns_done=turns, overflow=False, ctx=ctx, prefill_tok=prefill_tok,
                prefill_s=prefill_s, decode_s=decode_s, decode_tok=turns * model_out,
                kv_gb=ctx * KV_BYTES / 1e9)

if __name__ == "__main__":
    for cache in (False, True):
        r = episode(cache=cache)
        print(f"prefix cache={cache}: prefill {r['prefill_tok']:,} tok ({r['prefill_s']:.1f} s), "
              f"decode {r['decode_tok']:,} tok ({r['decode_s']:.1f} s), final ctx {r['ctx']:,} tok, KV {r['kv_gb']:.1f} GB")
    # how many turns fit in the 128k window?
    t = 1
    while not episode(turns=t, cache=True)["overflow"]:
        t += 1
    print(f"turn at which the 131,072-token window overflows: {t}")
    print(f"Course-8B's native 8,192-token window holds {(8_192 - 4_000) // 2_000} full turns after the system prompt")
