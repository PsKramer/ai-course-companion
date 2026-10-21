"""Exercise 5.1.1 (starter): chat template, loss mask and loss averaging for supervised fine-tuning (SFT).
Run: python ex_5_1_1_solution.py   (numpy only, < 1 s)"""
import numpy as np

# A tiny chat template in the style of current open models. Special markers are single tokens.
SPECIAL = {"<|bos|>": 256, "<|system|>": 257, "<|user|>": 258, "<|assistant|>": 259, "<|end|>": 260}
VOCAB = 261  # 256 byte tokens + 5 special tokens

def render(messages):
    """Return a list of (text, trainable) segments for one conversation.
    Only assistant content and the <|end|> that closes an assistant turn are trainable."""
    segs = [("<|bos|>", False)]
    for m in messages:
        segs.append((f"<|{m['role']}|>", False))
        is_asst = m["role"] == "assistant"
        segs.append((m["content"], is_asst))
        segs.append(("<|end|>", is_asst))     # the model must learn to stop
    return segs

def encode(text):
    if text in SPECIAL:
        return [SPECIAL[text]]
    return list(text.encode("utf-8"))           # byte-level tokens 0..255

def tokenize_with_mask(segs):
    """Return (ids, mask) with mask[i] = 1 where token i is a training *target*.
    Inputs are ids[:-1], targets ids[1:], so the mask is aligned to targets."""
    # TODO: encode each segment, build ids and a per-token trainable flag, return ids and the flags shifted to align with targets ids[1:]
    raise NotImplementedError("TODO: tokenize_with_mask")

def masked_ce(logits, targets, mask, mode="token"):
    """Cross-entropy for a batch of sequences.
    logits: list of [T_i, V] arrays, targets: list of [T_i], mask: list of [T_i].
    mode='token': sum of masked token losses / total masked tokens in the batch.
    mode='sequence': mean over sequences of (masked mean within the sequence)."""
    # TODO: log-softmax, gather target log-probs, apply the mask; 'token' divides by all masked tokens in the batch, 'sequence' averages per-sequence means
    raise NotImplementedError("TODO: masked_ce")

def token_weights(mask, mode):
    """Weight each trainable token gets in the batch loss (for comparing the two modes)."""
    n = np.array([m.sum() for m in mask])
    if mode == "token":
        return [m / n.sum() for m in mask]
    return [m / (len(mask) * k) for m, k in zip(mask, n)]

if __name__ == "__main__":
    convs = [
        [{"role": "system", "content": "You are a concise assistant."},
         {"role": "user", "content": "What is 17 * 3?"},
         {"role": "assistant", "content": "51."}],
        [{"role": "user", "content": "Explain gradient clipping in one paragraph."},
         {"role": "assistant", "content": "Gradient clipping rescales the full gradient vector whenever its norm "
          "exceeds a threshold, usually 1.0, so one bad batch cannot produce a huge update. It keeps the "
          "direction and caps only the length, which is why it is cheap and almost always on."}],
    ]
    rng = np.random.default_rng(0)
    ids_l, tg_l, m_l, lg_l = [], [], [], []
    for c in convs:
        ids, m = tokenize_with_mask(render(c))
        ids_l.append(ids); tg_l.append(ids[1:]); m_l.append(m)
        lg_l.append(rng.normal(size=(len(ids) - 1, VOCAB)))
        print(f"tokens {len(ids):4d}  trainable targets {int(m.sum()):4d}  ({m.mean():.0%})")
    # sanity: the first trainable target is the first byte of the assistant reply
    first = int(np.argmax(m_l[0])); assert tg_l[0][first] == ord("5") and m_l[0][first - 1] == 0
    assert tg_l[0][np.nonzero(m_l[0])[0][-1]] == SPECIAL["<|end|>"]
    for mode in ("token", "sequence"):
        w = token_weights(m_l, mode)
        print(f"{mode:8s}-mean: loss {masked_ce(lg_l, tg_l, m_l, mode):.3f}; weight per token: short reply "
              f"{w[0][m_l[0] > 0][0]:.4f}, long reply {w[1][m_l[1] > 0][0]:.4f}")
    # unmasked training: fraction of the gradient spent on prompt and template tokens
    all_t = sum(len(m) for m in m_l); trn = sum(m.sum() for m in m_l)
    print(f"without the mask, {1 - trn / all_t:.0%} of target tokens would be prompt/template tokens")
    print("OK")
