"""Exercise 2.5.2 (starter): turn an image into patch tokens and audio-like frames into residual-VQ codes.
numpy only, CPU, < 30 s.  Fill in the four TODOs, then run: python ex_2_5_2_starter.py
"""
import numpy as np

def patchify(img, p):
    """img [H, W, C] -> patches [(H/p)*(W/p), p*p*C], row-major over the patch grid (ViT order)."""
    # TODO 1: reshape to [gh, p, gw, p, C], move the two grid axes to the front, flatten each patch.
    raise NotImplementedError

def merge_2x2(tokens, gh, gw):
    """Concatenate each 2x2 block of neighbouring patch vectors: [gh*gw, k] -> [gh*gw/4, 4k] (Qwen-VL style)."""
    # TODO 2: same trick one level up: group the [gh, gw] grid of k-vectors into 2x2 blocks and concatenate.
    raise NotImplementedError

def kmeans(x, K, iters=15, rng=None):
    rng = rng or np.random.default_rng(0)
    C = x[rng.choice(len(x), K, replace=False)].copy()
    for _ in range(iters):
        a = ((x[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        for j in range(K):
            if (a == j).any():
                C[j] = x[a == j].mean(0)
    return C

def rvq_train(x, n_stages, K):
    """Residual VQ: stage s quantizes what stages 0..s-1 failed to explain."""
    # TODO 3: for each stage, fit kmeans(r, K) to the current residual r, quantize r to its nearest centroid,
    #         subtract, and keep the codebook. Return the list of codebooks.
    raise NotImplementedError

def rvq_encode(x, books):
    # TODO 4: same loop with fixed codebooks; return integer codes of shape [n_frames, n_stages].
    raise NotImplementedError

def rvq_decode(codes, books):
    return sum(C[codes[:, s]] for s, C in enumerate(books))

if __name__ == "__main__":
  try:
      rng = np.random.default_rng(0)
      # --- images: 448 x 448 RGB, 16-pixel patches, 2x2 merge (Qwen3-VL-style numbers) ---
      img = rng.random((448, 448, 3), dtype=np.float32)
      P = patchify(img, 16)
      gh = gw = 448 // 16
      M = merge_2x2(P, gh, gw)
      W = rng.standard_normal((M.shape[1], 4096)).astype(np.float32) / np.sqrt(M.shape[1])  # projection to d=4096
      print("patches", P.shape, "-> merged", M.shape, "-> LLM tokens", (M @ W).shape)
      assert np.allclose(P[1], img[0:16, 16:32].reshape(-1))    # patch 1 = top row, second column
      # --- audio-like frames: 8-dim latent per frame; 60 s at 12.5 frames/s = 750 frames ---
      scale = np.array([3, 2, 1.5, 1, .8, .5, .3, .2], np.float32)
      train = rng.standard_normal((3000, 8)).astype(np.float32) * scale     # codec training data
      frames = rng.standard_normal((750, 8)).astype(np.float32) * scale     # 60 s of held-out "audio"
      books = rvq_train(train, n_stages=8, K=64)
      codes = rvq_encode(frames, books)
      print("codes", codes.shape, "(frames x codebooks)")
      var = frames.var()
      for n in (1, 2, 4, 8):
          err = ((frames - rvq_decode(codes[:, :n], books[:n])) ** 2).mean() / var
          print(f"  {n} codebooks: relative MSE {err:.3f}, bitrate {12.5 * n * np.log2(64):.0f} bit/s")
      print("OK")
  except NotImplementedError:
    print("Fill in the TODOs in patchify(), merge_2x2(), rvq_train() and rvq_encode().")
