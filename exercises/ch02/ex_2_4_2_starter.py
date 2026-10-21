"""Exercise 2.4.2 (starter): how vocabulary size trades sequence length against embedding size.
Trains the byte-level BPE from Exercise 2.3.1 at several vocab sizes on English prose (Python standard-library
docstrings, available on any machine) and measures compression on held-out docstrings and on a German passage.
Fill in the two TODOs, then run: python ex_2_4_2_starter.py   (CPU, ~1 min)
"""
import ast, os, glob, time
from ex_2_3_1_solution import train_bpe, encode

def docstring_corpus():
    files = sorted(glob.glob(os.path.join(os.path.dirname(os.__file__), "*.py")))
    out = []
    for f in files:
        try:
            tree = ast.parse(open(f, encoding="utf-8").read())
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
                d = ast.get_docstring(node)
                if d and len(d) > 80:
                    out.append(d)
    return out

GERMAN = ("Die Tokenisierung zerlegt einen Text in kleinere Einheiten. Ein Sprachmodell sieht niemals Buchstaben, "
          "sondern nur ganze Zahlen aus einem festen Vokabular. Wenn das Vokabular hauptsaechlich mit englischen "
          "Texten trainiert wurde, werden deutsche Woerter in viele kurze Stuecke zerschnitten. ") * 3

def bytes_per_token(text, merges):
    # TODO 1: UTF-8 bytes divided by the number of BPE tokens (use encode() from Exercise 2.3.1).
    raise NotImplementedError

def embedding_share(V, d=4096, N_other=8.03e9 - 2 * 128256 * 4096):
    """Course-8B with vocabulary V: embedding + unembedding params (2*V*d, untied) as a share of N."""
    # TODO 2: return (embedding + unembedding parameter count, its share of the total N).
    raise NotImplementedError

if __name__ == "__main__":
  try:
      docs = docstring_corpus()
      train = "\n".join(docs[0::2])[:400_000]          # ~400 KB of English prose for training
      test = "\n".join(docs[1::2])[:100_000]           # held-out English
      print(f"train {len(train.encode())/1e3:.0f} KB, test {len(test.encode())/1e3:.0f} KB")
      t0 = time.time()
      merges_full, _ = train_bpe(train, 256 + 2000)    # merge list is nested: the first m merges = vocab 256+m
      print(f"trained 2000 merges in {time.time()-t0:.0f} s")
      print(f"{'V':>6} {'EN bytes/token':>15} {'DE bytes/token':>15}")
      for m in (0, 100, 500, 1000, 2000):
          merges = dict(list(merges_full.items())[:m])
          print(f"{256+m:>6} {bytes_per_token(test, merges):>15.2f} {bytes_per_token(GERMAN, merges):>15.2f}")
      print("\nCourse-8B embedding + unembedding (d = 4096, untied):")
      for V in (32_000, 128_256, 262_144):
          emb, share = embedding_share(V)
          print(f"  V = {V:>7,}: {emb/1e9:.2f}B params = {share*100:.1f}% of N, {2*emb/1e9:.1f} GB in bf16")
      print("OK")
  except NotImplementedError:
    print("Fill in the TODOs in bytes_per_token() and embedding_share().")
