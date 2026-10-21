const WIDGETS=[
{id:'ch01-pipeline-map',title:'The full pipeline, stage by stage',desc:'Ch1 — tap a stage to see its input, output artifact, dominant cost, and knobs.'},
{id:'ch01-train-cost',title:'Training compute & cost: C ≈ 6·N·D',desc:'Ch1 — FLOPs, GPU-hours, days, and dollars for a pre-training run.'},
{id:'ch01-decode-roofline',title:'One decode step: memory- or compute-bound?',desc:'Ch1 — bytes vs FLOPs per decode step, KV cache, and the batch-size knee.'},
{id:'ch02-minhash-lsh',title:'Near-duplicate detection: MinHash + LSH',desc:'Ch2 (2.1): shingles, Jaccard similarity, MinHash signatures, banding S-curve.'},
{id:'ch02-bpe-stepper',title:'Byte-pair encoding, one merge at a time',desc:'Ch2 (2.3): train byte-level BPE step by step and watch tokens fuse.'},
{id:'ch08-test-time-budget',title:'Test-time budget: best-of-n sampling',desc:'Ch8 (8.1): coverage, waves, wall-clock and cost of parallel samples.'},
{id:'ch08-agent-context-cost',title:'Agent context cost per turn',desc:'Ch8 (8.2): prefill and decode cost of a multi-turn agent, with and without a prefix cache.'},
{id:'ch08-efficiency-stack',title:'The efficiency stack',desc:'Ch8 (8.6): weights format, KV format, drafting and hybrid layers, lever by lever.'}];
const ANIMS=[{id:'ch01-life-of-token',title:'Life of a token (prefill, decode, KV cache growth)'}];
