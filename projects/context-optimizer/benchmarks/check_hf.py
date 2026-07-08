"""Time flan-t5-base and flan-t5-small — the real CPU-fast options."""
import sys, time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "src"))

sample = (
    "It is a truth universally acknowledged, that a single man in possession of a "
    "good fortune must be in want of a wife. However little known the feelings or views "
    "of such a man may be on his first entering a neighbourhood, this truth is so well "
    "fixed in the minds of the surrounding families, that he is considered as the "
    "rightful property of some one or other of their daughters. "
    * 8
)

for model_id in ["google/flan-t5-base", "google/flan-t5-small"]:
    from context_optimizer.providers.hf_summarizer import build
    print(f"\n=== {model_id} ===")
    t0 = time.perf_counter()
    llm = build(model=model_id)
    # warm-up call (first PyTorch call compiles kernels)
    _ = llm.invoke(sample[:500])
    warm = time.perf_counter() - t0
    print(f"Warmup : {warm:.1f}s")

    # timed call
    t0 = time.perf_counter()
    r = llm.invoke(sample)
    elapsed = time.perf_counter() - t0
    print(f"Time   : {elapsed:.2f}s  |  {len(r.content.split())} words")
    print(f"Output : {r.content[:200]}")

    for corpus_mb, block_mb in [(18, 0.1), (100, 0.5), (400, 0.5)]:
        n_blocks = int(corpus_mb / block_mb)
        t_min = n_blocks * elapsed / 60
        print(f"  {corpus_mb:>4} MB / {block_mb} MB blocks = {n_blocks:,} blocks  ~{t_min:.0f} min ingestion")

    # Reset for next model (lazy-loaded internally)
    del llm
    import gc; gc.collect()
