"""Quick smoke test: measure BART-large-cnn per-block latency on this CPU."""

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

os.environ["CONTEXT_OPTIMIZER_COMPRESSOR_PROVIDER"] = "hf"
os.environ["CONTEXT_OPTIMIZER_COMPRESSOR_MODEL"] = "facebook/bart-large-cnn"

from context_optimizer.compressor import _build_local_llm

print("Building LLM...", flush=True)
llm = _build_local_llm()
print(f"LLM: {type(llm).__name__}  model={getattr(llm,'_model_name','?')}", flush=True)

corpus = open(
    r"c:\repos\ai-portfolio\projects\context-optimizer\benchmarks\data\corpus\gutenberg_combined.txt",
    encoding="utf-8",
    errors="replace",
).read(3200)
print(f"Input sample: {len(corpus)} chars", flush=True)

for i in range(3):
    t = time.perf_counter()
    r = llm.invoke(corpus)
    elapsed = time.perf_counter() - t
    print(f"Block {i+1}: {elapsed:.1f}s  output={r.content[:100]!r}", flush=True)

print("Done.", flush=True)
