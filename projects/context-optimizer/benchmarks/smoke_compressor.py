"""Smoke test for chunk logger + parallelism gate."""
import json
import os
import sys

sys.path.insert(0, "src")
from context_optimizer.compressor import _effective_workers, _log_chunk

# ── Parallelism gate ──────────────────────────────────────────────────────────
os.environ.pop("OLLAMA_NUM_PARALLEL", None)
w = _effective_workers(4, explicit_llm=False)
assert w == 1, f"Expected 1, got {w}"
print(f"Gate (no env):          4 requested -> {w} effective [OK]")

os.environ["OLLAMA_NUM_PARALLEL"] = "4"
w = _effective_workers(4, explicit_llm=False)
assert w == 4, f"Expected 4, got {w}"
print(f"Gate (OLLAMA_NUM_P=4):  4 requested -> {w} effective [OK]")

os.environ.pop("OLLAMA_NUM_PARALLEL", None)
os.environ["CONTEXT_OPTIMIZER_COMPRESSOR_PROVIDER"] = "groq"
w = _effective_workers(4, explicit_llm=False)
assert w == 4, f"Expected 4, got {w}"
print(f"Gate (groq provider):   4 requested -> {w} effective [OK]")
os.environ.pop("CONTEXT_OPTIMIZER_COMPRESSOR_PROVIDER", None)

# explicit_llm always passes through
w = _effective_workers(8, explicit_llm=True)
assert w == 8
print(f"Gate (explicit llm):    8 requested -> {w} effective [OK]")

# ── Chunk logger ──────────────────────────────────────────────────────────────
tmp = "benchmarks/data/_smoke_test.jsonl"
os.makedirs("benchmarks/data", exist_ok=True)
os.environ["COMPRESSOR_LOG_FILE"] = tmp

_log_chunk("chunk_000001", "moby-dick", 2.345, 512, 87, 0.17)
_log_chunk("chunk_000002", "moby-dick", 3.1,   480, 92, 0.19, error=None)
_log_chunk("chunk_000003", "dracula",   9.9,   512,  0, 0.0,  error="timeout")

lines = open(tmp, encoding="utf-8").readlines()
assert len(lines) == 3, f"Expected 3 log lines, got {len(lines)}"
rec = json.loads(lines[0])
assert rec["chunk_id"] == "chunk_000001"
assert rec["elapsed_s"] == 2.345
assert rec["label"] == "moby-dick"
assert "error" not in rec
rec3 = json.loads(lines[2])
assert rec3["error"] == "timeout"
os.remove(tmp)
print(f"Logger: 3 records written and verified [OK]")

print("\nAll smoke tests passed.")
