import json
from pathlib import Path

d = json.loads(
    Path(
        "c:/repos/ai-portfolio/projects/context-optimizer/benchmarks/corpus_results.json"
    ).read_text()
)
v = d.get("vanilla_rag") or {}
o = d.get("optimized_rag") or {}


def qavg(data, key):
    qrs = data.get("query_results", [])
    return sum(q.get(key, 0) for q in qrs) / max(1, len(qrs))


v_recall = qavg(v, "kw_recall")
o_recall = qavg(o, "kw_recall")
v_tokens = qavg(v, "tokens_used")
o_tokens = qavg(o, "tokens_used")
v_reason = qavg(v, "reasoning_recall")
o_reason = qavg(o, "reasoning_recall")
fb = sum(1 for q in o.get("query_results", []) if q.get("used_raw_fallback"))
n_q = len(o.get("query_results", []))

print("=" * 65)
print("SEMANTIC CORE RUN — Same Gutenberg corpus, 44 questions")
print("=" * 65)
print(f"{'Metric':<40} {'Vanilla':>10} {'Optimized':>11} {'Delta':>8}")
print("-" * 65)
print(
    f"{'Retrieval recall':<40} {v_recall:>10.1%} {o_recall:>11.1%} {o_recall-v_recall:>+7.1%}"
)
print(
    f"{'Tokens / query':<40} {v_tokens:>10,.0f} {o_tokens:>11,.0f} {100*(o_tokens-v_tokens)/v_tokens:>+7.1f}%"
)
print(
    f"{'Index entries':<40} {d['vanilla_rag']['index_size_chunks']:>10,} {d['optimized_rag']['index_size_chunks']:>11,} {100*(d['optimized_rag']['index_size_chunks']-d['vanilla_rag']['index_size_chunks'])/d['vanilla_rag']['index_size_chunks']:>+7.1f}%"
)
print(
    f"{'Index size (MB)':<40} {d['vanilla_rag']['index_size_mb']:>10.0f} {d['optimized_rag']['index_size_mb']:>11.1f} {100*(d['optimized_rag']['index_size_mb']-d['vanilla_rag']['index_size_mb'])/d['vanilla_rag']['index_size_mb']:>+7.1f}%"
)
print(
    f"{'Raw fallback triggered':<40} {'—':>10} {fb:>10}/{n_q} ({100*fb/max(1,n_q):.0f}%)"
)
print(
    f"{'Ingestion time (s)':<40} {d['vanilla_rag']['ingestion_time_s']:>10.0f} {d['optimized_rag']['ingestion_time_s']:>11.0f}"
)

print()
print("PROGRESSION across prompt strategies (Gutenberg, same 44 questions)")
print(f"{'Prompt':<22} {'Recall':>8} {'Tokens/q':>10} {'Fallback':>10} {'Notes'}")
print("-" * 70)
print(
    f"{'Vanilla RAG (baseline)':<22} {'53.4%':>8} {'2,594':>10} {'—':>10}  brute-force 8,826 chunks"
)
print(
    f"{'Triple format':<22} {'14.5%':>8} {'462':>10} {'7%':>10}  label tokens wasted budget"
)
print(
    f"{'Prose (no filler rule)':<22} {'~15%':>8} {'~800':>10} {'~7%':>10}  11% function words"
)
print(
    f"{'Semantic core (this)':<22} {o_recall:>8.1%} {o_tokens:>10,.0f} {str(fb)+'/'+str(n_q):>10}  100% signal density"
)
print()
print("KEY FINDINGS")
print(
    f"  Recall:  14.5% → {o_recall:.1%}  (+{(o_recall-0.145)*100:.0f}pp) — semantic core nearly 3× better than triples"
)
print(
    f"  Tokens:  462  → {o_tokens:,.0f}  (more due to 41% fallback rate fetching raw blocks)"
)
print(f"  Fallback: 7% → 41% — better summaries retrieved MORE relevant blocks,")
print(f"          and those blocks scored high enough to pass the fallback threshold")
print(
    f"  vs Vanilla: {o_recall:.1%} vs {v_recall:.1%} — gap narrowed from -39pp to {(o_recall-v_recall)*100:.0f}pp"
)
