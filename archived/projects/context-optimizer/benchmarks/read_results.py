import json
import pathlib

p = pathlib.Path(__file__).parent / "corpus_results.json"
d = json.loads(p.read_text())

print(f"Run date : {d['run_date']}")
print(f"Corpus   : {d['corpus']}")

v = d.get("vanilla_rag") or {}
o = d.get("optimized_rag") or {}


def section(name, data):
    qrs = data.get("query_results", [])
    if not qrs:
        print(f"\n{name}: no query results")
        return
    recalls = [q["kw_recall"] for q in qrs]
    r_recalls = [q["reasoning_recall"] for q in qrs if q.get("reasoning_answer")]
    faiths = [q["faithfulness"] for q in qrs if q.get("reasoning_answer")]
    tokens = [q["tokens_used"] for q in qrs]
    lats = [q["latency_ms"] for q in qrs]
    fb = sum(1 for q in qrs if q.get("used_raw_fallback"))
    print(f"\n{name}")
    print(f"  Questions            : {len(qrs)}")
    print(f"  Ingestion time (s)   : {data.get('ingestion_time_s', 0):.1f}")
    print(f"  Index entries        : {data.get('index_size_chunks', 0):,}")
    print(f"  Index size (MB)      : {data.get('index_size_mb', 0):.1f}")
    print(f"  Avg retrieval recall : {sum(recalls)/len(recalls):.1%}")
    if r_recalls:
        print(f"  Avg reasoning recall : {sum(r_recalls)/len(r_recalls):.1%}")
        print(f"  Avg faithfulness     : {sum(faiths)/len(faiths):.1%}")
    print(f"  Avg tokens/query     : {sum(tokens)/len(tokens):,.0f}")
    print(f"  Avg latency (ms)     : {sum(lats)/len(lats):.1f}")
    print(f"  Raw fallback count   : {fb}/{len(qrs)}")


section("VANILLA RAG", v)
section("OPTIMIZED RAG (extractive, 200MB corpus)", o)

# Head-to-head if both present
v_qrs = v.get("query_results", [])
o_qrs = o.get("query_results", [])
if v_qrs and o_qrs:
    vr = sum(q["kw_recall"] for q in v_qrs) / len(v_qrs)
    or_ = sum(q["kw_recall"] for q in o_qrs) / len(o_qrs)
    vt = sum(q["tokens_used"] for q in v_qrs) / len(v_qrs)
    ot = sum(q["tokens_used"] for q in o_qrs) / len(o_qrs)
    vi = v.get("ingestion_time_s", 0)
    oi = o.get("ingestion_time_s", 0)
    print(f"\n{'─'*50}")
    print(f"HEAD-TO-HEAD")
    print(f"  Recall       : vanilla={vr:.1%}  opt={or_:.1%}  delta={or_-vr:+.1%}")
    print(
        f"  Tokens/query : vanilla={vt:,.0f}  opt={ot:,.0f}  delta={100*(ot-vt)/vt:+.1f}%"
    )
    print(
        f"  Ingestion(s) : vanilla={vi:.1f}  opt={oi:.1f}  delta={100*(oi-vi)/vi:+.1f}%"
    )
