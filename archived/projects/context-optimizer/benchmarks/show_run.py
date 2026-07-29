import json
from pathlib import Path

d = json.loads(Path("c:/repos/ai-portfolio/projects/context-optimizer/benchmarks/corpus_results.json").read_text())
o = d["optimized_rag"]
qrs = o["query_results"]

print(f"Run date  : {d['run_date']}")
print(f"Corpus    : clean enwik9 (Wikipedia plain text, XML stripped)")
print(f"Ingestion : {o['ingestion_time_s']:.0f}s  |  {o['index_size_chunks']} blocks  |  {o['index_size_mb']:.1f} MB index")
print()

recalls    = [q["kw_recall"] for q in qrs]
r_recalls  = [q["reasoning_recall"] for q in qrs]
faiths     = [q["faithfulness"] for q in qrs]
tokens     = [q["tokens_used"] for q in qrs]
lats       = [q["latency_ms"] for q in qrs]

print(f"RESULTS (50 questions, top-k=5)")
print(f"  Retrieval recall  : {sum(recalls)/len(recalls):.1%}  (do retrieved triples contain expected keywords?)")
print(f"  Reasoning recall  : {sum(r_recalls)/len(r_recalls):.1%}  (can llama3.2:3b answer FROM the triples?)")
print(f"  Faithfulness      : {sum(faiths)/len(faiths):.1%}  (answer grounded in retrieved context?)")
print(f"  Avg tokens/query  : {sum(tokens)/len(tokens):.0f}  (vs vanilla RAG ~2,560)")
print(f"  Avg latency       : {sum(lats)/len(lats):.0f} ms")
print(f"  Raw fallback      : 0/{len(qrs)}")
print()

print("SAMPLE RETRIEVED TRIPLES (what was actually stored in ChromaDB):")
for i, q in enumerate(qrs[:5]):
    print(f"\n  [{i}] Query: {q['query'][:60]}")
    print(f"      Retrieved: {q['answer_snippet'][:130]}")
    print(f"      ret={q['kw_recall']:.0%}  reason={q['reasoning_recall']:.0%}  faith={q['faithfulness']:.0%}")

print()
print("WHY RECALL IS STILL LOW:")
print("  Questions are Wikipedia-sourced: 'What is Anarchism?', 'What is Autism?'")
print("  Expected keywords = words from Wikipedia article summaries.")
print("  Retrieved triples = summaries of 50MB of cleaned article BODY text.")
print("  Mismatch: questions expect encyclopedia-style definitions,")
print("  triples contain entity/date/relationship facts from body paragraphs.")
print("  A query 'What is Anarchism?' needs the definition — the triples may")
print("  have 'REL:Bakunin->influenced->anarchist_movement;DATE:1840s'")
print("  which is correct content but doesn't match the keyword-based judge.")
