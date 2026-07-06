#!/usr/bin/env python3
"""
Retrieval Benchmark — verifies the three core claims from the architecture design.

Runs entirely offline (no Ollama, no internet required).
Uses sentence-transformers (all-MiniLM-L6-v2) for embeddings and
extractive compression instead of an LLM.

Three experiments
-----------------
1. **Token Compression Ratio**
   Measures how much the extractive compressor reduces token count while
   retaining content.  Target: ≥ 60 % reduction (ratio ≤ 0.40).

2. **Summary-Blurring Recall** (the key problem from the design conversation)
   Builds a summary-only index and a parent-child index from the same corpus.
   Runs 20 "granular" queries — specific proper nouns / low-salience details
   that are likely to be dropped by a summariser but preserved in raw text.
   Measures Recall@3 for both modes.
   The parent-child mode should outperform summary-only on these queries.

3. **K-Means Ingestion Cost Reduction**
   For a 500-sentence corpus, compares:
   - Per-chunk compression: every 200-token sub-chunk = 1 LLM-equivalent call
   - Cluster-then-compress: N/target_cluster_size LLM calls
   Measures the call-count savings (%) and confirms cluster assignments are
   semantically coherent (intra-cluster vocabulary overlap).

Usage
-----
    python retrieval_benchmark.py               # all three experiments
    python retrieval_benchmark.py --exp 1       # only experiment 1
    python retrieval_benchmark.py --exp 2 3     # experiments 2 and 3

Requirements
------------
    pip install sentence-transformers chromadb scikit-learn
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import NamedTuple

# ── Project path setup ────────────────────────────────────────────────────────
_BENCH_DIR = Path(__file__).parent
_SRC_DIR = _BENCH_DIR.parent / "src"
sys.path.insert(0, str(_SRC_DIR))

# ── Fixed corpus — no internet required ──────────────────────────────────────
# A synthetic but realistic literary + technical corpus.
# Granular-detail queries target the bracketed tokens [DETAIL] below — items
# that an abstractive summariser might drop but a raw-sub-chunk index preserves.

_CORPUS_SENTENCES: list[str] = [
    # ── Literary passages (Moby Dick flavour) ────────────────────────────────
    "Ishmael, penniless and restless, decides to join a whaling voyage as his cure for melancholy.",
    "He arrives in New Bedford and shares a bed at the Spouter-Inn with the tattooed harpooner Queequeg.",
    "Queequeg carries a [shrunken head] and a [tomahawk pipe] everywhere he goes.",
    "Captain Ahab paces the quarterdeck on a [carved ivory leg] fashioned from a sperm whale's jaw.",
    "Ahab nails a [gold doubloon] to the mast as a prize for the first man who sights Moby Dick.",
    "Starbuck, the first mate, discovers [barrels of oil] leaking but Ahab refuses to stop.",
    "The ship's carpenter fashions a new [ivory prosthetic leg] after the old one splinters in a gam.",
    "Fedallah, the Parsee, prophesies that Ahab will see [hemp only] before he dies.",
    "The Pequod encounters a ship called the Rachel whose captain lost his [twelve-year-old son].",
    "Moby Dick breaches for the first time and [crushes one of the whaleboats] in his jaws.",
    "On the second day Moby Dick [destroys two more boats] and bites off Fedallah's line.",
    "Ahab's harpoon is forged in [pagan blood] and baptised in the name of the devil.",
    "Pip, the cabin boy, falls overboard and goes [mad from floating alone] in the ocean.",
    "The carpenter carves an elaborate [life-buoy coffin] that later saves Ishmael.",
    "On the third day Ahab is tangled in the harpoon line and dragged to the deep.",
    # ── Technical passages (incident-response flavour) ────────────────────────
    "Service payment-api began returning HTTP [504 Gateway Timeout] at 14:32 UTC.",
    "Root cause: CosmosDB primary replica [21012 connection limit] hit under peak load.",
    "The retry storm generated [42 million] redundant requests within 90 seconds.",
    "Circuit breaker for cosmos-primary opened after [threshold=500] errors per minute.",
    "Rollback to [version 2.3.1] restored normal latency within eight minutes.",
    "Memory utilisation on node pool [aks-nodepool2] spiked to [94 %] before the restart.",
    "The on-call engineer received a PagerDuty alert referencing [runbook #RT-1042].",
    "A [missing index] on the OrderItems table caused a full table scan on every request.",
    "Deployment [canary-release-v4.7.2] was promoted after the [P99 latency] dropped to 12 ms.",
    "The SLO breach window lasted [47 minutes] — within the [30-day error budget].",
    # ── Science passages ─────────────────────────────────────────────────────
    "The [double-slit experiment] demonstrates wave-particle duality in quantum mechanics.",
    "Planck's constant h equals [6.626 × 10⁻³⁴ J·s] — the quantum of action.",
    "Marie Curie isolated [polonium] and [radium] from pitchblende ore in 1898.",
    "The first controlled nuclear chain reaction occurred on [2 December 1942] in Chicago.",
    "CRISPR-Cas9 was adapted for mammalian gene editing by [Jennifer Doudna] and Emmanuelle Charpentier.",
    "The [Hubble constant] measures the expansion rate of the universe at roughly 70 km/s/Mpc.",
    "Water's anomalous density maximum occurs at [4 °C] — denser than ice at 0 °C.",
    "The Krebs cycle produces [3 NADH, 1 FADH2, and 1 GTP] per acetyl-CoA turn.",
    "Alexander Fleming noticed [mould contamination] killing bacteria on a Petri dish in 1928.",
    "The [mitochondrial Eve] hypothesis traces all maternal human lineages to a single woman in Africa.",
    # ── More literary (Pride and Prejudice flavour) ───────────────────────────
    "Elizabeth Bennet first meets Darcy at a [Netherfield ball] where he refuses to dance.",
    "Wickham tells Elizabeth that Darcy cheated him out of [a clerical living] worth £1,000.",
    "Lady Catherine de Bourgh visits Longbourn to warn Elizabeth away from Darcy at [Rosings Park].",
    "Darcy's first letter explains that he separated Bingley from Jane and that Wickham was a [gambler].",
    "Mr Collins proposes to Elizabeth and is rejected; he immediately proposes to [Charlotte Lucas].",
    "Lydia elopes with Wickham and is discovered hiding in [Gracechurch Street, London].",
    "Darcy secretly pays Wickham [£10,000] and settles his debts to secure the marriage.",
    "Elizabeth visits Pemberley, Darcy's estate in [Derbyshire], and changes her opinion of him.",
    "Bingley proposes to Jane at [Netherfield] after Darcy confesses to interfering previously.",
    "The novel ends with Elizabeth and Darcy settling at Pemberley, and Jane at [Netherfield].",
    # ── More technical (database / infra flavour) ─────────────────────────────
    "PostgreSQL [VACUUM FULL] reclaims dead tuple storage but requires an [exclusive table lock].",
    "Redis sentinel uses a [quorum of 2] to promote a replica when the master fails.",
    "Kafka consumer groups rebalance when a consumer joins and the [session.timeout.ms=10000] expires.",
    "The [WAL (Write-Ahead Log)] ensures durability by flushing log records before data pages.",
    "B-tree indexes in MySQL degrade when the [fill factor] exceeds 90 % on heavily updated tables.",
]

# ── Granular-detail queries (the 'summary blurring' test cases) ───────────────
# These target specific low-salience details from [bracketed] tokens above.
# An LLM summary would likely capture character names and plot arcs but drop
# object-level details like "gold doubloon", "ivory leg", "tomahawk pipe".
_GRANULAR_QUERIES: list[tuple[str, str]] = [
    # (query, expected_chunk_keyword_present_in)
    ("shrunken head tomahawk pipe Queequeg", "tomahawk pipe"),
    ("ivory leg prosthetic carved sperm whale", "carved ivory leg"),
    ("gold doubloon prize mast Moby Dick", "gold doubloon"),
    ("leaking oil barrels Starbuck refuses", "barrels of oil"),
    ("hemp prophecy Parsee Fedallah", "hemp only"),
    ("twelve year old son Rachel captain", "twelve-year-old son"),
    ("pagan blood harpoon baptised devil", "pagan blood"),
    ("life-buoy coffin saves Ishmael", "life-buoy coffin"),
    ("504 Gateway Timeout payment-api 14:32 UTC", "504 Gateway Timeout"),
    ("21012 connection limit CosmosDB replica", "21012 connection limit"),
    ("42 million redundant requests retry storm", "42 million"),
    ("runbook RT-1042 PagerDuty on-call", "runbook #RT-1042"),
    ("missing index full table scan OrderItems", "missing index"),
    ("canary release v4.7.2 P99 latency 12ms", "canary-release-v4.7.2"),
    ("47 minutes SLO breach error budget", "47 minutes"),
    ("6.626e-34 Planck constant quantum action", "6.626"),
    ("4 degrees Celsius water density maximum ice", "4 °C"),
    ("3 NADH 1 FADH2 GTP Krebs cycle", "3 NADH"),
    ("Gracechurch Street Lydia Wickham London", "Gracechurch Street"),
    ("WAL Write-Ahead Log flush log records durability", "Write-Ahead Log"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────


class BenchStats(NamedTuple):
    recall_at_k: float
    avg_latency_ms: float
    n_queries: int
    k: int


def _keyword_hit(
    results: list[dict], keyword: str, field: str = "compressed_summary"
) -> bool:
    """Return True if *keyword* appears (case-insensitive) in any result's field."""
    kw = keyword.lower()
    for r in results:
        text = r.get(field, "") or ""
        if kw in text.lower():
            return True
        # Also check raw_text if available
        raw = r.get("raw_text", "") or ""
        if kw in raw.lower():
            return True
    return False


def _run_queries(
    retriever,
    queries: list[tuple[str, str]],
    top_k: int = 3,
    use_child_index: bool = False,
) -> BenchStats:
    """Execute queries and compute recall."""
    hits = 0
    total_latency = 0.0
    for query, keyword in queries:
        t0 = time.perf_counter()
        if use_child_index:
            results = retriever.search_with_child_index(
                query, top_k=top_k, use_cache=False
            )
        else:
            results = retriever.search(query, top_k=top_k, use_cache=False)
        latency = (time.perf_counter() - t0) * 1000
        total_latency += latency
        if _keyword_hit(results, keyword):
            hits += 1
    return BenchStats(
        recall_at_k=hits / len(queries),
        avg_latency_ms=total_latency / len(queries),
        n_queries=len(queries),
        k=top_k,
    )


def _bar(pct: float, width: int = 30) -> str:
    filled = round(pct * width)
    return "█" * filled + "░" * (width - filled)


def _print_table(rows: list[tuple], headers: list[str]) -> None:
    col_widths = [
        max(len(str(r[i])) for r in ([headers] + rows)) for i in range(len(headers))
    ]
    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_widths) + " |"
    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(c) for c in row]))
    print(sep)


# ── Experiment 1: Token Compression Ratio ─────────────────────────────────────


def exp1_compression_ratio() -> None:
    print("\n" + "=" * 70)
    print("EXPERIMENT 1 — Token Compression Ratio (extractive, no LLM)")
    print("=" * 70)

    from context_optimizer.compressor import (
        _estimate_tokens,
        compress_chunk_extractive,
        split_into_sub_chunks,
    )

    full_text = "\n".join(_CORPUS_SENTENCES)
    sub_chunks = split_into_sub_chunks(full_text, sub_chunk_tokens=200)

    rows = []
    total_orig = 0
    total_comp = 0
    for i, sc in enumerate(sub_chunks):
        c = compress_chunk_extractive(sc, chunk_id=f"sc_{i:03d}", ratio=0.35)
        total_orig += c.original_tokens
        total_comp += c.compressed_tokens
        rows.append(
            (
                f"sub-chunk {i:02d}",
                c.original_tokens,
                c.compressed_tokens,
                f"{c.compression_ratio:.1%}",
            )
        )

    overall_ratio = total_comp / total_orig if total_orig else 1.0
    rows.append(("TOTAL", total_orig, total_comp, f"{overall_ratio:.1%}"))

    _print_table(rows, ["Chunk", "Orig tokens", "Comp tokens", "Ratio"])
    print(f"\n  Overall compression ratio : {overall_ratio:.1%}")
    print(f"  Token reduction           : {100*(1-overall_ratio):.1f}%")
    goal_met = overall_ratio <= 0.40
    print(f"  Target (<= 40%)           : {'PASS' if goal_met else 'FAIL'}")
    if not goal_met:
        print(
            f"  [Note] Ratio {overall_ratio:.1%} > 40% -- may need a lower `ratio` param"
        )


# ── Experiment 2: Summary-Blurring Recall ─────────────────────────────────────


def exp2_summary_blurring_recall() -> None:
    print("\n" + "=" * 70)
    print("EXPERIMENT 2 — Summary-Blurring Recall (parent-child vs summary-only)")
    print("=" * 70)
    print("  Queries target specific low-salience details (object names, numbers)")
    print("  that an LLM summariser would likely drop from a parent summary.")
    print("  The parent-child index should catch these via raw sub-chunk hits.\n")

    try:
        from context_optimizer.cached_retriever import CachedChromaRetriever
        from context_optimizer.compressor import (
            compress_chunk_extractive,
            split_into_sub_chunks,
        )
    except ImportError as e:
        print(f"  [SKIP] Import error: {e}")
        return

    tmp_dir = tempfile.mkdtemp(prefix="co_bench2_")
    try:
        # Build compressed chunks from the fixed corpus
        full_text = "\n".join(_CORPUS_SENTENCES)
        sub_chunks_text = split_into_sub_chunks(full_text, sub_chunk_tokens=200)
        chunks = [
            compress_chunk_extractive(sc, chunk_id=f"chunk_{i:03d}", ratio=0.35)
            for i, sc in enumerate(sub_chunks_text)
        ]

        print(
            f"  Corpus    : {len(_CORPUS_SENTENCES)} sentences → {len(chunks)} parent chunks"
        )
        print(f"  Queries   : {len(_GRANULAR_QUERIES)} granular-detail queries\n")

        # ── Mode A: Summary-only ──────────────────────────────────────────────
        retriever_a = CachedChromaRetriever(
            collection_name="bench2_summary",
            persist_directory=tmp_dir + "/a",
            embedding_model_name="all-MiniLM-L6-v2",
            embedding_backend="sentence-transformers",
        )
        retriever_a.add_chunks(chunks)

        # ── Mode B: Parent-child ──────────────────────────────────────────────
        retriever_b = CachedChromaRetriever(
            collection_name="bench2_child",
            persist_directory=tmp_dir + "/b",
            embedding_model_name="all-MiniLM-L6-v2",
            embedding_backend="sentence-transformers",
        )
        retriever_b.add_chunks(chunks)
        n_children = retriever_b.add_raw_sub_chunks(chunks, sub_chunk_tokens=100)
        print(f"  Child index: {n_children} sub-chunks added\n")

        # ── Run queries ───────────────────────────────────────────────────────
        print("  Running summary-only queries ...")
        stats_a = _run_queries(
            retriever_a, _GRANULAR_QUERIES, top_k=3, use_child_index=False
        )

        print("  Running parent-child queries ...")
        stats_b = _run_queries(
            retriever_b, _GRANULAR_QUERIES, top_k=3, use_child_index=True
        )

        print()
        _print_table(
            [
                (
                    "Summary-only",
                    f"{stats_a.recall_at_k:.0%}",
                    f"{stats_a.avg_latency_ms:.1f} ms",
                    _bar(stats_a.recall_at_k),
                ),
                (
                    "Parent-child",
                    f"{stats_b.recall_at_k:.0%}",
                    f"{stats_b.avg_latency_ms:.1f} ms",
                    _bar(stats_b.recall_at_k),
                ),
            ],
            ["Mode", f"Recall@{stats_a.k}", "Avg latency", "Recall bar"],
        )

        delta = stats_b.recall_at_k - stats_a.recall_at_k
        print(f"\n  Recall improvement (parent-child over summary-only): {delta:+.0%}")
        if delta > 0:
            print("  Result: parent-child INDEX FIXES blurring [OK]")
        elif delta == 0:
            print(
                "  Result: no difference -- extractive compressor already preserves keywords"
            )
        else:
            print(
                "  Result: unexpected regression -- check sub-chunk size and embedding model"
            )

        # ── Per-query breakdown ───────────────────────────────────────────────
        print("\n  Per-query breakdown:")
        header = ["#", "Query (first 40 chars)", "Keyword", "Summary", "Child"]
        rows = []
        for i, (query, kw) in enumerate(_GRANULAR_QUERIES):
            r_sum = retriever_a.search(query, top_k=3, use_cache=False)
            r_chi = retriever_b.search_with_child_index(query, top_k=3, use_cache=False)
            hit_s = "HIT " if _keyword_hit(r_sum, kw) else "MISS"
            hit_c = "HIT " if _keyword_hit(r_chi, kw) else "MISS"
            rows.append((i + 1, query[:40], kw[:25], hit_s, hit_c))
        _print_table(rows, header)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Experiment 3: K-Means Ingestion Cost Reduction ────────────────────────────


def exp3_kmeans_cost_reduction() -> None:
    print("\n" + "=" * 70)
    print("EXPERIMENT 3 — K-Means Ingestion Cost Reduction")
    print("=" * 70)
    print("  Compares LLM-call count for per-chunk vs. cluster-then-compress.")
    print("  Uses TF-IDF + MiniBatchKMeans -- no LLM, no network required.\n")

    try:
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        print("  [SKIP] scikit-learn not installed: pip install scikit-learn")
        return

    from context_optimizer.compressor import _estimate_tokens, split_into_sub_chunks

    # Use a larger synthetic corpus for meaningful clustering
    corpus_lines = _CORPUS_SENTENCES * 10  # 500 sentences

    full_text = "\n".join(corpus_lines)
    sub_chunks = split_into_sub_chunks(full_text, sub_chunk_tokens=200)

    # Baseline: per-sub-chunk compression = 1 call per sub-chunk
    baseline_calls = len(sub_chunks)

    # K-Means approach: 1 call per cluster
    target_cluster_sizes = [10, 25, 50]
    rows = []
    for tcs in target_cluster_sizes:
        n_clusters = max(1, len(sub_chunks) // tcs)

        t0 = time.perf_counter()
        vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")
        tfidf = vectorizer.fit_transform(sub_chunks)
        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=3,
            batch_size=min(1024, len(sub_chunks)),
        )
        labels = km.fit_predict(tfidf)
        elapsed = time.perf_counter() - t0

        savings_pct = (1 - n_clusters / baseline_calls) * 100

        # Measure intra-cluster coherence: avg pairwise vocabulary overlap within clusters
        clusters: dict[int, list[str]] = {}
        for i, lbl in enumerate(labels):
            clusters.setdefault(int(lbl), []).append(sub_chunks[i])

        overlaps = []
        for texts in clusters.values():
            if len(texts) < 2:
                continue
            words_sets = [set(t.lower().split()) for t in texts]
            # Jaccard similarity: mean pairwise
            pairs = 0
            overlap_sum = 0.0
            for a in range(len(words_sets)):
                for b in range(a + 1, min(a + 5, len(words_sets))):
                    inter = len(words_sets[a] & words_sets[b])
                    uni = len(words_sets[a] | words_sets[b])
                    if uni:
                        overlap_sum += inter / uni
                        pairs += 1
            if pairs:
                overlaps.append(overlap_sum / pairs)
        avg_coherence = sum(overlaps) / len(overlaps) if overlaps else 0.0

        rows.append(
            (
                tcs,
                len(sub_chunks),
                n_clusters,
                f"{savings_pct:.1f}%",
                f"{avg_coherence:.3f}",
                f"{elapsed*1000:.0f} ms",
            )
        )

    _print_table(
        rows,
        [
            "Target cluster size",
            "Sub-chunks (calls w/o cluster)",
            "Clusters (calls w/ cluster)",
            "LLM call savings",
            "Intra-cluster coherence",
            "Cluster time",
        ],
    )

    print("\n  Coherence > 0.05 = semantically related sentences grouped together.")
    print("  Savings column shows how many fewer LLM calls clustering requires.\n")

    # Validate: show a sample cluster
    target = 25
    n_c = max(1, len(sub_chunks) // target)
    vectorizer2 = TfidfVectorizer(max_features=2000, stop_words="english")
    tfidf2 = vectorizer2.fit_transform(sub_chunks)
    km2 = MiniBatchKMeans(
        n_clusters=n_c,
        random_state=42,
        n_init=3,
        batch_size=min(1024, len(sub_chunks)),
    )
    labels2 = km2.fit_predict(tfidf2)

    clusters2: dict[int, list[str]] = {}
    for i, lbl in enumerate(labels2):
        clusters2.setdefault(int(lbl), []).append(sub_chunks[i])

    # Print the largest cluster as a sample
    largest_k = max(clusters2, key=lambda k: len(clusters2[k]))
    sample = clusters2[largest_k]
    print(f"  Sample cluster (id={largest_k}, {len(sample)} sub-chunks):")
    for sc in sample[:3]:
        print(f"    • {sc[:90].strip()}")
    if len(sample) > 3:
        print(f"    … (+{len(sample)-3} more)")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Context-optimizer retrieval benchmark (offline, no LLM required)"
    )
    parser.add_argument(
        "--exp",
        nargs="*",
        type=int,
        choices=[1, 2, 3],
        default=[1, 2, 3],
        help="Which experiments to run (default: all)",
    )
    args = parser.parse_args()
    exps = set(args.exp)

    print("=" * 70)
    print("Context-Optimizer Retrieval Benchmark")
    print("Verifies architecture claims from design conversation")
    print("=" * 70)

    if 1 in exps:
        exp1_compression_ratio()
    if 2 in exps:
        exp2_summary_blurring_recall()
    if 3 in exps:
        exp3_kmeans_cost_reduction()

    print("\n" + "=" * 70)
    print("Benchmark complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
