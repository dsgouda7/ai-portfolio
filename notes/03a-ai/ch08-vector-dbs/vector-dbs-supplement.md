# Vector Databases — Production Tuning, Benchmarking, and Operational Patterns

> **Companion to:** [VectorDBs.md](vector-dbs.md)  
> This document enriches the main notes with: index selection decision trees, production tuning recipes, recall vs. latency tradeoff analysis, operational pitfalls, filtering strategies, and a structured interview Q&A.

---

## 1. The ANN Tradeoff Triangle

Every vector index design navigates three competing constraints simultaneously. You cannot optimize all three at once:

```
                       RECALL
                      (accuracy)
                          ▲
                         /|\
                        / | \
                       /  |  \
                      /   |   \
                     /    |   \
                    / HNSW|    \
                   /      |     \
                  /       |      \
                 /   IVF  | ScaNN \
                /─────────┼────────\
               /    PQ    |  DiskANN\
              /           |          \
             ◄────────────┼────────────►
          SPEED                      MEMORY
         (low latency)              (low footprint)

  IVF-PQ lives near the SPEED-MEMORY corner (fast + compact but lower recall)
  HNSW lives near the RECALL-SPEED corner (highest recall + fast but memory-hungry)
  DiskANN lives near the RECALL-MEMORY corner (high recall + low RAM via SSD)
```

**Practical choosing rule:**
- Need highest recall at lowest latency, RAM is plentiful → **HNSW**
- RAM-constrained, batch workload okay → **IVF or IVF-PQ**
- Billions of vectors on commodity hardware → **DiskANN**
- Research/benchmarking baseline → **Flat (brute-force)**

---

## 2. HNSW Parameter Tuning Recipes

The main notes list the parameters (`M`, `efConstruction`, `efSearch`). Here are concrete starting points calibrated to dataset size:

### Small corpus (< 100K vectors)

```sql
-- pgvector
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

SET hnsw.ef_search = 40;
```

### Medium corpus (100K–10M vectors)

```sql
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 32, ef_construction = 128);

SET hnsw.ef_search = 80;
```

### Large corpus (> 10M vectors)

```sql
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 64, ef_construction = 200);

SET hnsw.ef_search = 120;
```

### Effect of M on memory

```
M = 16:  ~1.1–1.5× raw vector size in total index memory
M = 32:  ~1.2–2.0× raw vector size in total index memory
M = 64:  ~1.5–3.0× raw vector size in total index memory

For 1M vectors × 1536 dims × 4 bytes = 6.1 GB raw
With M=32: expect ~8–12 GB for the full HNSW index
```

**Tuning strategy:** Start with `M=16, efConstruction=64`, measure recall@10 on a test set, then increase `M` in doubling steps until recall plateaus. Increase `efSearch` independently to trade query latency for recall at query time without rebuilding the index.

---

## 3. IVF Tuning Recipes

### Choosing nlist (number of clusters)

```
Rule of thumb:
  nlist ≈ √N      (N = number of vectors)

Examples:
  1M vectors  → nlist ≈ 1,000
  10M vectors → nlist ≈ 3,162
  100M vectors→ nlist ≈ 10,000

Minimum data per cluster for good quality:
  Each cluster should contain at least 39 × nlist vectors at training time
  (i.e., train on at least 39 × nlist representative samples)
```

### Choosing nprobe (clusters searched per query)

```
nprobe / nlist = recall target

nprobe = 1%  of nlist → ~50–60% recall  (very fast, poor quality)
nprobe = 5%  of nlist → ~80–85% recall  (good for RAG with re-ranking)
nprobe = 10% of nlist → ~90–95% recall  (general purpose)
nprobe = 50% of nlist → ~98%+ recall    (high precision, slower)
nprobe = nlist        → 100% recall     (equivalent to brute-force)
```

**Azure Cosmos DB defaults (from main notes):**

```
rows < 10K  → nprobe = nlist  (full search; dataset small enough)
rows < 1M   → nprobe = rows / 1,000
rows > 1M   → nprobe = √(rows)
```

---

## 4. Recall vs. Latency Benchmarking — How to Do It

Never accept vendor-quoted recall numbers at face value. Measure on your own data and queries.

### The ANN Benchmark Protocol

```python
import numpy as np
import time

def benchmark_index(index, query_vectors, ground_truth_ids, k=10):
    """
    Measures recall@k and average query latency.
    ground_truth_ids[i] = list of the true k nearest neighbor IDs for query i
    """
    correct = 0
    total = 0
    latencies = []

    for i, query in enumerate(query_vectors):
        start = time.perf_counter()
        result_ids = index.search(query, k)          # your index's search method
        elapsed = time.perf_counter() - start
        latencies.append(elapsed * 1000)             # ms

        true_ids = set(ground_truth_ids[i])
        found_ids = set(result_ids)
        correct += len(true_ids & found_ids)
        total += k

    recall_at_k = correct / total
    p50_latency = np.percentile(latencies, 50)
    p99_latency = np.percentile(latencies, 99)

    return {
        f"recall@{k}": recall_at_k,
        "p50_latency_ms": p50_latency,
        "p99_latency_ms": p99_latency
    }
```

**What to measure:**
- `recall@1` — is the single best result correct? (Strictest)
- `recall@10` — is the true result anywhere in top 10? (Standard RAG metric)
- `p50` and `p99` latency — p99 matters more for user-facing systems
- Memory footprint — `top` or `nvidia-smi` during load

---

## 5. The Filtering Problem — Four Strategies Compared

Filtering (combining vector search with metadata predicates like `category = 'rail'`) is one of the hardest problems in vector database design. The main notes introduce iterative filtering for DiskANN. Here is the full landscape:

### Strategy 1: Post-Filtering

```
ANN search → top-K results → apply metadata filter → ≤K results returned
```

**Problem:** If the filter is selective (e.g., only 1% of documents match), you may need to fetch top-1000 from ANN just to return 10 that pass the filter. Recall plummets and cost spikes.

**When acceptable:** Filter selectivity > 20% (i.e., >20% of vectors pass the filter).

### Strategy 2: Pre-Filtering

```
Metadata filter → candidate set → ANN search within candidates → top-K
```

**Problem:** The ANN index is built over the entire corpus. Restricting to a subset may not have a pre-built index, forcing a scan of the filtered subset.

**When acceptable:** The filtered subset is large enough (> 10K vectors) to justify ANN search within it, and the database supports sub-index structures.

### Strategy 3: Iterative Filtering (DiskANN, Azure Cosmos DB)

Apply predicates **during graph traversal** — at each hop, only follow edges to nodes that pass the filter:

```
Graph traversal:
  Visit node A → passes filter → add to candidates
  Visit node B → fails filter  → skip, but still follow its edges
  Visit node C → passes filter → add to candidates
  ...continue until k valid candidates found
```

**Advantage:** Never over-fetches. Accurate result count even with highly selective filters.

**Trade-off:** Graph traversal may explore more nodes than unfiltered search to find k valid results.

### Strategy 4: Separate Indexes per Segment

Pre-build a separate ANN index for each major filter value:

```
index_category_rail   = HNSW index over rail documents only
index_category_air    = HNSW index over airline documents only
index_category_road   = HNSW index over road documents only

Query with filter category='rail' → search index_category_rail only
```

**When to use:** When filter values are a small, fixed set (e.g., product categories, tenants in a multi-tenant SaaS). Provides perfect recall for each category but multiplies index storage.

---

## 6. Quantization Deep Dive — When to Compress and What You Lose

### Half-Precision (float16)

```
float32: 4 bytes per dimension
float16: 2 bytes per dimension → 2× memory savings

768-dim float32: 3,072 bytes per vector
768-dim float16: 1,536 bytes per vector

100M vectors × 768 dims:
  float32: 307 GB
  float16: 154 GB  ← fits on a server that float32 could not
```

**Recall impact:** Typically < 0.5% recall degradation for most embedding models. The precision loss in distance calculations is negligible because cosine similarity is insensitive to small rounding errors.

**Support:** pgvector >= 0.7.0 with HNSW and IVF. Allows up to **4,000 dimensions** (vs. 2,000 for float32 on an 8KB page).

### Binary Quantization (1-bit)

```
Each float → 1 bit (sign only)

1536-dim float32: 6,144 bytes
1536-dim binary:     192 bytes → 32× compression

Distance metric: Hamming distance (XOR + popcount) — extremely fast on modern CPUs
```

**Recall impact:** Significant — typically 5–15% recall drop. Only viable with a two-phase pipeline: binary ANN for coarse retrieval → exact float32 re-ranking of top candidates.

**Best for:** Very large image or audio embedding corpora where RAM is the hard constraint.

### Product Quantization vs. Binary — Quick Decision

```
Memory savings needed:  2×  →  float16
                        4–8× →  PQ (m=32–64 subspaces)
                       32×   →  Binary + re-ranking
                      192×   →  Binary (with significant recall tradeoff)
```

---

## 7. Multi-Tenancy in Vector Databases

A common production scenario: a single vector database serves multiple tenants (e.g., a SaaS with per-customer knowledge bases). Poorly designed multi-tenancy leaks data between tenants or produces noisy results.

### Namespace / Collection per Tenant

```
tenant_001_collection → {doc embeddings for tenant 001}
tenant_002_collection → {doc embeddings for tenant 002}
```

**Pros:** Perfect isolation; separate index parameters per tenant.  
**Cons:** Large number of indexes; index per tenant is wasteful if most tenants have small corpora.

### Shared Index + Tenant Metadata Filter

```
Single index with all vectors
Each vector has metadata: {"tenant_id": "001", "category": "contracts"}

Query: embed(query) + filter: tenant_id = "001"
```

**Pros:** Single index is efficient for small-to-medium tenants.  
**Cons:** Filtering overhead; a very large tenant can skew the ANN graph quality for smaller tenants.

### Hybrid: Shared Small, Dedicated Large

```
tenants with < 10K docs → shared index + filter
tenants with > 10K docs → dedicated index
```

This is the pattern used by Pinecone namespaces and Weaviate multi-tenancy.

---

## 8. Common Production Pitfalls

### Pitfall 1: Indexing Before Loading Data

HNSW and IVF build their structures around the data that exists at index-creation time. Creating the index first and then inserting data means the index was built on an empty or sparse dataset — the structure is wrong from the start.

**Rule:** Always load all existing data **before** creating the index. For streaming ingestion, create the index on the seed dataset, then rely on the database's incremental insert support (HNSW supports this; IVF requires periodic rebuild).

### Pitfall 2: Not Normalizing Before Index Creation

If you plan to use dot product as your distance metric (faster than cosine), you must normalize your vectors **before inserting them** into the index. Forgetting normalization at ingestion but normalizing at query time produces silently wrong results — the math won't error, but results will be meaningless.

### Pitfall 3: Setting `nprobe=1` in Production IVF

The default `nprobe` for IVF in many libraries is 1 — meaning only 1 cluster is searched per query. This gives very low recall (often 50–60%) but looks fine in unit tests on small datasets where all vectors are in one cluster. Always tune `nprobe` on production-scale data.

### Pitfall 4: Ignoring Index Build Time

HNSW build time scales as **O(N log N)** but with a high constant — building an HNSW index over 100M vectors can take hours. Plan for this in your data pipeline. For streaming data, use background index builds with a brute-force fallback for unindexed segments.

### Pitfall 5: Memory Estimation Errors

Many teams underestimate HNSW memory:

```
Naive estimate: N × D × 4 bytes (just the vectors)

Correct estimate: N × D × 4 bytes (vectors) + N × M × 8 bytes (graph edges)

For N=10M, D=1536, M=32:
  Vectors:    10M × 1536 × 4 = 61.4 GB
  Graph:      10M × 32   × 8 =  2.6 GB
  Total:                      ~64 GB  (not 61.4 GB)
```

The graph overhead is modest for typical M values but grows linearly with M.

---

## 9. Vector DB Landscape — Quick Selection Guide

```
┌──────────────────────────────────────────────────────────────────────┐
│                  VECTOR DATABASE SELECTION GUIDE                      │
│                                                                        │
│  Already running PostgreSQL?                                           │
│    YES → pgvector (IVFFlat, HNSW, DiskANN)                           │
│                                                                        │
│  Already running SQL Server 2025+?                                    │
│    YES → Native vector type + DiskANN                                 │
│                                                                        │
│  Need to store vectors alongside JSON documents?                      │
│    YES → Azure Cosmos DB (DiskANN, pre-filtering)                     │
│                                                                        │
│  Need full-text + vector hybrid search?                               │
│    YES → Azure AI Search or Weaviate                                  │
│                                                                        │
│  Billion-scale, RAM-constrained, on commodity hardware?               │
│    YES → DiskANN (Cosmos DB or standalone)                            │
│                                                                        │
│  Fully managed, minimal ops, standard English embeddings?             │
│    YES → Pinecone                                                      │
│                                                                        │
│  Open-source, need maximum flexibility + Kubernetes?                  │
│    YES → Milvus                                                        │
│                                                                        │
│  Rapid prototyping / in-process / no server needed?                   │
│    YES → FAISS (library, not a database)                              │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 10. Interview Q&A — Vector Databases and ANN Indexing

**Q: Why can't you use a traditional B-tree index for vector search?**  
A: B-trees rely on total ordering — every value can be compared as greater than or less than another. Vectors have no total order; their "closeness" is measured by a distance metric (cosine, L2) across all dimensions simultaneously. B-trees would need to compare all dimensions at each node, effectively degenerating to brute-force. Additionally, the curse of dimensionality makes tree-based spatial indexes (like kd-trees) nearly as slow as brute-force in high dimensions (>20–50 dims).

**Q: Explain HNSW. What is the "small world" property and why does it help?**  
A: HNSW builds a multi-layer graph where each node (vector) connects to its M nearest neighbors. The top layers are sparse with long-range connections (like highways); the bottom layer is dense with local connections. The "small world" property means any two nodes are connected via a short path — typically O(log N) hops. Search starts at the top layer, greedily walks toward the query, then drops to lower layers for refinement. This gives O(log N) search time with high recall even in high dimensions.

**Q: What is the recall-latency tradeoff in HNSW and how do you control it without rebuilding the index?**  
A: The `efSearch` parameter controls the size of the candidate set explored during search. Higher `efSearch` → more candidates examined → higher recall but more latency. Crucially, `efSearch` is a query-time parameter — you can change it without rebuilding the index. `efConstruction` is set at build time and affects index quality but cannot be changed afterward.

**Q: What is product quantization and what problem does it solve?**  
A: PQ compresses vectors by splitting each high-dimensional vector into m sub-vectors, then encoding each sub-vector as a cluster ID from a learned codebook. A 768-dim float32 vector (3,072 bytes) can be reduced to 16 bytes (one byte per sub-vector with m=16) — a 192× compression. This allows billion-scale datasets to fit in RAM that would otherwise require hundreds of GB. The tradeoff is reduced recall due to lossy compression; oversampling + re-ranking with full-precision vectors is used to recover accuracy.

**Q: What is the difference between post-filtering and iterative filtering?**  
A: Post-filtering runs ANN search over all vectors, then discards results that fail the metadata predicate. This is simple but can return far fewer than k results when the filter is selective, requiring over-fetching. Iterative filtering (used by DiskANN) applies predicates during graph traversal — only following nodes that pass the filter, ensuring exactly k valid results are returned without over-fetching. It's more efficient for selective filters at the cost of potentially traversing more graph edges.

**Q: Why does HNSW require more memory than IVF?**  
A: HNSW stores the entire graph in RAM — each node maintains M bidirectional links to its neighbors. For N vectors with M=32, this means 32N additional pointers in memory. IVF only stores the vectors grouped into clusters and a small array of cluster centroids; no graph overhead. The memory difference becomes significant at scale: for 10M vectors, HNSW needs ~2.6 GB just for the graph edges (at M=32).

**Q: What is the curse of dimensionality and why does it matter for vector search?**  
A: In high-dimensional spaces, data points become roughly equidistant from each other — the ratio of the farthest to the nearest neighbor approaches 1 as dimensionality grows. This makes distance-based nearest neighbor search increasingly less meaningful (all points look equally "close"). It also renders traditional spatial tree indexes ineffective — they partition space based on coordinate splits that stop being discriminative in high dimensions. This is why specialized ANN indexes (HNSW, IVF) that exploit the data distribution are necessary.

**Q: When would you use DiskANN over HNSW?**  
A: DiskANN when the dataset is too large to fit its graph in RAM — it stores the graph on SSD and caches a small working set in memory. It achieves recall comparable to HNSW (especially with its oversampling + full-precision re-ranking) at a fraction of the RAM cost. The tradeoff is higher query latency due to SSD I/O. HNSW is preferable when the full graph fits in RAM and lowest possible latency is the priority.
