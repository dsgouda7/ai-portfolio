# Vector Databases — Indexing, Scaling, and Production Architecture

**Status:** Complete

The engineering chapter for production retrieval: approximate nearest neighbor (ANN) indexing, why brute-force search doesn't scale to 100M documents, the trade-off topology of HNSW vs. IVF vs. DiskANN vs. PQ, distance metrics, and the architecture decisions behind Pinecone, Weaviate, Qdrant, and pgvector.

## Contents

- [vector-dbs.md](vector-dbs.md) — Core chapter content
- [vector-dbs-supplement.md](vector-dbs-supplement.md) — Implementation patterns and benchmarks
  - Why O(N) brute-force search hits a wall at 500K–1M documents
  - Flat index (exact) → IVF (inverted file) → HNSW (hierarchical navigable small world) → PQ (product quantization)
  - DiskANN: billion-scale search with disk-resident indices
  - Distance metrics: cosine, dot product, L2 — when each is correct
  - Filtering under ANN: pre-filter vs. post-filter vs. in-filter
  - Production architecture: managed vs. self-hosted vs. pgvector
  - 7 misconceptions about vector databases

## Learning Objectives

After completing this chapter, you should be able to:

1. **Understand ANN index trade-offs**
   - Explain why brute-force search is infeasible at 100M+ documents and quantify the latency degradation
   - Describe how IVF uses inverted file clustering to reduce search scope
   - Explain HNSW graph structure and why its multi-layer design achieves O(log N) expected search time
   - Understand PQ as lossy compression of vectors enabling billion-scale indices in RAM

2. **Choose the right index for a workload**
   - Select between HNSW, IVF, and DiskANN based on dataset size, latency budget, and hardware constraints
   - Explain the recall–latency–memory trade-off triangle and how index parameters (ef, nprobe, m) adjust it
   - Understand when pgvector's flat IVFFLAT is sufficient vs. when a dedicated vector DB is required

3. **Design a production retrieval architecture**
   - Architect a vector search system for 10M, 100M, and 1B document scales
   - Apply pre-filter vs. post-filter vs. in-filter strategies for metadata constraints
   - Select between Pinecone (fully managed), Weaviate/Qdrant (open-source cloud), and pgvector (in-Postgres)

## Prerequisites

- **Ch.07** — RAG and Embeddings (vectors are the output of embedding models; you need to understand what you're indexing before choosing how to index it)
- **Ch.04** *(helpful)* — Memory constraints apply to vector index RAM requirements as much as to model weights

## Key Concepts

| Concept | Analogy | Why It Matters |
|---|---|---|
| **Flat index (brute force)** | Checking every book in the library | Exact recall; infeasible above ~100K vectors |
| **IVF (Inverted File Index)** | First narrow down to the right shelf, then search | Clusters vectors; searches only top K clusters (nprobe parameter) |
| **HNSW** | "Six degrees of separation" graph search | O(log N) expected complexity; best recall–latency in RAM for <100M vectors |
| **PQ (Product Quantization)** | Compressing vectors from 32 floats to 4 bytes | 8× memory reduction; enables billion-scale in-RAM indices |
| **DiskANN** | HNSW where most of the graph lives on SSD | Billion-scale search without billion-dollar RAM budgets |
| **Pre-filter** | Filter the database before searching | Correct recall; only works efficiently with metadata-aware indices |
| **Post-filter** | Search then discard non-matching results | Simple; recall degrades when filtered set is small |
| **Cosine vs. dot product** | Angle vs. length × angle | Use cosine for normalized embeddings; dot product for unnormalized scores |

## Key Stats

**Brute-force latency scaling (768-dim vectors, single-threaded):**
- 50,000 documents → ~15 ms
- 500,000 documents → ~150 ms
- 5,000,000 documents → ~1,500 ms (1.5 seconds)
- 100,000,000 documents → **~30 seconds** — unusable for interactive queries

**HNSW at 100M documents:** ~20–50 ms with 95%+ recall at ef=200 (Qdrant benchmarks).

## Quick Start

```bash
code notes/03-llm/ch08-vector-dbs/vector-dbs.md
```

## Common Questions

**Q: Should I use pgvector or a dedicated vector database?**
A: pgvector (PostgreSQL extension) is appropriate for <5M vectors when you already run Postgres and want to avoid operational complexity. Above 5M vectors, or when you need filtered search with high recall guarantees, dedicated systems (Weaviate, Qdrant, Pinecone) outperform pgvector on latency and recall. The rule: start with pgvector, migrate when you hit the ceiling.

**Q: Why does HNSW use so much RAM?**
A: HNSW stores the graph structure (neighbor lists per node) in addition to the raw vectors. For 1M vectors at 768 dimensions (fp32): ~3GB vectors + ~2GB graph ≈ 5GB RAM. PQ compresses the vector storage to ~200MB but keeps the graph — net ~2.2GB. DiskANN externalizes the graph to SSD, cutting RAM to ~400MB.

**Q: What's the difference between recall and accuracy in ANN search?**
A: Recall@K: fraction of the true top-K nearest neighbors returned by the ANN algorithm. Accuracy: typically not used (there's no "correct answer" label). HNSW at default settings achieves 95–99% recall@10, meaning it finds 9–10 of the true 10 nearest neighbors per query. The 1–5% miss rate is the trade-off for O(log N) vs O(N) complexity.

## Track Complete

This is the final chapter of the **notes/03-ai** LLM track.

**Track summary:** Ch.00 bridged ML foundations → Ch.01–02 built the transformer + inference mental model → Ch.03–04 covered training and deployment economics → Ch.05–06 covered behavioral control → Ch.07–08 covered knowledge augmentation at scale.

**Continue to:** [notes/04-multimodal-ai](../../04-multimodal-ai/README.md) — Extends the LLM foundation to vision, audio, and cross-modal reasoning (CLIP, Whisper, diffusion models, GPT-4V).
