# RAG and Embeddings — Grounding LLMs in Retrieved Knowledge

**Status:** Complete

Retrieval-Augmented Generation (RAG) is the dominant architecture for production LLM systems with knowledge requirements beyond the training cutoff. This chapter covers the full retrieval stack: embedding semantics, chunking strategies, dense retrieval, hybrid search (BM25 + dense), HyDE (Hypothetical Document Embeddings), and intent routing.

## Contents

- [rag-and-embeddings.md](rag-and-embeddings.md) — Core chapter content
- [rag-and-embeddings-supplement.md](rag-and-embeddings-supplement.md) — Extended implementation patterns
  - Embedding fundamentals: Word2vec → Sentence-BERT → modern bi-encoders
  - Chunking strategies and their accuracy impacts
  - Dense retrieval with cosine similarity
  - Hybrid search: combining BM25 (keyword) and dense (semantic) scores
  - HyDE: generating hypothetical answers to improve query embeddings
  - Intent routing: dispatching to the right retrieval strategy
  - 7 common RAG misconceptions

## Learning Objectives

After completing this chapter, you should be able to:

1. **Understand embeddings as a retrieval mechanism**
   - Explain how sentence embeddings encode semantic similarity (not keyword overlap)
   - Describe the bi-encoder architecture (Sentence-BERT) and why it enables fast retrieval at scale
   - Distinguish embedding models from generation models (different training objectives)

2. **Design a retrieval pipeline**
   - Choose chunking strategy based on document structure (fixed-size vs. semantic vs. hierarchical)
   - Implement hybrid search by combining BM25 and dense retrieval scores with Reciprocal Rank Fusion
   - Apply HyDE to improve retrieval for short or ambiguous queries

3. **Reason about RAG failure modes**
   - Explain why retrieved context can still produce hallucinations (model ignores context)
   - Describe when RAG helps (knowledge-intensive tasks) vs. when it doesn't (reasoning-intensive tasks)
   - Design intent routing to dispatch queries to the right retrieval strategy

## Prerequisites

- **Ch.01** — Attention mechanics (cross-attention between query tokens and retrieved document tokens is how the model reads retrieved context)
- **Ch.04** — Context window budgeting (retrieved chunks consume context tokens; VRAM constraints bound how much you can retrieve)
- **Ch.05** — Prompt engineering (retrieved context is injected into the prompt; understanding prompt structure matters)

## Key Concepts

| Concept | Analogy | Why It Matters |
|---|---|---|
| **Embedding** | Coordinates in meaning-space | Enables semantic search: "fast car" finds "high-speed vehicle" |
| **Chunking** | Dividing a book into indexed cards | Too large: noisy retrieval. Too small: loses context. |
| **Dense retrieval** | Semantic nearest-neighbor search | Finds conceptually similar passages regardless of keyword overlap |
| **BM25 (sparse retrieval)** | TF-IDF keyword matching | Excellent for exact terms (names, codes, IDs); poor for paraphrase |
| **Hybrid search** | Combining a librarian and a semantic index | Covers both keyword-exact and concept-similar queries |
| **HyDE** | Ask "what would the answer look like?" then retrieve on that | Bridges the lexical gap between short queries and long documents |
| **Intent routing** | Dispatcher before retrieval | Different queries need different retrieval strategies (vector vs. SQL vs. web) |

## Key Stat

**Hallucination reduction from RAG on knowledge-intensive tasks:**
- Baseline (no RAG): ~38% hallucination rate on factual QA
- With RAG: ~4% hallucination rate (Lewis et al., 2020, Facebook AI)

The 38%→4% figure is dataset-specific (NaturalQuestions, TriviaQA) but directionally robust across knowledge-intensive benchmarks.

## ML → LLM Bridge

If you completed **notes/02**, these connections are direct:

- **Ch.07 (Contrastive Learning)**: Sentence-BERT embedding training uses a contrastive objective identical in structure to SimCLR — (sentence, paraphrase) as a positive pair, random sentences as negatives. The NT-Xent loss minimizes distance between paraphrases and maximizes distance between semantically different sentences. The embeddings from Ch.07 are the dense retrieval backbone.
- **Ch.08 (Self-supervised learning / MAE)**: Embedding models trained with masked language modeling (e.g., BERT base) are direct descendants of the MAE pretraining paradigm from notes/02 Ch.8 — recover masked tokens from context. The resulting representations encode position in semantic space.

## Quick Start

```bash
code notes/03-ai/ch07-rag-and-embeddings/rag-and-embeddings.md
```

## Common Questions

**Q: Does RAG replace fine-tuning?**
A: They solve different problems. RAG provides dynamic, up-to-date factual knowledge that the model didn't see in training. Fine-tuning bakes in consistent *behavior* (style, format, domain terminology). Most production systems use both: fine-tuned instruct model + RAG for knowledge retrieval.

**Q: Why not just increase context length to include all documents?**
A: (1) Quadratic attention cost — processing 1M tokens of context is feasible but expensive. (2) "Lost in the middle" phenomenon: models attend strongly to beginning and end of long contexts, weakly to the middle. Retrieving the 3–5 most relevant chunks is usually better than including everything. (3) Context windows have hard limits; most enterprise corpora exceed 1M tokens.

**Q: What's the difference between an embedding model and a generation model?**
A: Embedding models (SBERT, text-embedding-ada-002) are typically encoder-only or bi-encoder architectures trained to produce semantically meaningful vectors. They don't generate text. Generation models (GPT-4, Claude) produce token-by-token text. RAG pipelines use both: the embedding model retrieves; the generation model answers using retrieved context.

## Next Chapter

[Ch.08 — Vector Databases](../ch08-vector-dbs/README.md): Ch.07 covers *what* to retrieve and *why*. Ch.08 covers *how to store and search at scale* — ANN index types (HNSW, IVF, DiskANN, PQ), distance metrics, and the production architecture decisions that make vector search viable at 100M+ documents.
