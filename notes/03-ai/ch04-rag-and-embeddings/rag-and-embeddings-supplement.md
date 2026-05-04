# RAG & Embeddings — Advanced Patterns, Evaluation, and Failure Modes

> **Companion to:** [RAGAndEmbeddings.md](rag-and-embeddings.md)  
> This document adds: advanced RAG patterns (HyDE, FLARE, query decomposition), evaluation frameworks (RAGAS), sparse vs. dense embeddings, common failure modes, and a structured interview Q&A.

---

## 1. Why Basic RAG Breaks — Common Production Failure Modes

Before advancing to sophisticated techniques, understand where vanilla RAG (chunk → embed → retrieve → generate) fails. Most failures fit into three categories:

### Failure Mode 1: Retrieval Failures (Wrong Chunks Retrieved)

**Symptom:** The answer exists in the corpus but the returned chunks don't contain it.

| Root Cause | Description | Fix |
|-----------|-------------|-----|
| Semantic gap | Query phrasing ≠ document phrasing (e.g., "myocardial infarction" vs. "heart attack") | HyDE, query expansion, synonym injection |
| Chunk too large | The relevant sentence is diluted inside a 1,024-token chunk | Smaller chunks or hierarchical chunking |
| Wrong metric | Using dot product on non-normalized vectors | Normalize or switch to cosine |
| Vocabulary mismatch | Technical acronyms not in embedding training set | Domain-specific fine-tuned embeddings |

### Failure Mode 2: Utilization Failures (Right Chunks, Wrong Answer)

**Symptom:** Retrieved chunks are relevant, but the LLM still generates a wrong or incomplete answer.

| Root Cause | Description | Fix |
|-----------|-------------|-----|
| Lost-in-the-middle | LLM attends primarily to beginning and end of context, ignoring middle chunks | Reorder: put most relevant chunks first and last |
| Context overflow | Too many chunks; relevant detail drowns in noise | Reduce k; add re-ranking |
| Conflicting chunks | Two retrieved chunks contradict each other | Source filtering; add chunk provenance to prompt |
| No relevant answer in corpus | Corpus doesn't have the answer; LLM hallucinates | "I don't know" fallback + citation requirement |

### Failure Mode 3: Generation Failures (Faithful Retrieval, Unfaithful Response)

**Symptom:** The LLM ignores the retrieved context and answers from parametric memory.

**Fix:** Use explicit grounding instructions in the system prompt: *"Base your answer ONLY on the provided context. If the context does not contain sufficient information, say 'I don't have enough information in the provided documents.'"*

---

## 2. Advanced RAG Patterns

### 2.1 HyDE — Hypothetical Document Embeddings

**The problem:** The embedding of a user question and the embedding of the answer document are inherently asymmetric — questions are short and interrogative; documents are long and declarative. This mismatch degrades cosine similarity scores.

**The solution:** Instead of embedding the raw query, ask the LLM to generate a **hypothetical document** that would answer the query, then embed that hypothetical document for retrieval.

```
User Query:  "Does the Veggie Feast pizza contain any nuts?"

Step 1 — Generate hypothetical answer:
LLM: "The Veggie Feast pizza does not contain nuts. Its allergens include
      dairy (mozzarella) and gluten (standard base). A gluten-free base is
      available on request."

Step 2 — Embed the hypothetical answer (not the query)
embed("The Veggie Feast pizza does not contain nuts...dairy...gluten...")

Step 3 — Search the vector store with this embedding
```

**Why it works:** The hypothetical answer lives in the same semantic space as real document chunks — both are declarative statements about the topic. Cosine similarity is thus more meaningful.

**Tradeoff:** One extra LLM call per query. For high-throughput systems, this may be too expensive. Use on queries where a simple embedding retrieval returns poor results.

---

### 2.2 FLARE — Forward-Looking Active Retrieval

**The problem:** A single upfront retrieval step may miss information needed for later parts of a multi-sentence response. By the time the model is generating sentence 4, the context it retrieved at step 0 may be stale or insufficient.

**The solution:** The model generates its response **token-by-token** and monitors its own confidence. When it detects low-confidence text (measured by low token probabilities), it **pauses, issues a retrieval query, and regenerates the low-confidence passage** with the new context:

```
Generation in progress:
"The Veggie Feast contains dairy and gluten. For a nut-free option, the [LOW CONFIDENCE →
                                                                         PAUSE AND RETRIEVE]"
→ Retrieval query: "Mamma Rosa nut allergen kitchen policy"
→ Retrieved: "All Mamma Rosa pizzas are prepared in a nut-free kitchen."
→ Continue: "...entire Mamma Rosa kitchen is nut-free, so cross-contamination risk is minimal."
```

**When to use:** Long-form document generation (reports, summaries) where different sections require different grounding evidence.

---

### 2.3 Query Decomposition

**The problem:** Complex multi-part questions embed multiple retrieval needs in one query. A single embedding of the entire query produces a muddled vector that retrieves mediocre results for each sub-need.

**The solution:** Decompose the user query into atomic sub-queries, retrieve for each independently, then synthesize:

```
Original: "Compare the chunking strategy and distance metric defaults used by 
           LangChain vs. Semantic Kernel for RAG applications."

Decomposed:
  Q1: "LangChain default chunking strategy"
  Q2: "Semantic Kernel default chunking strategy"  
  Q3: "LangChain distance metric defaults for vector search"
  Q4: "Semantic Kernel distance metric defaults for vector search"

→ Retrieve top-3 for each sub-query (12 chunks total)
→ Synthesize across all chunks
```

**Two variants:**
- **Sequential decomposition:** Each sub-query may depend on the result of the previous (useful when sub-task B requires knowing the answer to sub-task A)
- **Parallel decomposition:** All sub-queries are independent and can be retrieved simultaneously

---

### 2.4 Step-Back Prompting

Before searching, ask the model to generate a **more general version** of the query — a "step back" to the underlying principle — and retrieve on that too. This catches cases where the corpus has general explanations but not the exact specific fact.

```
Specific: "What is the nprobe default for IVF when rows > 1M?"
Step-back: "How does IVF index tuning work in pgvector?"

→ Retrieve on both; merge results
```

---

### 2.5 Contextual Retrieval (Anthropic, 2024)

Before embedding each chunk, prefix it with an LLM-generated statement that situates the chunk in its document context:

```
Original chunk: "Revenue grew by 3% over the previous quarter."

Contextualized chunk: 
"This excerpt is from ACME Corp's Q2 2023 SEC filing (Section 4: Financial Highlights). 
Revenue grew by 3% over the previous quarter."
```

Anthropic reported a **67% reduction in retrieval failures** when combined with BM25 hybrid retrieval. The cost: one LLM call per chunk at ingestion time. Prompt caching can reduce this by ~90%.

---

## 3. Sparse vs. Dense Embeddings — The Full Picture

The main notes focus on dense embeddings (transformer encoders producing a fixed-size float vector). For production RAG, understanding the sparse alternative is equally important.

### Dense Embeddings (Covered in Main Notes)

- Encodes **semantic meaning** into a continuous high-dimensional vector
- Two semantically identical sentences in different words get similar vectors
- Good at paraphrase matching and conceptual retrieval
- Poor at exact keyword matching (the exact word "DiskANN" may not be reliably distinguished from "HNSW")

### Sparse Embeddings (Learned Sparse Retrieval)

Sparse embeddings represent text as a **bag-of-weighted-terms** — similar to TF-IDF. Weights can be produced by a learned neural model (e.g., **SPLADE**, trained with FLOPS regularization) or by classical term-scoring functions (e.g., **BM25**, a retrieval function — not a neural model):

> **BM25 vs SPLADE:** BM25 is a probabilistic term-frequency scoring *function* with no trainable parameters. SPLADE is a true learned sparse embedding model that produces vocabulary-dimension vectors via a masked-LM head. They are not the same kind of thing.

```
Dense vector (1536 dims):  [0.12, -0.33, 0.04, ..., 0.87]  ← 1536 floats

Sparse vector (50k vocab): {"cat": 2.1, "feline": 1.8,       ← only non-zero terms
                             "mammal": 0.9, "purring": 0.4}    stored
```

**Key differences:**

| Aspect | Dense | Sparse |
|--------|-------|--------|
| **Matching type** | Semantic / conceptual | Lexical / keyword |
| **Dimensionality** | 768–3072 floats | Up to vocab size (50k+), >90% zeros |
| **Storage** | Fixed per vector | Compressed (only non-zero terms) |
| **Out-of-vocabulary** | Generalizes | Fails on unknown terms |
| **Exact match** | May miss it | Reliable |
| **Cross-lingual** | Possible (multilingual models) | Language-dependent |

### When Dense Fails, Sparse Succeeds

```
Query: "What is DiskANN?"
Dense retrieval returns: docs about HNSW, IVF (semantically related ANN methods)
Sparse retrieval returns: docs containing the literal string "DiskANN" ← exact match needed
```

### Hybrid is the Production Standard

Production RAG systems (Azure AI Search, Pinecone, Weaviate) combine both in a **hybrid retrieval** pipeline:

```
User Query
    │
    ├─► Dense (semantic) search → ranked list A
    └─► Sparse (BM25/lexical) search → ranked list B
           │
    Reciprocal Rank Fusion (RRF)  
           │
    └─► Merged ranked list → Top-K chunks → LLM
```

**RRF formula:** `score(d) = Σ 1 / (k + rank_i(d))` where k=60 is standard.  
This is robust to outlier rank positions and doesn't require score normalization across methods.

---

## 4. RAGAS — Evaluating Your RAG Pipeline

You cannot improve what you don't measure. RAGAS (Retrieval-Augmented Generation Assessment) is the standard evaluation framework — it assesses four independent dimensions:

### The Four RAGAS Metrics

| Metric | Measures | Formula Intuition | Goal |
|--------|----------|-------------------|------|
| **Faithfulness** | Does the answer contain only claims supported by the retrieved context? | Fraction of answer claims entailed by context | 1.0 |
| **Answer Relevancy** | Is the answer relevant to the question? | Similarity of question to N reverse-engineered questions from the answer | 1.0 |
| **Context Precision** | Are the retrieved chunks all relevant? (Precision) | Fraction of retrieved chunks that are relevant | 1.0 |
| **Context Recall** | Were all relevant facts retrieved? (Recall) | Fraction of ground-truth facts covered by retrieved chunks | 1.0 |

### Diagnosing with RAGAS

```
High Faithfulness, Low Context Recall → Retrieval problem: relevant chunks not retrieved
Low Faithfulness, High Context Recall → Generation problem: LLM ignoring retrieved context  
Low Context Precision              → Too many irrelevant chunks retrieved (reduce k or add re-ranking)
Low Answer Relevancy               → LLM is answering a different question than what was asked
```

### Implementing a Basic RAGAS Evaluation Loop

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

# Each item: {"question": ..., "answer": ..., "contexts": [...], "ground_truth": ...}
dataset = load_your_evaluation_dataset()

results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)

print(results.to_pandas())
```

**Pro tip:** Build a **golden eval set** of 50–100 representative queries with manually verified correct answers. Run RAGAS on every pipeline config change before deploying.

---

## 5. The Lost-in-the-Middle Problem (Empirical Finding)

A 2023 Stanford study ("Lost in the Middle") found that LLMs perform significantly worse when the relevant answer appears in the **middle** of a long context window compared to the beginning or end. Performance follows a U-shaped curve:

```
Recall (%)
    ▲
100 │█                                                   █
 90 │  █                                               █
 80 │    █                                           █
 70 │      ████                               ████
 60 │          ████                       ████
 50 │              ████████████████████
    └──────────────────────────────────────────────────►
    Position 1   Position 5    Position 10   Position 20 (last)
    (beginning)                               (end)
```

**Practical fix:** Place the most relevant retrieved chunk **first and last**. Less critical context goes in the middle. This is called **"Lost-in-Middle-aware reordering."**

LangChain implements this with `LongContextReorder`:
```python
from langchain.document_transformers import LongContextReorder
reorder = LongContextReorder()
reordered_docs = reorder.transform_documents(docs)
```

---

## 6. Embedding Model Selection Cheatsheet

| Use Case | Recommended Model | Why |
|----------|-------------------|-----|
| General English RAG (cost-sensitive) | `text-embedding-3-small` | Best MTEB/$ ratio |
| High-accuracy English RAG | `text-embedding-3-large` | Highest MTEB score |
| Multilingual corpus | `multilingual-e5-large` | 100+ languages |
| On-premises / private data | `all-MiniLM-L6-v2` | Fast, no API calls |
| Code retrieval | `code-search-babbage-code-001` | Trained on code corpora |
| Image+text hybrid RAG | `voyage-multimodal-3`, CLIP | Unified embedding space |

**The dimensionality trick (Matryoshka embeddings):** `text-embedding-3-*` models support **dimension truncation** — you can use only the first 256 or 512 dimensions. This reduces storage and search cost with a small accuracy tradeoff:

```python
# Request 256-dim instead of full 1536-dim
response = openai.embeddings.create(
    input="your text",
    model="text-embedding-3-small",
    dimensions=256  # truncate to 256 dims
)
```

---

## 7. RAG Architecture Decision Tree

```
Is your corpus > 10M tokens?
    YES → You need a vector database (not just in-memory or pgvector)
    NO  → pgvector or in-memory FAISS may suffice

Do queries mix exact keywords with semantic meaning?
    YES → Hybrid retrieval (dense + sparse / BM25 + vector)
    NO  → Dense-only may suffice

Do you need answer faithfulness guarantees?
    YES → Add a citation/source requirement in the prompt + RAGAS evaluation
    NO  → Standard RAG prompt

Is retrieval quality currently poor?
    YES → Try in order: larger chunks → smaller chunks → semantic chunking → HyDE → query decomposition
    NO  → Optimize for cost

Are you dealing with multi-hop questions?
    YES → Add query decomposition + multi-step retrieval
    NO  → Single-step retrieval is fine

Is your document set updating frequently?
    YES → Streaming ingestion + HNSW (supports real-time inserts) 
    NO  → Batch ingestion + IVF or DiskANN
```

---

## 8. Interview Q&A — RAG and Embeddings

**Q: Explain the difference between the ingestion pipeline and the query pipeline in RAG.**  
A: The ingestion pipeline runs offline: load documents → clean → chunk → embed → store in vector DB. The query pipeline runs at inference time: embed the user query → similarity search → retrieve top-k chunks → compose a prompt with retrieved context → LLM generates a grounded answer. The critical constraint is that both pipelines must use the same embedding model and normalization.

**Q: What is semantic dilution and how does chunking address it?**  
A: Embedding models produce a single fixed-size vector regardless of input length. A 3,000-word document chunk and a 100-word chunk both produce one vector. The long chunk's vector is an "average" over all its topics, making it less discriminative for similarity search. Chunking breaks the document into focused segments so each vector represents a narrower, more searchable concept.

**Q: What is HyDE and why does it improve retrieval?**  
A: HyDE generates a hypothetical answer to the user's query, then embeds that answer for retrieval. This resolves the semantic asymmetry between short interrogative queries and long declarative documents — the hypothetical answer lives in the same linguistic register as real document chunks.

**Q: Explain hybrid retrieval. Why is it better than pure vector search?**  
A: Hybrid retrieval combines dense (semantic) vector search with sparse (BM25/lexical) search, merging results via Reciprocal Rank Fusion. It excels because dense search finds paraphrases and conceptual matches while sparse search finds exact keyword matches. Neither alone handles all query types — hybrid covers both.

**Q: What are the four RAGAS metrics? What does each diagnose?**  
A: Faithfulness (are claims grounded in context?), Answer Relevancy (does the answer address the question?), Context Precision (are retrieved chunks all relevant?), Context Recall (was all necessary information retrieved?). Together they pinpoint whether failures are in retrieval or generation.

**Q: What is the lost-in-the-middle problem?**  
A: LLMs attend disproportionately to content at the beginning and end of long contexts. Relevant information in the middle of a large context window is often underweighted, degrading answer quality. Fix: reorder retrieved chunks so the most important are placed first and last.

**Q: What is the difference between CLS pooling and mean pooling?**  
A: CLS pooling uses the final hidden state of the `[CLS]` token as the sentence representation — efficient but depends on the model being trained with this objective. Mean pooling averages all token embeddings (excluding padding) — often outperforms CLS because it incorporates signal from every token. Most modern embedding models (SentenceTransformers) use mean pooling.

**Q: When should you use contextual retrieval vs. standard chunking?**  
A: Use contextual retrieval when your corpus has many chunks that are meaningful only in the context of the broader document (e.g., "Revenue grew by 3% quarter-over-quarter" is meaningless without knowing which company and which quarter). The extra LLM call per chunk costs money but can reduce retrieval failures by ~67%.

**Q: How does chunk size affect ANN recall and LLM context window usage?**  
A: Smaller chunks → more vectors in the index → higher chance a relevant chunk is in top-k (better recall) but each chunk has less context for answer generation. Larger chunks → fewer vectors → possibly lower recall but each chunk contains more context. You must also ensure total retrieved context (k × chunk_size) fits within the LLM's context window minus prompt/answer overhead.
