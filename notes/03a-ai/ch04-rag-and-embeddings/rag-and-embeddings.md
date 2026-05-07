# End-to-End Guide to Embeddings in RAG Pipelines: From Ingestion to Query

> In the spring of 2013, a young Google researcher named **Tomas Mikolov** was running out of patience with neural language models that couldn't generalize. He had an idea that felt almost too simple: train a shallow network not to *understand* language, but just to *predict neighbors* — what words tend to appear near this word? The result was **Word2vec**. What surprised even Mikolov was that the learned vectors had geometry: *king − man + woman ≈ queen*. Meaning had become arithmetic.
>
> The field spent the next six years refining the recipe. **GloVe** (Pennington, Socher, Manning, Stanford 2014) combined global co-occurrence statistics with local prediction. **fastText** (Bojanowski et al., Facebook 2016) added subword structure so "gluten-free" and "glutenfree" stopped being strangers. But all of these were word-level: each word got one fixed vector regardless of context. *Bank* in "river bank" and *bank* in "bank account" were given the exact same number. The vectors were beautiful, but brittle.
>
> The fix came in 2019 when **Nils Reimers and Iryna Gurevych** at TU Darmstadt published **Sentence-BERT**: a siamese transformer trained with contrastive loss to push semantically similar sentences together and dissimilar ones apart. For the first time, a model could reliably answer *"are these two paragraphs about the same thing?"* — and return a number you could actually trust.
>
> Then **Patrick Lewis and a team at Facebook AI** connected the final wire. In their **2020 RAG paper**, they asked: what if you plugged a retriever into the front of a generative model? Instead of making the LLM memorize every fact during training, let it *look things up* at inference time — search a corpus, read the relevant chunks, then answer. The model could now cite sources instead of hallucinating. By 2023, this pattern — embed a corpus offline, retrieve the nearest chunks at query time, hand them to the LLM — had become the default architecture for any AI system that touches private data.
>
> **You are now building that architecture.** The hallucinated facts you saw in Ch.3 — the 38% wrong answers in the wiki experiment — exist because the model answers from training memory, not from your organization's actual documents. This chapter fixes that. You will embed an internal engineering wiki corpus, build a vector index, and wire a semantic retriever in front of the LLM. By the end, you will ground every factual claim in retrieved documents and hallucination rate will fall from 38% to 4%.
>
> **Where you are in the curriculum.** This is the chapter where you learn what *exactly* gets stored in a vector index, how an embedding model decides two pieces of text are similar, and how a query is matched against millions of chunks. The next chapter — [VectorDBs](../ch05-vector-dbs) — takes the index itself apart (HNSW, IVF, DiskANN). Together they are the foundation for everything else: agents that retrieve before they answer, evaluation pipelines that check grounding, and the entire RAG project under [`projects/ai/rag-pipeline`](../../../projects/ai/rag_pipeline).
>
> **Notation.** $\mathbf{e} \in \mathbb{R}^d$ — embedding vector of dimension $d$; $\text{sim}(\mathbf{q}, \mathbf{k}) = \frac{\mathbf{q} \cdot \mathbf{k}}{\|\mathbf{q}\|\|\mathbf{k}\|}$ — cosine similarity between query and chunk; $k$ — number of retrieved chunks (top-$k$); $c$ — chunk size in tokens.

***

***

## 0 · The Investigation — The Memory Problem

> 🔬 **AI Literacy Kit — Chapter 4:** You've shown that LLMs reason step-by-step. Now: what happens when they don't know your data? Chapter 4 is **"The Memory Problem"** — build a RAG experiment over an internal document corpus; measure hallucination rate before vs. after grounding; prove that retrieval is the answer to domain knowledge gaps.

**The investigation scenario:**

You load your company's internal engineering wiki (200 Markdown files, ~80k words) and ask GPT-4 and Claude questions from it — first without RAG, then with a full retrieval pipeline. You measure the hallucination rate on a 50-question test set.

**Baseline: no grounding**

```
Query: "What's the SLA for our authentication service?"

GPT-4 (no RAG):
"Typical SLA targets for authentication services are 99.9% uptime (three nines),
with p99 latency under 200ms."

Actual company SLA (from wiki): 99.95% uptime, p99 under 50ms, MTTR < 5 minutes
```

**Test results across 50 questions:**
- Without RAG: **38%** of factual answers contain at least one hallucinated number or claim
- With RAG (after this chapter): **4%** hallucination rate

**What this chapter unlocks:**

🚀 **Retrieval-Augmented Generation solves the domain knowledge problem:**
1. **Embeddings**: Convert documents to dense vectors capturing semantic meaning
2. **Chunking strategy**: Fixed-size vs. sentence vs. semantic chunking trade-offs
3. **Retrieval pipeline**: Embed query → cosine similarity → top-k chunks → pass to LLM
4. **Grounding prompt**: "Answer only from the provided context; say I don't know if not found"
5. **Evaluation**: Hallucination rate, answer faithfulness, context precision/recall

✅ **AI Literacy Kit finding after Ch.4:**
RAG reduces hallucination rate from 38% → 4% on internal documents. The remaining 4% are retrieval failures — the relevant document wasn't retrieved. Fixing retrieval is the focus of Ch.5.

***

## 1 · Core Idea

Failures #1 through #4 in §0 share a single root cause: the LLM answers from training memory, not from the organization's actual documents. RAG is the architectural fix that replaces invented answers with retrieved ones.

**Retrieval-Augmented Generation (RAG) is a two-phase pipeline that grounds LLM responses in external knowledge.** Instead of relying solely on parametric memory (what the model learned during training), RAG fetches relevant documents at query time and includes them in the prompt context. This eliminates hallucination of facts that can be looked up.

**For the investigation:** Without RAG, the model invents policy details and SLA values. With RAG, every factual claim is grounded in retrieved chunks from the wiki — the model cannot fabricate what it reads directly from context.

> 💡 **Core idea → investigation:** Replacing hallucinated wiki facts with retrieved ones is the direct path from the §0 38% hallucination rate to the 4% target. Every percentage point improvement here reduces the risk of wrong technical decisions being made from AI-generated answers.

---

## 1.5 · The Practitioner Workflow — Your 5-Phase RAG Pipeline

The §0 finding — "38% of factual answers contain at least one hallucinated claim" — is answered by this workflow: a concrete 5-phase sequence that converts a static document corpus into a grounded, queryable knowledge base.

> ⚠️ **Two ways to read this chapter:**
> - **Theory-first (recommended for learning):** Read §0→§18 sequentially to understand the concepts, then use this workflow as your reference
> - **Workflow-first (practitioners with existing knowledge):** Use this diagram as a jump-to guide when working with real data

**Before diving into embeddings and vector databases, understand the end-to-end workflow you'll follow with every RAG implementation:**

> 📊 **What you'll build by the end:** A production RAG system with ingestion pipeline (chunk → embed → index) and query pipeline (embed query → retrieve → generate) achieving <5% error rate on factual queries.

```
Phase 1: CHUNK              Phase 2: EMBED              Phase 3: STORE              Phase 4: RETRIEVE           Phase 5: GENERATE
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Split documents:            Convert text to vectors:     Build searchable index:     Match query to docs:        LLM synthesizes answer:

• Load raw docs (PDF,       • Load embedding model      • Initialize vector DB       • Embed user query          • Compose prompt with
  HTML, JSON, CSV)           (OpenAI, sentence-trans)    (Chroma, Pinecone, FAISS)   (same model as Phase 2!)     retrieved chunks
• Recursive splitting       • Batch embed all chunks    • Insert chunk vectors +     • ANN search (HNSW/IVF)     • Generate grounded
  (400-512 tokens)          • L2-normalize vectors        metadata                   • Optional: Re-rank top-k     response
• 10-20% overlap           • Store for indexing        • Build ANN index           • Return top-k chunks        • Cite sources

→ DECISION:                → DECISION:                 → DECISION:                 → DECISION:                 → DECISION:
  Which chunking strategy?   Which embedding model?      Which vector DB?            How many chunks (k)?        Include citations?
  • Fixed: Fast baseline     • OpenAI ada-002:           • Chroma: Local, free       • k=5: Typical default      • Yes: Better trust
  • Recursive: Respects        $0.10/1M tokens            prototyping                • k=10: More context         & debuggability
    structure (80% use this)  • sentence-trans:          • Pinecone: Managed         • k=20: Broad coverage      • No: Faster inference
  • Semantic: +2-3pp           Free, local, slower         cloud, $70/mo                but dilutes relevance
    recall, slow ingestion    • BGE: SOTA open-source    • FAISS: In-memory,
                                                           fastest for small scale
```

> 💡 **How to use this workflow:** Execute Phase 1→2→3 once during ingestion (offline, can take hours). Then Phase 4→5 run per query (online, must complete in <3s p95). The sections above teach WHY each phase works; refer back here for WHAT to do.

> ➡️ Phase 1–3 (ingestion) detail in §5; Phase 4–5 (query) detail in §6. Ch.5 covers the ANN index underlying Phase 3 (HNSW, IVF, DiskANN).

---

## 2 · Running Example: The Wiki Knowledge Problem

The §0 test case — "What's the SLA for our authentication service?" — is the concrete form of every domain-knowledge failure: the model answers confidently from training data, not from the organization's actual policy documents. This section traces that exact failure so every concept in §3–§13 has a before/after anchor.

**Scenario:** An LLM is asked about the organization's authentication service SLA. Without RAG, it responds with "typical SLA targets are 99.9% uptime" — but the actual company SLA (from the internal wiki) is 99.95% uptime, p99 under 50ms, MTTR < 5 minutes. The model cannot know the org-specific value from training alone.

**Why CoT reasoning alone isn't enough:** Chain-of-thought helps with logic ("if A < B and B < C, then A < C"), not facts ("what is the actual organization SLA?"). The model has access to reasoning machinery but no mechanism to look up private documents.

**What this chapter unlocks:** RAG pipeline that grounds every factual claim in retrieved documents:
1. **Ingestion time:** Chunk wiki (200 docs) → Embed with `text-embedding-3-small` → Index in Chroma vector DB
2. **Query time:** Embed "authentication service SLA" → Retrieve top-5 chunks → LLM sees actual wiki content in context → Generate answer: "99.95% uptime, p99 under 50ms, MTTR < 5 minutes"

**Expected improvement:** Hallucination rate 38% → 4% (all policy and technical spec hallucinations eliminated from the retrieved corpus).

> 💡 **Running example → investigation:** The "99.9% vs 99.95% uptime" discrepancy is a hallucination on a critical technical spec. Across the 50-question test set, 19 such hallucinations appeared without RAG — any one of which could lead to a wrong architectural decision. RAG eliminates this class of error.

> ➡️ The ingestion half of this fix (chunk → embed → index) is built in §5; the query half (embed query → retrieve → generate) is built in §6.

---

## 3 · Embeddings Fundamentals: Transforming Text into Vectors

**You need embeddings to solve semantic mismatch** — when an engineer asks "service uptime targets," wiki documents say "99.95% availability" without the word "uptime." Keyword search misses it; embeddings find it. **Embeddings transform text into vectors where meaning becomes measurable** — similar concepts cluster together in high-dimensional space, so "uptime" and "availability" become neighbors even with different exact wording.

**For the investigation:** When a query asks about "authentication service SLA," you need to retrieve wiki documents that discuss response time budgets, uptime commitments, and on-call policies — even though they may use different terminology. That's semantic similarity, and it's what embeddings deliver.

Modern embedding models are built on **transformer encoder** architectures. Unlike decoder-only models (GPT, Llama) that generate text autoregressively, encoder models process the entire input simultaneously and produce contextual representations for each token.

### How Text Embeddings Are Created

The transformer encoder consists of stacked self-attention layers. Each layer allows every token to attend to every other token, building increasingly abstract representations. The process follows a specific flow:

1. **Tokenization:** Input text is split into tokens, with special tokens added (e.g., `[CLS]` and `[SEP]`)
2. **Token + Position Embeddings:** Each token receives an initial embedding (typically 768-dimensional)
3. **Self-Attention Layers:** The input passes through 6–12 stacked self-attention and feed-forward layers
4. **Final Hidden States:** The model outputs a matrix of contextual token embeddings (e.g., `[768-dim] × N tokens`)
5. **Pooling:** A pooling strategy collapses the per-token representations into a single fixed-size vector

The self-attention mechanism computes attention scores between all token pairs, requiring **O(n²) operations** for a sequence of *n* tokens — which is why most embedding models have sequence length limits (512–8,192 tokens).

### Pooling Strategies

After the transformer produces per-token representations, a pooling strategy collapses them into a single vector. The choice of pooling strategy significantly affects embedding quality:

| Strategy | Mechanism | When Used |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| **\[CLS] Token Pooling** | Uses the final hidden state of the special `[CLS]` token as the sentence embedding. Simple but effective for models trained with this objective | BERT-family models trained with classification heads |
| **Mean Pooling** | Averages all token embeddings (excluding padding). Often outperforms \[CLS] pooling because it incorporates information from every position equally | Most modern embedding models (SentenceTransformers, etc.) |
| **Max Pooling** | Takes the maximum value across tokens for each dimension. Captures the strongest signal for each feature | Certain specialized retrieval tasks |
| **Last Token Pooling** | Uses the final non-padding token. Common in decoder-based embeddings (e5-mistral, GritLM) where the last token aggregates bidirectional context through causal attention modifications | Decoder-based embedding models |

Implementation in code is straightforward:

```python
# CLS pooling
embedding = hidden_states[0] # [CLS] is first token

# Mean pooling
embedding = mean(hidden_states * attention_mask)

# Max pooling
embedding = max(hidden_states, dim=0)

# Last token pooling
embedding = hidden_states[last_non_pad_idx]
```
### Training Objective: Contrastive Learning

Embedding models are **not** trained to predict tokens. They are trained with **contrastive learning objectives** that teach the model to produce similar vectors for semantically similar text and dissimilar vectors for unrelated text. The most common training objective is **InfoNCE** (Noise Contrastive Estimation), which treats embedding as a classification problem: given a query, identify the correct positive from a set of negatives:

 L = -log(
 exp(sim(q, p⁺) / τ)
 ─────────────────────────────────────
 exp(sim(q, p⁺) / τ) + Σᵢ exp(sim(q, pᵢ⁻) / τ)
 )

Where `q` = query embedding, `p⁺` = positive (similar) embedding, `pᵢ⁻` = negative embeddings, and `τ` = temperature parameter.

**In practice:** This formula is why "service uptime" and "availability target" cluster close together in embedding space — the model learned from millions of (question, answer) pairs that these phrases co-occur in similar contexts. **For the investigation:** This training objective is why semantic search works at all — your retrieval quality depends on how well the embedding model was trained with contrastive learning.

### Dimensionality and Storage Costs

**For the wiki experiment with 200 documents, storage is cheap — but for an enterprise knowledge base with millions of pages, dimensionality matters.** Embedding dimensions represent a tradeoff between **expressiveness**, **storage**, and **computational cost**. Every piece of text passed to a model like `text-embedding-ada-002` is converted into a vector containing exactly **1,536** floating-point values. From a technical perspective, 1,536 dimensions represent a balance between expressiveness and efficiency — the vector is large enough to encode meaningful semantic detail but small enough to store and search efficiently in most systems.

**Business impact:** If you're embedding 200 wiki docs × 7 chunks each = 1,400 vectors × 6 KB = **8.4 MB total storage** (negligible). But at 100,000 docs × 7 chunks each = 700,000 vectors × 6 KB = **4.2 GB** — now dimensionality and quantization matter for cost.

| Dimensions | Size per Vector (float32) | Example Model |
| ---------- | ------------------------- | ------------------------------- |
| **384** | 1.5 KB | MiniLM, Instructor-XL |
| **768** | 3.0 KB | BERT base, MPNet |
| **1,024** | 4.0 KB | Various proprietary models |
| **1,536** | 6.0 KB | OpenAI `text-embedding-ada-002` |

**Concrete storage calculation:** Storing 10 million vectors at 1,536 dimensions in float32 requires:
`10,000,000 × 1,536 × 4 bytes = 61.4 GB`

Developers need to account for this size when planning storage and memory usage, especially when embedding millions of documents. Vector databases such as Milvus or Zilliz Cloud are well suited for handling vectors of this dimensionality and support indexing methods optimized for dense vectors that can scale horizontally as data grows.

### Model Comparison: OpenAI Embedding Family

| Model | Dimensions | Max Tokens | MIRACL avg | MTEB avg | Price |
| -------------------------- | ---------- | ---------- | ---------- | -------- | ----------------- |
| **text-embedding-3-large** | 3,072 | 8,191 | 54.9 | 64.6 | $0.13 / 1M tokens |
| **text-embedding-ada-002** | 1,536 | 8,191 | 31.4 | 61.0 | $0.10 / 1M tokens |
| **text-embedding-3-small** | 1,536 | 8,191 | 44.0 | 62.3 | $0.02 / 1M tokens |
Note that `text-embedding-3-small` achieves higher MIRACL and MTEB scores than `text-embedding-ada-002` at **one-fifth the cost** ($0.02 vs. $0.10 per 1M tokens), making model selection an important engineering decision.

### Query vs. Document Prefixes

Some embedding models use **different prefixes** for queries versus documents to improve asymmetric retrieval:

 BGE: "Represent this sentence for searching..."
 E5: "query: " for queries, "passage: " for docs
 Nomic: "search_query: " and "search_document: "
 GTE: No prefix required
 MiniLM: No prefix required

Using wrong prefixes degrades retrieval quality — always check model documentation.

**Code Snippet — Phase 2: Batch embed documents with sentence-transformers:**

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model (384-dim, fast, good quality)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Document chunks from Phase 1 (chunking)
chunks = [
    "Margherita Pizza (Large): $14.99, 920 calories",
    "Pepperoni Pizza (Large): $15.99, 1,040 calories",
    "Veggie Garden Pizza (Large): $14.49, 780 calories"
]

# DECISION LOGIC: Batch processing for efficiency
batch_size = 32  # Typical: 16-64 depending on GPU memory

# Generate embeddings (returns numpy array: n_chunks × 384)
embeddings = model.encode(
    chunks,
    batch_size=batch_size,
    show_progress_bar=True,
    normalize_embeddings=True  # L2-normalize for dot product retrieval
)

print(f"Embedded {len(chunks)} chunks")
print(f"Embedding shape: {embeddings.shape}")  # → (3, 384)
print(f"Vector magnitude (after normalization): {np.linalg.norm(embeddings[0]):.4f}")  # → 1.0000

# Store embeddings for Phase 3 (indexing)
np.save('menu_embeddings.npy', embeddings)
```

> 💡 **Industry Standard:** `sentence-transformers` library
> **When to use:** Always for local/free embeddings. SOTA models: `all-MiniLM-L6-v2` (fast, 384-dim), `all-mpnet-base-v2` (higher quality, 768-dim), or `BGE` models (multilingual, SOTA).
> **Common alternatives:** OpenAI `text-embedding-3-small` (API, $0.02/1M tokens), Cohere `embed-v3` (multilingual), `voyage-2` (high-quality retrieval).
> **Production pattern:** sentence-transformers for prototyping → OpenAI API for production if budget allows.

> 💡 **Embed verdict:** 1,500 menu chunks converted to normalized 384-dim vectors — dot product now equals cosine similarity, enabling fast ANN index queries.
> ➡️ Query time must use the same embedding model as ingestion; a model mismatch silently degrades retrieval to near-random results — Ch.5 covers managing this at index-upgrade time.

***

## 4 · Normalization: Why It Matters for Production Systems

**You'll hit this in production when retrieval latency becomes a bottleneck.** For the wiki experiment, with ~1,400 embedded document chunks, retrieval takes ~10ms. Scale to 1.4 million chunks (an enterprise knowledge base), and brute-force cosine similarity becomes a 10-second query — unacceptable.

**For normalized embeddings, dot product equals cosine similarity** — this is why production systems often normalize embeddings at index time and use dot product at query time. **Business payoff:** Dot product is ~2× faster than cosine similarity (one square root elimination), and it enables HNSW/IVF index optimizations that wouldn't work otherwise (see Ch.5 Vector DBs).

### L2 Normalization

 norm = √Σᵢ(vᵢ²)
 v_normalized = v / norm

 After normalization: ||v_normalized|| = 1
 Vector lies on the unit hypersphere

### Benefits of Normalization

**Metric Equivalence:** For normalized vectors, dot product = cosine similarity, and L2 distance is a monotonic transformation of both. You can use the **fastest metric (dot product)** without sacrificing accuracy.

**Index Efficiency:** Many vector indexes (HNSW, IVF) are optimized for specific metrics. Normalizing allows consistent use of inner product indexes across all your data.

**Bounded Similarity:** Normalized embeddings have similarity in \[-1, 1], making threshold-based filtering predictable. Raw embeddings can have unbounded dot products.

**Length Invariance:** Document length affects raw embedding magnitude. Normalization removes this bias.

Many embedding models output normalized vectors by default — check your model's documentation.

### Metric Selection Guide

| Scenario | Recommended Metric |
| ------------------------------------- | --------------------- |
| Normalized embeddings, speed critical | **Dot Product** |
| Non-normalized embeddings | **Cosine Similarity** |
| Using HNSW or IVF indexes | **L2 or Dot Product** |

**Interview-critical insight:** For normalized vectors: `d² = 2(1 - cos(θ))`, meaning Euclidean and cosine become equivalent. This means `argmin L2 == argmax dot product` when vectors are unit-length. This is how FAISS, Milvus, pgvector, and Pinecone support cosine search internally — they normalize vectors and use inner product search.

**For the investigation:** Your prototype uses cosine similarity directly. When you move to production scale (Ch.5), you'll switch to normalized vectors + dot product + HNSW index to hit the <3s latency constraint.

> 💡 **Normalization → latency:** L2-normalizing at index time and using dot product at query time is a free ~2× latency improvement — at 1.5 million chunks that's the difference between 10-second queries (failed p95 budget) and <200ms (within budget). The normalization step costs nothing at ingestion; skipping it costs everything at scale.

***

## 3. Embeddings Beyond Text: Images and Audio

**The wiki investigation doesn't need image embeddings yet** — but if the organization wants to add visual search over architecture diagrams or "find documentation matching this screenshot," you'll need multimodal embeddings. For now, this is optional depth; skip to § 4 if you're focused on the core RAG pipeline.

> 📖 **Optional: Multimodal Embeddings for Future Features**
>
> While text embeddings are the primary focus of most RAG systems, the concept extends to other modalities. If your roadmap includes visual search ("find architecture diagrams matching this description") or audio transcription search ("find the meeting recording where we discussed this architectural decision"), you'll need these techniques.

### Image Embeddings (CLIP and Multimodal Models)

**Multimodal RAG** integrates additional modalities into traditional text-based RAG, enhancing LLMs' question-answering by providing extra context and grounding textual data for improved understanding. The approach of directly embedding images for similarity search bypasses the lossy process of text captioning, boosting retrieval accuracy.

The objective is to embed images and text into a **unified vector space** to enable simultaneous vector searches across both media types. Two primary methods exist:

1. **Multimodal embedding model** (e.g., CLIP, `voyage-multimodal-3`): Embeds both text and images into the same vector space directly. The model `voyage-multimodal-3` has a token limit of 32,000 — far greater than models like CLIP or ImageBind.
2. **LLM-based summarization**: Uses a multimodal LLM to summarize images into text, then passes those summaries to a text embedding model.

A practical implementation using CLIP and FAISS:

```python
# Load CLIP model
model, preprocess = clip.load("ViT-B/32", device="cpu")

# Create image embeddings
image_features = model.encode_image(image_input).float()

# Build FAISS index with inner product
index = faiss.IndexFlatIP(image_features.shape[1])
index.add(image_features)

# Search: embed query image, find nearest neighbors
distances, indices = index.search(query_embedding.reshape(1, -1), 2)
```
Using CLIP-based embeddings further allows fine-tuning with specific data or updating with unseen images.

### Audio Embeddings

Audio encoders (such as OpenAI's Whisper or specialized audio embedding models) convert audio clips into fixed-dimensional vectors, enabling keyword search in audio corpora and audio-based retrieval in multimodal RAG systems.

> 💡 **Multimodal embeddings verdict:** Image/audio RAG is a future feature for the investigation — not needed for the 4% hallucination target. When the organization adds visual search over architecture diagrams, `voyage-multimodal-3` replaces the text-only pipeline with no architecture change needed.

***

## 4. The Two-Phase RAG Pipeline: Ingestion Time vs. Query Time

The §0 system has a query path but no ingestion path — that asymmetry is why every factual answer is invented. Distinguishing which phase a problem lives in is the first debugging question in every production RAG incident.

A complete RAG pipeline consists of two main phases:

**Ingestion Phase (Offline):**
`Documents → Chunking → Embedding → Index/Store in Vector Database`

**Retrieval Phase (Runtime):**
`User Query → Embed Query → Similarity Search → Top-K Results → LLM Generation → Response`

The RAG pipeline connects the model to the information it needs at query time. When a user asks a question, the pipeline retrieves the most relevant documents, prepares that text as context, and includes it in the prompt so the model can generate an answer grounded in retrieved material rather than training data alone.

### Comparison: Ingestion-Time vs. Query-Time Embedding

| Characteristic | Ingestion-Time Embedding | Query-Time Embedding |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| **When it runs** | Offline / batch | Real-time, per-request |
| **Input** | Document chunks (100s–1000s of tokens each) | User query (typically 10–100 tokens) |
| **Volume** | Millions of chunks in a corpus | One query at a time (or small batches) |
| **Latency tolerance** | High (hours acceptable for bulk processing) | Low (sub-second response expected) |
| **Model choice** | Can use larger, higher-quality models | Speed may dictate smaller or API-based models |
| **Caching utility** | Cache entire corpus embeddings permanently | Limited (queries are diverse) |
| **Normalization** | Applied once at storage time | Must match document normalization |

> 💡 **Two-phase split verdict:** Separating ingestion (one-time, offline) from query (per-question, <3s p95) keeps embedding cost at ~$0.000012/query — only the 12-token user query is embedded per question. Re-embedding the corpus per query would cost $0.024/query, making it impractical to scale beyond a small test set.

***

## 5 · Ingestion-Time Pipeline: Preparing the Knowledge Base

The §0 experiment has no ingestion step — there is no offline process that converts the internal wiki into something searchable. This section builds it: document loading, cleaning, chunking, embedding, and indexing run once so every query can retrieve in milliseconds.

### 5.1 Document Loading & Preprocessing

Ingestion is the process of loading documents from multiple sources and in multiple formats, transforming them into a structured form suitable for embedding and retrieval. Raw documents — whether PDFs, Word documents, web pages, or database records — must be transformed into manageable chunks.

**Common sources include:** local files (PDF, DOCX, TXT, CSV), web pages and APIs, databases and data warehouses, cloud storage (S3, GCS, Azure Blob), and enterprise systems (SharePoint, Confluence, Notion).

**Cleaning steps** for a clean corpus that improves both retrieval accuracy and embedding quality:

1. **Remove Noise:** Headers, footers, navigation, ads
2. **Normalize Structure:** Convert to consistent format (JSONL, Parquet)
3. **Preserve Metadata:** Source, timestamp, author, document type
4. **Handle Special Characters:** Normalize encoding (UTF-8)
5. **Maintain Logical Structure:** Keep headings, lists, tables

### 5.2 Metadata Enrichment

Each chunk should carry metadata such as topic, date, and source, so that retrieval can filter and rank results more effectively. Track which document each chunk came from, its position in the document, version information, and any relevant tags — this enables filtering, deduplication, and better citation in your responses.

**Important metadata fields:**

| Field | Purpose |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **source\_id** | Unique document identifier |
| **title** | Document title |
| **created\_date** | Document creation timestamp |
| **author** | Document creator |
| **document\_type** | PDF, email, webpage |
| **topic/category** | For domain filtering |
| **page\_number** | For PDF sources |
| **chunk\_id** | Unique identifier for the chunk |

### 5.3 Chunking (covered in detail in Section 6)

### 5.4 Embedding and Indexing

Each chunk is fed into an embedding model to produce a vector representation. The chunking process directly feeds into the embedding step, where each chunk is converted into vector representations that capture semantic meaning. Once embedded, vectors are inserted into a vector database with their associated metadata. ANN indexes (HNSW, IVF, DiskANN) then enable fast similarity search over these vectors.

**Code Snippet — Phase 3: Index embeddings in Chroma vector database:**

```python
import chromadb
from chromadb.utils import embedding_functions

# Initialize Chroma client (local persistent storage)
client = chromadb.PersistentClient(path="./chroma_db")

# Use same embedding model as Phase 2 (CRITICAL!)
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create collection (like a table in SQL)
collection = client.get_or_create_collection(
    name="pizzabot_menu",
    embedding_function=embedding_fn,
    metadata={"description": "PizzaBot menu corpus"}
)

# Load chunks and metadata from Phase 1
chunks = [
    "Margherita Pizza (Large): $14.99, 920 calories",
    "Pepperoni Pizza (Large): $15.99, 1,040 calories",
    "Veggie Garden Pizza (Large): $14.49, 780 calories"
]

metadatas = [
    {"source": "menu.json", "category": "pizza", "size": "large"},
    {"source": "menu.json", "category": "pizza", "size": "large"},
    {"source": "menu.json", "category": "pizza", "size": "large"}
]

# DECISION LOGIC: Batch insert for efficiency
collection.add(
    documents=chunks,      # Text chunks (will be auto-embedded)
    metadatas=metadatas,   # Structured metadata for filtering
    ids=[f"chunk_{i}" for i in range(len(chunks))]  # Unique IDs
)

print(f"✅ Indexed {collection.count()} chunks in vector DB")
print(f"📊 Collection metadata: {collection.metadata}")
```

> 💡 **Industry Standard:** `chromadb` (local prototyping), `pinecone` (managed cloud), `weaviate` (hybrid search), `qdrant` (production-grade open-source)
> **When to use Chroma:** Prototyping, <100k documents, local development, no infrastructure setup
> **When to use Pinecone:** Production, >1M documents, managed scaling, $70/month+
> **When to use FAISS:** Research, in-memory only, fastest for batch experiments
> **See also:** [Ch.5 Vector DBs](../ch05-vector-dbs) for HNSW/IVF index details

**Batch vs. Streaming Ingestion:**

* **Pre-chunking** (batch) is the most common method. It processes documents asynchronously by breaking them into smaller pieces before embedding and storing them in the vector database. This enables fast retrieval at query time since all chunks are pre-computed.
* **Post-chunking** (streaming) takes a different approach by embedding entire documents first, then performing chunking at query time only on the documents that are actually retrieved. The chunked results can be cached, so the system becomes faster over time.

> 💡 **Index verdict:** 1,500 chunks stored with 384-dim vectors and metadata — brute-force exact search handles this corpus at <10 ms query time.
> ➡️ Metadata filtering cuts the searched space from 1,500 to ~800 pizza chunks; without it every query scans drinks and sides, diluting precision and wasting compute.

***

## 6 · Query-Time Pipeline: From Question to Answer

**This is where latency matters.** When a customer asks "gluten-free options under $15," they expect a response in 2-3 seconds. Your query pipeline must: embed the query (∼50ms), search the index (∼200ms for 1,500 chunks, ∼2s for 1.5M chunks without optimization), and generate an answer (∼1s). **Total budget: <3s p95** to avoid abandonment.

When a user asks a question, the query pipeline activates:

**Step 1 — Query Embedding:** The user's query is processed by the **same** embedding model used at ingestion time to produce a query vector. This step happens in real time and must be low-latency. Engineering considerations include:

* **Batching:** Aggregating multiple concurrent queries into one batch for the embedding model to increase throughput
* **Caching:** Storing embeddings for frequently repeated queries (though cache hit rates in open-ended Q\&A tend to be low)
* **Embeddings must be regenerated when:** source document changes, embedding model changes, chunking strategy changes, or model fine-tuning occurs

**For the investigation:** Query embedding costs ~$0.000012 per query (12 tokens × $0.02 / 1M tokens with `text-embedding-3-small`). At 10,000 daily queries, that's **$0.12/day** — negligible. Latency is the real constraint: OpenAI's API typically returns embeddings in 30-50ms, well within your <3s p95 budget.

**Code Snippet — Phase 4: Query-time retrieval with Chroma:**

```python
import chromadb
from chromadb.utils import embedding_functions

# Load same client and collection from Phase 3
client = chromadb.PersistentClient(path="./chroma_db")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # SAME model as ingestion!
)
collection = client.get_collection(
    name="pizzabot_menu",
    embedding_function=embedding_fn
)

# User query from chat interface
user_query = "What gluten-free options are under 800 calories?"

# DECISION LOGIC: Retrieve top-k with metadata filtering
results = collection.query(
    query_texts=[user_query],         # Embed query (auto, uses embedding_fn)
    n_results=5,                      # Top-5 most similar chunks
    where={"category": "pizza"},      # Filter: only pizza items
    include=["documents", "metadatas", "distances"]
)

# Display retrieved chunks
for i, (doc, meta, dist) in enumerate(zip(
    results['documents'][0],
    results['metadatas'][0],
    results['distances'][0]
)):
    similarity = 1 - dist  # Convert L2 distance to similarity
    print(f"\n[Rank {i+1}] Similarity: {similarity:.3f}")
    print(f"Chunk: {doc}")
    print(f"Metadata: {meta}")

# Expected output:
# [Rank 1] Similarity: 0.89
# Chunk: Veggie Garden Pizza (Large): $14.49, 780 calories. Gluten-free crust available (+$3.00).
# Metadata: {'source': 'menu.json', 'category': 'pizza', 'size': 'large'}
```

> 💡 **Industry Standard:** Hybrid retrieval with BM25 + dense embeddings
> **Pattern:** Run both keyword search (BM25) and semantic search (dense vectors) in parallel, merge with Reciprocal Rank Fusion (RRF)
> **When to use:** Always in production — catches both exact matches ("SKU-1234") and paraphrases ("gluten-free")
> **Implementation:** `rank-bm25` library (Python) or Elasticsearch for keyword, vector DB for dense, merge with RRF formula
> **See also:** [Cohere Rerank API](https://docs.cohere.com/docs/reranking) for cross-encoder re-ranking

> ### ⚠️ Critical Constraint: Query and Document Embeddings Must Use the Exact Same Model
>
> **You cannot use two different embedding models in a RAG pipeline — one for ingesting documents and a different one for embedding the user's query at retrieval time.**
>
> Here is why this is a hard technical requirement:
>
> **1. Vector space incompatibility.** Every embedding model learns its own unique high-dimensional vector space during training. `text-embedding-3-small` and `text-embedding-ada-002` both output 1,536-dimensional vectors, but those 1,536 numbers mean completely different things — the axes of each model's space are independent. A cosine similarity computed between a query vector from Model A and a document vector from Model B is **numerically meaningless** — the math runs without error, but the result does not reflect semantic similarity at all.
>
> **2. Dimensional mismatch.** If the two models have different output dimensions (e.g., Model A produces 768-dim vectors, Model B produces 1,536-dim vectors), the similarity search will fail outright — the vector database cannot compare vectors of different lengths.
>
> **3. Token counting is also model-specific.** The same text tokenizes to a different number of tokens depending on the model's vocabulary. Chunk size limits (e.g., "max 512 tokens") are only meaningful relative to the specific tokenizer of the model being used.
>
> **Mental model:** Think of each embedding model as a different language. The documents stored in the vector DB are all written in "Language A." If you translate the user's query into "Language B" at retrieval time, looking for the closest match in the Language A library produces nonsense results — even if both languages have the same alphabet (same vector dimensions).
>
> **Operational implication:** Treat the embedding model as a **schema-level dependency** of your vector index. If you need to upgrade the embedding model, you must **re-embed the entire corpus** and rebuild the index from scratch. There is no in-place migration path.
>
> | Scenario | Valid? | Consequence |
> |----------|--------|-------------|
> | Ingest with `text-embedding-3-small`, query with `text-embedding-3-small` | ✅ Yes | Correct semantic similarity |
> | Ingest with `text-embedding-ada-002`, query with `text-embedding-3-small` | ❌ No | Meaningless similarity scores |
> | Ingest with `multilingual-e5-large` (768-dim), query with `text-embedding-ada-002` (1536-dim) | ❌ No | Dimensional mismatch — hard error |
> | Upgrade model mid-index (some docs with old model, new docs with new model) | ❌ No | Mixed vector spaces; partial garbage results |

**Step 2 — Similarity Search:** The query vector is used to perform **nearest neighbor search** in the vector database, retrieving the top *k* most similar chunk vectors. The distance metric used must match how embeddings were stored. For normalized embeddings, use dot product (which equals cosine similarity).

**Step 3 — (Optional) Re-ranking:** Bi-encoder retrieval is fast but approximate — the query and document are embedded independently and compared by cosine similarity alone. A **cross-encoder** reads the query and candidate chunk *together* as a single sequence, letting every token attend to every other token, then outputs a scalar relevance score. It is slower but far more accurate.

**Technical definition:** A cross-encoder takes `[CLS] query [SEP] chunk [SEP]` as input and produces a relevance score in [0, 1]. Because the model sees both texts simultaneously, it catches subtleties cosine similarity cannot — a question asking about a *process* versus a *definition* of the same term, or a chunk that is semantically adjacent but contextually useless.

**Intuition:** Bi-encoder retrieval is like matching resumes by keyword: fast, scalable, but it misses nuance. Cross-encoder re-ranking is like having a senior engineer actually read each candidate: slower, but far more accurate about fit.

**For the investigation:** After retrieving 20 chunks for "authentication service SLA requirements," a cross-encoder re-ranks them so chunks about *actual SLA commitments and metrics* rise above generic chunks that merely mention "authentication" in passing. The bi-encoder returned the 20 most *semantically similar* chunks; the cross-encoder re-ranks to the 5 most *relevant* ones.

**Architecture pattern:** Use bi-encoder for top-100 recall (milliseconds), cross-encoder to re-rank to top-5 (adds ~200–400 ms per query). Never run a cross-encoder against the full corpus — at 1.5 million chunks that is a 10-minute query.

> ⚠️ **Re-ranking trap:** High cosine similarity ≠ high relevance. A chunk reading "our menu includes gluten-free options" scores 0.82 similarity against "gluten-free options under 800 calories," but it contains no calorie data — it is useless to the LLM. The cross-encoder catches this; the bi-encoder does not. See § 11 (ColBERT) for token-level late interaction as an alternative re-ranker.

**Step 4 — LLM Prompt Composition:** The top relevant chunks are inserted into the LLM's prompt as reference context. The LLM then generates a grounded answer.

> 💡 **Retrieve verdict:** Semantic search returns Veggie Garden Pizza (similarity 0.89) as top match for "gluten-free under 800 calories" — 91% recall at 10 ms on 1,500 chunks.
> ➡️ The remaining 9% miss rate comes from items with non-standard descriptions; hybrid BM25 + dense retrieval (§ 6.5) closes that gap by catching exact-wording cases dense search misses.

***

## 6.5 · Hybrid Search: BM25 + Dense Retrieval

**The investigation's failure mode you haven't hit yet:** An engineer types "AUTH-SVC-9B" — an internal service identifier for the authentication component. Dense retrieval returns nothing useful because no embedding model was trained on that code. A keyword search returns an exact match in milliseconds. You need both.

### BM25: Sparse Retrieval

**Technical definition:** BM25 (Best Match 25) is a probabilistic ranking function that scores documents by term frequency (how often a query term appears in the document) and inverse document frequency (how rare that term is across the full corpus), with a saturation term to prevent long documents from dominating:

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d) \cdot (k_1 + 1)}{f(t,d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

Where $f(t,d)$ = term frequency of token $t$ in document $d$, $k_1 = 1.2$ controls term-frequency saturation (diminishing returns from repeated terms), $b = 0.75$ controls length normalization, and $\text{avgdl}$ = average document length across the corpus.

**Intuition:** BM25 is keyword search done right. It rewards documents containing your exact query terms, penalizes documents that repeat them obsessively, and adjusts scores so short FAQ entries aren't drowned out by long menu descriptions. It requires no training, no GPU, and no warm-up — just an inverted index built once at ingestion time.

**For the investigation:** BM25 excels when an engineer uses exact terminology: "AUTH-SVC-9B," "MTTR," "99.95%," or "Incident-2024-0231." Dense retrieval misses these because embedding models map rare tokens to generic representations. BM25 catches them through exact term matching.

| Retriever | Strengths | Weaknesses |
|-----------|-----------|------------|
| **BM25 (sparse)** | Exact matches, rare terms, acronyms, SKUs, proper nouns; no training required | Misses paraphrases: "low-calorie" vs. "light option" |
| **Dense (bi-encoder)** | Semantic similarity, paraphrases, intent; handles out-of-vocabulary gracefully | Struggles with rare proper nouns, exact model numbers, domain-specific codes |

### Reciprocal Rank Fusion (RRF)

**Technical definition:** RRF merges two rank lists by replacing raw scores with rank-based weights. For each document, sum its contribution across all rankers:

$$\text{RRF}(d) = \sum_{\text{ranker}} \frac{1}{k + \text{rank}_i(d)}$$

Where $k = 60$ is a smoothing constant that prevents top-ranked documents from dominating, and $\text{rank}_i(d)$ is the 1-indexed position of document $d$ in ranker $i$'s result list.

**Intuition:** BM25 returns a TF-IDF-weighted score that might range from 0 to 80. Dense search returns a cosine similarity between 0 and 1. You cannot add these directly — the scales are incompatible. RRF sidesteps the problem entirely by caring only about *rank order*, not raw score. A document that ranks 3rd in BM25 and 5th in dense search gets a stable combined score regardless of what either raw value was.

**For the investigation:** Hybrid search finds "Authentication Service SLA: 99.95% uptime, p99 < 50ms, MTTR < 5 minutes" even when the engineer types "auth service reliability guarantees." BM25 catches "MTTR" (exact match); dense search catches "reliability guarantees" (semantic similarity). RRF merges both lists — the chunk that ranks high in *both* floats to the top.

```python
from rank_bm25 import BM25Okapi
import numpy as np

def hybrid_rrf(
    query: str,
    dense_ranking: list,   # list of doc IDs ordered by dense similarity (best first)
    corpus: list,          # list of raw text strings indexed by doc ID
    k: int = 60,
    top_n: int = 5
) -> list:
    """Merge dense retrieval and BM25 results with Reciprocal Rank Fusion."""
    # BM25 retrieval
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_ranking = list(np.argsort(bm25_scores)[::-1])  # descending

    # RRF scoring: accumulate 1/(k + rank) across both rankers
    rrf_scores: dict = {}
    for rank, idx in enumerate(bm25_ranking):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, idx in enumerate(dense_ranking):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank + 1)

    # Return top-n doc IDs sorted by combined RRF score
    return sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_n]
```

> 💡 **Hybrid search verdict:** BM25 handles "AUTH-SVC-9B" and exact service codes; dense retrieval handles "authentication reliability." RRF combines both rank lists without score normalization — no calibration step needed. The investigation's 91% semantic-only recall becomes 94% with hybrid, closing the gap on service codes, incident identifiers, and exact-match queries.
> ➡️ Production stacks: Elasticsearch or OpenSearch (BM25) + Pinecone/Qdrant (dense) with RRF at the application layer — or Weaviate/Qdrant with built-in hybrid search that handles both in a single API call.

***

## 7 · Why Chunking Is Required

**You can't embed the entire 200-doc wiki as one vector** — `text-embedding-ada-002` has an 8,191 token limit, and your wiki is 80,000+ tokens. Even if it fit, a single vector averaging over "authentication SLA" + "deployment procedures" + "incident runbooks" + "onboarding guides" would dilute every query.

Chunking for RAG is the process of breaking large documents into smaller, manageable pieces before converting them into embeddings for retrieval. It might sound like a minor preprocessing step, but it is actually one of the biggest levers available to improve a RAG system's performance — **a well-tuned chunking strategy can improve retrieval accuracy by 40%** compared to naive approaches.

### Three Fundamental Reasons Chunking Is Necessary

**1. Embedding Model Token Limits**

Embedding models have maximum input lengths — commonly 512 to 8,192 tokens. OpenAI's `text-embedding-ada-002` accepts a maximum of **8,191 tokens**. Source documents routinely exceed these limits. A 20-page technical document might contain 15,000+ tokens — physically impossible to embed in a single pass. Chunking ensures each piece fits within model constraints.

**For the investigation:** Your wiki spans ~80,000 tokens across 200 docs. Individual runbooks are 3,000–8,000 tokens each. Without chunking, you can't even start the ingestion pipeline.

**2. Semantic Dilution Problem**

Embedding models produce **a single fixed-size vector** regardless of input length. A 200-word chunk and a 2,000-word chunk both become one vector. This means **larger chunks can dilute specific information**, while smaller chunks may lose important context.

**For the investigation:** If you embed entire wiki documents as one chunk, the query "authentication SLA" retrieves a vector that averages over the full wiki — the model can't tell if the SLA page is more relevant than an onboarding guide. **You need document-section granularity** to get precise retrieval.

A common source of inaccurate answers lies in a structural conflict within the traditional "chunk-embed-retrieve" pipeline: using a single-granularity, fixed-size text chunk to perform two inherently conflicting tasks — **semantic matching (recall)**, where smaller chunks (100–256 tokens) are needed for precise similarity search, and **context understanding (utilization)**, where larger chunks (1,024+ tokens) are needed for coherent LLM generation. This creates a difficult tradeoff between "precise but fragmented" and "complete but vague".

**3. Retrieval Precision and Storage Efficiency**

When a retrieval system underperforms, most developers immediately blame the embedding model or the vector database. But the real issue is often hiding in plain sight — even a perfect retrieval system fails if it searches over poorly prepared data. Chunks need to accomplish two things simultaneously: they must be easy for vector search to find, and they must give the LLM enough context to generate useful answers.

**For the investigation:** You debugged an elevated hallucination rate for two weeks, suspecting the embedding model. Turns out the issue was chunking — you split documents at arbitrary 500-character boundaries, so SLA headers got separated from their values. Queries about uptime targets retrieved the header without the actual number. **Fixing chunking to respect section boundaries dropped the hallucination rate significantly** — no model changes needed.

> 💡 **Chunking necessity verdict:** Without section-level chunking, "Authentication SLA" headers and "99.95% uptime" values land in separate vectors — an SLA query retrieves the header without the value. Fixing chunking strategy drives significant hallucination rate improvements, with zero model changes, zero re-training, and zero API cost increase.

***

## 8. Chunking Strategies

**Your choice here determines whether you hit the 5% error target or stay stuck at 10-15%.** There is no single best approach. Each strategy trades off context preservation against retrieval precision in different ways.

**For the investigation:** Start with recursive character splitting (respects section boundaries), then test semantic chunking if your budget allows (2-3 percentage point recall improvement). Don't use fixed-size chunking — it will cut policy sections in half.

### Fixed-Size Chunking

The simplest method: split text into uniform segments based on character, word, or token count. Fast to implement, predictable to manage. The downside: it has zero awareness of document structure — sentences get cut mid-word, paragraphs break in awkward places, and related ideas end up scattered across chunks. Use this as a starting point when prototyping or when you don't know your data well yet.

### Recursive Character Splitting

The **default choice for about 80% of RAG applications**. It uses a hierarchy of separators to find natural boundaries, attempting splits at paragraph breaks first, then sentences, then spaces. LangChain's `RecursiveCharacterTextSplitter` is the most common implementation, with default separators of `["\n\n", "\n", " ", ""]`.

In Chroma's research, **recursive splitting achieved 88 to 89% recall with 400-token chunks** using `text-embedding-3-large`.

**For the investigation:** Your wiki docs have natural section boundaries (Markdown headers), runbooks have step structures, and policy docs have clause boundaries. Recursive splitting respects all of these — fixed-size chunking doesn't.

**Code Snippet — Phase 1: Recursive character splitting with LangChain:**

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import JSONLoader
import json

# Load raw menu document
with open('menu.json', 'r') as f:
    menu_data = json.load(f)

# Extract text from JSON structure
documents = []
for item in menu_data['items']:
    # Combine fields for rich context
    text = f"{item['name']} ({item['size']}): ${item['price']}, {item['calories']} calories"
    if 'allergens' in item:
        text += f". Allergens: {', '.join(item['allergens'])}"
    if 'gluten_free_available' in item and item['gluten_free_available']:
        text += ". Gluten-free crust available (+$3.00)."
    documents.append(text)

# DECISION LOGIC: Configure splitter for menu data
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,              # Target: 400 tokens (good for factoid queries)
    chunk_overlap=60,            # 15% overlap (60/400) prevents boundary issues
    length_function=len,         # Count by characters (approximate tokens)
    separators=["\n\n", "\n", ". ", " ", ""]  # Hierarchy: paragraph → sentence → word
)

# Split documents
chunks = splitter.create_documents(documents)

print(f"✅ Split {len(documents)} items into {len(chunks)} chunks")
print(f"📊 Avg chunk length: {sum(len(c.page_content) for c in chunks) / len(chunks):.0f} chars")

# Sample output
for i, chunk in enumerate(chunks[:3]):
    print(f"\n[Chunk {i+1}]")
    print(chunk.page_content[:150] + "...")
```

> 💡 **Industry Standard:** `LangChain RecursiveCharacterTextSplitter` or `LlamaIndex SentenceSplitter`
> **When to use:** First choice for 80% of RAG applications — respects structure, fast, predictable
> **Common alternatives:** `SemanticChunker` (groups by meaning, 2-3pp better recall but slower), `NLTKTextSplitter` (sentence-aware), `MarkdownTextSplitter` (preserves headers)
> **Production pattern:** Start with recursive splitting → measure recall → upgrade to semantic if needed

### Sentence-Based Chunking

Each chunk contains complete sentences. No thought gets cut off mid-expression. This works well when queries align with sentence-level information, like customer support questions or conversational AI. The tradeoff: sentence lengths vary wildly, leading to inconsistent chunk sizes — a single sentence might be 10 tokens or 200 tokens.

### Semantic Chunking

Instead of splitting on structure, semantic chunking uses embedding similarity to group related content together:

1. Split text into sentences
2. Generate embeddings for each sentence
3. Compare similarity between consecutive sentences
4. Create chunk boundaries where similarity drops significantly

This approach can improve recall by **2 to 3 percentage points** over recursive splitting, according to Chroma's benchmarks. LLM-based semantic chunking achieved the highest scores in multiple tests, with **0.919 recall** in one study.

**Cost tradeoff:** Every sentence needs its own embedding. For a 10,000-word document, you might generate 200 to 300 embeddings just for the chunking step — expensive if using API-based embedding services.

**For the investigation:** Your wiki is ~80,000 tokens → ~2,000 sentences → ~$0.16 in embedding costs for semantic chunking (negligible). If the 2-3 point recall improvement drops your hallucination rate from 6% → 4%, that's a meaningful reduction in incorrect technical decisions being made from AI-generated answers — at a one-time $0.16 cost. Do it.

### Agentic Chunking

The newest approach: let an LLM analyze each document and decide where to split it. The model can understand semantic meaning, identify topic transitions, and respect content structure like section headings and step-by-step instructions. It produces the highest-quality chunks but is also the **slowest and most expensive** method — you're making an LLM call for document segmentation before you even start the retrieval pipeline.

> 💡 **Chunk verdict:** Wiki corpus split into ~1,400 chunks at 400 tokens with 60-token overlap — recursive splitter preserves section boundaries and achieves 91% retrieval recall.
> ➡️ Dropping below 256 tokens fragments multi-section entries like runbook procedures; going above 512 dilutes embeddings across unrelated topics — both degrade recall toward 89%.

***

## 9 · Chunk Size Selection

**Your hallucination rate depends on getting chunk size right.** Too small (128 tokens), and "Authentication SLA: 99.95% uptime, p99 < 50ms" gets split into 3 chunks — the SLA query retrieves only the header. Too large (2,048 tokens), and the embedding averages over an entire runbook — similarity scores become mushy, retrieval precision drops.

The optimal chunk size isn't a single magic number — it depends on document types, query patterns, and what you're trying to accomplish. Research from multiple sources points to a general sweet spot between **128 and 512 tokens** for most use cases.

**For the investigation:** You tested 256, 400, and 512 tokens on 50 test queries from the wiki test set. Results:
- 256 tokens: 89% recall, but many chunks missing context (SLA headers without values)
- 400 tokens: 91% recall, best balance
- 512 tokens: 90% recall (slightly worse — chunks too generic)

**Winner: 400 tokens.** This is your production setting.

### Size Recommendations by Query Type

| Query Type | Recommended Size | Rationale |
| -------------------------------------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Factoid queries** (names, dates, specific facts) | 256–512 tokens | Chunk should contain the answer and minimal surrounding noise |
| **Analytical queries** (explanations, comparisons) | 1,024+ tokens or page-level | LLM needs broader context to synthesize a coherent response |
| **Mixed queries** | 400–512 tokens | Balanced middle ground |

### Size Recommendations by Document Type

| Document Type | Chunk Size (tokens) |
| ---------------------- | ------------------------------------------------------------------------------------------------------ |
| **FAQs** | 128–256 (matches question-answer pairs naturally) |
| **Technical docs** | 300–500 (preserves step-by-step procedures) |
| **Research papers** | 512–1,024 (maintains complex arguments) |
| **Legal documents** | 600–1,000 (keeps clause integrity) |
| **Code documentation** | 200–400 (aligns with function-level context) |

### How Chunk Size Interacts with Other System Parameters

**Embedding dimension and distance metrics:** All chunks become the same-dimensional vector regardless of chunk length. However, smaller chunks produce vectors that are more "focused" on a narrow semantic topic, which often yields sharper cosine similarity scores against a focused query. Larger chunks produce vectors that blend multiple semantic themes, potentially reducing discriminative power.

**ANN recall:** With more chunks (from smaller sizes), the ANN index has more candidate vectors. This can improve recall\@k (more chances to find the relevant passage) but increases index size and potentially query latency. The interaction is: `total_vectors ≈ corpus_tokens / (chunk_size - overlap)`.

**LLM context window:** The sum of retrieved chunk lengths plus query and prompt overhead must fit within the LLM's token limit. For a 4,096-token LLM window with \~1,000 tokens reserved for answer generation and \~500 for prompt overhead, approximately 2,500 tokens remain for retrieved context — allowing 5 chunks of 500 tokens each, or only 2 chunks of 1,000+ tokens each.

> 💡 **Chunk size verdict:** 400-token chunks deliver 91% recall at <10ms query latency on the wiki's ~1,400-chunk corpus — the sweet spot that achieves the 4% hallucination target. Under 256 tokens: recall degrades ~2 pp as multi-section wiki entries split; over 512 tokens: precision degrades ~1 pp as unrelated topics blur together. Both lift hallucination rate above the 4% threshold.

***

## 10. Chunk Overlap: Preventing Boundary Information Loss

**Overlap is an insurance policy against boundary problems.** When chunks share some content with their neighbors, the risk of important information getting split across chunks is reduced.

### How Much Overlap to Use

NVIDIA tested 10%, 15%, and 20% overlap values and found **15% performed best** on their FinanceBench dataset with 1,024-token chunks. Industry practice generally recommends **10 to 20% overlap** as a starting point.

For a 500-token chunk, that means **50 to 100 tokens** of overlap with adjacent chunks.

```plaintext
 Chunk 1: [─────────────────────|====]
 Chunk 2: [====|─────────────────────|====]
 Chunk 3: [====|─────────────────────]

 ──── = unique content per chunk
 ==== = overlapping region (10-20% of chunk size)

 If a key sentence falls in an overlap zone,
 it appears in BOTH adjacent chunks → retrieval succeeds.
```

### Overlap Tradeoffs
| Chunk Size (tokens) | Overlap (tokens) | Overlap % | Notes |
| ------------------- | ---------------- | --------- | ------------------------------------------------------------------------ |
| 256 | 25–50 | 10–20% | Minimal context bleed; good for factual Q\&A |
| 512 | 50–100 | 10–20% | Most common default |
| 1,024 | 100–200 | 10–20% | Use for complex documents; monitor cost |

### When Overlap Hurts

Overlap is not always beneficial. For very short, self-contained documents (like FAQ entries or product descriptions under 500 tokens), chunking itself may hurt performance — embedding the whole thing is preferable. Not all documents need chunking, and over-chunking short, focused content can actually degrade results.

> 💡 **Overlap verdict:** 15% overlap is the cheapest error-rate insurance in the pipeline — it adds ~15% storage while eliminating the boundary-miss failure mode responsible for 3–5 pp of real-world menu RAG error rates, where prices and calorie counts routinely straddle paragraph boundaries.

***

## 11. Advanced Chunking and Retrieval Techniques

**These techniques are for when recursive splitting + 400-token chunks + 15% overlap plateaus at 88% recall and you need 92%.** For most production systems (including the investigation's current scale), the techniques in § 8-10 are sufficient. Read this section when you're optimizing the last 5% of retrieval quality.

### Late Chunking

Traditional chunking embeds each chunk independently, which means each piece loses context from the rest of the document. **Late chunking flips this:** embed the entire document first, then segment the token-level embeddings into chunks afterward. This preserves global context within each chunk's embedding without requiring extra LLM calls. Research shows late chunking works particularly well as documents approach **8,000 tokens** in length.

**Tradeoff:** It requires embedding models that support long context windows, and you process more tokens upfront.

**For the investigation:** Your longest wiki docs are ~8,000 tokens — good candidates for late chunking. But test them against recursive splitting first; the complexity might not be worth the gain.

### Contextual Retrieval

Anthropic introduced this technique in 2024. Before embedding each chunk, an LLM generates a **contextual summary** that situates the chunk within the broader document. For example, a chunk that originally said\*"Revenue grew by 3% over the previous quarter"\* becomes\*"This chunk is from ACME Corp's Q2 2023 SEC filing. Revenue grew by 3% over the previous quarter."\*.

In Anthropic's tests, this reduced retrieval failures by up to **67%** when combined with re-ranking. The cost consideration: you're making an LLM call for every chunk. Prompt caching can reduce this by up to **90%**, but it's still more expensive than basic chunking.

### Hierarchical Chunking

Create multiple chunk sizes with **parent-child relationships**. Index a chapter, its sections, its paragraphs, and its sentences. At query time, search across all levels and use the results to navigate to the most relevant granularity. This approach is ideal for very large, complex documents like textbooks or legal contracts — you can answer both high-level summary questions and highly specific detail questions from the same knowledge base. LlamaIndex's `HierarchicalNodeParser` makes implementation straightforward in Python.

### Multi-Vector Representations

Standard single-vector embeddings can sometimes fall short when dealing with long, multifaceted documents. A single vector representing an entire document might average out specific details or fail to capture distinct topics discussed within different sections. **Multi-vector embeddings** represent a single document using multiple distinct vectors, allowing for capturing different facets, sections, or granular pieces of information within the document.

Common strategies include:

1. **Chunk-based Segmentation:** The document is divided into smaller chunks, each independently embedded
2. **Propositional Segmentation:** Documents are broken into individual propositions or claims — very fine-grained but more complex to implement
3. **Summary and Detail Vectors:** One vector might represent a summary; other vectors represent specific detailed sections

During retrieval, the query vector is compared against all individual vectors associated with documents. Relevance scores can be aggregated by taking the maximum similarity score, the average of the top-k highest-scoring sub-vectors, or a more sophisticated aggregation function.

**Advantages:** Enhanced granularity, improved handling of long documents, better topic separation. **Tradeoffs:** Increased storage (multiple vectors per document), more complex retrieval logic, and higher computational cost at query time.

### ColBERT: Late Interaction for Fine-Grained Matching

**ColBERT** (Contextualized Late Interaction over BERT) creates contextualized embeddings for **each token** in both the query and document. Rather than collapsing each to a single vector:

1. **Query Encoding:** Each query token → a contextualized embedding {q₁, q₂, ..., qₘ}
2. **Document Encoding:** Each document token → a contextualized embedding {d₁, d₂, ..., dₙ} (done offline)
3. **MaxSim:** For each query token qᵢ, compute maximum similarity with all document tokens: `MaxSim(qᵢ, D) = max_{j=1…n}(qᵢᵀ dⱼ)`
4. **Scoring:** Final relevance = sum of MaxSim scores: `Score(Q, D) = Σᵢ₌₁ᵐ MaxSim(qᵢ, D)`

This token-level interaction allows ColBERT to identify documents where many query terms are strongly represented, capturing subtle lexical and semantic matches that might be missed by single-vector approaches. ColBERT often demonstrates superior performance in re-ranking tasks — however, using ColBERT as a standalone first-pass retriever over a very large corpus can be computationally intensive, making it more practical as a **re-ranker** over a smaller candidate set.

**Storage tradeoff:** Storing contextualized embeddings for every token in every document results in a significantly larger index size compared to document-level or chunk-level embeddings.

> 💡 **Advanced techniques verdict:** These techniques are only cost-justified after basic RAG plateaus. Contextual retrieval's 67% failure reduction is worth ~$0.01 per ingestion run — negligible against the hallucination rate gain from moving below 4%. ColBERT re-ranking adds 200–400ms latency in exchange for precision gains; budget for it only after cross-encoder re-ranking shows measurable lift on your test set.

***

## 12. Choosing the Right Chunking Strategy: Decision Framework

**Don't guess — your hallucination rate depends on this choice.** For most applications, start with `RecursiveCharacterTextSplitter` at **400 to 512 tokens** with **10 to 20% overlap**. This provides a solid baseline that you can optimize from.

**For the investigation:** You started with fixed-size 500-character chunks (naive), saw elevated hallucination rate, switched to recursive 400-token chunks (respects section boundaries), hallucination rate dropped significantly. That's the difference between "acceptable prototype" and "production-ready system."

### Decision Tree

1. **Is your document short and focused?** (Under 500 tokens, single topic) → Don't chunk at all. Embed the whole thing
2. **Is your document well-structured?** (Clear headers, sections, paragraphs) → Use recursive character splitting or document-based chunking that respects structure
3. **Are your queries fact-based or analytical?** → Fact-based: smaller chunks (256–512 tokens). Analytical: larger chunks (512–1,024 tokens)
4. **Is retrieval quality critical and budget flexible?** → Test semantic or LLM-based chunking
5. **Are you dealing with PDFs or visual documents?** → Consider page-level chunking or specialized AI document analysis tools that respect layout

### Evaluation Methodology

**Don't guess — measure.** You need quantitative evidence that RAG reduces hallucinations.

1. **Create a test dataset:** 50 to 100 representative documents with 20 to 30 realistic queries. Include edge cases from expected usage
2. **Define success metrics:** **Recall\@k** measures whether relevant chunks appear in your top k results. **Precision** measures how many retrieved chunks are actually useful. **MRR** (Mean Reciprocal Rank) shows how highly relevant results rank
3. **Test 2 to 3 strategies:** At minimum, compare your current approach against one alternative — recursive splitting vs. semantic chunking, or 256-token vs. 512-token chunks
4. **Evaluate with humans and LLMs:** Automated metrics catch obvious problems. Human review catches things metrics miss, like whether the retrieved context actually enables good answers
5. **Monitor in production:** The queries you designed for might not match real user behavior. Track retrieval performance over time and iterate

**For the investigation:** You created a test set of 50 queries from the wiki test set. Measured recall@5 for three chunking strategies:
- Fixed 500-char: 78% recall (failed on multi-section queries)
- Recursive 400-token: 91% recall (production choice)
- Semantic chunking: 93% recall (2 point gain, but 3× slower ingestion)

You picked recursive for production (fast, good enough), and documented semantic chunking as a future optimization if hallucination rate plateaus.

### Common Chunking Mistakes

**You'll make at least 2 of these 5 mistakes in your first RAG implementation.** Based on production RAG systems, these are the most common errors:

* **Using only default settings** without testing alternatives — might be costing 20% retrieval accuracy
  - **Investigation mistake:** Used default LangChain 500-char chunks for 2 weeks, never tested alternatives. Hallucination rate was stuck at elevated levels. Spent 1 hour testing 256/400/512 tokens → found 400 works best → hallucination rate dropped to near target.
* **Ignoring document structure** — naive character splitting doesn't care if it breaks a sentence mid-word or separates a heading from its content
  - **Investigation mistake:** Fixed-size chunking split SLA headers from their values into different chunks. Queries about uptime targets retrieved headers without the actual numbers.
* **Zero overlap** — boundary misses are a real problem
  - **Investigation mistake:** No overlap meant if a query about "authentication SLA" landed at a chunk boundary, it missed half the relevant text. Adding 15% overlap fixed it.
* **Over-chunking everything** — not all documents need chunking. Documents under 200,000 tokens might work better stuffed directly into the prompt
  - **Investigation note:** Short wiki entries (under 200 tokens) don't need chunking. Embed each one as a single chunk.
* **No metadata** — without tracking which document each chunk came from, you lose the ability to filter, deduplicate, and cite
  - **Investigation mistake:** Couldn't tell if retrieved chunk came from runbooks, SLA docs, or onboarding guides. Added `source` metadata field → now can filter "only search SLA documents" for uptime queries.

> 💡 **Decision framework verdict:** Systematic evaluation of 3 chunking strategies (2 hours) moved hallucination rate from 38% toward the 4% target. Skipping evaluation costs weeks of debugging — the difference between a 3-month and a 3-week time-to-production.

***

## 13 · Putting It All Together: Full Pipeline Walkthrough

Every §0 failure traces to a gap in the pipeline — no ingestion, no retrieval, no grounding. This section closes all three gaps: ingestion output becomes query input, and the code path runs end-to-end from "internal engineering wiki" to "99.95% uptime, p99 < 50ms, MTTR < 5 minutes" with no invented numbers.

### Pipeline in ASCII

```plaintext
INGESTION PATH (Offline)
========================

 ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
 │ Raw Docs│───▶│ Clean & │───▶│ Chunk │───▶│ Embed │───▶│ Vector │
 │ (PDF, │ │ Extract │ │ (400-512 │ │ (Model: │ │ DB + │
 │ HTML…) │ │ + Meta │ │ tokens) │ │ ada-002) │ │ ANN Index│
 └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘

QUERY PATH (Runtime)
====================

 ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
 │ User │───▶│ Embed │───▶│ ANN │───▶│ (Optional│───▶│ LLM │
 │ Query │ │ Query │ │ Search │ │ Re-rank) │ │ Generate │
 │ │ │ (same │ │ top-K │ │ ColBERT/ │ │ Answer │
 │ │ │ model) │ │ chunks │ │ x-encoder│ │ │
 └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
```

### Practical Example with Code

**Code Snippet — Phase 5: Complete RAG pipeline with prompt template:**

```python
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

# Phase 4: Retrieve relevant chunks (from earlier)
client_chroma = chromadb.PersistentClient(path="./chroma_db")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
collection = client_chroma.get_collection(
    name="pizzabot_menu",
    embedding_function=embedding_fn
)

user_query = "What's the calorie count for a large Margherita pizza?"

# Retrieve top-3 chunks
results = collection.query(
    query_texts=[user_query],
    n_results=3,
    include=["documents", "metadatas"]
)

# Phase 5: Compose prompt with retrieved context
retrieved_chunks = results['documents'][0]
context = "\n\n".join([f"[Source {i+1}]\n{chunk}"
                       for i, chunk in enumerate(retrieved_chunks)])

# DECISION LOGIC: Prompt template with citation requirement
system_prompt = """You are PizzaBot, Mamma Rosa's ordering assistant.
Answer questions using ONLY the provided menu context.
Always cite the source number (e.g., "According to Source 1...").
If the answer isn't in the context, say "I don't have that information in the menu."""

user_prompt = f"""Context from menu database:
{context}

User question: {user_query}

Answer:"""

# Generate grounded response
client_openai = OpenAI()
response = client_openai.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0.0  # Deterministic for factual queries
)

answer = response.choices[0].message.content
print(f"PizzaBot: {answer}")

# Expected output:
# "According to Source 1, a large Margherita pizza is 920 calories."
```

> 💡 **Industry Standard:** `LangChain RetrievalQA` chain or `LlamaIndex QueryEngine`
> **When to use:** Always in production — handles prompt composition, citation tracking, error handling
> **Common alternatives:** Custom prompt templates (shown above, more control), `Haystack` pipelines (European open-source ecosystem)
> **Production pattern:** Start with custom templates → migrate to LangChain/LlamaIndex when complexity grows
> **Advanced features:** Conversational memory (multi-turn dialogue), streaming responses, source attribution UI

> 💡 **Generate verdict:** Full RAG pipeline achieves 95% answer accuracy and 1.26 s average latency (p95 2.8 s) at $0.008/conv — all five pipeline constraints in the green zone.
> ➡️ The 5% residual errors are ambiguous queries where top-k chunks contain conflicting facts; §11 re-ranking and Ch.8 RAGAS faithfulness scoring address this systematically.

***

## 14. The Anisotropy Problem

**You'll notice this when debugging "why did the model retrieve this irrelevant chunk?"** — sometimes chunks with 0.7 cosine similarity are less relevant than chunks with 0.65 similarity, because the embedding space isn't uniformly distributed.

Embedding spaces often exhibit **anisotropy** — embeddings cluster in a narrow cone rather than uniformly filling the space. This can cause:

* Similarity thresholds to be misleading: a 0.7 cosine similarity might only indicate weak relatedness
* Over-reliance on absolute similarity values rather than relative ranking

**For the investigation:** Don't set hard cutoffs like "only return chunks with similarity > 0.75" — instead, always return top-k and let relative ranking decide. If the best match is 0.62, that's still the best you've got.

**Mitigations** include:

* **Whitening:** Transform embeddings to have unit covariance (can improve retrieval)
* **Model choice:** Some models are explicitly trained to reduce anisotropy
* **Relative ranking:** Focus on relative similarity rather than absolute values (this is what you should use)

> 💡 **Anisotropy verdict:** Hard similarity thresholds (e.g., "only return chunks with score > 0.75") silently drop 20–30% of valid chunks in clustered embedding spaces — inflating error rate by 3–5 pp without any retrieval failure appearing in logs. Use relative top-k ranking instead; the best match at 0.62 is still the best you have.

***

## 15. Quantization of Embeddings

**The wiki investigation prototype doesn't need quantization** — the embeddings fit comfortably in RAM. But when you scale to an enterprise knowledge base with millions of documents, quantization becomes critical for cost control.

Full-precision embeddings (float32) consume significant storage and memory. **Quantization** reduces precision to shrink memory footprint and accelerate similarity computation, with controllable quality tradeoffs:

**Business case for enterprise scale:**
- 90 GB float32 embeddings → **$180/month in vector DB storage** (AWS pricing)
- 22.5 GB int8 embeddings → **$45/month** (4× reduction)
- **Savings: $135/month** with <2% recall degradation
- At 10,000 daily queries, this is **$1.35 saved per 1,000 queries** — small but adds up

| Precision | Bytes per Dimension | Relative Size | Use Case |
| ----------- | ------------------- | --------------- | -------------------------------- |
| **float32** | 4 bytes | 1.0× (baseline) | Maximum fidelity |
| **float16** | 2 bytes | 0.5× | Good balance of size and quality |
| **int8** | 1 byte | 0.25× | Large-scale production systems |

**Scalar quantization (int8)** maps float32 values to 8-bit integers, reducing storage by 4×. When used with oversampling (retrieving more candidates from the compressed index before refining with full-precision vectors), the recall loss can be minimized.

> 💡 **Quantization verdict:** int8 quantization delivers 4× storage reduction with <2% recall degradation — at enterprise scale that's $135/month saved with no measurable impact on the 4% hallucination target. Irrelevant at the wiki experiment's current scale; critical when expanding to full enterprise knowledge bases.

---

## 17. Summary: Key Engineering Decisions

**These are the 8 decisions that determined whether the investigation hit the 4% hallucination target or stayed stuck at 38%.** A well-tuned chunking strategy can improve retrieval accuracy by **40%** compared to naive approaches. The difference between a good and bad chunking strategy can mean a 40% improvement in retrieval accuracy, according to recent benchmarks. Getting this right makes everything downstream improve.

**Your production checklist:**

| Decision | Default Recommendation | When to Deviate |
| --------------------- | ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- |
| **Chunk size** | 400–512 tokens | Smaller (128–256) for factoid Q\&A; larger (512–1,024) for analytical queries |
| **Chunk overlap** | 10–20% (50–100 tokens at 500-token chunks) | Increase for noisy boundary-heavy docs; decrease for very short docs |
| **Chunking strategy** | Recursive character splitting (80% of RAG apps) | Semantic chunking if budget allows (+2–3pp recall); agentic for highest quality |
| **Embedding model** | Match to your use case; newer `text-embedding-3-small` often outperforms `ada-002` at lower cost | Larger models for higher accuracy requirements |
| **Normalization** | L2-normalize at index time, use dot product at query time | Skip only if model already normalizes |
| **Distance metric** | Dot product for normalized; cosine for non-normalized | L2 for specific index requirements |
| **Metadata** | Always attach document\_id, chunk\_id, position, source | Add domain-specific fields as needed |
| **Re-ranking** | Add if baseline recall is insufficient | ColBERT for token-level precision; cross-encoder for highest accuracy |

**The key takeaways from production experience (yours and ours):**

* Start with recursive splitting at 400 to 512 tokens and 10 to 20% overlap **(investigation: 400 tokens, 15% overlap)**
* Match chunk size to query type: smaller for facts, larger for analysis **(investigation: 400 tokens for policy docs, 300 for short wiki entries)**
* Always test on your actual data and measure with your real queries **(investigation: 50 test queries from wiki test set)**
* Consider advanced techniques like contextual retrieval or late chunking when basic approaches plateau **(investigation: recursive splitting was enough to hit 91% recall)**

**Cost-benefit reality check:**
- Spending 4 hours testing chunking strategies → **saved 2 months of debugging "why is RAG inaccurate?"**
- Hallucination rate improvement (38% → 4%) → every percentage point reduction means fewer wrong technical decisions made from AI-generated answers
- Accuracy target **achieved** → investigation proceeds to Ch.5 to close the remaining retrieval gaps

> 💡 **Summary verdict:** Getting chunking strategy and chunk size right is the single highest-leverage RAG decision — responsible for the 78% → 91% recall improvement that drove hallucination rate from 38% toward the 4% target. The remaining seven decisions in the table are optimizations; chunking is the foundation.

---

## 18 · Progress Check — What We Can Solve Now

🎉 **MAJOR BREAKTHROUGH**: Wiki hallucinations grounded by retrieval!

**Unlocked capabilities:**
- ✅ **Embedding-based retrieval**: Wiki corpus searchable via semantic similarity
- ✅ **Two-phase RAG pipeline**: Ingestion (chunk → embed → index) + Query (embed query → retrieve → generate)
- ✅ **Grounded answers**: LLM sees actual wiki content in context, cannot fabricate facts
- ✅ **Chunking strategy**: 400-token recursive splits with 15% overlap for wiki documents
- ✅ **Hybrid retrieval**: BM25 (keyword) + dense embeddings (semantic) with RRF fusion

**Experiment results:**

| Metric | Without RAG | With RAG | Finding |
|--------|-------------|----------|---------|
| Hallucination rate (50-question test) | 38% | 4% | ✅ Target hit |
| Query latency (p95) | N/A | <3s | ✅ Acceptable |
| Retrieval cost | N/A | ~$0.000012/query | ✅ Negligible |
| Residual failures | — | 4% | ❌ Retrieval misses |

**What we can solve:**

✅ **Wiki factual queries with high accuracy**:
```
User: "What's the SLA for our authentication service?"

RAG system (with wiki corpus):
[Ingestion: 200 wiki documents already chunked, embedded, indexed in vector DB]

Query embedding: embed("authentication service SLA")
Vector search: retrieve top-5 chunks by cosine similarity

Retrieved chunk #1 (similarity: 0.96):
"Authentication Service SLA
  Uptime target: 99.95%
  p99 latency: <50ms
  MTTR: <5 minutes
  On-call rotation: Platform Team, 24/7"

LLM sees chunk in context:
Thought: "The retrieved SLA document shows 99.95% uptime and p99 <50ms."
Answer: "Our authentication service SLA targets 99.95% uptime,
        p99 latency under 50ms, and MTTR under 5 minutes."

Result: ✅ Correct! Grounded in actual wiki content!
```

✅ **Technical specification queries with accurate values**:
```
User: "What's the maximum request size for the API gateway?"

RAG system:
Query embedding: embed("API gateway request size limit")
Hybrid retrieval: BM25("request size") + dense_search(query_embedding)
RRF fusion: merge both rank lists

Retrieved chunks:
1. "API Gateway Configuration
    Max request body: 10 MB
    Max response body: 50 MB
    Timeout: 30 seconds
    Rate limit: 1000 req/min per client"
2. "Large Payload Handling: For payloads >10MB, use the multipart upload endpoint."
3. "Common 413 errors: Check if payload exceeds 10MB limit before debugging auth."

LLM sees all 3 chunks:
Thought: "Multiple chunks confirm 10MB limit for request body.
         Secondary chunk explains what to do when it's exceeded."
Answer: "The API gateway accepts request bodies up to 10MB.
        For larger payloads, use the multipart upload endpoint.
        Exceeding 10MB returns a 413 error."

Result: ✅ Accurate! All values grounded in retrieved wiki chunks!
```

✅ **Policy and procedure queries without hallucination**:
```
User: "What's the process for production deployments?"

RAG system:
Query embedding: embed("production deployment process")
Retrieved chunks:
1. "Production Deployment Checklist: (1) PR reviewed by 2 engineers..."
2. "Deployment Freeze Windows: No deployments Friday 4pm to Monday 9am"
3. "Rollback Procedure: Run ./scripts/rollback.sh <version> within 15min of deploy"

LLM sees all chunks:
Thought: "Deployment process has multiple steps: PR review, freeze windows, rollback."
Answer: "Production deployments require 2-engineer PR review. Deployments are
        frozen Friday 4pm through Monday 9am. If issues arise, use
        ./scripts/rollback.sh <version> within 15 minutes of deployment."

Result: ✅ Correct! No fabricated steps or wrong windows!
```

❌ **What remains open:**

- **Retrieval failures (4% residual)** — the relevant document exists but wasn't retrieved
  - Dense retrieval misses documents with exact-match terms (service codes, incident IDs)
  - Hybrid search helps but recall is bounded by index quality
  - Ch.5 (Vector DBs) covers HNSW, IVF, and DiskANN for better index recall

- **No incremental index updates** — if the wiki is updated, the index is stale
  - Need incremental ingestion pipeline for live documentation
  - Ch.5 covers index update strategies

**Investigation findings summary:**
- **Hallucination rate**: 38% → 4% on the 50-question test set (**TARGET HIT!**)
- **Residual errors**: All retrieval failures — the relevant doc wasn't retrieved (addressed in Ch.5)
- **Query latency**: <3s p95 — vector retrieval is fast
- **Cost**: ~$0.000012/query — negligible

**Why this matters:**
1. **Domain knowledge gap closed**: The model now answers from the organization's actual documents, not training memory
2. **Hallucination rate at target**: 4% is at the investigation's 5% threshold
3. **Residual gap is addressable**: The remaining 4% are retrieval failures — fixed by better indexing (Ch.5)
4. **Scales to enterprise**: The same architecture handles wikis, runbooks, policy docs, and codebases

**Next chapter**: [Vector DBs](../ch05-vector-dbs) explains the index structures (HNSW, IVF, DiskANN) behind the RAG retrieval — and how to push retrieval recall from 91% toward 96%, closing the remaining gap.

**Key interview concepts from this chapter:**

| Must know | Likely asked | Trap to avoid |
|---|---|---|
| The two-phase RAG pipeline: **ingestion** (chunk → embed → store in vector DB) vs **query** (embed query → ANN retrieve → re-rank → generate); failures in either phase compound downstream | "Walk me through a RAG pipeline end-to-end" — interviewers expect both phases, not just retrieval | Describing RAG as just "search then generate" without mentioning chunking, embedding, and the index |
| Chunking strategy matters: **fixed-size** (fast, baseline), **sentence/recursive** (respects natural boundaries, most common in prod), **semantic** (highest recall, groups by meaning, slowest); chunk size drives the recall-vs-context tradeoff | "What chunking strategy would you use for a legal document Q&A system?" | Saying chunk size doesn't matter — a 40% recall difference between naive and tuned chunking is well-documented |
| Cosine similarity = dot product on L2-normalised vectors; for un-normalised embeddings they differ; most production pipelines normalise at index time and use dot product (faster) at query time | "When do you use cosine vs. dot product similarity?" | Saying "cosine similarity and dot product are the same thing" — they're only equivalent after normalisation |
| Sparse retrieval (BM25): keyword-based, fast, no index warm-up, excels on rare proper nouns and exact matches. Dense retrieval (bi-encoder): semantic, catches paraphrases, misses OOV terms. Hybrid search uses **Reciprocal Rank Fusion (RRF)** to merge both rank lists with no score normalisation required | "When would you prefer BM25 over a dense retriever?" | Saying dense retrieval is strictly better — on domain-specific jargon, acronyms, or cold-start (no training data), BM25 often wins |
| Re-ranker (cross-encoder): reads the full query + chunk together, much more accurate than bi-encoder but O(k) LLM calls; bi-encoder retrieval is approximate but sub-millisecond. Architecture: use bi-encoder for top-100 recall, cross-encoder to re-rank to top-5 | "Why not use a cross-encoder for all retrieval?" | Forgetting that a cross-encoder has to see every candidate — you can't run it against millions of chunks |
| Hybrid search with RRF formula: $score = \sum_{ranker} \frac{1}{k + rank_i}$ where $k=60$ is a smoothing constant; merges sparse and dense rank lists without needing to normalise heterogeneous scores | "How does RRF combine results from different retrievers?" | Trying to combine raw BM25 and cosine scores directly — their scales are incompatible without normalisation |
| **Trap:** "embedding similarity = relevance" — a chunk can be semantically close to the query yet contextually useless (e.g., a definition chunk when the query needs a procedure); re-ranking and result diversity both matter | "Why might high-similarity retrieved chunks still produce a bad RAG answer?" | Assuming top-k by cosine is always the right answer to pass to the LLM |

---

## What RAG Can't Fix — When to Reach for Fine-Tuning

**You spent 3 weeks tuning RAG and still have a 5% error rate.** Before investing another month in advanced chunking strategies, check if the problem is actually RAG's job to solve.

RAG solves a *knowledge* problem: the model doesn't have certain facts, so you supply them at inference time. It does not solve *behaviour* problems — how the model reasons, formats its output, or expresses a personality.

**For the investigation:** Your 4% residual hallucination rate breaks down as:
- 3% retrieval failures → **RAG can fix** (better chunking, hybrid search, improved index — Ch.5)
- 1% behaviour errors → **Fine-tuning can fix** (model ignores "answer only from provided context" instruction)

Don't waste time improving RAG for the 1% that need fine-tuning.

```
RAG fixes                           Fine-tuning fixes
─────────────────────────────────   ─────────────────────────────────────────────
✅ Missing private facts             ✅ Wrong tone / persona despite detailed prompt
✅ Stale knowledge (cutoff date)     ✅ Model ignores retrieved context
✅ Hallucinated menu prices/data     ✅ Consistent structured output (JSON, XML)
✅ Live inventory / recent events    ✅ Internal DSL or notation never in training data
                                    ✅ High-volume cost reduction (distillation)
```

**The test:** put the correct answer in the context and ask the model again. If it still fails, that is a behaviour problem — RAG won't help, fine-tuning will. If it succeeds with the context but fails without it, that is a knowledge problem — RAG is the right tool.

**For the investigation diagnosis:**
```
Error: Model says "99.9% uptime" (wrong SLA, actual wiki value: 99.95%)

Test 1: Add correct SLA to context manually
  Prompt: "[CONTEXT: Auth SLA: 99.95% uptime] What is the authentication SLA?"
  Response: "99.9% uptime"
  → Model ignores retrieved context → **Fine-tuning problem** (teach model to follow context)

Error: Model says "99.9% uptime" (wrong SLA, actual wiki value: 99.95%)

Test 1: Add correct SLA to context manually
  Prompt: "[CONTEXT: Auth SLA: 99.95% uptime] What is the authentication SLA?"
  Response: "99.95% uptime"
  → Model uses context correctly → **RAG problem** (improve chunking or retrieval)
```

**They are composable.** A fine-tuned model can sit behind a RAG pipeline. The fine-tune teaches the model *how to behave*; RAG tells it *what to say* for each query.

**For the investigation roadmap:**
- Ch.4 (this chapter): RAG fixes knowledge gaps → 4% hallucination rate ✅
- Ch.8 (Fine-Tuning): LoRA fixes behaviour issues → model consistently follows retrieved context

> Details, LoRA math, and the full decision framework: [FineTuning.md](../ch10_fine_tuning/fine-tuning.md)

> 💡 **RAG vs. fine-tuning verdict:** The diagnostic test (put the correct answer in context; does the model use it?) costs 5 minutes. Getting this wrong costs 4+ weeks: improving RAG for a behaviour problem leaves hallucination rate unchanged; applying fine-tuning for a knowledge problem is equally futile. The 3% vs. 1% residual split saves a month of misdirected effort.

---

## References

[1] https://stackviv.ai/blog/chunking-strategies-rag
[2] https://dev.to/derrickryangiggs/rag-pipeline-deep-dive-ingestion-chunking-embedding-and-vector-search-2877
[3] https://onyxlab.ai/docs/getting-started/embeddings
[4] https://developers.openai.com/cookbook/examples/custom_image_embedding_search
[5] https://medium.com/kx-systems/guide-to-multimodal-rag-for-images-and-text-10dab36e3117
[6] https://zilliz.com/ai-models/text-embedding-ada-002
[7] https://milvus.io/ai-quick-reference/what-is-the-vector-size-of-textembeddingada002
[8] https://apxml.com/courses/optimizing-rag-for-production/chapter-2-advanced-retrieval-optimization/advanced-document-representations-rag

## Bridge to Next Chapter

RAG grounds LLM answers in the organization's actual documents — but retrieval at prototype scale is brute-force: every query does a linear scan over all chunks. **Vector DBs (Ch.5)** replaces that with HNSW and IVF indexes that return approximate nearest neighbours in sub-millisecond time, scaling from ~1,400 wiki chunks to millions without hitting the latency constraint (<3 s p95).

## Illustrations

![RAG and embeddings — semantic clustering, ingestion pipeline, query pipeline, chunk-size tradeoff](img/RAG%20and%20Embeddings.png)
