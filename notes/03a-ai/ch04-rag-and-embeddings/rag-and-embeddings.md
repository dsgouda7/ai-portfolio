# RAG and Embeddings: Grounding LLMs in Retrieved Knowledge

> In the spring of 2013, a young Google researcher named **Tomas Mikolov** was running out of patience with neural language models that couldn't generalize. He had an idea that felt almost too simple: train a shallow network not to *understand* language, but just to *predict neighbors* — what words tend to appear near this word? The result was **Word2vec**. What surprised even Mikolov was that the learned vectors had geometry: *king − man + woman ≈ queen*. Meaning had become arithmetic.
>
> The field spent the next six years refining the recipe. **GloVe** (Pennington, Socher, Manning, Stanford 2014) combined global co-occurrence statistics with local prediction. **fastText** (Bojanowski et al., Facebook 2016) added subword structure so "gluten-free" and "glutenfree" stopped being strangers. But all of these were word-level: each word got one fixed vector regardless of context. *Bank* in "river bank" and *bank* in "bank account" were given the exact same number. The vectors were beautiful, but brittle.
>
> The fix came in 2019 when **Nils Reimers and Iryna Gurevych** at TU Darmstadt published **Sentence-BERT**: a siamese transformer trained with contrastive loss to push semantically similar sentences together and dissimilar ones apart. For the first time, a model could reliably answer *"are these two paragraphs about the same thing?"* — and return a number you could actually trust.
>
> Then **Patrick Lewis and a team at Facebook AI** connected the final wire. In their **2020 RAG paper**, they asked: what if you plugged a retriever into the front of a generative model? Instead of making the LLM memorize every fact during training, let it *look things up* at inference time — search a corpus, read the relevant chunks, then answer. The model could now cite sources instead of hallucinating. By 2023, this pattern — embed a corpus offline, retrieve the nearest chunks at query time, hand them to the LLM — had become the default architecture for any AI system that touches private data.
>
> **This chapter explains why that architecture works.** The hallucinated facts you saw in Ch.3 exist because the model answers from parametric memory, not from the organization's actual documents. This chapter traces the path from *"LLMs hallucinate private data"* to *"retrieval fixes the gap"* — embeddings as the bridge, encoder vs. decoder models, contrastive learning as the secret sauce, and the full RAG pipeline from chunking to generation.
>
> **Where you are in the curriculum.** This is the chapter where you learn what *exactly* gets stored in a vector index, how an embedding model decides two pieces of text are similar, and how a query is matched against millions of chunks. The next chapter — [VectorDBs](../ch05-vector-dbs) — takes the index itself apart (HNSW, IVF, DiskANN). Together they are the foundation for everything else: agents that retrieve before they answer, evaluation pipelines that check grounding, and the entire RAG project under [`projects/ai/rag-pipeline`](../../../projects/ai/rag_pipeline).
>
> **Notation.** $\mathbf{e} \in \mathbb{R}^d$ — embedding vector of dimension $d$; $\text{sim}(\mathbf{q}, \mathbf{k}) = \frac{\mathbf{q} \cdot \mathbf{k}}{\|\mathbf{q}\|\|\mathbf{k}\|}$ — cosine similarity between query and chunk; $k$ — number of retrieved chunks (top-$k$); $c$ — chunk size in tokens.

***

***

## 0 · The Hallucination Problem

LLMs trained on public corpora answer questions about their training data — Wikipedia, books, web crawls. But when you ask about your organization's authentication service SLA, the model has no choice but to guess. The pretraining corpus is frozen at training time and does not include your internal wiki, runbooks, or policy documents. **The model invents a plausible-sounding answer from training memory instead of admitting ignorance.**

**Baseline: no grounding**

```
Query: "What's the SLA for our authentication service?"

GPT-4 (no RAG):
"Typical SLA targets for authentication services are 99.9% uptime (three nines),
with p99 latency under 200ms."

Actual company SLA (from internal wiki): 99.95% uptime, p99 under 50ms, MTTR < 5 minutes
```

The model returned a *reasonable* industry baseline, but it is wrong for this organization. Across a 50-question test set drawn from an internal engineering wiki, **38% of factual answers contain at least one hallucinated number or claim**. Chain-of-thought prompting cannot fix this — reasoning machinery helps with logic ("if A < B and B < C, then A < C"), not facts ("what is the actual company SLA?").

**The root cause:** The model has no mechanism to look up private documents at inference time. It can only answer from parametric memory — the knowledge baked into its weights during pretraining.

**The fix:** **Retrieval-Augmented Generation (RAG).** Instead of relying solely on parametric memory, fetch relevant documents at query time and include them in the prompt context. The model reads the actual SLA document and answers from it, eliminating hallucination of facts that can be looked up.

**Test results:**
- Without RAG: **38%** hallucination rate on factual queries
- With RAG (after implementing this chapter): **4%** hallucination rate

The remaining 4% are retrieval failures — the relevant document exists but wasn't retrieved. Fixing retrieval is the focus of Ch.5.

> 💡 **Problem statement:** LLMs hallucinate because the pretraining corpus is frozen and private data was never in it. Retrieval is the grounding fix: don't train the knowledge in, look it up at query time.

***

***

## 1 · Embeddings: Text → Fixed Vector Where Semantic Distance Is Measurable

**The retrieval problem:** You need to find documents about "service uptime targets" when the wiki says "99.95% availability" — different words, same meaning. Keyword search fails. You need semantic similarity.

**Embeddings solve this** by transforming text into vectors where meaning becomes measurable. Similar concepts cluster together in high-dimensional space, so "uptime" and "availability" become neighbors even with different exact wording. **Text → fixed vector where semantic distance = cosine similarity.**

Word2vec (2013) proved that word vectors could encode meaning: *king − man + woman ≈ queen*. But these were word-level — one vector per word regardless of context. *Bank* in "river bank" and *bank* in "bank account" got the same vector. GloVe (2014) and fastText (2016) refined the approach but kept the word-level limitation.

Sentence-BERT (2019) fixed this with **contextual embeddings** — entire sentences collapsed to a single vector that captures their meaning in context. Modern embedding models are built on **transformer encoder** architectures. Unlike decoder-only models (GPT, Llama) that generate text autoregressively, encoder models process the entire input simultaneously and produce contextual representations for each token.

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
***

## 2 · Encoder vs Decoder: BERT vs GPT, Bidirectional vs Causal

**Not all transformers are created equal.** GPT and BERT are both transformers, but they're architecturally opposite — and that matters for embeddings.

**Decoder models (GPT family):**
- **Causal attention:** Each token can only attend to previous tokens (left-to-right)
- **Trained objective:** Predict the next token autoregressively
- **Use case:** Text generation — "complete this sentence"
- **Embedding quality:** Weaker for similarity tasks because the model never sees future context

**Encoder models (BERT family):**
- **Bidirectional attention:** Each token attends to all tokens (left and right)
- **Trained objective:** Masked language modeling — "fill in the blank"
- **Use case:** Understanding and classification — "are these two sentences similar?"
- **Embedding quality:** Ideal for embeddings because every token has full context

Modern embedding models (Sentence-BERT, E5, BGE) are **encoder-based** because bidirectional attention produces richer representations. A decoder can generate text but struggles to compare semantic similarity — it sees "The bank is by the river" token-by-token, never having future context to disambiguate "bank." An encoder sees the full sentence simultaneously and can encode that this "bank" relates to rivers, not money.

**Why this matters for RAG:** Your retriever needs to answer "are these two pieces of text about the same thing?" — a bidirectional task. Encoders are architecturally designed for it; decoders are not.

> 💡 **BERT vs GPT for embeddings:** BERT-family (encoders) are ideal for RAG retrieval because bidirectional attention sees full context. GPT-family (decoders) excel at generation but produce weaker embeddings due to causal masking. Use encoders for retrieval, decoders for generation.

***

## 3 · Contrastive Learning: InfoNCE, Why It Works

Embedding models are **not** trained to predict tokens like GPT. They are trained with **contrastive learning objectives** that teach the model to produce similar vectors for semantically similar text and dissimilar vectors for unrelated text.

**The training setup:** Given a query sentence and a batch of candidates, exactly one is the correct match (positive), and the rest are negatives. The model learns to push the query embedding close to the positive and far from the negatives.

The most common training objective is **InfoNCE** (Noise Contrastive Estimation), which treats embedding as a classification problem: given a query, identify the correct positive from a set of negatives:

$$L = -\log\left(
\frac{\exp(\text{sim}(q, p^+) / \tau)}{\exp(\text{sim}(q, p^+) / \tau) + \sum_i \exp(\text{sim}(q, p_i^-) / \tau)}
\right)$$

Where $q$ = query embedding, $p^+$ = positive (similar) embedding, $p_i^-$ = negative embeddings, and $\tau$ = temperature parameter.

**Intuition:** The model learns to maximize similarity between query and positive while minimizing similarity to negatives. Over millions of training examples, semantically related text clusters together in embedding space.

**Why "service uptime" and "availability target" cluster together:** The embedding model saw millions of (question, answer) pairs during training where these phrases co-occurred in similar contexts. Contrastive learning pulled them close in vector space.

**Why this matters for RAG:** This training objective is why semantic search works at all. Your retrieval quality depends on how well the embedding model was trained with contrastive learning — models trained on domain-specific data (medical, legal, code) perform better on those domains because their contrastive training aligned the embedding space to those concepts.

> 💡 **Contrastive learning:** InfoNCE loss trains embeddings to cluster semantically similar text. This is why RAG retrieval can find "authentication SLA" documents even when the query uses different wording ("service uptime targets") — the embedding space learned that these concepts are related.

***

## 4 · The RAG Pipeline: Ingestion + Query Flow

RAG is a **two-phase architecture** that separates offline corpus preparation from online query execution:

**Phase 1: Ingestion (Offline)**

1. **Chunk:** Split documents into 400-512 token chunks with 10-20% overlap
2. **Embed:** Convert each chunk to a dense vector using an encoder model (e.g., `text-embedding-3-small`)
3. **Store:** Index vectors in a vector database (Chroma, Pinecone, FAISS) with metadata

**Phase 2: Query (Runtime)**

1. **Embed:** Convert user query to a vector using the **same** embedding model
2. **Retrieve:** ANN search finds top-k most similar chunks by cosine similarity
3. **Augment:** Insert retrieved chunks into LLM prompt as context
4. **Generate:** LLM answers from retrieved context, grounding the response

**Pipeline diagram:**

```
INGESTION (Offline)                    QUERY (Runtime)
══════════════════════                 ═══════════════

Documents                              User Query
    ↓                                      ↓
Chunk (400 tokens)                     Embed Query
    ↓                                  (same model!)
Embed All Chunks                           ↓
    ↓                                  ANN Search
Store in Vector DB                     (top-k chunks)
                                           ↓
                                       Augment Prompt
                                           ↓
                                       LLM Generate
                                           ↓
                                       Grounded Answer
```

**Critical constraint:** Query and document embeddings **must use the exact same model**. `text-embedding-3-small` and `text-embedding-ada-002` both output 1,536-dimensional vectors, but those dimensions mean completely different things — the axes of each model's space are independent. Mixing models produces numerically meaningless similarity scores.

**Chunking strategy matters:** A 40% recall difference exists between naive fixed-size chunking and well-tuned recursive splitting. Chunk size determines the recall-vs-context tradeoff:

- **256 tokens:** High precision, but multi-section entries get fragmented
- **400-512 tokens:** Sweet spot for most use cases (80% of production RAG systems)
- **1,024+ tokens:** Better for analytical queries, but dilutes embeddings

**Chunk overlap (10-20%):** Prevents boundary information loss. If a key sentence falls on a chunk boundary, overlap ensures it appears in multiple chunks.

**Hybrid search (BM25 + dense retrieval):** Combines keyword matching (exact terms, acronyms, codes) with semantic similarity (paraphrases, synonyms). **Reciprocal Rank Fusion (RRF)** merges results:

$$\text{RRF}(d) = \sum_{\text{ranker}} \frac{1}{k + \text{rank}_i(d)}$$

Where $k = 60$ is a smoothing constant. This merges heterogeneous scores (BM25 vs. cosine) without normalization.

> 💡 **RAG pipeline:** Chunk → embed → store (offline), then embed query → retrieve → augment → generate (online). Mixing embedding models breaks retrieval; chunking strategy determines recall; hybrid search combines exact + semantic matching.

***

## 5 · HyDE: Hypothetical Document Embeddings

**The semantic gap problem:** A user asks "What's our authentication service SLA?" — phrased as a question. The wiki document says "Authentication Service SLA: 99.95% uptime, p99 <50ms" — phrased as a statement. These have different sentence structures and embeddings, reducing similarity even though they're about the same topic.

**HyDE (Hypothetical Document Embeddings)** solves this by **embedding a hypothetical answer instead of the raw question**. Instead of directly embedding the query, generate what an answer might look like, then embed that. Hypothetical answers are structurally similar to actual documents, closing the phrasing gap.

**The algorithm:**

1. **User query:** "What's our authentication service SLA?"
2. **Generate hypothetical answer:** Use an LLM to generate a plausible answer (even if hallucinated): "The authentication service SLA is 99.9% uptime with p99 latency under 200ms."
3. **Embed hypothetical answer:** This embedding is structurally similar to actual wiki documents
4. **Retrieve:** Search using the hypothetical answer embedding
5. **Generate real answer:** LLM sees the actual retrieved chunks and corrects any hallucinations from step 2

**Why it works:** The hypothetical answer embedding clusters near actual document chunks because they share structural patterns ("The X is Y" vs. "What is X?"). The hallucinated numbers in step 2 don't matter — retrieval finds the actual documents, and the LLM reads the correct values in step 5.

**Cost tradeoff:** HyDE adds one extra LLM call per query (hypothetical answer generation). For a 50-token generation at $0.03/1M tokens (GPT-4-mini), that's $0.0000015 per query — negligible. The retrieval quality improvement (2-5 percentage points in recall) often justifies the cost.

**When to use HyDE:**
- ✅ Questions phrased differently than documents (Q&A style vs. declarative docs)
- ✅ Queries where semantic gap reduces retrieval quality
- ❌ Exact keyword searches (BM25 already handles these)
- ❌ Ultra-low-latency requirements (<100ms p95)

> 💡 **HyDE:** Embed a hypothetical answer instead of the raw question, closing the phrasing gap between queries and documents. Adds one LLM call per query but improves recall by 2-5 percentage points when questions and documents have structural mismatches.

***

## 6 · Failure Modes

RAG reduces hallucination from 38% → 4%, but the remaining 4% traces to specific failure modes. Understanding these is critical for debugging production systems.

### Common RAG Failure Modes

| Failure Mode | Cause | Symptom | Fix |
|--------------|-------|---------|-----|
| **Semantic gap** | Query and relevant documents use different terminology | Relevant chunks not retrieved; model answers from parametric memory or says "I don't know" | Use hybrid search (BM25 + dense); consider domain-specific embedding model; HyDE for structural mismatches |
| **Chunk size mismatch** | Chunks too small → context fragmented; chunks too large → embeddings diluted | Retrieved chunks don't contain complete information; similarity scores mushy | Test 256/400/512 tokens on your corpus; 400-512 is the sweet spot for most cases |
| **Lost-in-the-middle** | LLM ignores context buried in the middle of a long prompt | Model answers from parametric memory despite relevant chunks in context | Re-rank chunks (put highest-confidence first); reduce k (top-5 instead of top-20); use models with better long-context handling |
| **Unfaithful generation** | Model hallucinates despite seeing correct information in retrieved chunks | Answer contradicts retrieved context | Fine-tune for instruction following; add "cite your sources" prompt constraint; use structured output formats (JSON) |
| **Retrieval failure** | Relevant document exists but wasn't retrieved | Model says "I don't know" or invents an answer; correct document not in top-k | Improve chunking strategy; increase k; hybrid search; better embedding model; check index quality (Ch.5) |
| **Stale index** | Document corpus updated but embeddings not re-indexed | Retrieved chunks contain outdated information | Implement incremental index updates; version control embeddings; track document freshness metadata |

**Debugging workflow:**

1. **Check if retrieval succeeded:** Log the top-k chunks for failed queries. If the correct chunk is present, it's a generation problem. If absent, it's a retrieval problem.
2. **Test with manual context:** Put the correct answer in the prompt manually. If the model still fails, fine-tune for instruction following. If it succeeds, improve retrieval.
3. **Measure recall@k:** What fraction of test queries retrieve the relevant chunk in top-k? If <90%, focus on chunking, embedding model, or hybrid search.

> 💡 **Failure modes:** The 4% residual errors split into retrieval failures (3%) and generation failures (1%). Fix retrieval with better chunking/hybrid search/re-ranking. Fix generation with fine-tuning for instruction following.

***

## 7 · Key Distinctions (Interview Table)

These are the concepts interviewers expect you to know cold. Each row represents a common interview question.

| Must know | Likely asked | Trap to avoid |
|---|---|---|
| The two-phase RAG pipeline: **ingestion** (chunk → embed → store in vector DB) vs **query** (embed query → ANN retrieve → re-rank → generate); failures in either phase compound downstream | "Walk me through a RAG pipeline end-to-end" — interviewers expect both phases, not just retrieval | Describing RAG as just "search then generate" without mentioning chunking, embedding, and the index |
| Chunking strategy matters: **fixed-size** (fast, baseline), **sentence/recursive** (respects natural boundaries, most common in prod), **semantic** (highest recall, groups by meaning, slowest); chunk size drives the recall-vs-context tradeoff | "What chunking strategy would you use for a legal document Q&A system?" | Saying chunk size doesn't matter — a 40% recall difference between naive and tuned chunking is well-documented |
| Cosine similarity = dot product on L2-normalized vectors; for un-normalized embeddings they differ; most production pipelines normalize at index time and use dot product (faster) at query time | "When do you use cosine vs. dot product similarity?" | Saying "cosine similarity and dot product are the same thing" — they're only equivalent after normalization |
| Sparse retrieval (BM25): keyword-based, fast, no index warm-up, excels on rare proper nouns and exact matches. Dense retrieval (bi-encoder): semantic, catches paraphrases, misses OOV terms. Hybrid search uses **Reciprocal Rank Fusion (RRF)** to merge both rank lists with no score normalization required | "When would you prefer BM25 over a dense retriever?" | Saying dense retrieval is strictly better — on domain-specific jargon, acronyms, or cold-start (no training data), BM25 often wins |
| Re-ranker (cross-encoder): reads the full query + chunk together, much more accurate than bi-encoder but O(k) LLM calls; bi-encoder retrieval is approximate but sub-millisecond. Architecture: use bi-encoder for top-100 recall, cross-encoder to re-rank to top-5 | "Why not use a cross-encoder for all retrieval?" | Forgetting that a cross-encoder has to see every candidate — you can't run it against millions of chunks |
| Hybrid search with RRF formula: $\text{score} = \sum_{\text{ranker}} \frac{1}{k + \text{rank}_i}$ where $k=60$ is a smoothing constant; merges sparse and dense rank lists without needing to normalize heterogeneous scores | "How does RRF combine results from different retrievers?" | Trying to combine raw BM25 and cosine scores directly — their scales are incompatible without normalization |
| **Encoder vs Decoder for embeddings:** BERT (encoder, bidirectional attention, ideal for similarity) vs GPT (decoder, causal attention, optimized for generation). Use encoders for RAG retrieval | "Why is BERT better than GPT for embeddings?" | Saying "all transformers are the same" — causal masking in decoders produces weaker embeddings because tokens never see future context |
| **Contrastive learning (InfoNCE):** Embedding models are trained to push similar text close and dissimilar text apart. This is why semantic search works — the embedding space learned which concepts are related | "How are embedding models trained?" | Saying embeddings are trained to predict the next token like GPT — embedding models use contrastive objectives, not autoregressive prediction |
| **HyDE:** Embed a hypothetical answer instead of the raw query to close the phrasing gap between questions and declarative documents. Adds one LLM call per query but improves recall when queries and docs have structural mismatches | "What is HyDE and when would you use it?" | Thinking HyDE embeddings replace the retrieval step entirely — HyDE generates a hypothetical answer, embeds it, then uses that for retrieval |
| **Failure mode: embedding similarity ≠ relevance** — a chunk can be semantically close to the query yet contextually useless (e.g., a definition chunk when the query needs a procedure); re-ranking and result diversity both matter | "Why might high-similarity retrieved chunks still produce a bad RAG answer?" | Assuming top-k by cosine is always the right answer to pass to the LLM |

***

## 8 · Bridge to Next Chapter

RAG grounds LLM answers in the organization's actual documents — but retrieval at prototype scale is brute-force: every query does a linear scan over all chunks. **Vector DBs (Ch.5)** replaces that with HNSW and IVF indexes that return approximate nearest neighbors in sub-millisecond time, scaling from ~1,400 wiki chunks to millions without hitting the latency constraint (<3s p95).

**What you've learned:**
- Embeddings convert text to vectors where semantic distance is measurable
- Encoder models (BERT) are ideal for embeddings; decoder models (GPT) are ideal for generation
- Contrastive learning (InfoNCE) is why embedding models cluster semantically similar text
- RAG pipeline: chunk → embed → store (ingestion), embed query → retrieve → augment → generate (query)
- HyDE closes the phrasing gap by embedding hypothetical answers instead of raw queries
- Failure modes: semantic gap, chunk size, lost-in-middle, unfaithful generation, retrieval failure

**What's next:**
- Ch.5 (Vector DBs): HNSW, IVF, DiskANN — the index structures that make RAG scale
- Ch.6 (RAG Evaluation): Measuring retrieval quality, faithfulness, and answer correctness
- Ch.8 (Fine-Tuning): When RAG can't fix behavior problems (instruction following, tone, format)

---

## Bridge

RAG grounds LLM answers in retrieved documents, reducing hallucination from 38% to 4%. You now understand *what* embeddings are, *how* they cluster semantically similar text, and *why* the two-phase pipeline (ingestion + query) works. But there's a scaling wall: brute-force retrieval over 50K chunks takes 1.5 seconds — unacceptable for production.

The next chapter — [Vector Databases](../ch05-vector-dbs/vector-dbs.md) — solves this with approximate nearest neighbor (ANN) indexes. You'll learn why traditional spatial indexes fail in high dimensions (curse of dimensionality), how HNSW achieves O(log N) graph traversal, when IVF cluster-based search wins, and how DiskANN moves billion-scale indexes to SSD without sacrificing recall. This is what makes RAG fast enough to deploy.

## Illustrations

![RAG and embeddings — semantic clustering, ingestion pipeline, query pipeline, chunk-size tradeoff](img/RAG%20and%20Embeddings.png)
