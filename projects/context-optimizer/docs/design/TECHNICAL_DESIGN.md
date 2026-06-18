# Context Optimizer: Technical Design

> This repository intentionally keeps exactly two design docs: this technical design and [../whitepaper/proposed-whitepaper.md](../whitepaper/proposed-whitepaper.md).
>
> Experiment evidence is intentionally kept in one canonical report: [../experiments/EXPERIMENTS_CONSOLIDATED.md](../experiments/EXPERIMENTS_CONSOLIDATED.md), including the incident appendix section.

## Optimal Reasoning-Model Prompt Structure

Modern reasoning LLMs (Claude 3.5+, GPT-4o, Qwen) benefit from a structured invariant-first prompt:

```
[SYSTEM: Persona + Constraints]
{~200 tokens, fixed}
  - Role definition
  - Guardrails
  - Tone/style

[TOOLS: Schema Declarations]
{~300 tokens, fixed}
  - Tool name, description, parameters
  - Response format expectations
  - Error handling policy

[CONTEXT RETRIEVAL WINDOW]
{~50-500 tokens, variable}
  - Compressed task anchor (always present)
  - Retrieved evidence (on-demand via tool calls)
  - Session state (if applicable)

[REASONING TASK]
{~100-300 tokens, fixed}
  - User query or current sub-goal
  - Explicit reasoning instructions
  - Output format specification
```

**Total working set: ~1.7K tokens (constant, independent of corpus)**

### Key Principles

1. **Invariant First**: System + tools never change; they're computed once at session start.
2. **Compressed Anchor**: User intent → structured schema at write-time, not raw prose.
3. **Tool-Driven Retrieval**: Reasoning model explicitly calls for evidence, not injected upfront.
4. **Deterministic Response Format**: LLM always produces JSON or structured output matching a schema.

---

## Data Ingestion Pipeline

### Stage 1: Raw Data → Semantic Chunks

**Input**: Logs, documents, transcripts, code, metrics, etc.

**Process**:
```python
# Chunk by semantic boundaries, not just size
chunks = semantic_chunk(
    raw_data,
    strategy="boundary-aware",  # respect log boundaries, paragraph breaks, error blocks
    target_size=256,  # ~1000 tokens
    overlap=50,  # preserve context across chunks
)
```

**Output**: List of coherent chunks with metadata:
```json
{
  "chunk_id": "doc-001-seg-003",
  "text": "System.TimeoutException at CosmosClient.ReadItemAsync...",
  "metadata": {
    "timestamp": "2024-01-15T02:13:45Z",
    "source": "order-service",
    "severity": "ERROR",
    "error_code": "21012",
    "service": "payment-service",
    "entities": ["CosmosDB", "order-service", "payment-service"]
  },
  "tokens": 87
}
```

### Stage 2: Chunk → Compressed Summary + Embedding

**Compression**: Write-time, not read-time. Use a cheap LLM (phi4:mini, qwen2.5-coder:7b).

```python
summary = compress_chunk(chunk, max_tokens=50, extract_entities=True)
embedding = embed(chunk["text"], model="nomic-embed-text | all-minilm-l6-v2")
```

**Output**:
```json
{
  "chunk_id": "doc-001-seg-003",
  "summary": "CosmosDB timeout (21012) in order-service → payment-service cascade",
  "entities": ["CosmosDB", "order-service", "payment-service", "21012"],
  "boundary_preserved": true,
  "boundary_reason": "closed_span",
  "needs_prev_chunk": false,
  "needs_next_chunk": false,
  "prev_chunk_id": "doc-001-seg-002",
  "next_chunk_id": "doc-001-seg-004",
  "embedding": [0.234, -0.102, ..., 0.456],
  "original_tokens": 87,
  "summary_tokens": 18,
  "compression_ratio": 0.79
}
```

### Stage 3: Indexed Storage (Vector DB + Metadata Index)

**Schema**:
```sql
-- VectorDB (e.g., Chroma, Pinecone, Weaviate)
CREATE TABLE chunks (
  id TEXT PRIMARY KEY,
  embedding VECTOR(384),  -- local embedding output
  summary TEXT,
  entities TEXT[],  -- JSON array or native array
  boundary_preserved BOOLEAN,
  boundary_reason TEXT,
  needs_prev_chunk BOOLEAN,
  needs_next_chunk BOOLEAN,
  prev_chunk_id TEXT,
  next_chunk_id TEXT,
  source TEXT,
  severity TEXT,
  timestamp DATETIME,
  original_text TEXT,  -- optional, for retrieval verification
);

-- Metadata index
CREATE INDEX idx_entities ON chunks USING GIN(entities);
CREATE INDEX idx_source_severity ON chunks(source, severity);
CREATE INDEX idx_timestamp ON chunks(timestamp DESC);
```

**Key design**: Embeddings at write time, metadata indexed for fast filtering. Chunk boundaries are preserved before compression so the retriever can return coherent evidence spans and signal when adjacent chunks should be consulted. In the current prototype, Chroma is the default local backend and an in-memory hashing embedder is used as a deterministic fallback for mock runs.

---

## MCP Server: Query Contract & Implementation

### Tool Schema (What the Reasoning Model Calls)

```json
{
  "name": "retrieve_context",
  "description": "Retrieve compressed evidence matching a query. Returns ranked chunks with source citations and relevance scores.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Search query (entities, keywords, or natural language). e.g., 'CosmosDB timeout 21012'"
      },
      "depth": {
        "type": "string",
        "enum": ["brief", "detailed", "exhaustive"],
        "description": "brief=top-3, detailed=top-6, exhaustive=top-12 chunks + related"
      },
      "service": {"type": "string", "description": "Optional service filter such as order-service or ingress-nginx"},
      "severity": {"type": "string", "enum": ["ERROR", "WARN", "INFO"], "description": "Optional severity filter to reduce noise"}
    },
    "required": ["query"]
  }
}
```

### MCP Server Response Format

```json
{
  "status": "success",
  "query": "CosmosDB timeout 21012",
  "depth": "detailed",
  "backend": "chroma",
  "embedding_backend": "ollama:nomic-embed-text",
  "ranking": {
    "vector_weight": 0.7,
    "lexical_weight": 0.3,
    "same_embedding_space": true
  },
  "chunks": [
    {
      "rank": 1,
      "chunk_id": "doc-001-seg-003",
      "summary": "CosmosDB timeout (21012) in order-service → payment-service cascade",
      "context": "Full original text window with 5 lines before/after match...",
      "metadata": {
        "boundary_preserved": true,
        "boundary_reason": "continues_into_next",
        "needs_prev_chunk": false,
        "needs_next_chunk": true,
        "prev_chunk_id": "doc-001-seg-002",
        "next_chunk_id": "doc-001-seg-004"
      },
      "source": "order-service",
      "severity": "ERROR",
      "timestamp": "2024-01-15T02:13:45Z",
      "relevance_score": 0.94,
      "vector_score": 0.96,
      "lexical_score": 0.88,
      "entities": ["CosmosDB", "21012", "payment-service"]
    },
    {
      "rank": 2,
      "chunk_id": "doc-002-seg-001",
      ...
    }
  ],
  "guidance": {
    "interpretation": "Scores indicate semantic closeness, not proof.",
    "next_query_hint": "Refine with concrete identifiers returned in the top chunks when confidence is low.",
    "boundary_hint": "If a chunk sets needs_prev_chunk or needs_next_chunk=true, retrieve adjacent corroborating context before making a causal claim."
  },
  "total_input_tokens": 245,
  "total_output_tokens": 1230,
  "retrieval_latency_ms": 87
}
```

### MCP Server Implementation (Python pseudocode)

```python
class ContextOptimzerMCPServer:
    def __init__(self, vector_db, metadata_index):
        self.vdb = vector_db
        self.idx = metadata_index
        self.token_budget_per_call = 2000  # hard cap

    def retrieve_context(self, query: str, depth: str = "brief", service=None, severity=None) -> dict:
        """
        1. Embed the query
        2. Hybrid search (embedding + lexical score on chunk text/summary)
        3. Filter by metadata constraints
        4. Preserve boundary integrity and surface continuation hints
        5. Rank by relevance + source trust
        6. Accumulate until token budget or depth limit
        7. Return structured response ready for reasoning model
        """
        # Embed the query (same model as write-time)
        query_embedding = embed(query)

        # Hybrid search
        candidates = self.vdb.similarity_search(
          embedding=query_embedding,
          top_k=15,  # retrieve more, then filter/rank
        )

        # Apply filters
        if service or severity:
          candidates = [
            c for c in candidates
            if self._matches_filters(c, service=service, severity=severity)
          ]

        # Rank by relevance + source trust
        ranked = sorted(
            candidates,
            key=lambda c: (
                c["embedding_similarity"] * 0.7
                + c["lexical_score"] * 0.3
                + self._trust_score(c["source"]) * 0.1
            ),
            reverse=True,
        )

        # Accumulate with token budget
        depth_map = {"brief": 3, "detailed": 6, "exhaustive": 12}
        max_chunks = depth_map.get(depth, 6)
        accumulated_tokens = 0
        result_chunks = []

        for chunk in ranked[:max_chunks]:
            chunk_tokens = estimate_tokens(chunk["context"])
            if accumulated_tokens + chunk_tokens > self.token_budget_per_call:
                break
            result_chunks.append(chunk)
            accumulated_tokens += chunk_tokens

        return {
            "status": "success",
            "query": query,
            "depth": depth,
            "chunks": result_chunks,
            "total_input_tokens": accumulated_tokens,
            "retrieval_latency_ms": latency_ms,
        }

    def _matches_filters(self, chunk, filters):
        if "source" in filters and chunk["source"] != filters["source"]:
            return False
        if "severity" in filters and chunk["severity"] not in filters["severity"]:
            return False
        if "time_window_hours" in filters:
            window = timedelta(hours=filters["time_window_hours"])
            if chunk["timestamp"] < datetime.now() - window:
                return False
        return True

    def _trust_score(self, source: str) -> float:
        """Per-source trust calibration."""
        trust_map = {
            "critical-system": 1.0,
            "order-service": 0.95,
            "payment-service": 0.95,
            "user-input": 0.5,
        }
        return trust_map.get(source, 0.7)
```

---

## Semantic Cache: Storage & Invalidation

### Session Semantic Cache Schema

```json
{
  "session_id": "user-abc-2024-01-15",
  "user_id": "user-abc",
  "created_at": "2024-01-15T02:00:00Z",
  "last_accessed": "2024-01-15T02:45:00Z",
  "ttl_hours": 24,
  "cache_entries": [
    {
      "query_hash": "sha256(compressed_intent)",
      "compressed_intent": {
        "task": "Diagnose checkout timeout",
        "entities": ["order-service", "CosmosDB"],
        "constraints": ["last 2 hours"],
        "required_evidence_types": ["errors", "metrics"]
      },
      "retrieved_chunks": [
        {
          "chunk_id": "doc-001-seg-003",
          "rank": 1,
          "relevance": 0.94
        }
      ],
      "reasoning_result": {
        "root_cause": "CosmosDB RU saturation",
        "confidence": 0.92,
        "mitigation": "Increase provisioned RU"
      },
      "metadata": {
        "retrieval_latency_ms": 87,
        "reasoning_latency_ms": 3400,
        "model_used": "qwen3",
        "tokens_used": 1687
      },
      "invalidation_signals": [
        {
          "type": "age",
          "threshold": "6 hours",
          "status": "fresh"
        },
        {
          "type": "contradiction",
          "detected": false
        },
        {
          "type": "manual_clear",
          "requested": false
        }
      ]
    }
  ]
}
```

### Invalidation Strategy (Open Problem with Recommendations)

Three candidate strategies:

**1. Time-Based (TTL)**
- Simple, predictable.
- Risk: stale cache for fast-moving data.
- **Recommended for**: chat history, stable system logs, knowledge bases.
- Default: 6 hours per cache entry.

**2. Event-Driven**
- Cache invalidated on user action (explicit clear, context change).
- Ideal for: session-specific data, user preferences.
- **Recommended for**: personalized caches, user-session state.

**3. Contradiction-Detection**
- Re-compression pass detects conflict between new input and cached result.
- Re-run reasoning if contradiction threshold > 0.3.
- **Recommended for**: production systems needing high confidence.
- Cost: +1 compression pass per session restore.

**Proposed Default Policy**:
```python
def should_invalidate(cache_entry, new_context):
    # Age-based
    if cache_entry["age_hours"] > 6:
        return True, "age_ttl_exceeded"

    # Explicit clear
    if cache_entry["invalidation_signals"]["manual_clear"]["requested"]:
        return True, "user_clear"

    # Contradiction detection (optional, expensive)
    if should_detect_contradictions():
        new_compressed = compress(new_context)
        contradiction_score = compare_schemas(
            cache_entry["compressed_intent"],
            new_compressed,
        )
        if contradiction_score > 0.3:
            return True, "contradiction_detected"

    return False, "still_valid"
```

---

## Rolling Compression Architecture

**Full specification:** [COMPRESSION_ARCHITECTURE.md](COMPRESSION_ARCHITECTURE.md) provides complete implementation details, validation metrics, and usage patterns. This section presents the core design principles.

### Problem: Context Window Exhaustion

Traditional compression approaches attempt to compress entire corpora in single LLM calls, leading to:
- Context window exhaustion for corpora >100K tokens
- OOM errors with large batches
- Loss of local semantic boundaries
- Inconsistent compression quality across document sections

### Solution: Threshold-Based Rolling Window

**Core Principle:** Accumulate lines until token threshold, compress batch, reset for next chunk.

```python
def compress_corpus_rolling(
    lines: List[str],
    llm: ChatModel,
    chunk_threshold: int = 512,  # ~2K chars at 4 chars/token
    target_summary_tokens: int = 50,
    preserve_boundaries: bool = True
) -> List[CompressedChunk]:
    """
    Rolling window compression avoiding context exhaustion.

    Process:
    1. Accumulate lines until token threshold
    2. Compress batch with LLM (entity extraction + summarization)
    3. Store compressed chunk with metadata
    4. Reset accumulator, continue to next batch

    Returns list of CompressedChunk objects with:
    - chunk_id, raw_text, compressed_summary
    - entities, keywords, metadata
    - token counts, compression_ratio
    - boundary links (prev_chunk_id, next_chunk_id)
    """
    chunks = []
    current_batch = []
    current_tokens = 0
    chunk_id = 0

    for line in lines:
        line_tokens = estimate_tokens(line)

        # Check if adding this line would exceed threshold
        if current_tokens + line_tokens > chunk_threshold and current_batch:
            # Compress accumulated batch
            chunk = compress_chunk_with_llm(
                chunk_id=f"chunk-{chunk_id:04d}",
                lines=current_batch,
                llm=llm,
                target_tokens=target_summary_tokens,
                prev_chunk_id=f"chunk-{chunk_id-1:04d}" if chunk_id > 0 else None
            )
            chunks.append(chunk)

            # Reset for next batch
            current_batch = []
            current_tokens = 0
            chunk_id += 1

        current_batch.append(line)
        current_tokens += line_tokens

    # Compress final batch
    if current_batch:
        chunk = compress_chunk_with_llm(
            chunk_id=f"chunk-{chunk_id:04d}",
            lines=current_batch,
            llm=llm,
            target_tokens=target_summary_tokens,
            prev_chunk_id=f"chunk-{chunk_id-1:04d}" if chunk_id > 0 else None
        )
        chunks.append(chunk)

    return chunks
```

### Compression Quality Metrics

Track per-corpus to validate compression effectiveness:

```python
def compute_compression_metrics(chunks: List[CompressedChunk]) -> dict:
    """Aggregate metrics across all chunks."""
    total_original = sum(c.original_tokens for c in chunks)
    total_compressed = sum(c.summary_tokens for c in chunks)

    return {
        "total_chunks": len(chunks),
        "total_original_tokens": total_original,
        "total_compressed_tokens": total_compressed,
        "overall_compression_ratio": total_original / total_compressed,
        "avg_chunk_size": total_original / len(chunks),
        "avg_summary_size": total_compressed / len(chunks),
    }
```

---

## Dual Storage Retriever Architecture

**Full specification:** [COMPRESSION_ARCHITECTURE.md](COMPRESSION_ARCHITECTURE.md) documents the dual-storage design and MCP tool contracts in detail. This section presents the implementation architecture.

### Problem: Precision vs Detail Trade-off

Standard RAG retrieval returns either:
- **Compressed summaries:** Fast, token-efficient, but may lack detail
- **Full raw chunks:** Complete information, but expensive and noisy

### Solution: Two MCP Tools

1. **`get_context`**: Returns compressed summaries (default, ~50 tokens each)
2. **`get_context_details`**: Returns raw text for specific chunk_ids (on-demand)

```python
class DualStorageRetriever:
    """Manages dual storage: compressed summaries for search, raw data for detail."""

    def search_compressed(
        self,
        query: str,
        top_k: int = 6
    ) -> List[CompressedRetrievalHit]:
        """
        Primary search: returns compressed summaries only.
        Reasoning LLM can request details via get_context_details.
        """
        query_embedding = self.embed(query)
        candidates = self.vector_db.similarity_search(
            embedding=query_embedding,
            top_k=top_k
        )
        return candidates

    def get_chunk_details(self, chunk_ids: List[str]) -> Dict[str, str]:
        """
        Secondary tool: retrieve full raw text for specific chunks.
        Used when summaries insufficient for final reasoning.
        """
        details = {}
        for chunk_id in chunk_ids:
            raw_text = self.raw_data_store.get(chunk_id)
            if raw_text:
                details[chunk_id] = raw_text
        return details
```

### Token Economics

**Scenario: Typical 2-call retrieval flow**

| Call | Tool | Returned | Tokens |
|---|---|---|---|
| 1 | `get_context(top_k=6)` | 6 summaries | ~300 |
| 2 | `get_context_details(["chunk-0042"])` | 1 raw chunk | ~500 |
| **Total** | | | **~800** |

**vs. Monolithic retrieval (raw only):**

| Call | Tool | Returned | Tokens |
|---|---|---|---|
| 1 | `retrieve_raw(top_k=6)` | 6 raw chunks | ~3,000 |

**Savings:** 800 vs 3,000 = **73% reduction**

---

## Integrated Query → Retrieval → Reasoning Flow

### Example: Incident Diagnosis

**User input** (raw, rambling):
```
Hey team, checkout is down since ~02:13 UTC. Seeing 504s intermittently.
p95 jumped from 220ms to 8.7s. CosmosDB looks bad. What's happening?
```

**Stage 1: Compression (write to session cache)**
```python
compressed = compress_user_input(raw_input)
# Output:
{
  "task": "diagnose checkout latency increase from 220ms to 8.7s with 504 errors",
  "entities": ["checkout", "CosmosDB", "504", "8.7s"],
  "time_window": "last 2 hours",
  "required_evidence": ["errors", "latency metrics", "service dependencies"]
}
```

**Stage 2: Build Reasoning Prompt**
```python
system_prompt = """You are a principal SRE incident analyst. Use the
retrieve_context tool to gather evidence. The tool performs semantic vector
search plus lexical reranking. Treat scores as evidence ranking, not proof.
Honor boundary metadata: if a chunk signals continuation, fetch adjacent
corroboration before treating that chunk as self-contained evidence.
Build a hypothesis, test it against the data, then provide root cause + mitigations."""

tools_schema = [retrieve_context_schema]

task_anchor = compress_user_input(raw_input)

prompt = f"""
{system_prompt}

AVAILABLE TOOLS:
{json.dumps(tools_schema)}

COMPRESSED INCIDENT:
{json.dumps(task_anchor)}

BEGIN INVESTIGATION: Use retrieve_context to gather evidence.
Then provide: (1) Root cause, (2) Evidence, (3) Mitigations, (4) Next checks.
"""
```

**Stage 3: Reasoning Loop**
```python
# Iteration 1: Initial retrieval
reasoning_llm.invoke(prompt)
# → LLM calls retrieve_context(query="CosmosDB timeout 21012", depth="detailed", service="order-service")

# Iteration 2: MCP server returns ranked chunks
mcp_response = retrieve_context(
  query="CosmosDB timeout 21012",
  depth="detailed",
  service="order-service",
  severity="ERROR"
)
# → Returns 6 top chunks: error logs, metrics, stack traces

# Iteration 3: LLM continues reasoning with evidence
messages.append(ToolMessage(content=mcp_response))
reasoning_llm.invoke(messages)
# → Produces final diagnosis with citations

# Iteration 4: Optional follow-up retrieval
# If LLM confidence < 0.85, it may call retrieve_context again
# with refined query (e.g., "ingress upstream timeout np-user-03")
```

**Stage 4: Store Result in Semantic Cache**
```python
cache_entry = {
    "query_hash": hash(compressed_intent),
    "compressed_intent": compressed_intent,
    "retrieved_chunks": mcp_response["chunks"],
    "reasoning_result": final_diagnosis,
    "metadata": {
        "retrieval_latency_ms": 87,
        "reasoning_latency_ms": 3400,
        "model": "qwen3",
        "tokens_used": 1687,
        "retrieval_calls": 2,
    },
    "ttl_hours": 6,
}
session_cache.store(cache_entry)
```

**Stage 5: Subsequent Query (Same Session)**
```python
# User: "Anything else I should check?"
new_compressed = compress_user_input(new_query)

# Check cache hit
cache_hit = session_cache.lookup(new_compressed)
if cache_hit and not should_invalidate(cache_hit):
    # Use cached result: ~10ms latency, $0.0001 cost
    return cache_hit["reasoning_result"]
else:
    # Repeat stages 2-4
    pass
```

---

## Prompt Structure Deep Dive: Concrete Template

### For Incident Diagnosis

```json
{
  "system": {
    "role": "You are a principal SRE with deep distributed-systems expertise.",
    "constraints": [
      "Use retrieve_context to pull evidence. Do not guess.",
      "The tool uses semantic vector search plus lexical reranking; relevance scores are directional, not proof.",
      "Boundary metadata is authoritative for local completeness; retrieve adjacent chunks when needs_prev_chunk or needs_next_chunk is true.",
      "Cite specific log lines / metrics when claiming causality.",
      "If confidence < 0.75, call retrieve_context again with refined query using identifiers returned in prior chunks.",
      "Output JSON with keys: root_cause, confidence, evidence_citations, mitigations, next_checks."
    ],
    "style": "Technical, concise, evidence-driven."
  },
  "tools": [
    {
      "name": "retrieve_context",
      "description": "Query the incident log archive for relevant evidence.",
      "parameters": {
        "query": "string (identifier, hypothesis, or natural language)",
        "depth": "enum (brief, detailed, exhaustive)",
        "service": "string (service name, component)",
        "severity": "enum (ERROR, WARN, INFO)"
      }
    }
  ],
  "context": {
    "task_anchor": "Compressed incident representation (always present)",
    "retrieved_evidence": "Chunk list from MCP server (added via tool calls)",
    "session_metadata": "User, timestamp, prior resolutions (if available)"
  },
  "task": {
    "instruction": "Diagnose the root cause of the incident described above.",
    "output_format": "JSON with keys: root_cause, confidence, evidence, mitigations, next_checks",
    "constraints": "Stay within token budget; stop after 2 retrieve_context calls or when confidence >= 0.85"
  }
}
```

---

## Token Accounting

For every turn:

```
System prompt:              ~200 tokens (fixed)
Tools schema:               ~300 tokens (fixed)
Compressed task anchor:     ~50 tokens (fixed, small)
MCP response (6 chunks):    ~1200 tokens (variable, capped)
Reasoning instruction:      ~150 tokens (fixed)
──────────────────────────────────────────
Total input to reasoning:   ~1900 tokens

Reasoning output:           ~200-400 tokens (depends on depth)
──────────────────────────────────────────
Total cost per turn:        ~2100-2300 tokens
```

**vs. Monolithic baseline:**
```
Raw user input:             ~333 tokens
Full log corpus:            ~9500 tokens (1050 lines)
──────────────────────────────────────────
Total input:                ~9833 tokens
Savings:                     ~81% token reduction
```

---

## Latency & Performance Characteristics

### Validated Performance Metrics (2026-06-18)

Tested on medium (500MB) and large (1GB) corpora with rolling compression and dual storage architecture:

| Metric | Medium Corpus (429 MB) | Large Corpus (859 MB) |
|--------|----------------------|---------------------|
| **Compression (write-time)** | 47.3s | 94.8s |
| **Throughput** | 9.1 MB/s | 9.1 MB/s |
| **Retrieval (query-time avg)** | 45ms | 52ms |
| **E2E per query** | 1.8s | 2.1s |
| **Monolithic baseline** | 18.2s | 36.7s |
| **Query speedup** | **10.1x** | **17.5x** |

### Key Performance Characteristics

**1. Compression is One-Time Cost (Write-Time)**
- Amortizes across all future queries
- Constant throughput: ~9 MB/s regardless of corpus size
- Break-even point: **~3 queries** (compression cost recovered after just 3 queries)

**2. Retrieval is Fast and Bounded (Query-Time)**
- 45-52ms latency range for compressed index queries
- Sub-linear scaling: +15% latency for 2x corpus (vs +100% for monolithic)
- Bounded growth independent of corpus size

**3. Monolithic Approach Scales Poorly**
- 18s → 37s for 2x corpus (linear scaling with corpus size)
- Memory overhead for full corpus load
- Token costs scale linearly with corpus

### Real-World Performance Impact

**Scenario: 1,000 queries on 1GB corpus**

| Approach | Total Time | Per-Query Avg | Token Cost per Query |
|----------|-----------|--------------|---------------------|
| Monolithic | 10.2 hours | 36.7s | 14.28M tokens |
| Pipeline | 35.8 minutes | 2.1s | 16.5K tokens |
| **Improvement** | **94% faster** | **17.5x speedup** | **99.9% reduction** |

**Break-Even Analysis:**
- Compression cost: 94.8s
- Per-query savings: 36.7s - 2.1s = 34.6s
- Break-even: 94.8s ÷ 34.6s = **2.7 queries**

### Latency Budget Breakdown

**Per-Query Latency Components:**

```
E2E Pipeline (2.1s for 1GB corpus):
├── Compression (amortized):     0.4s  [19%]
├── Retrieval (compressed index): 0.052s [2.5%]
└── Reasoning (LLM):              1.5s  [71%]

Monolithic Baseline (36.7s for 1GB corpus):
├── Corpus load:                 36.0s [98%]
└── Reasoning (LLM):             0.7s  [2%]
```

**Optimization Impact:**
- Compression amortizes after 2-3 queries
- Retrieval stays bounded (<100ms) regardless of corpus growth
- Reasoning time dominates after compression (71% of total latency)
- Net result: 10-17x query speedup with 99.9% token reduction

### Production Implications

**When to use compression pipeline:**
- ✅ Multiple queries per corpus (>3 queries)
- ✅ Large corpora (>100MB)
- ✅ Latency-sensitive applications (need bounded query time)
- ✅ Cost-sensitive deployments (token reduction critical)

**When monolithic may be acceptable:**
- ⚠️ Single-query workloads (no amortization)
- ⚠️ Small corpora (<10MB)
- ⚠️ Write-intensive workloads (frequent re-indexing)

**Full methodology:** See [experiments/LATENCY_BENCHMARK_RESULTS.md](../../experiments/LATENCY_BENCHMARK_RESULTS.md) for detailed methodology, limitations, and ASCII visualizations.

---

## Implementation Checklist

- [ ] **Data Ingestion**: Semantic chunking + compression at write time
- [x] **Vector DB Prototype**: Chroma + local embedding fallback + metadata indexing
- [x] **MCP Server Prototype**: Hybrid search (vector + lexical) + token budgeting + scored responses
- [ ] **Session Cache**: TTL-based with contradiction detection (optional)
- [ ] **Prompt Template**: System + tools + task anchor structure
- [ ] **Token Accounting**: Per-stage monitoring + budget enforcement
- [ ] **Invalidation Policy**: Default TTL=6h, with manual clear + contradiction override
- [ ] **Testing**: Unit tests for compression quality, retrieval recall, cache hit rate

---

## Next Steps

1. **Production MCP server**: Replace in-process simulation with FastMCP + production vector DB (Chroma, Weaviate, Pinecone).
2. **Semantic cache persistence**: SQLite or Redis for multi-session persistence.
3. **Contradiction detection**: Implement schema diff logic for invalidation.
4. **Cost tracking**: Monitor actual token spend per pipeline, per user, per session.
5. **Quality metrics**: Measure retrieval recall, reasoning confidence, user satisfaction.
