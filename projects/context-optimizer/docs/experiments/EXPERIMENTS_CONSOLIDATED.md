# Consolidated Experiment Report: Chat-Assistant Context Architecture

> **Purpose:** Validate Pipe C (MCP Pull) for chat-assistant tasks over long external memory,
> with boundary-preserving semantic retrieval and task-specific indexing.

**Run time:** 2026-06-18 18:03 UTC

---

## Scope

- Focus: chat assistants, not coding agents
- Pattern: compress intent -> retrieve evidence -> reason over bounded context
- Retrieval contract: typed MCP pull with relevance scoring and boundary hints

---

## Corpus Definitions by Experiment Family

| Domain | Corpus | Stored Index Fields | Evaluation Goal |
|---|---|---|---|
| Large Book/Document QA | Project Gutenberg books + open-access technical docs (chapter/section indexed) | doc_id, chapter_id/section_id, heading_path, entities, prev_chunk_id, next_chunk_id | Long-range factual and comparative QA with source-grounded citations |
| Episodic Chat Memory | Prior conversation sessions/transcripts with turn-level metadata | session_id, turn_id, speaker, topic_tags, commitments, unresolved_threads, adjacency links | Continuity, prior-decision recall, and contradiction avoidance |
| Terms/Fine-Print Assistant | Public terms/privacy docs and clause-level policy text | doc_type, clause_id, obligations, exceptions, penalties, jurisdiction, cross_refs | Risk-focused clause QA and plain-language policy interpretation |
| Social Sentiment/Abuse Analytics | Large social-text streams with sentiment/toxicity annotations | community, time_bucket, sentiment, abuse_type, topic_cluster, sample_ids | Trend analysis, moderation risk summaries, and week-over-week changes |

---

## Test Summary

| Test | Domain | Avg Token Reduction | Avg Quality Parity | Notes |
|---|---|---|---|---|
| Book/Document QA — Chapter-Aware | books-docs | 4.5% | 0.83 | [LOW] Marginal savings |
| Episodic Memory QA — Previous Chats | chat-memory | -68.1% | 0.80 | [NEG] Overhead dominates at this corpus size |
| Terms QA — Fine-Print Obligations | terms-fine-print | -23.7% | 0.70 | [NEG] Overhead dominates at this corpus size |
| Social Analytics QA — Sentiment/Abuse | social-analytics | -23.1% | 0.77 | [NEG] Overhead dominates at this corpus size |

---

## Test Results (Detailed)

### Book/Document QA — Chapter-Aware

**Domain:** books-docs

**Question 1:** Where does the narrative first revise earlier assumptions, and how is that resolved later?

| Metric | Monolithic | Pipe C | Delta |
|---|---|---|---|
| Tokens | 247 | 236 | **4.5% saved** |
| Latency | 0.010s | 0.000s | -0.010s |
| Retrieved Lines | — | 5 | — |
| Tool Calls | 0 | 2 | — |

**Quality Scores:**
- Structural: 0.90 (expected: ≥0.5)
- Citations: 0.80 (expected: ≥0.3)
- Specificity: 0.80 (expected: ≥0.3)
- **Overall Parity:** 0.83

**Pipe C Answer (excerpt):**
> Book evidence: revision appears at book:ch35 (written explanation) and final alignment at book:ch61. Paper evidence: paper:s2 reports multi-head O(n^2); paper:s4 reports cross-attention O(n*m); paper:s7 states fusion tas

**Monolithic Answer (excerpt):**
> Assumptions are revised in the explanatory letter chapter and resolved by the final chapter. In the survey, multi-head is O(n^2) and cross-attention is O(n*m), with cross-attention often better for fusion.

**Question 2:** How do multi-head and cross-attention differ for fusion tasks and computational cost?

| Metric | Monolithic | Pipe C | Delta |
|---|---|---|---|
| Tokens | 246 | 235 | **4.5% saved** |
| Latency | 0.010s | 0.000s | -0.010s |
| Retrieved Lines | — | 5 | — |
| Tool Calls | 0 | 2 | — |

**Quality Scores:**
- Structural: 0.90 (expected: ≥0.5)
- Citations: 0.80 (expected: ≥0.3)
- Specificity: 0.80 (expected: ≥0.3)
- **Overall Parity:** 0.83

**Pipe C Answer (excerpt):**
> Book evidence: revision appears at book:ch35 (written explanation) and final alignment at book:ch61. Paper evidence: paper:s2 reports multi-head O(n^2); paper:s4 reports cross-attention O(n*m); paper:s7 states fusion tas

**Monolithic Answer (excerpt):**
> Assumptions are revised in the explanatory letter chapter and resolved by the final chapter. In the survey, multi-head is O(n^2) and cross-attention is O(n*m), with cross-attention often better for fusion.

### Episodic Memory QA — Previous Chats

**Domain:** chat-memory

**Question 1:** What architecture choices did we lock in across prior sessions?

| Metric | Monolithic | Pipe C | Delta |
|---|---|---|---|
| Tokens | 166 | 279 | **-68.1% saved** |
| Latency | 0.010s | 0.000s | -0.010s |
| Retrieved Lines | — | 6 | — |
| Tool Calls | 0 | 2 | — |

**Quality Scores:**
- Structural: 0.80 (expected: ≥0.5)
- Citations: 0.80 (expected: ≥0.3)
- Specificity: 0.80 (expected: ≥0.3)
- **Overall Parity:** 0.80

**Pipe C Answer (excerpt):**
> Locked decisions: use Chroma locally (s01/t03), preserve chunk boundaries with adjacency hints (s01/t08), focus on the Pipe C MCP pull path (s02/t02, s02/t07). Scope update: benchmark chat assistants, not coding agents (

**Monolithic Answer (excerpt):**
> We chose Pipe C and planned additional benchmark domains. There was also a decision around vector retrieval and memory continuity.

**Question 2:** What did we agree to benchmark next and what scope was excluded?

| Metric | Monolithic | Pipe C | Delta |
|---|---|---|---|
| Tokens | 166 | 279 | **-68.1% saved** |
| Latency | 0.010s | 0.000s | -0.010s |
| Retrieved Lines | — | 6 | — |
| Tool Calls | 0 | 2 | — |

**Quality Scores:**
- Structural: 0.80 (expected: ≥0.5)
- Citations: 0.80 (expected: ≥0.3)
- Specificity: 0.80 (expected: ≥0.3)
- **Overall Parity:** 0.80

**Pipe C Answer (excerpt):**
> Locked decisions: use Chroma locally (s01/t03), preserve chunk boundaries with adjacency hints (s01/t08), focus on the Pipe C MCP pull path (s02/t02, s02/t07). Scope update: benchmark chat assistants, not coding agents (

**Monolithic Answer (excerpt):**
> We chose Pipe C and planned additional benchmark domains. There was also a decision around vector retrieval and memory continuity.

### Terms QA — Fine-Print Obligations

**Domain:** terms-fine-print

**Question 1:** Can the service auto-renew and change terms without direct notice?

| Metric | Monolithic | Pipe C | Delta |
|---|---|---|---|
| Tokens | 177 | 219 | **-23.7% saved** |
| Latency | 0.010s | 0.000s | -0.010s |
| Retrieved Lines | — | 4 | — |
| Tool Calls | 0 | 2 | — |

**Quality Scores:**
- Structural: 0.50 (expected: ≥0.5)
- Citations: 0.80 (expected: ≥0.3)
- Specificity: 0.80 (expected: ≥0.3)
- **Overall Parity:** 0.70

**Pipe C Answer (excerpt):**
> Yes. Auto-renew is explicit (tos:3.2) and material changes may be posted without direct email notice (tos:9.2). Dispute risk: binding arbitration with class-action waiver (tos:5.1). Termination risk: immediate suspension

**Monolithic Answer (excerpt):**
> There are auto-renew and arbitration clauses. The provider can suspend accounts for abuse. Terms updates may appear in product.

**Question 2:** What dispute and account-termination risks should a user know first?

| Metric | Monolithic | Pipe C | Delta |
|---|---|---|---|
| Tokens | 178 | 220 | **-23.6% saved** |
| Latency | 0.010s | 0.000s | -0.010s |
| Retrieved Lines | — | 4 | — |
| Tool Calls | 0 | 2 | — |

**Quality Scores:**
- Structural: 0.50 (expected: ≥0.5)
- Citations: 0.80 (expected: ≥0.3)
- Specificity: 0.80 (expected: ≥0.3)
- **Overall Parity:** 0.70

**Pipe C Answer (excerpt):**
> Yes. Auto-renew is explicit (tos:3.2) and material changes may be posted without direct email notice (tos:9.2). Dispute risk: binding arbitration with class-action waiver (tos:5.1). Termination risk: immediate suspension

**Monolithic Answer (excerpt):**
> There are auto-renew and arbitration clauses. The provider can suspend accounts for abuse. Terms updates may appear in product.

### Social Analytics QA — Sentiment/Abuse

**Domain:** social-analytics

**Question 1:** Which abuse categories increased week-over-week in tech community posts?

| Metric | Monolithic | Pipe C | Delta |
|---|---|---|---|
| Tokens | 193 | 238 | **-23.3% saved** |
| Latency | 0.010s | 0.000s | -0.010s |
| Retrieved Lines | — | 4 | — |
| Tool Calls | 0 | 2 | — |

**Quality Scores:**
- Structural: 0.70 (expected: ≥0.5)
- Citations: 0.80 (expected: ≥0.3)
- Specificity: 0.80 (expected: ≥0.3)
- **Overall Parity:** 0.77

**Pipe C Answer (excerpt):**
> Week-over-week in tech: threat incidents appeared in 2026-W21 (new), insults persisted, harassment remained present. Risk summary for latest week: high negativity concentration in tech with insult+threat patterns; priori

**Monolithic Answer (excerpt):**
> Tech has more negative comments with insults and harassment. Latest week still shows harmful language and elevated moderation risk.

**Question 2:** What moderation risk summary should we produce for the latest week?

| Metric | Monolithic | Pipe C | Delta |
|---|---|---|---|
| Tokens | 192 | 236 | **-22.9% saved** |
| Latency | 0.010s | 0.000s | -0.010s |
| Retrieved Lines | — | 4 | — |
| Tool Calls | 0 | 2 | — |

**Quality Scores:**
- Structural: 0.70 (expected: ≥0.5)
- Citations: 0.80 (expected: ≥0.3)
- Specificity: 0.80 (expected: ≥0.3)
- **Overall Parity:** 0.77

**Pipe C Answer (excerpt):**
> Week-over-week in tech: threat incidents appeared in 2026-W21 (new), insults persisted, harassment remained present. Risk summary for latest week: high negativity concentration in tech with insult+threat patterns; priori

**Monolithic Answer (excerpt):**
> Tech has more negative comments with insults and harassment. Latest week still shows harmful language and elevated moderation risk.

---

## Key Findings

### 1. Architecture Behavior

- Average token reduction across all suites: **-27.6%**
- Average quality parity across all suites: **0.78**
- Pipe C quality remains strong when evidence retrieval is selective and structured.
- Overhead appears on small corpora where retrieval shell and index context dominate prompt budget.

### 2. What Improves Results

- Boundary-preserving chunks reduce local truncation errors and missing-evidence claims.
- Task-specific metadata (chapter/turn/clause/time bucket) improves retrieval precision.
- Tool-aware prompting helps the reasoner refine queries instead of guessing from weak context.

### 3. Residual Risks

- Small contexts can produce negative savings due to fixed shell/tool overhead.
- Quality still depends on embedding fidelity and index freshness.
- Multi-hop answers require explicit retrieval refinement loops to avoid shallow synthesis.

### 4. Production Guidance

- Use Pipe C for large corpora where selective retrieval removes most irrelevant context.
- Maintain domain-specific indexes: chapter/section, session/turn, clause/risk, time/community.
- Keep boundary metadata and neighbor links in storage and retrieval outputs.
- Add cache invalidation by TTL + update events before production deployment.

---

## Next Steps

1. Add real corpora ingestion for each suite (Gutenberg/docs, chat transcripts, terms policies, social datasets).
2. Run N>=3 trials per question and report confidence intervals.
3. Evaluate retrieval recall/precision and citation correctness separately from answer quality.
4. Add event-driven cache invalidation and stale-index detection.
5. Extend with multi-hop tool-call tests where one retrieval result triggers a follow-up query.

---

## Hypothesis Validation

**H4 (chat-assistant scope):** Pipe C can maintain answer quality while reducing prompt size
for large memory-retrieval tasks when context selection is high and indexing is task-aware.

**Evidence:**
- Average token reduction: -27.6% across all suites
- Average quality parity: 0.78 across all suites
- Domains covered: 4 assistant-focused families

**Conclusion:** Partially supported in this mock run. Pipe C is strongest when corpus scale and
retrieval selectivity are high; additional large-corpus runs are required for final confirmation.

---

<!-- INCIDENT_APPENDIX_START -->
## Incident Benchmark Appendix (Auto-Generated)

> **Important:** Results produced in mock mode are illustrative estimates.
> Structural quality scores reflect the design intent of each pipeline.
> Run with `--provider ollama` for real LLM inference and LLM-as-judge scoring.

---

## Run Metadata

| Key | Value |
|---|---|
| Run time (UTC) | 2026-06-18 18:03 UTC |
| Provider | `mock` |
| Small model (compression) | `mock-paraphraser` |
| Reasoning model | `mock-reasoner` |
| Log corpus size | 1,050 lines |
| Incident prompt tokens (est.) | 333 |
| Full corpus tokens (est.) | 38,249 |

---

## Input: Same Incident Prompt (All Pipes)

```
Hey team, sorry this is a bit all over the place because I have been on this for hours and this has
turned into a full-on fire. Since around 02:13 UTC the checkout flow has been intermittently timing out
and support is flooded. We run on AKS, ingress-nginx in front, api-gateway then order-service and
payment-service. Clients report 504 then sometimes 499. Prometheus shows p95 latency climbed from 220ms
to 8.7s, error_rate is at 17.6%, and CPU on a few pods is normal which is weird. CosmosDB dependency
calls in application insights look bad. I keep seeing timeout error code 21012 and a lot of retries.

We had a deployment at 01:55 UTC but only for recommendation-service so I do not think it should impact
checkout, but maybe noisy neighbors? Also there was a weird spike in ingress warnings around
"upstream timed out while reading response header" on aks-prod-eastus nodepool np-user-03. One trace
references 10.42.7.19 and another mentions 10.42.8.44. I also saw a stack trace from order-service:
System.TimeoutException at CosmosClient.ReadItemAsync, then downstream call cancellation in
PaymentConnector.SubmitAsync.

I am honestly not sure if this is network, CosmosDB RU starvation, bad retry policy, or something in
ingress connection handling. Can you help figure out what is likely happening and what to check first?
```

---

## 1. Token Efficiency Comparison

| Pipeline | Prompt Tokens Sent | Tool Calls | Retrieved Lines | Total Latency (s) | vs Pipe A |
|---|---|---|---|---|---|
| Pipe A — Monolithic (baseline) | **38,582** | 0 | 1050 | 0.000 | — |
| Pipe OOTB — Standard LangChain RAG | **4,012** | 0 | 104 | 0.010 | 10.4% |
| Pipe C — MCP Pull (structured shell) | **4,344** | 3 | 173 | 0.052 | 11.3% |

> Token counts are estimated at 4 chars/token. Real counts will vary by model tokeniser.

---

## 2. Quality Evaluation

### Structural Quality Scores (keyword / heuristic, no LLM)

| Pipeline | Keyword Coverage | Specificity | Root Cause Found | Service Coverage | Mitigations | Next Steps | Overall |
|---|---|---|---|---|---|---|---|
| Pipe A — Monolithic (baseline) | 25% | 0% | ❌ | 0% | ✅ | ❌ | **0.175** |
| Pipe OOTB — Standard LangChain RAG | 50% | 10% | ✅ | 0% | ✅ | ✅ | **0.520** |
| Pipe C — MCP Pull (structured shell) | 88% | 80% | ✅ | 25% | ✅ | ✅ | **0.810** |

_LLM judge errors (check Ollama availability):_
- Pipe A — Monolithic (baseline): `Skipped in mock mode — run with --provider ollama for LLM judge scores`
- Pipe OOTB — Standard LangChain RAG: `Skipped in mock mode — run with --provider ollama for LLM judge scores`
- Pipe C — MCP Pull (structured shell): `Skipped in mock mode — run with --provider ollama for LLM judge scores`


---

## 3. Per-Pipeline Answers

### Pipe A — Monolithic (baseline)

```
There appears to be some issue with the checkout flow. The logs show multiple errors. Database connections look bad. There are timeout errors in the system. The latency has increased significantly. The deployment at 01:55 may be related. You should check your database and potentially restart some services. Error rate is high. Consider rolling back.
```

**Token budget:** 38,582 tokens  |  **Latency:** 0.000s  |  **Tool calls:** 0

### Pipe OOTB — Standard LangChain RAG

```
Based on the retrieved logs, the system is experiencing timeouts. The CosmosDB connection seems to be having issues based on the error patterns in the logs. The ingress controller is also reporting upstream timeouts. The checkout service is returning 504 errors. You should investigate the CosmosDB configuration and possibly increase the timeout limits. The retry policy may also be contributing to the problem.
```

**Token budget:** 4,012 tokens  |  **Latency:** 0.010s  |  **Tool calls:** 0

### Pipe C — MCP Pull (structured shell)

```
Root cause (high confidence): CosmosDB RU saturation causing substatus=21012 timeouts. The cascade path is: CosmosClient.ReadItemAsync timeout → order-service failure → PaymentConnector.SubmitAsync cancellation. Ingress logs confirm 'upstream timed out while reading response header' on pods 10.42.7.19 and 10.42.8.44 (nodepool np-user-03).

Supporting evidence from targeted log retrieval:
- 21012 substatus appears on CosmosDB read path (partition=tenant-445, ru_charge=128.44)
- p95 climbed from 220ms to 8.7s; error_rate=17.6%
- Stack trace: System.TimeoutException at ReadItemAsync:line 214 → SubmitAsync:line 87

Immediate mitigations:
1. Increase CosmosDB provisioned RU or enable autoscale
2. Add jitter + exponential backoff to retry policy (current retries amplify RU pressure)
3. Set ingress proxy_read_timeout > 8s for /v1/checkout upstream
4. Isolate order-service CosmosDB calls behind circuit breaker

Next observability checks:
- RU consumption graph per partition (last 2h)
- Retry burst metrics on order-service
- CosmosDB connection pool saturation
- Ingress upstream queue depth on np-user-03
```

**Token budget:** 4,344 tokens  |  **Latency:** 0.052s  |  **Tool calls:** 3

_Extra telemetry: {"mcp_tokens_consumed": 3468, "shell_tokens": 876, "total_context_tokens": 4344}_

---

## 4. Key Observations

### Token Efficiency

- **Pipe A** sends the full log corpus to the reasoning model: O(corpus size) tokens.
- **Pipe OOTB** uses TF-IDF retrieval to reduce token load, but without compression the
  query quality depends on raw user phrasing and retrieval may miss precise error codes.
- **Pipe C** maintains a fixed structured shell (~1.7k token contract) and pulls context
  only when the reasoning model requests it via typed MCP tool calls.

### Quality

- In mock mode, quality differences reflect deliberate calibration of canned responses
  to illustrate what each architecture would likely produce in practice.
- In Ollama mode, structural and LLM-judge scores reflect actual model outputs.
- Pipe C is expected to produce more specific answers because the retrieval queries
  are derived from the compressed identifiers, targeting precise error codes rather
  than the raw user prose.

### Trade-off Summary

| Pipe | Token cost | Quality potential | Latency overhead | Complexity |
|---|---|---|---|---|
| A  | Highest (O corpus) | Baseline | Lowest | Lowest |
| OOTB | Medium (top-k chunks) | Similar to A | Low | Low |
| C  | Bounded (shell + MCP) | Best (typed pull) | +1 compress + N MCP RTTs | Highest |

---

## 5. Open Problems and Next Steps

1. **Cache invalidation strategy** — session-persisted semantic cache has no TTL policy yet.
2. **Real embedding quality test** — compare retrieval recall on compressed-before-index
   vs raw-indexed corpus.
3. **MCP server process** — replace in-process simulation with a real FastMCP server.
4. **Domain transfer** — run same pipelines on chat-assistant datasets (books/docs, episodic memory, terms, social analytics).
5. **LLM judge calibration** — run human eval alongside LLM judge to validate correlation.

_Generated: 2026-06-18 18:03 UTC | Provider: mock_
<!-- INCIDENT_APPENDIX_END -->
