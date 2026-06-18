# Tri-Stage Context Decomposition for Token-Efficient LLM Systems

## A Proposed Architecture for Text, Chat, Audio, Video, and Multimodal Reasoning

**Status:** Concept and design proposal (not an empirical claim)

**Authors:** Context Optimizer Project

**Document policy:** This whitepaper and [../design/TECHNICAL_DESIGN.md](../design/TECHNICAL_DESIGN.md) are the only maintained design documents in the project.

---

## Abstract

Large language model (LLM) systems are often built with monolithic prompting, where raw user input and large context corpora are concatenated into a single reasoning prompt. This strategy can cause token growth, degraded relevance, and brittle runtime behavior as input history and data sources scale. We propose a tri-stage architecture that decomposes inference into: (i) low-cost compression, (ii) targeted evidence retrieval, and (iii) final reasoning. The architecture is modality-agnostic and is intended to transfer beyond incident triage to generic chat, long audio, long video, and multimodal conversational systems.

This document deliberately presents a **hypothesis-driven design** rather than validated benchmark outcomes. Earlier internal numbers are treated as provisional and are not used as evidence claims. We focus on system structure, complexity arguments, extension strategies, and an evaluation protocol for future validation.

---

## Keywords

LLM systems, context engineering, token efficiency, staged inference, multimodal retrieval, low-cost model routing, recurrent compression, architecture proposal

---

## 1. Introduction

Most production chat systems still approximate reasoning quality by providing more context. This creates a scaling pattern where prompt size and cost increase with conversation length and corpus size. We propose reframing context construction as a decomposition problem:

1. **What is the smallest structured representation of the user goal?**
2. **What evidence is required to answer that goal?**
3. **Which model should perform final reasoning over the selected evidence?**

The guiding principle is to use inexpensive intermediate computation to constrain expensive reasoning.

---

## 2. Problem Formulation

Let $x$ denote user input (or a multimodal event stream), and let $C$ denote a large external context store (memory, documents, transcripts, scene features, tool outputs).

Naive monolithic prompting constructs:

$$
P_{mono} = f(x, C)
$$

with reasoning token budget approximately increasing with $|C|$.

We propose staged decomposition:

$$
z = g_{cheap}(x), \quad E = r(z, C), \quad y = h_{reason}(z, E)
$$

where:

- $g_{cheap}$: low-cost compressor (small LLM, RNN, compact encoder)
- $r$: targeted retriever/router
- $h_{reason}$: high-capability reasoning model

Desired property (to be tested):

$$
\text{tokens}(h_{reason}) \approx O(1) \text{ w.r.t. } |C|
$$

under fixed retrieval budgets and stable compression quality.

---

## 3. Proposed Architecture

### 3.1 Stage A: Low-Cost Compression

Input is transformed into a constrained intermediate representation:

- intent and task type
- hard constraints
- entities and identifiers
- evidence requirements
- uncertainty/confidence markers

For generic chat this can be a schema such as:

- `intent`
- `constraints`
- `entities`
- `preferred_style`
- `required_evidence_types`
- `missing_information`

For long media, compression may be hierarchical:

- audio: chunk-level semantic states aggregated by recurrent summarizers
- video: frame/scene summaries plus temporal state memory
- multimodal: cross-modal latent alignment before retrieval

### 3.2 Stage B: Targeted Evidence Retrieval and Routing

The compressed representation drives retrieval from only the required sources:

- chat memory store
- product knowledge base
- tool/API outputs
- transcript segments
- scene/event indexes

Budgeted retrieval constraints can include:

- max contexts per source
- freshness weighting
- source trust tiers
- diversity penalty to reduce duplicates

### 3.3 Stage C: Final Reasoning

The reasoning model consumes only:

- compressed latent brief
- selected evidence pack
- explicit citation/grounding instructions

This stage can be replaced or upgraded independently of Stage A and Stage B.

### 3.4 Refined Variant: MCP-Based Pull Architecture

The tri-stage design above is a *push* model — context is assembled before the reasoning call. A more robust variant inverts this into a *pull* model where the reasoning model retrieves context on demand during inference via a typed MCP (Model Context Protocol) server interface.

**Structured shell (~1.7k tokens, fixed cost):**

The prompt presented to the reasoning model is a stable contract:

```
system:   persona | instructions | safety constraints
tools:    retrieve_context(query, depth, service, severity)
          write_to_cache(content)
          get_session_memory(session_id)
task:     <compressed task anchor>   ← last line, always present
```

The 1.7k budget is not a data budget; it is an interface budget. No raw context is ever injected here.

**MCP server with typed tool contracts:**

Tool signatures are co-designed with the reasoning model's calling conventions. The model can specify not just *what* it needs but *how much* and *under what constraints*, e.g., `retrieve_context(query="CosmosDB timeout 21012", depth="brief", service="order-service")`. This makes every retrieval call a typed, logged, auditable event rather than an opaque context injection.

**Pre-compressed semantic vector store:**

Context is compressed by a cheap LLM or summarizer at *write time*, chunked semantically, embedded across the full token span of each chunk, and then indexed. At retrieval time the MCP server returns already-distilled chunks ranked by semantic similarity plus lexical support. The reasoning model never touches the full raw corpus.

**Boundary-preserving compression:**

Compression should occur *inside* stable source boundaries rather than across them. Each stored chunk retains its original span metadata plus continuation hints such as `needs_prev_chunk` and `needs_next_chunk`. This allows the reasoning model to detect when a retrieved chunk is locally incomplete and request adjacent corroboration before asserting causality.

**Tool-aware reasoning policy:**

The reasoning model is explicitly taught how the retrieval surface behaves. Broad symptom queries are appropriate when the root cause is unknown; identifiers returned in high-scoring chunks should seed narrower follow-up queries. Relevance scores are evidence-prioritisation hints rather than proofs of causality and should be corroborated before final conclusions.

**Feedback-driven re-compression:**

If accumulated tool-response tokens across a turn exceed a configurable threshold, the content is routed back through the compression pipeline rather than truncated. This produces a structurally consistent re-compressed representation, not an arbitrary trim. This is deliberately different from compaction strategies used in most agentic frameworks, which truncate without schema guarantees.

**Session-persisted semantic cache:**

The semantic cache is compressed before persistence and loaded per session on restore. This acts as a personal episodic memory layer — cheaper than full conversation replay, more stable than raw embeddings, and bounded in storage cost by the compression pass. Invalidation policy is discussed in Section 7.

**Pass-through policy for small seeds:**

If the compressed task anchor is already within token budget with no retrieval required, the MCP call is skipped and the payload is sent direct. The MCP path is an optimisation, not a mandatory stage.

---

## 4. Scientific Positioning and Claims Discipline

This manuscript uses **proposal language only**:

- We do **not** claim validated latency or token savings here.
- We do **not** treat prior exploratory metrics as publishable evidence.
- We propose testable hypotheses and a reproducible evaluation plan.

### Proposed hypotheses

1. **H1:** staged decomposition reduces reasoning-token variance across large corpus growth.
2. **H2:** staged decomposition improves grounding precision versus monolithic prompting at equal reasoning-token budget.
3. **H3:** the same decomposition transfers across text, audio, video, and multimodal chat with modality-specific compressors.
4. **H4:** a bounded context window with MCP-mediated on-demand retrieval can achieve answer quality comparable to a full-context heavy-duty LLM call, with the trade-off of additional per-turn latency from retrieval round trips. This latency cost is hypothesised to be economically preferable to the token cost of monolithic prompting at scale.

---

## 5. Transfer to Generic Chat and Multimodal Systems

### 5.1 Generic Chat Applications

Mapping from the current design to chat:

- compression: user turn + recent state -> structured intent graph
- retrieval: memory/doc/tool selection from intent graph
- reasoning: bounded response generation with citations

This reduces dependence on full conversation replay and supports cost-predictable multi-turn operation.

### 5.2 Long Audio

Proposed stack:

1. ASR stream -> token chunks
2. low-cost recurrent semantic compressor (RNN/state-space/small transformer)
3. segment retrieval from transcript and metadata index
4. final reasoning LLM over selected spans

### 5.3 Long Video

Proposed stack:

1. frame/clip encoder + scene boundary detection
2. temporal compressor producing event timeline tokens
3. retrieval over scene graph + transcript + OCR cues
4. final reasoning LLM over selected event windows

### 5.4 Multimodal Chat

Use cross-modal compression to construct a shared latent brief (text, audio, video references), then retrieve only linked evidence and route to a final reasoning model.

---

## 6. Complexity and Scaling Intuition

Assume per-turn limits:

- compression output size bounded by $k_z$
- retrieval budget bounded by $k_r$ items and max chunk size $k_c$

Then reasoning input is bounded by:

$$
T_{reason} \leq k_z + k_r \cdot k_c
$$

which is independent of raw corpus size if budgets are enforced.

Practical caveat: if compression quality degrades, retrieval recall may collapse, creating downstream quality failure.

---

## 7. Risk and Failure Modes

1. **Over-compression risk:** critical details removed.
2. **Routing error risk:** wrong sources queried.
3. **Stale evidence risk:** retrieved data is outdated.
4. **Cross-modal alignment risk:** text/audio/video entities are mismatched.
5. **Confidence miscalibration:** system answers instead of requesting clarification.
6. **Session cache invalidation (open problem):** the persisted semantic cache requires an invalidation strategy. Three candidate approaches are: time-to-live (TTL), user-triggered explicit clear, and contradiction-detection triggered by a re-compression pass that identifies conflicts between new input and cached content. The right policy depends on domain and staleness tolerance. **This design does not yet specify a default invalidation strategy and treats it as a required open problem before production deployment.**

Mitigations (items 1–5):

- schema validation and required-field guards
- retrieval diagnostics with source-level traces
- freshness and trust weighting
- abstain/clarify policy when confidence is low
- adversarial test suites for modality edge cases

---

## 8. Proposed Evaluation Protocol (Future Work)

### 8.1 Benchmarks

- generic chat transcripts
- long-form meeting audio
- long-form instructional video
- multimodal assistant tasks

### 8.2 Metrics

- reasoning tokens/turn
- end-to-end latency and p95
- grounding/citation precision
- answer acceptance rate
- hallucination incidence
- cost per resolved task

### 8.3 Experimental Design

**Core hypothesis under test (H4):**
Can a bounded context window with MCP-mediated retrieval achieve answer quality comparable to a full-context monolithic LLM call, with the trade-off of additional per-turn retrieval latency? The claim is that this latency cost is economically preferable at scale.

Reference implementation for experiments: `context_optimizer_benchmark.py` and `scalability_test.py` in this repository provide the baseline harness. Adapting those scripts to the MCP pull variant is the next experimental step.

**Baseline comparison:**
- Pipe A: monolithic prompt, full context, no retrieval
- Pipe C: structured 1.7k shell + MCP pull retrieval, pre-compressed semantic store

**Controlled variables:** model family, temperature, corpus, task set

**Primary outcome metric:** answer quality delta (human eval or automated scoring) per unit of reasoning-token spend

**Ablations:**
- remove MCP retrieval (pass-through only, fixed 1.7k)
- remove pre-compression at write time (raw indexed)
- vary MCP call budget cap (1, 2, 4, unlimited calls per turn)
- swap compressor class (small LLM vs RNN vs extractive summarizer)

---

## 9. Patentability Assessment (Preliminary, Non-Legal)

**Short answer:** potentially patent-eligible **if** the claims are drafted around a concrete technical method and demonstrable system effects, not a high-level AI idea.

### Why it might qualify

- specific staged control architecture (compression-routing-reasoning)
- modality-agnostic extension with explicit intermediate representations
- budget-constrained retrieval orchestration and failure gating
- measurable systems objectives (token boundedness, runtime predictability)

### Why it might fail

- broad claims may be considered obvious combinations of summarization + retrieval
- prior art in RAG, query rewriting, and hierarchical summarization may overlap
- lack of validated technical effect can weaken prosecution

### Stronger patent strategy

1. Claim a concrete control loop with schema constraints and confidence gating.
2. Claim modality-transfer mechanism (text/audio/video/multimodal) via shared intermediate form.
3. Claim retrieval-budget enforcement and fallback semantics as system behavior.
4. Support with ablation evidence showing non-trivial technical improvement.

**Important:** this is not legal advice. A patent attorney should run a prior-art search and draft jurisdiction-specific claims.

---

## 10. Conclusion

This whitepaper proposes a disciplined context-engineering architecture that decomposes inference into compression, retrieval, and reasoning stages. The design is intended to scale across generic chat and long multimodal inputs by enforcing bounded reasoning context through low-cost intermediate models. The contribution at this stage is architectural: a testable systems hypothesis with clear transfer paths and an evaluation framework, rather than a final empirical claim.

---

## Appendix A: Implementation Roadmap

**Complementary document**: [../design/TECHNICAL_DESIGN.md](../design/TECHNICAL_DESIGN.md) provides the tactical design for production implementation, including:

- **Optimal prompt structure** for reasoning LLMs (system → tools → compressed anchor → task)
- **Data ingestion pipeline**: semantic chunking → compression at write-time → indexing
- **MCP server schema**: hybrid retrieval (vector + BM25) + token budgeting + trust scoring
- **Semantic cache design**: TTL-based invalidation with optional contradiction detection
- **Token accounting**: per-stage monitoring and budget enforcement
- **Concrete patterns**: integration flow, code examples, implementation checklist

This whitepaper is the *what and why*; the technical design is the *how*.

---

## Figures

- Figure 1: [whitepaper/figures/figure1_system_overview.png](figures/figure1_system_overview.png)
- Figure 2: [whitepaper/figures/figure2_token_scaling_hypothesis.png](figures/figure2_token_scaling_hypothesis.png)
- Figure 3: [whitepaper/figures/figure3_modality_transfer_map.png](figures/figure3_modality_transfer_map.png)
- Figure 4: [whitepaper/figures/figure4_mcp_pull_architecture.png](figures/figure4_mcp_pull_architecture.png)

To regenerate:

```bash
python docs/whitepaper/scripts/generate_whitepaper_figures.py
```
