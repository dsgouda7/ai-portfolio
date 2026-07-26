# Improvement Plan — Inference Systems

**Audited:** 2026-07-26 | **Audience fit:** 7/10

## Overall Assessment

Strong architectural foundation: the Riverside scenario grounds every optimization in real stakes, the toy KV-cache implementation is executable and verifiable, and the closing compounding cell quantifies cumulative throughput across all Parts. The notebook's biggest structural issue is that the "bottleneck chain" breaks after Part 2. Only the KV→CB transition has an explicit "this solves X, but now Y is the new bottleneck" bridge. Parts 3–5 open with technical definitions rather than problem statements, turning the notebook into a feature list rather than a progressive constraint-solving story.

---

## Strengths (preserve these)

- **Riverside opening** — specific and vivid: 10 req/min → 1,000 req/min; not vague "scale your system"
- **KV→CB "Missing piece" bridge** — the model all other transitions should replicate
- **KV cache predict-first question** — well-crafted options force commitment; option (c) captures the nuance
- **Closing compounding decision cell** — quantifies cumulative effect of each optimization; reconnects every Part to Riverside target
- **KV cache toy model** — real measured speedup, `torch.equal` identity assertion builds trust
- **Speculative decoding acceptance-rate sweep** — shows quantitative threshold (>70%) for meaningful speedup
- **"Your Turn" draft-length exercise** — well-scoped, teaches the draft-length vs. acceptance-rate tradeoff
- **Tier 1/2/3 scoping taxonomy**
- **PagedAttention level of detail** — OS virtual memory analogy, utilization math, right abstraction level

---

## Gaps & Recommended Changes

### Gap 1 — Missing bottleneck bridges on 3 of 5 section transitions — Priority: High

**Problem:** Only the KV→CB transition has an explicit "this solves X, but now Y is the bottleneck" bridge. CB→PA, PA→SD, and SD→P/D transitions have none.

**Recommendation:** Add a "Missing piece" cell after each section:
- *After CB → before PA:* "Continuous batching worked. But now you're running 100 concurrent requests, each with KV cache reserved at max sequence length. You're hitting OOM at 200 concurrent users — not because of the model size, but because of memory fragmentation. That's what PagedAttention fixes."
- *After PA → before SD:* "PagedAttention solved the OOM. But each individual request is still slow — decode is serial, so 200 tokens at 10ms each is 2s per request. The bottleneck shifted from concurrency to per-request throughput. That's what speculative decoding addresses."
- *After SD → before P/D:* "Speculative decoding improved token throughput. But you're now seeing TTFT spikes specifically on requests with long system prompts, while short-context requests are fast. The system is showing two distinct latency profiles. That's the prefill/decode asymmetry."

---

### Gap 2 — KV cache lacks a familiar analogy before the code — Priority: High

**Problem:** The principle "compute K and V once, store, reuse" has no bridge to a concept the audience already knows. Engineers from a backend background immediately recognize database caching; without that bridge, KV cache lands as "attention math optimization" rather than "oh, it's just caching."

**Recommendation:** Add one sentence before the predict-first question:
> "Think of it like a database query cache: the database stores the result of an expensive query on the first call and skips the query on subsequent ones. KV cache does the same for attention — compute keys and values for each prompt token once during prefill, write them to cache, and on every decode step only compute the one new token's K/V pair."

---

### Gap 3 — Continuous batching code chart shows identical bars for both approaches — Priority: Medium

**Problem:** The matplotlib cell produces two identical bar histograms of the same 20 request lengths, labeled "Static batching (idle slots)" and "Continuous batching (always busy)." A learner sees no difference between them, which actively undermines the mental model.

**Recommendation:** Replace the two bar charts with a Gantt-style plot: x-axis = decode steps, y-axis = batch slots (1–8), color = active-request vs. idle. The simulation already tracks which slots are occupied; render that directly. This makes the code chart consistent with what `continuous-batching-vs-static.png` shows.

---

### Gap 4 — Speculative decoding: "why parallel verification is faster" is absent — Priority: High

**Problem:** "Verify in parallel" is stated but not explained. Without understanding that transformers inherently produce N output logits for N input tokens in one forward pass, "parallel verification" sounds like a trick.

**Recommendation:** Add 3 sentences before the predict-first question:
> "Here's why this works. A transformer is inherently parallel — given N input tokens, it produces N output logits in one forward pass. Autoregressive generation artificially enforces serial order. Speculative decoding breaks that constraint: the draft model proposes K tokens, and the large verifier does what transformers naturally do — processes all K positions simultaneously in one shot. If the draft was right, you get K tokens for the price of one verifier call."

---

### Gap 5 — Prefill vs. decode: no concrete Riverside example before the timing data — Priority: Medium

**Problem:** Part 5 opens with "High arithmetic intensity (large matmuls). Compute-bound." — abstract for an engineer whose on-call alert fires on TTFT spikes for some requests and not others.

**Recommendation:** Add 2 sentences at the top of Part 5:
> "Riverside's editing prompt includes a 500-token system preamble. Processing those 500 tokens before the first output token is generated is prefill — compute-bound, runs once, scales linearly with prompt length. If your TTFT alert fires when users add more context but TPOT stays flat, that's a prefill bottleneck. If TPOT is slow across all requests regardless of prompt length, it's a decode bottleneck."

---

## Do NOT Change

- KV→CB "Missing piece" bridge (model for Gap 1's additions)
- Closing compounding decision cell
- Speculative decoding acceptance-rate sweep and "Your Turn" exercise
- PagedAttention level of detail — OS analogy, stop before block-level kernel implementation
- Tier 1/2/3 scoping taxonomy
- KV cache toy model — `SimpleCausalAttention` + `torch.equal` identity assertion
