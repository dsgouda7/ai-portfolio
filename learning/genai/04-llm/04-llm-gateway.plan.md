# Plan: `04-llm-gateway.ipynb` → authoring-guide parity

## 1 · Current state

`04-llm-gateway.ipynb` (74 cells) teaches LLM gateway patterns — unified provider abstraction,
routing strategies (round-robin, least-busy, latency-aware EMA, cost-aware, weighted), rate
limiting (token bucket, sliding window), fallback chains, response caching, and a final
production-style `Gateway` class — bridged at the end to a real library (LiteLLM).

It already implements almost every gold-standard pattern from Sections 1–9 of the authoring
guide, at a level of polish comparable to `01-llm-finetuning.ipynb`:

- **Narrative framing (Section 8):** threaded throughout via "Riverside House," the same
  publishing-firm scenario from the fine-tuning notebook, now needing to combine its own
  in-house model with hosted providers. Every Part opens with a one-line "Riverside's question
  for this section" callout.
- **Roadmap table** in the title cell, restated as a completed table in "What This Notebook
  Covered," followed by a "Key Insights to Keep" bullet list — matches Section 2/5.7.
- **Predict-first cells** before non-obvious reveals (routing lock-on, fallback chain math,
  cache hit rate, gateway batch behavior).
- **Code Walkthrough cells** (Section 9.1) after every dense code cell.
- **Common Pitfalls + runnable Quick Health Check cells** (Section 8.5) for rate limiting,
  fallback, and caching — each demonstrates a real failure mode with executed code, not just a
  warning.
- **A quantitative combination grid** (Section 9.3/10.3) already exists for "Rate Limiter
  Config × Traffic Pattern" — qualitative markdown table, dual heatmap, printed observations.
- **A `FuncAnimation`** for the token bucket (Section 9.4), with a static legend before the loop.
- **An honest, real-numbers-only Scorecard + closing Decision** (Section 8.2/8.7) — every number
  is pulled from variables computed earlier in the same run, and the recap ("What This Notebook
  Covered") is correctly placed *before* the closing decision, not after.
- **Simulated providers with an explicit try/except-free, no-network-call design**, and the
  LiteLLM bridge at the end wraps any real API usage in `if api_key:` / `try/except` so the
  notebook runs end-to-end with zero keys and zero real network calls.

## 2 · Checklist scoring (Sections 6, 8.8, 9.5, 10.8)

| Item | Status |
|---|---|
| Roadmap table, single running example (Riverside's fleet) | ✅ |
| Claims measured, not asserted | ✅ (scorecard, health checks) |
| 🔮 Predict-first before non-obvious reveals | ✅ |
| 🧪 Your-turn exercises near the concept | ✅ |
| Reflection cells ("What Just Happened") planting next question | ✅ |
| Toy→real bridge | ✅ (LiteLLM mapping table) |
| Math always glossed (EMA formula, multiplicative reliability) | ✅ |
| Code Walkthrough after dense cells | ✅ |
| Named scenario + constraint shaping every section | ✅ |
| Real numbers from in-notebook objects only | ✅ |
| Honest mixed/failure results stated plainly | ✅ (fallback quality-risk finding) |
| Pitfalls → runnable health check | ✅ |
| Ends on decision, recap placed before it | ✅ |
| **Table of Contents** (>40 cells) | ❌ **missing** |
| **Per-subplot legends**, esp. list-comprehension-colored bars | ⚠️ **partial** — several multi-color bar charts have no legend even though line/scatter charts consistently do |
| **Qualitative pros/cons matrix** for the routing strategies axis | ❌ **missing** — Part 3 has a "Strategy / Idea / Optimizes For" table but no Pros/Cons |
| Cross-reference hygiene ("cell below/above") | ✅ mostly — only stable adjacent references, no stale distance claims |
| **Structural bug: duplicate/misplaced section header** | ❌ **found** — see below |
| Progressive disclosure (30+ line multi-concept cells split) | ⚠️ not done, but every long cell already has a Code Walkthrough (Section 9.1's alternative treatment); judged lower-value to split further in this pass |

### Structural bug found

A markdown cell reading `## Part 4 — Rate Limiting & Throttling` (short version, no algorithm
comparison table) is misplaced **immediately after** the "Your Turn: Tune the EMA Smoothing
Factor" code cell — i.e. *before* the cost-aware/weighted-routing code cell, which is still
Part 3 content. A second, fuller `## Part 4 — Rate Limiting & Throttling` cell (with the
token-bucket-vs-sliding-window comparison table) correctly appears later, right before the
actual `TokenBucket`/`SlidingWindowLimiter` code. This is duplicate content left over from an
earlier edit, and its position actively breaks navigation (a reader sees "Part 4" twice, the
first time out of order).

It also happens to sit in the exact spot where the roadmap promises "Cost-Aware & Weighted
Routing" gets its own strategy write-up (matching the existing "### Strategy 1 —" / "### Strategy
2 —" pattern for round-robin/least-busy and latency-aware) — but no such intro markdown cell
exists before the cost-aware/weighted-routing code. **Fix: replace the misplaced cell's content**
with the missing "### Strategy 3 & 4 — Cost-Aware and Weighted Routing" intro, which both removes
the duplicate/out-of-place header and fills the real content gap in one edit.

## 3 · Ordered changes for this pass

1. **Fix the structural bug** — replace the misplaced early "Part 4" markdown cell with a proper
   "### Strategy 3 & 4 — Cost-Aware and Weighted Routing" intro cell (Riverside's question,
   what the two strategies trade off), matching the Strategy 1/2 pattern already established.
2. **Add a Table of Contents** cell (Section 10.1) right after the title cell — numbered list of
   every `##` Part plus the handful of `###` subsections worth a direct jump (Common Pitfalls,
   Strategy N intros, Popular Gateways/Production Checklist), GitHub-slug anchors, with the
   `Ctrl+F`/outline-panel fallback caveat line.
3. **Add a qualitative Pros/Cons table** for the five routing strategies (Section 10.3), in Part
   3 right after the existing "Strategy / Idea / Optimizes For" table and before "Strategy 1"
   begins — forward-pointer to the Scorecard, which later gives real measured numbers for some
   of these strategies (congestion reduction, cost savings).
4. **Add missing legends** (Section 10.2) to the clearest violations — multi-color categorical
   bar charts built from a color list/dict with zero legend at all (the exact "easiest to
   accidentally ship without one" case the guide calls out):
   - Round-robin vs. least-busy bar chart (2 single-series panels, no legend)
   - Premium-vs-cheapest cost bar chart (2-color, no legend) — appears twice
   - Fallback outcome bar chart (4-color via dict comprehension, no legend) — appears twice
   - Gateway Dashboard "Requests Served" and "Cache Hits vs. Misses" panels (no legend)
   Left untouched: charts that already have a legend/colorbar (latency scatter, weighted
   routing, rate-limiter dual heatmap, token-bucket animation, cumulative-cost lines) — adding a
   redundant one-entry legend to every remaining single-series subplot was judged low value
   relative to effort for this pass (see Section 5, deferred items).
5. **Cross-reference hygiene** — re-verify all four "cell above/below"/"chart below" hits after
   the edits above land, since inserting the TOC and editing the Strategy-3/4 cell shifts
   absolute cell positions (though not the *relative adjacency* any of the four hits depend on).

## 4 · Deliberately out of scope for this pass

- **Splitting the long `MockLLMProvider`/`call_with_fallback`/`Gateway` cells** into
  progressive-disclosure pieces (Section 10.6). Each is already followed by a Code Walkthrough
  cell (Section 9.1), which the guide treats as an acceptable alternative treatment, and each is
  a single cohesive concept (one class, one function) rather than several separately-nameable
  steps chained together — the clearest case for 10.6 splitting (data prep → tokenize → train)
  doesn't really apply to this notebook's code shape.
- **Exhaustive one-entry legends on every remaining single-series subplot.** The guide asks for
  this "for consistency," but with ~15+ such subplots in this notebook, doing all of them was
  judged to be mechanical, low-insight churn relative to the explicit instruction not to chase
  100% literal parity in one pass. The highest-value subset (multi-color, zero-legend bar
  charts) is fixed instead.
- **litellm as a hard dependency.** The existing `if api_key: ... import litellm ... except
  Exception` pattern is already correctly defensive and is left untouched — no install attempted.
