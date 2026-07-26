# Improvement Plan — Quantization in Depth

**Audited:** 2026-07-26 | **Audience fit:** 7/10

## Overall Assessment

Strong narrative spine: `MACBOOK_VRAM_GB` is a live constant referenced in code comparisons, the intro table maps each Part to a named Riverside question, and the closing decision is concrete and opinionated ("RECOMMENDATION: GGUF Q4_K_M via llama.cpp. No Python environment needed on author MacBooks. ~30 tokens/second on Apple M3"). The main weakness is intuition sequencing: key insights arrive after the formula. Part 6 (NF4/QLoRA) is a training-context digression sitting as the penultimate Part before the closing decision, breaking the deployment narrative at its most critical moment.

---

## Strengths (preserve these)

- **`MACBOOK_VRAM_GB` as a live constant** — constraint is empirical in code comparisons, not decorative
- **Intro table** — maps each Part to a named Riverside question; converts 7 sub-topics into one decision journey
- **"🔮 Predict first" in Part 2** — committing before seeing the perplexity result; well-crafted three-option prompt
- **Model-size table in Part 4 with "Fits 16GB?" column** — scannable in 30 seconds; engineer can immediately locate their model
- **Closing decision** — concrete, opinionated, names the recommended format, speed, and memory footprint
- **Tier 1/2/3 scope management**
- **"When to Use What" decision matrix** — keyed on constraint, not on method

---

## Gaps & Recommended Changes

### Gap 1 — Rounding error: formula leads, intuition follows — Priority: High

**Problem:** Part 1 opens with the LaTeX quantization formula before any "here's how to think about it" framing. The `images/quantization-rounding-error.png` diagram is in a separate cell *after* the formula.

**Recommendation:** Add a 3-sentence visceral hook *before* the formula:
> "Your weight tensor contains millions of floats — 0.01847, −0.03221, 0.00491. int8 offers exactly 256 possible values to represent all of them. You are about to snap every float to its nearest slot. The gap between where a float lives and its nearest slot is the rounding error — everything downstream (perplexity, editing quality) traces back to how bad those gaps are across billions of weights."
Then move the image above the formula so the reader sees the visualization before the algebra.

---

### Gap 2 — GPTQ's Hessian: mechanism present, sensitivity frame absent — Priority: High

**Problem:** Part 4 explains GPTQ as "uses the Hessian to redistribute error across remaining weights." Mechanically correct. But "smarter rounding that accounts for how sensitive each weight is" — the core intuition — is absent. "Sensitive" appears zero times in Part 4.

**Recommendation:** Add one sentence before the "Why GPTQ works" paragraph:
> "Not all weights matter equally — the Hessian measures exactly this. A weight with low curvature (changing it barely shifts the loss) can be rounded aggressively. A weight with high curvature (changing it spikes the loss) must be rounded carefully. GPTQ uses this sensitivity map to decide which weights can absorb error and which cannot — that's the core insight behind why int4 works where naive rounding fails."

---

### Gap 3 — GGUF table: recommendation given, decision framework not — Priority: Medium

**Problem:** The Part 5 code prints the GGUF table and immediately states "RECOMMENDATION: Q4_K_M." No guidance on how to read the tradeoff for a different memory budget or quality tolerance.

**Recommendation:** Add a "how to navigate this table" block before the recommendation print:
> "Decision framework: Start from the 'Fits 16GB?' column — find the heaviest format that fits with ≥2 GB headroom. Then ask: does the extra memory buy meaningful quality? If perplexity delta < 0.5, the lighter format is probably better. For Riverside: Q4_K_M (4.1 GB, Δ+0.3) vs Q5_K_M (5.0 GB, Δ+0.2) — the 0.1 point improvement isn't worth 0.9 GB of headroom. Substitute your own memory budget and quality threshold."

---

### Gap 4 — Part 6 (NF4) breaks the deployment narrative — Priority: Medium

**Problem:** The Riverside question for Part 6 is "How does NF4 connect to what we did in 04-llm?" — the only backward-looking question in the intro table. The closing decision does not mention NF4, meaning Part 6 contributes nothing to the Riverside recommendation. It's a detour placed right before the finish line.

**Recommendation:** Either:
- **(a)** Move Part 6 to an appendix after Part 7 with a skip-ahead note: "If your goal is inference deployment, skip to Part 7."
- **(b)** Reframe the Riverside question: "If Riverside later decides to fine-tune the model on proprietary manuscripts, can they do it on the same 16 GB MacBook?" — making Part 6 a natural extension of the deployment journey.

---

### Gap 5 — "What actually breaks" quantified in perplexity, not described qualitatively — Priority: Medium

**Problem:** The audience's stated need is "understand what actually breaks." The notebook answers with perplexity deltas, which are a proxy that can't translate directly to editing quality.

**Recommendation:** In the closing decision cell, add a calibration sentence:
> "In literary-editing blind tests, trained editors cannot reliably identify Q4_K_M vs. bf16 output at perplexity deltas below approximately +1.0. Q4_K_M's +0.3 is well inside that threshold. At +3 points — reached by aggressive int3 quantization — grammatical accuracy begins to visibly degrade and stylistic consistency drops."

---

### Gap 6 — Part 4 → Part 5 transition has a silent causal gap — Priority: Low

**Problem:** Part 4 ends demonstrating GPTQ int4 at 3.5 GB — a working solution. Part 5 opens with "GGUF is a binary format..." without explaining why Part 5 exists when Part 4 already solved the memory problem.

**Recommendation:** Open Part 5 with one sentence: "GPTQ assumed Python and PyTorch are available on the target machine. Riverside cannot meet that assumption — author MacBooks must stay clean environments. GGUF solves this: a self-contained binary format that llama.cpp runs directly via Apple Metal, no Python required."

---

## Do NOT Change

- The intro table (Part / Concept / Riverside question)
- `MACBOOK_VRAM_GB` as a live constant in code comparisons
- "🔮 Predict first" interactivity in Part 2
- "Why naive int4 fails" paragraph in Part 4 (correct failure-first pedagogy)
- Tier 1/2/3 scope management in Part 7
- Cross-reference to `04-llm` for QLoRA context in Part 6
- "When to Use What" decision matrix
- Model-size table in Part 4 with "Fits 16GB?" column
