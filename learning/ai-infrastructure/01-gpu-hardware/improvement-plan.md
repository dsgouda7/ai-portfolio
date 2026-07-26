# Improvement Plan — GPU Hardware Foundations

**Audited:** 2026-07-26 | **Audience fit:** 6/10

## Overall Assessment

Right architecture: concrete business scenario ($80k/month InferenceBase), predict-first questions, CPU-only fallback with attributed reference numbers, and a Part 6 that genuinely follows from the prior parts. What's missing is the intuitive layer that must precede every technical definition for engineers who use GPUs as black boxes. Three of the four hardest concepts (SIMT, memory hierarchy, roofline) are introduced formula-first or table-first with no analogy to anchor the vocabulary. The running example `(B=8, S=128, D=256)` disappears after the first timing cell. Two factual errors (Llama-7B vs 8B inconsistency, A100 budget math wrong).

---

## Strengths (preserve these)

- **InferenceBase scenario** — $80k/month pain, specific model, specific hardware candidates; earns reader trust immediately
- **Predict-first question in Part 1** — the notebook's best pedagogical moment
- **CPU-only graceful fallback with published reference numbers** — rare and correct
- **Roofline chart** — four GPUs, log-log axes, annotated LLM workload points, ridge dotlines; the clearest roofline diagram in any tutorial
- **Part 6 "Checklist so far" table** — links each requirement to the Part that established it
- **Tier 1/2/3 scope disclaimer**
- **`gpus` dict shared between Part 3 and Part 6** — reader can trace numbers across parts

---

## Gaps & Recommended Changes

### Gap 1 — No SIMT analogy before the technical definition — Priority: High

**Problem:** Part 1 jumps to "1 instruction, 32 threads execute together (a warp)" with no preceding mental model.

**Recommendation:** Add 3 sentences before the SIMD/SIMT bullet list:
> "Imagine 32 factory workers on an assembly line. A foreman shouts one instruction — 'tighten bolt' — and all 32 workers tighten their bolt at the same moment. That's a GPU warp. If even one worker must do something different from the others, the whole group stalls — that's warp divergence."

---

### Gap 2 — Memory hierarchy table lands cold — Priority: High

**Problem:** Part 2 opens with a four-row table with raw bandwidth numbers before the reader has any frame for what those numbers mean.

**Recommendation:** Add 4 sentences before the table:
> "Think of the memory hierarchy as containers at different distances from the chef (the compute cores). Registers are ingredients on the cutting board — tiny supply, grabbed in one motion. SRAM is the kitchen counter — small but fast. HBM is the walk-in fridge at the back of the restaurant — big capacity, but every trip costs time. When inference reads 16 GB of Llama-3-8B weights from HBM, it's making thousands of trips to the walk-in fridge per token."

---

### Gap 3 — Roofline model is formula-first, intuition-last — Priority: High

**Problem:** Part 3 opens with $\text{Performance} = \min(\text{peak TFLOPS},\ \text{bandwidth} \times \text{AI})$ before arithmetic intensity is explained in plain language.

**Recommendation:** Add 4 sentences before the formula:
> "Before the math — imagine a factory: a brilliant machinist (GPU cores) and one slow truck delivering raw materials (memory bandwidth). If the machinist finishes each batch before the next truck arrives, he's sitting idle — the truck is the bottleneck. Arithmetic intensity measures how much machining you do per truck delivery (FLOP per byte). Low AI = memory-bound."

---

### Gap 4 — Running example vanishes after Part 1 — Priority: Medium

**Problem:** `(B=8, S=128, D=256)` promised as running example throughout. After the first timing cell: Part 1 speedup chart uses `(4, s, s)`; Part 4 uses `D_MODEL=4096`; Part 5 uses `M=4096`.

**Recommendation:** Either (a) annotate the running example's arithmetic intensity on the roofline chart, or (b) update the intro note: "Running example: `(B=8, S=128, D=256)` for Part 1 timing; model-scale numbers (Llama-3-8B) used from Part 2 onward, where toy dimensions produce unrepresentative arithmetic intensities."

---

### Gap 5 — InferenceBase narrative thread drops in Parts 4 and 5 — Priority: Medium

**Problem:** After Part 3's explicit "For InferenceBase's use case: HBM bandwidth matters most," neither Part 4 nor Part 5 mentions InferenceBase. They become self-contained technical explorations.

**Recommendation:** One sentence at the opening of each Part:
- **Part 4:** "For InferenceBase, this is where batch size turns from a training concern into a cost lever: how many requests can be batched per forward pass determines tok/s-per-dollar."
- **Part 5:** "For InferenceBase, this matters specifically for the KV cache: `K.transpose(-2,-1)` creates a non-contiguous view on every forward pass for every token generated."

---

### Gap 6 — Warp divergence image has no textual setup — Priority: Medium

**Problem:** Part 4's image shows 20 teal threads and 12 grayed-out threads. Part 4's text covers occupancy and batching — warp divergence is never mentioned.

**Recommendation:** Add 2 sentences before the image reference: "There is a second occupancy-killer: warp divergence. When threads in the same warp take different `if/else` branches, the GPU runs both branches serially with one set masked off — halving throughput."

---

### Gap 7 — Part 4 uses Llama-7B dimensions while the scenario is Llama-3-8B — Priority: High (factual error)

**Problem:** `D_MODEL = 4096  # Llama-7B hidden dim`, `model_gb = 14.0  # 7B params`, prints "Throughput analysis for Llama-7B." The business scenario is Llama-3-8B (16 GB at bf16).

**Recommendation:** Change `14.0` → `16.0`, update the comment and print label. Llama-3-8B also uses `hidden_dim=4096`, so only the label and `model_gb` need updating.

---

### Gap 8 — Part 6 A100 budget conclusion is arithmetically false — Priority: Critical (factual error)

**Problem:** `cost_mo: 10.0` ($/hr) × 730 hr = $7,300/month. The code prints "exceeds $15k budget" — false. $7,300 < $15,000. A100 actually fits the budget.

**Recommendation:** Change the A100 rejection rationale from budget to value:
```python
print(f"    Monthly: ~${monthly_cost_a100:,.0f} — fits the $15k budget,")
print(f"    but delivers only {a100['bandwidth_tbs']/rtx4090['bandwidth_tbs']:.1f}× more throughput")
print(f"    at {a100['cost_mo']/rtx4090['cost_mo']:.0f}× the hourly cost.")
print(f"    → RTX 4090 wins on tok/s per dollar for inference-only workloads.")
```
Also rename `cost_mo` → `cost_hr` to eliminate the naming ambiguity.

---

## Do NOT Change

- Predict-First question structure in Part 1
- CPU-only graceful fallback with reference numbers
- Roofline chart design (log-log axes, annotated workload points, $/hr in legend)
- Part 6 "Checklist so far" table
- Tier 1/2/3 scope disclaimer
- "When to Use What" final table
- `gpus` dict shared between Part 3 and Part 6
- Part 5 coalescing benchmark (`A @ B` vs `A.t() @ B` vs `A.t().contiguous() @ B`)
