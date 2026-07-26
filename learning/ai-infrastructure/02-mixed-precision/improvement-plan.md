# Improvement Plan — Mixed Precision and Memory Math

**Audited:** 2026-07-26 | **Audience fit:** 6/10

## Overall Assessment

Strong structural bones: Riverside A10G constraint drives every Part, the TOC maps each Part to a Riverside question, predict-first questions create active engagement, and the closing Part 6 payoff answers the original question with computed numbers. Between the bones and the payoff, three core pedagogical contracts are broken: the fp16 overflow demonstration is promised but not delivered, GradScaler is described mechanically without an analogy, and gradient checkpointing receives complexity-theory notation (O(L) → O(√L)) that means nothing to an ML engineer. Parts 4 and 5 let the Riverside thread go slack.

---

## Strengths (preserve these)

- **Riverside narrative frame** — 24 GB, 3 days, 3 named models; TOC table maps each Part to a Riverside question
- **Predict-first (🔮) questions in Parts 2 and 4** — active-learning moments before code runs; Part 4 question is especially well-crafted
- **Part 3 "What just happened" bridge cell** — explicitly connects Part 3 output to Part 4 motivation
- **Part 6 payoff** — all three candidate models at all relevant precision + technique combinations with ✓/✗ against 24 GB limit
- **"When to Use What" decision table** — decision-oriented, not description-oriented
- **Tier 1/2/3 coverage framework** — honest about what was built vs. explained vs. named

---

## Gaps & Recommended Changes

### Gap 1 — Memory math: formula before intuition — Priority: High

**Problem:** Part 1 opens with the four-component table (`params × bytes_per_param`, etc.) before any "here's how to think about it" framing.

**Recommendation:** Add 3–4 prose sentences before the table:
> "Think of each parameter as a single number stored on the GPU — like one cell in a spreadsheet. fp32 gives each cell 4 bytes; fp16/bf16 gives it 2 bytes. A 1B-parameter model in fp32 needs 4 GB just to hold those numbers at rest. Training multiplies that: space for gradients (one per weight), optimizer momentum terms (two per weight for Adam), and intermediate forward-pass values. The table below formalises each component."

---

### Gap 2 — fp16 overflow: promised but not delivered — Priority: High

**Problem:** Code runs fp16 training and prints "This run didn't overflow (small model/short sequence)." The 🔮 predict-first question builds an expectation that the learner does not see met. The code wraps the attempt in `torch.autocast` which routes operations to more stable paths, making failure even less likely.

**Recommendation:** Add a synthetic overflow cell *before* the model demo:
```python
x = torch.tensor(70000.0, dtype=torch.float16)
print(f"70,000 in fp16:   {x}")        # → inf  (overflow)
g = torch.tensor(0.00005, dtype=torch.float16)
print(f"0.00005 in fp16:  {g}")        # → 0.0  (underflow)
```
This makes overflow/underflow *observable* before the model runs. The model demo then becomes confirmation, not proof.

---

### Gap 3 — GradScaler: API calls without a mental model — Priority: High

**Problem:** The description says "scales the loss before backward; unscales before optimizer step" — accurate but mechanical. The code then shows `scaler.scale(loss).backward()`, `scaler.step()`, `scaler.update()` with no mental model bridging description to code.

**Recommendation:** Add 2 sentences immediately before the code block:
> "Imagine your gradients are whispered numbers — so small that fp16 rounds them to zero. GradScaler *shouts* them first: it multiplies the loss by 65,536 before the backward pass so the gradients are large enough for fp16 to represent, then divides back by 65,536 before the optimizer uses them — the optimizer never sees inflated values."

---

### Gap 4 — Gradient checkpointing: O(L) notation for an ML audience — Priority: Medium

**Problem:** "Memory saved: O(L) → O(√L) where L = number of layers" — correct algorithmically but provides no intuition for engineers who don't think in asymptotic complexity.

**Recommendation:** Replace with: "Instead of keeping all N layer activations in memory simultaneously, checkpointing keeps only every √N-th layer's output — and recomputes the in-between ones during the backward pass. For a 24-layer model, that's roughly 5 layers saved instead of 24 — at the cost of running the forward pass once more. Rule of thumb: saves ~40% of activation memory, costs ~30% more compute."

---

### Gap 5 — Riverside narrative drops in Parts 4 and 5 — Priority: Medium

**Problem:** Parts 4 and 5 use anonymous toy models with no connection to LLaMA-3-8B or the A10G. The Riverside constraint disappears.

**Recommendation:** One sentence at the opening of each Part:
- **Part 4:** "For Riverside's 8B model, activation memory grows as `batch × seq × hidden × layers × bytes = 8 × 512 × 4096 × 32 × 2 = ~4 GB` — the fourth-biggest component. Gradient checkpointing is the lever."
- **Part 5:** "Memory profiling for Riverside: before launching the training run on the A10G, verify peak VRAM won't exceed 23 GB (leaving 1 GB for the OS) by tracing one forward+backward pass."

---

## Do NOT Change

- Riverside A10G narrative frame and 24 GB constraint
- 🔮 predict-first questions in Parts 2 and 4 (format, position, three-option structure)
- Part 3 "What just happened" bridge cell
- Part 6 model-selection table with ✓/✗ against 24 GB limit
- Tier 1/2/3 coverage framework
- "When to Use What" closing table
