# Improvement Plan — RNN / LSTM Sequence Modeling

**Audited:** 2026-07-26 | **Audience fit:** 6/10

## Overall Assessment

Excellent bones: the Twinkle melody threads from the opening blockquote to the closing decision, the predict-first pattern appears before every Part, and the verification-by-assertion philosophy (embedding = one-hot proved, manual RNN = `nn.RNN` proved, gates ∈ (0,1) proved) builds trust through measurement rather than assertion. The notebook weakens at three critical teaching moments: (1) an implementation bug silently returns `nan` from the gradient experiment, breaking the visual proof; (2) the hidden-state equation appears before any analogical frame; (3) the "Twinkle 15 characters apart" in the intro blockquote contradicts the code's "8 characters apart" — a trust-breaking inconsistency on line 1.

---

## Strengths (preserve these)

- **Predict-first quiz pattern** before every Part — the highest-leverage pedagogical feature
- **Single running example end-to-end** — Twinkle melody appears in intro, corpus setup, BPTT, Part 5 test, closing decision
- **Verification by assertion** — embedding = one-hot, manual RNN = `nn.RNN`, gates ∈ (0,1) all proved by code
- **Part 5 Twinkle test** — head-to-head comparison; abstract LSTM vs. RNN claim cashed out in one countable number
- **Tier 1/2/3 scope declaration** — GRU, teacher forcing, bidirectional correctly scoped out
- **Closing decision block** — parameter counts, loss values, twinkle score: effective payoff format
- **Bridge-to-Transformers closing cell** — names both the serialization bottleneck and the fixed-bottleneck

---

## Gaps & Recommended Changes

### Gap 1 — Factual error: "15 characters apart" contradicts "8 characters apart" — Priority: Critical

**Problem:** The opening blockquote — the first thing any learner reads — states "twinkle appears twice, 15 characters apart." The corpus setup comment correctly states "8 characters apart." The CORPUS string `"Twinkle twinkle little star, how I w"` places the second "twinkle" at index 8.

**Recommendation:** Change opening blockquote from "15 characters apart" to "8 characters apart." One-word fix; removes a trust-breaking inconsistency in the first sentence.

---

### Gap 2 — Implementation bug: `measure_gradient_norms` silently returns `nan` — Priority: Critical

**Problem:** The function creates `h = torch.zeros(..., requires_grad=True)` (leaf tensor) then rebinds `h` inside the loop: `h = cell(x_seq[t].unsqueeze(0), h)`. After the loop, `h` is a non-leaf tensor; PyTorch only stores `.grad` for leaf tensors. The guard `if h.grad is not None else float('nan')` therefore returns `float('nan')` for every T > 0. `int(float('nan') * 50)` raises `ValueError` in the bar chart cell.

**Recommendation:** Save a reference to the initial leaf tensor before the loop:
```python
h_init = torch.zeros(1, hidden_size, requires_grad=True)
h = h_init
for t in range(T):
    h = cell(x_seq[t].unsqueeze(0), h)
loss = h.sum()
loss.backward()
grad_norm = h_init.grad.norm().item()
```
Apply the same pattern in the Your Turn cell.

---

### Gap 3 — Hidden state equation appears before any analogical frame — Priority: High

**Problem:** The roadmap table already displays `$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$` as the entry for Part 2. Part 2's header leads with the equation before any prose metaphor. The phrase "compressed memory" does not appear anywhere in the notebook.

**Recommendation:** Add a 3–4 sentence prose block immediately before the Part 2 equation, grounded in the melody:
> "Think of `h_t` as a small notecard the RNN carries from character to character. At each step it tears up the old notecard and writes a new one: a blend of what it remembered before and what it just read. After processing 'Twinkle tw', the notecard should still carry a faint signal — 'a twinkle-pattern started 8 steps back.' Here is the exact rule for rewriting the notecard:"
Then present the equation. Also remove the equation from the ToC row and replace with "hₜ as compressed memory."

---

### Gap 4 — Gradient experiment omits T=8 (the actual twinkle gap) — Priority: High

**Problem:** `measure_gradient_norms` is called with `seq_lengths = [5, 10, 20, 30, 50]`. T=8 — the exact number of steps between the two "twinkle" occurrences — is not in the list. The BPTT commentary asserts "gradient at step 0 is already tiny" for T=8 without the learner having seen that data point.

**Recommendation:** Change to `seq_lengths = [5, 8, 10, 20, 30, 50]`. Add one highlighted print line:
```python
idx8 = seq_lengths.index(8)
print(f"→ T=8 (the actual twinkle gap): norm={vanilla_norms[idx8]:.2e}"
      " — this is why the second 'twinkle' is hard to learn.")
```

---

### Gap 5 — FuncAnimation shows pre-computed bars, not a gradient dying — Priority: Medium

**Problem:** The animation reveals one more pre-calculated bar per frame (T=5 → T=10 → ... → T=50). The learner watches bars appear with decreasing heights — a static comparison animated. The concept (gradient signal fading as it travels backward through each timestep) is never conveyed as motion.

**Recommendation:** Replace the bar-reveal animation with one that shows gradient magnitude *per timestep* for a fixed T=50 run, animated from timestep 50 → 1. Mark the timestep-8 position with a vertical dashed line labeled "← the twinkle gap." Fallback: a static log-scale line chart showing gradient norm vs. steps-from-end, with the twinkle-gap marker.

---

### Gap 6 — LSTM gate explanations lack melody-specific questions — Priority: Medium

**Problem:** The Part 4 gate table has a Purpose column but no column anchoring each gate to a melody-specific question. The "four questions" framing arrives only in the closing "Key Insights" cell — retroactively.

**Recommendation:** Add a **Melody question** column to the Part 4 gate table at first introduction:
| Gate | Melody question |
|------|----------------|
| Forget | "Is the first 'twinkle' still relevant, or can I safely let it go?" |
| Input | "Is this new character important enough to write into long-term memory?" |
| Candidate | "If I do write something, what should it say about this character?" |
| Output | "Of everything I'm carrying, what matters for predicting the next character now?" |

---

### Gap 7 — `LSTMCell` fuses 4 gate matrices before the learner understands 4 separate gates — Priority: Medium

**Problem:** `self.W_gates = nn.Parameter(torch.randn(4 * H, H + I) * 0.1)` — all four gate matrices fused for efficiency, sliced by index arithmetic. The learner has just seen a table where each gate has its own W. The jump is an optimization insight, not a conceptual one.

**Recommendation:** Write the naive `LSTMCell` with four separate `nn.Linear` layers — one per gate — matching the table exactly. Add a comment block after the class definition explaining that PyTorch fuses these internally for efficiency, and why that's a performance optimization, not a conceptual change.

---

### Gap 8 — Rule-of-thumb threshold contradicts gradient experiment — Priority: Low

**Problem:** Closing rule: "Sequence length ≤ 15 tokens → vanilla RNN is sufficient." BPTT prose: "With T=8 steps, the gradient at step 0 is already tiny." Part 3 "What just happened": "vanilla RNNs can memorize patterns within ~10 steps." Three different thresholds (15, 8, ~10) for the same boundary.

**Recommendation:** Tighten to "≤ 10 tokens" (matching Part 3 language) and add parenthetical: "(at T=8 the RNN can still learn on this tiny corpus with enough epochs — but reliability drops at each step beyond ~10)."

---

## Do NOT Change

- Predict-first quiz pattern before every Part
- Twinkle melody as the single running example
- Verification-by-assertion philosophy
- Part 5 head-to-head Twinkle test
- Tier 1/2/3 scope table
- Closing decision block format
- Bridge-to-Transformers closing cell
- Your Turn exercise cell with `# ← CHANGE ME`
