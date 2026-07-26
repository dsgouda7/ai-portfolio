# Improvement Plan — Neural Networks and Backpropagation

**Audited:** 2026-07-26 | **Audience fit:** 7/10

## Overall Assessment

Strong structural spine: the XOR proof-by-contradiction is genuinely satisfying, backprop variable names mirror the math notation, and the Dropout train/eval proof is memorable. The notebook weakens in the middle. Part 3's chain rule formula arrives cold with no backward-pass diagram (despite the forward pass having a dedicated diagram). Part 4 presents depth-vs-width as a result to observe rather than an intuition to build first. The XOR/SmartVal thread breaks cleanly in Parts 4 and 5. The `XORNet` class is defined and immediately used with no walkthrough of `nn.Module`, `super().__init__()`, or `forward()` semantics.

---

## Strengths (preserve these)

- **Proof-by-contradiction for XOR** — constraints A–D → contradiction is clean and epistemically satisfying
- **"Predict first" active learning prompt** in Part 1 — right pattern-recognition hook
- **Backprop variable names** (`dL_dyhat`, `dyhat_dz2`, `dz2_dW2`) — mirror math notation, self-documenting
- **`torch.allclose(..., atol=1e-5)` assertion** in Part 3 — engineers believe what they can verify
- **"What is IDENTICAL / What CHANGES"** structure in Part 6 — clear, reassuring, demystifying
- **Dropout train/eval proof** — five-call observation shows stochastic zeros; concrete and memorable
- **Opening overview table and closing summary table** — strong scaffolding at both ends

---

## Gaps & Recommended Changes

### Gap 1 — Backprop: chain rule formula arrives cold, no backward-pass diagram — Priority: High

**Problem:** Part 3 opens with the full chain rule LaTeX block before any intuitive warm-up. The forward pass has `neural-network-forward-pass.png`; the backward pass has no visual equivalent. The asymmetry signals "forward is for understanding, backward is for experts" — the opposite of the intent.

**Recommendation:** Add before the formula: "Backprop asks: if I nudge W₂ slightly, how much does the loss change? Trace backward from the loss — at each layer, multiply how sensitive that layer's output is to its input by the accumulated sensitivity from the output side. That product-of-sensitivities is the chain rule." Then add a backward-pass diagram to `images-plan.md`: same 2→2→1 architecture, gradient arrows flowing right-to-left, annotated ∂L/∂W₂, ∂L/∂h, ∂L/∂W₁.

---

### Gap 2 — Depth vs. Width: result shown before intuition is built — Priority: High

**Problem:** Part 4 introduces depth vs. width with one functional sentence then drops into code. The reader observes numbers; they do not feel why.

**Justification:** "Depth beats width" is only memorable when the reader first thinks: each layer reshapes the space before handing it to the next, so two layers can detect local curvature AND global arm membership. Width gives you more of the same transformation, not a new one.

**Recommendation:** Add a 4-sentence intuition block before the code: "Width = more workers doing the same job: one transformation with more capacity. Depth = assembly line: each layer reshapes the space before the next one sees it. Detecting a spiral arm requires two sequential abstractions — 'is this point locally curved?' then 'which arm belongs to it?' That maps naturally to two hidden layers, not one fat one."

---

### Gap 3 — XOR thread breaks in Parts 4 and 5 without bridge sentences — Priority: Medium

**Problem:** Part 4 pivots to the spiral dataset with no explanation of why XOR can't demonstrate depth vs. width. Part 5 uses entirely synthetic tensors with zero connection to XOR or SmartVal AI. When Part 6 brings back the XOR parameter count, the payoff is weakened.

**Recommendation:** One sentence at the start of each: 
- Part 4: "XOR has only 4 points — too small to reveal how a network generalises; we need a harder problem where architecture choice makes a measurable difference."
- Part 5: "If SmartVal AI's network grew to thousands of neurons, two new failure modes emerge: neurons become over-dependent, and activations drift to extreme values."

---

### Gap 4 — Hidden-space transformation never visualised — Priority: Medium

**Problem:** "The hidden layer transforms the input into a new feature space where XOR IS linearly separable" is the most important claim in the notebook — stated in prose, never visualized. This is the defining "aha" moment for neural networks.

**Recommendation:** After the trained model in Part 2, add a 6-line cell: extract `h = relu(net.layer1(X_xor)).detach()`, plot the 4 XOR points in (h₁, h₂) space colored by label, draw the linear boundary. Caption: "Input space: XOR not separable. Hidden layer space: XOR is."

---

### Gap 5 — `XORNet` class defined without a PyTorch module walkthrough — Priority: Medium

**Problem:** `class XORNet(nn.Module)` is defined and used in a training loop with no explanation of `nn.Module` (parameter registry), `super().__init__()` (required for `.parameters()` to work), or why calling `model(x)` dispatches to `forward()`.

**Recommendation:** Add a 4-bullet markdown cell immediately before the `XORNet` class:
- `nn.Module` = PyTorch's parameter registry
- `super().__init__()` = registers this object so `.parameters()` finds all weights  
- `self.layer1 = nn.Linear(2, 2)` = registers a learnable layer as a tracked attribute
- `def forward(self, x)` = called automatically when you write `model(x)`

---

### Gap 6 — GPT-2 scale bridge informative but not awe-inspiring — Priority: Low

**Problem:** The 13M× ratio appears as `print()` output rather than being given visual prominence. The key closing insight is buried in terminal output, not rendered as a callout.

**Recommendation:** Move "If you understood gradient descent on 9 XOR weights, you understand it on 117 million GPT-2 weights" from a `print()` statement into a markdown blockquote immediately after the code cell. Add one concrete analogy: "GPT-2's vocabulary embedding table alone holds ~38 million parameters — more than 4 million XOR networks stacked together."

---

## Do NOT Change

- Proof-by-contradiction structure — constraints A–D, contradiction statement
- The "Predict first" prompt cell in Part 1
- Backprop variable naming: `dL_dyhat`, `dyhat_dz2`, `dz2_dW2`
- The `torch.allclose(..., atol=1e-5)` match assertion in Part 3
- Dropout 5-call train/eval proof
- "What is IDENTICAL / What CHANGES" two-section structure in Part 6
- Spiral dataset, `make_wide`, `make_deep`, `train_net` code
- Tier 1/2/3 scope section
