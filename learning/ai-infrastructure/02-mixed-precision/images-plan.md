# Images Plan — Mixed Precision and Memory Math

All images use the **dark graphite palette** (`#1e1e2e` background, `#cdd6f4` primary text,
accent colours from Catppuccin Mocha). Figures should be 1200 × 600 px at 150 dpi unless noted.

---

## memory-footprint-breakdown.png

**Purpose:** Stacked bar chart showing how 24 GB is consumed across the four components
(params, gradients, optimizer states, activations) for each of the three Riverside models
at bf16, side by side.

**Perchance / generation prompt:**
> A dark-graphite stacked bar chart, background #1e1e2e, three grouped bars labelled
> "GPT-2-Medium", "LLaMA-3-1B", "LLaMA-3-8B". Each bar is stacked with four segments:
> Parameters (#89b4fa blue), Gradients (#a6e3a1 green), Optimizer states (#fab387 peach),
> Activations (#f38ba8 red). A dashed horizontal line at y=24 GB labelled "A10G limit".
> Y-axis: "Memory (GB)", range 0–45. Title: "Full Fine-Tuning Memory at bf16 (batch=8, seq=512)".
> Clean sans-serif font, no chart junk.

**Used in:** Cell 5 output commentary; Cell 16 summary figure.

---

## fp32-fp16-bf16-number-line.png

**Purpose:** Side-by-side bit-field diagrams for fp32, fp16, and bf16 showing sign /
exponent / mantissa bit widths, with the max representable value annotated on each.
A small inset shows the "danger zone" where LLM gradients live (values > 65 504)
highlighted on the fp16 line as a red overflow region.

**Perchance / generation prompt:**
> Three horizontal bit-field bars stacked vertically on a dark graphite background
> (#1e1e2e). Top bar: fp32 — 1 grey sign bit, 8 orange exponent bits, 23 blue mantissa bits.
> Middle bar: fp16 — 1 grey bit, 5 orange bits, 10 blue bits, then a wide red "overflow zone"
> arrow labelled "> 65 504". Bottom bar: bf16 — 1 grey bit, 8 orange bits, 7 blue bits.
> Each bar is 600 px wide, bars are the same width so bit widths are proportional.
> Annotation below fp16: "max 65 504 — LLM gradients regularly exceed this!".
> Annotation below bf16: "same exponent range as fp32 → no overflow risk".
> Dark background, white labels, accent colours as above.

**Used in:** Cell 6 (Part 2 introduction).

---

## gradient-checkpointing-tradeoff.png

**Purpose:** A two-panel figure. Left panel: timeline of a standard forward+backward pass
with all activation tensors shown as coloured blocks pinned in memory simultaneously.
Right panel: the same timeline with gradient checkpointing — only one activation block
present at a time, but the forward arrows are doubled (recomputation). An annotation
arrow on the left says "peak memory ∝ L" and on the right "peak memory ∝ √L, compute ×1.3".

**Perchance / generation prompt:**
> A two-panel diagram on dark graphite (#1e1e2e). Left panel labelled "Standard backward pass":
> a vertical stack of 8 coloured rectangles (activation buffers, each labelled "act_i" in
> #cdd6f4) all simultaneously present, with a red bracket on the right labelled "peak = L buffers".
> Right panel labelled "Gradient checkpointing": only 1 rectangle present at a time; blue
> curved arrows showing "recompute" looping back through 2-3 layers; a green bracket labelled
> "peak ≈ √L buffers". Below each panel: "Memory: 100%" (left) and "Memory: ~55%" (right),
> "Compute: 100%" (left) and "Compute: ~130%" (right). Clean, no decorative elements.

**Used in:** Cell 11 (Part 4 introduction) and Cell 12 commentary.
