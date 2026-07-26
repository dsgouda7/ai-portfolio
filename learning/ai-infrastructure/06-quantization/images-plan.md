# Images Plan — Quantization in Depth

Images referenced by `quantization-in-depth.ipynb`. All are generated inside the
notebook itself (matplotlib) except the three conceptual diagrams below, which are
standalone illustrations for use in slides or README pages.

---

## `quantization-rounding-error.png`

**Purpose:** Visualise how int8 rounding error distributes across a weight tensor and
show the relationship between scale size and maximum error.

**Perchance prompt:**
> A clean, white-background technical diagram showing a number line from -1.0 to +1.0.
> Above the line, 256 evenly-spaced tick marks represent int8 quantization levels.
> A bell-curve density plot (steelblue fill) hovers above the line showing a normal
> distribution of neural-network weights centred at 0 with σ=0.02 — the curve towers
> near zero and barely reaches ±0.15. Small coral-coloured vertical error bars of uniform
> height (≈ scale/2) sit atop each tick mark, labelled "max rounding error = scale/2".
> Title: "int8 quantization: 256 uniform levels, rounding error bounded by scale/2".
> Style: textbook illustration, no shadows, minimal colour palette (steelblue, coral,
> black text), 1200×600 px.

---

## `gptq-vs-awq-perplexity.png`

**Purpose:** Compare GPTQ and AWQ perplexity degradation as bit-width drops from 8 → 4 → 3
for a 7B-class model (reference values from published benchmarks).

**Perchance prompt:**
> A line chart with two curves on a white background. X-axis: bit-width (8, 6, 4, 3,
> labelled "Quantization bit-width"). Y-axis: "Perplexity delta vs. bf16 baseline" from
> 0 to 8. First curve (steelblue, solid, labelled "GPTQ"): points at (8, 0.0), (6, 0.1),
> (4, 0.8), (3, 4.2). Second curve (coral, dashed, labelled "AWQ"): points at (8, 0.0),
> (6, 0.1), (4, 0.5), (3, 2.8). A horizontal grey dashed line at y=1.0 labelled
> "Acceptable threshold (Riverside)". Both curves share the same marker style (circles).
> Title: "GPTQ vs AWQ: perplexity degradation by bit-width (LLaMA-3-7B, WikiText-2)".
> Caption below: "Reference values — Frantar et al. 2022, Lin et al. 2023".
> Style: clean, no gridlines except light horizontal, white background, 1200×700 px.

---

## `gguf-quantization-formats.png`

**Purpose:** Visual summary of GGUF quantization formats, showing the size/quality/speed
tradeoff for a 7B model on Apple Silicon.

**Perchance prompt:**
> A horizontal grouped bar chart on a white background. Five GGUF formats on the Y-axis
> (from bottom to top: F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M). Three sets of bars per format
> in steelblue, coral, and mediumseagreen respectively:
>   - Steelblue: model size in GB (F16=14.0, Q8=7.7, Q6=6.1, Q5=5.0, Q4=4.1)
>   - Coral: perplexity delta × 5 for visual scale (F16=0, Q8=0, Q6=0.5, Q5=1.0, Q4=1.5)
>   - Mediumseagreen: tokens/second on M3 Pro (F16=8, Q8=15, Q6=20, Q5=25, Q4=30)
> Legend at top right. Title: "GGUF format comparison — LLaMA-3-7B on Apple M3 Pro".
> A vertical dashed coral line at x=14 labelled "Riverside 16 GB limit (2 GB headroom)".
> Style: publication-quality, no shadows, 1400×700 px.

---

## Notes

- All three images are **optional** — the notebook is fully self-contained with inline
  matplotlib figures.
- If generated, place them in an `images/` subdirectory next to the notebook.
- Notebook cells that plot their own figures (Cells 6 and 16) do not need standalone
  image files.
