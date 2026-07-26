# Images Plan — PyTorch Profiling

All images use the **dark graphite palette** (`#1e1e2e` background, `#cdd6f4` primary text,
accent colours from Catppuccin Mocha). Figures should be 1200 × 700 px at 150 dpi unless noted.

---

## profiler-timeline-annotated.png

**Purpose:** Visual companion to Part 1 — shows an annotated screenshot-style diagram of a
`torch.profiler` timeline with the four named regions (data_prep, forward, backward,
optimizer_step) highlighted as horizontal spans, and the top-5 operators annotated within
each region.

**Perchance prompt:**
```
Dark graphite profiler timeline diagram. Background #1e1e2e. Horizontal timeline bar divided into
four labelled segments from left to right: "data_prep" (#89b4fa steel blue, narrow), "forward"
(#f0a500 amber, medium), "backward" (#f38ba8 coral, widest — about 2.5x forward width),
"optimizer_step" (#a6e3a1 sea green, medium). Each segment has a bracket below listing 2–3
example operator names in small monospace text: forward shows "nn.Linear, LayerNorm, GELU";
backward shows "AddmmBackward, NativeLayerNormBackward, CudnnRnnBackward". A vertical dashed
line labeled "Step boundary" at the right edge. X-axis labeled "Wall time (ms)". A legend at
bottom-right with segment colours. Title: "torch.profiler — Single Training Step Regions".
Background #1e1e2e, gridlines #313244, label text #cdd6f4. Clean technical diagram style.
```

**Target script:** `img/gen_profiler_timeline.py`
**Output:** `img/profiler-timeline-annotated.png` (1400 × 600 px)

---

## compute-vs-memory-bound.png

**Purpose:** Visual companion to Part 3 — a roofline-style scatter showing matmul and softmax
plotted on arithmetic intensity (x) vs. effective throughput (y), with the bandwidth roof and
compute roof lines labelled, making it obvious that softmax sits far left of the ridge point.

**Perchance prompt:**
```
Dark graphite roofline diagram on log-log axes. Background #1e1e2e, gridlines #313244.
A single GPU roofline curve: rises linearly from lower-left (memory-bound slope, labeled
"Memory bandwidth roof — slope = BW"), then bends at the ridge point and flattens at peak
TFLOPS (labeled "Compute roof — TFLOPS"). Two large scatter dots: one labeled "matmul (Q@K^T)"
in coral (#f38ba8), plotted far right near compute roof; one labeled "softmax (S×S)" in cyan
(#89dceb), plotted far left on the memory-bound slope. A vertical dashed line at ridge point
labeled "Ridge point". Arrows pointing to each dot explaining the bottleneck type.
X-axis: "Arithmetic Intensity (FLOP/byte)", range 0.1–1000 (log scale).
Y-axis: "Effective Throughput (TFLOPS or GB/s)", log scale.
Title: "Attention Operations on the Roofline — Softmax is Memory-Bound".
Caption below: "Left of ridge = memory-bound; right = compute-bound".
Legend top-left. Clean sans-serif font, no gradients.
```

**Target script:** `img/gen_compute_vs_memory.py`
**Output:** `img/compute-vs-memory-bound.png` (1400 × 800 px)
