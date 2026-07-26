# Images Plan — GPU Hardware Foundations

Images referenced by `gpu-hardware-foundations.ipynb` and produced by future `img/gen_*.py` scripts.
Palette: dark graphite (#1e1e2e background, #cdd6f4 text, accent tones from Catppuccin Mocha).

---

## gpu-memory-hierarchy.png

**Purpose:** Visual companion to Part 2 — shows the GPU memory pyramid (Registers → L1/Shared → L2 → HBM) with annotated bandwidth and latency numbers.

**Perchance prompt:**
```
Dark graphite GPU memory hierarchy pyramid diagram. Four horizontal tiers stacked vertically.
Top tier (smallest): "Registers — 256 KB/SM — ~28 TB/s — 1 cycle" in bright cyan.
Second tier: "L1 / Shared Memory — 228 KB/SM — ~19 TB/s — 5 cycles" in teal.
Third tier: "L2 Cache — 40 MB — ~12 TB/s — 200 cycles" in amber.
Bottom tier (largest): "HBM (VRAM) — 24–80 GB — 0.9–3.4 TB/s — 600 cycles" in coral.
Background #1e1e2e, labels #cdd6f4. Arrows on left showing decreasing bandwidth downward.
Arrow on right showing increasing capacity downward. Caption: "Faster = Smaller = Closer to ALU".
Technical diagram style, no gradients, clean sans-serif font.
```

**Target script:** `img/gen_gpu_memory_hierarchy.py`
**Output:** `img/gpu-memory-hierarchy.png` (1200×800 px)

---

## roofline-model.png

**Purpose:** Static pre-built version of the roofline plot in Cell 11, suitable for README or slide deck export.

**Perchance prompt:**
```
Dark graphite roofline performance model chart on log-log axes. Four GPU roofline curves:
RTX 4090 (coral), A10G (steel blue), A100 80G (medium sea green), H100 (orange).
Each curve: rises linearly from lower-left (memory-bound), then flattens at peak TFLOPS (compute-bound).
Vertical dashed lines at each GPU's ridge point. X-axis: "Arithmetic Intensity (FLOP/byte)".
Y-axis: "Attainable Performance (TFLOPS)". Three black dots with labels: "LLM decode (AI≈2)",
"LLM prefill (AI≈50)", "Training batch=32 (AI≈150)". Background #1e1e2e, gridlines #313244.
Legend bottom-right. Caption below: "Left of ridge = memory-bound; right = compute-bound".
```

**Target script:** `img/gen_roofline.py`
**Output:** `img/roofline-model.png` (1400×800 px)

---

## warp-simt-execution.png

**Purpose:** Visual companion to Part 4 — illustrates 32 SIMT threads in a warp executing the same instruction on different data (contrast with CPU SIMD).

**Perchance prompt:**
```
Dark graphite side-by-side comparison diagram. Left panel "CPU SIMD": one wide core with
8 data lanes highlighted in amber, label "1 instruction × 8 data (AVX-512)".
Right panel "GPU SIMT": grid of 32 small square threads all highlighted simultaneously in cyan,
label "1 instruction × 32 threads (1 warp)". Between panels, a dividing line with label "vs."
Below GPU panel: second grid showing 32 threads dimmed/idle with label "Low occupancy: few active warps".
Third grid fully lit: label "High occupancy: many warps hide latency". Background #1e1e2e.
Clean technical illustration, no 3D effects, monospace font for labels.
```

**Target script:** `img/gen_warp_simt.py`
**Output:** `img/warp-simt-execution.png` (1400×700 px)

---

## Generation Notes

- All scripts use `matplotlib.use('Agg')` (headless, per animation-conventions).
- Export at 150 DPI minimum.
- Notebook cells reference images with relative paths: `![alt](img/filename.png)`.
- Scripts live in `img/` alongside outputs (per chapter-local convention in this repo).
