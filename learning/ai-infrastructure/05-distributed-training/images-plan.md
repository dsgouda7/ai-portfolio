# Images Plan — Distributed Training Notebook

All images live in an `images/` subdirectory alongside the notebook.
Generate via [Perchance AI Image Generator](https://perchance.org/ai-photo-generator)
or equivalent; use the prompts below verbatim.

---

## `ddp-gradient-allreduce.png`

**Purpose:** Illustrate the DDP all-reduce ring: 4 GPUs each with a local gradient,
arrows showing the ring-allreduce communication pattern, and the same averaged gradient
appearing on all GPUs after the operation.

**Perchance prompt:**
```
Technical diagram, clean white background, flat vector style. Four blue GPU boxes
arranged in a diamond/ring topology, each labeled "GPU 0", "GPU 1", "GPU 2", "GPU 3".
Each GPU shows a small bar chart representing a local gradient (different heights,
different colours). Thick curved arrows connect adjacent GPUs forming a ring. After the
ring, all four GPUs show identical bar charts (same height, green). Caption reads
"DDP all-reduce: ring communication → averaged gradients on every GPU". No photorealism,
diagram only, minimal colour palette (blue, green, grey).
```

---

## `fsdp-vs-ddp-memory.png`

**Purpose:** Side-by-side bar chart showing per-GPU memory for DDP vs FSDP at
N = 1, 2, 4, 8 GPUs for a 70B model, with a horizontal red line at the A100 80 GB limit.

**Perchance prompt:**
```
Clean data visualisation, white background, matplotlib style. Grouped bar chart with
four groups on the x-axis labeled "1 GPU", "2 GPUs", "4 GPUs", "8 GPUs". Each group
has two bars: one dark blue labeled "DDP (GB/GPU)" and one coral/orange labeled "FSDP
(GB/GPU)". DDP bars are constant height (840 GB). FSDP bars decrease from 840 to
420 to 210 to 105. A horizontal dashed red line at y=80 is labeled "A100 80 GB limit".
Title: "DDP vs FSDP per-GPU memory — 70B model". Y-axis: "Memory (GB)". Grid lines.
Professional, minimal chart design.
```

---

## `parallelism-strategy-matrix.png`

**Purpose:** 2×2 matrix diagram (model size vs. GPU count) showing which parallelism
strategy is recommended in each quadrant: single-GPU, FSDP, FSDP+TP, 3D.

**Perchance prompt:**
```
Technical diagram, clean white background, flat design. A 2×2 matrix grid.
X-axis label "Number of GPUs" with values "1", "2-4", "8+". Y-axis label "Model Size"
with values "≤3B", "3-13B", "13-70B", "70B+". Each cell is a rounded rectangle filled
with a distinct pastel colour (light blue, light green, light yellow, light orange)
containing: strategy name in bold (e.g. "Single GPU"), one-line description below it
(e.g. "+ gradient checkpointing"). Title at top: "Parallelism Strategy Selection Matrix".
Diagonal arrow from top-left to bottom-right labeled "Increasing complexity". Clean,
diagram-only, no photorealism.
```
