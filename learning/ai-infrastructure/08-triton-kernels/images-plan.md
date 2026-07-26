# Images Plan — 08 Triton Kernels

| Asset | Placement | Teaching job |
|---|---|---|
| `triton-grid-block-thread.png` | Part 1 intro (Cell 4) | CUDA grid → thread blocks → individual threads hierarchy; same layout used by Triton `@triton.jit` |
| `fused-vs-unfused-gelu.png` | Part 3 body (Cell 10) | Two-step unfused (read activations from HBM, write bias, read again, apply GELU, write) vs. one-step fused (single HBM read/write) |
| `autotune-block-size-sweep.png` | Part 5 body (Cell 14) | Bar chart: throughput (TFLOP/s) vs. block size (16, 32, 64, 128, 256) for the tiled matmul; optimal block size highlighted |

---

## Perchance Generation Prompts

```text
[triton-grid-block-thread.png]
Flat vector hierarchy diagram, wide 16:9, dark graphite background. Three levels:
Top: a large teal box labeled "CUDA Grid" containing a 4×4 grid of smaller amber
boxes labeled "Thread Blocks". Each block contains a 4×4 grid of tiny ivory squares
labeled "Threads". Bracket annotations show "gridDim", "blockDim". A separate callout
shows "tl.program_id(axis=0) → block index" and "tl.arange(0, BLOCK_SIZE) → thread
offsets within block". Ivory labels. No logos, no photorealism, no gradients, no tiny text.
```

```text
[fused-vs-unfused-gelu.png]
Flat vector pipeline diagram, wide 16:9, dark graphite background. Top half "Unfused
(2 kernels, 5 HBM accesses)": HBM box → load activations (coral slow arrow, labeled
"read X") → add bias (amber box, "Kernel 1") → write to HBM (coral, "write tmp") →
load again (coral, "read tmp") → apply GELU (amber box, "Kernel 2") → write to HBM
(coral, "write Y"). Five coral HBM access arrows labeled "5 accesses". Bottom half
"Fused Triton kernel (2 HBM accesses)": HBM box → load (teal arrow, "read X+bias")
→ add bias + GELU (amber SRAM block, "compute in registers") → write output (teal
arrow, "write Y"). Two teal arrows labeled "2 accesses". Large label "50% HBM
savings". Ivory labels. No logos, no photorealism, no gradients, no tiny text.
```

```text
[autotune-block-size-sweep.png]
Flat vector bar chart, wide 16:9, dark graphite background. X-axis: block size (16,
32, 64, 128, 256). Y-axis: throughput in TFLOP/s. Bars in muted teal, rising to a
peak at 128, then falling at 256 (showing diminishing returns as blocks exceed SRAM).
The 128 bar is amber and labeled "autotune winner (A100)". A horizontal ivory dashed
line shows "torch.matmul (cuBLAS) reference" throughput slightly above the 128 bar.
Subtitle text: "Optimal block size is GPU-specific — autotune finds it automatically".
Ivory axis labels. No logos, no photorealism, no gradients, no tiny text.
```
