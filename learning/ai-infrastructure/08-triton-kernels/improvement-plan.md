# Improvement Plan — Custom Kernels with Triton

**Audited:** 2026-07-26 | **Audience fit:** 7/10

## Overall Assessment

Strong architectural bones: clear 87% bandwidth target as the narrative thread, complete TRITON_AVAILABLE dual-path for GPU-less learners, genuine continuity with Ch1 and Ch4 via the prerequisite bridge table. The "Predict first" framing and Part 6 loop-closure are above the authoring standard for the track. Three problems prevent it from landing at full potential: one factual error (HBM savings math inconsistency across three locations in Part 3), one misleading algorithmic claim (Part 4 is not the Ch4 online softmax for practical sequence lengths), and one structural issue (Code Walkthrough appears after the kernel it explains).

---

## Strengths (preserve these)

- **Opening thread** — 87% target named before the first kernel; TOC maps each part to "what it proves"
- **Prerequisite bridge table** — links Ch1 (CUDA grid/block, warp occupancy) and Ch4 (tiling, online softmax) to Triton equivalents
- **`TRITON_AVAILABLE` dual-path** — annotated pseudocode path with genuine conceptual content for GPU-less learners
- **"Predict first" in Parts 2 and 5** — forces commitment; Part 5's "winner is hardware-specific" reveal is a clean payoff
- **Part 6 closes the loop** — shows `torch.compile → TorchDynamo → Triton` connecting daily PyTorch code to the kernels just written
- **Running example continuity** — `B=8, S=128, D=64` throughout anchors every benchmark in a familiar workload
- **Code comments** — `# STAYS IN REGISTERS`, `# load from HBM → registers`, `# ONE write to HBM per output tile` are doing real explanatory work
- **Decision tables at the end** — "When to Use What" and tier 1/2/3 give engineers a decision framework

---

## Gaps & Recommended Changes

### Gap 1 — HBM access counts are inconsistent across three locations — Priority: Critical (factual error)

**Problem:** Part 3 (Fused GELU+bias) gives three different answers:
- Markdown intro: "4 HBM accesses → **75% fewer HBM accesses**" — math error (4→2 = 50%, not 75%)
- Pseudocode path `print()`: "5 HBM accesses" unfused
- Code output `print()`: `n_bytes * 4` unfused → `HBM savings: 50%`

Part 4 (fused softmax) has the same layered inconsistency: closing decision prints "~75% vs. unfused," summary table says "75%", but code output implies 50%.

**Recommendation:** Pick one definition — "number of full-tensor reads or writes to HBM" — and apply consistently. For Part 3: unfused = 4 accesses (read X, write tmp, read tmp, write Y); fused = 2 (read X+bias together, write Y) → **50% savings**. Fix all instances of "75%" to "50%" in Part 3. Add one sentence defining "HBM access" at first use.

---

### Gap 2 — Code Walkthrough appears after the first kernel, not before it — Priority: High

**Problem:** "Code Walkthrough: Vector Addition Kernel — 4 Triton Primitives" appears *after* the Part 1 kernel code cell. Learners encounter `tl.program_id`, `tl.arange`, `tl.load`, `tl.store` without yet knowing their CUDA equivalents. For pseudocode-path learners, this is the vocabulary for all subsequent kernels.

**Recommendation:** Move the Code Walkthrough cell to appear *before* the Part 1 kernel code cell. Rename it "Before you read the kernel: 4 Triton primitives and their CUDA equivalents." This is a cell reorder, not a content change.

---

### Gap 3 — Part 4 softmax is not the Ch4 online softmax for practical sequence lengths — Priority: High

**Problem:** Part 4's intro says "implementing the online softmax from Ch4 in Triton." The actual kernel sets `BLOCK_SIZE = triton.next_power_of_2(N)` and loads the *entire row* into one block. This is a fused single-pass softmax, not the streaming two-pass algorithm from Ch4. The docstring even flags this: "For very long rows (N > BLOCK_SIZE), use an online softmax loop." At production sequence lengths (S=2048, 4096), this kernel would fail.

**Recommendation:** Update Part 4's intro:
> "This kernel fuses the softmax computation for sequences where the entire row fits in a single block (up to ~1024 elements). It demonstrates the **fusion** pattern — the same concept that makes FlashAttention fast. For production sequence lengths where S > SRAM capacity, FlashAttention uses the streaming two-pass algorithm from Ch4: a running `(max, sum)` state across tiles. See Ch4 for that extension; this kernel is its building block."
Update the prerequisite bridge table row accordingly.

---

### Gap 4 — Tiling insight lives in comments, not in a dedicated cell — Priority: Medium

**Problem:** The central insight of Part 2 — "all K-loop partial products stay in registers; only one HBM write per output tile" — lives in code comments and the "What just happened" cell. No standalone cell before the K-loop isolates this insight visually.

**Recommendation:** Add a markdown cell immediately before the K-loop in Part 2 with an ASCII two-column comparison:
```
Without tiling:                    With tiling (Triton):
k=0: compute → write HBM          acc = zeros         (registers)
k=1: read HBM, compute, write      acc += tile_k0      (registers)
k=2: read HBM, compute, write      acc += tile_k1      (registers)
...                                acc += tile_kN      (registers)
                                   write acc → HBM    ← once, at the end
```

---

### Gap 5 — 87% thread fades after the introduction — Priority: Medium

**Problem:** The 87% target appears in the opening, once in Part 4's output, and once in the closing cell. Parts 1, 2, and 3 do not connect their HBM savings to the 87% target — a learner finishing Part 3 cannot see how "fusion saves 50% HBM traffic" compounds toward 87%.

**Recommendation:** Add a single-sentence "bandwidth ledger" at the end of each Part's "What just happened" cell:
- Part 2: "Tiling eliminates all intermediate K-loop writes to HBM. This is the first of the two techniques behind the 87% figure."
- Part 3: "Fusion halves the read/write count per operation. Tiling + fusion together drive FlashAttention from ~40% to 87% HBM utilization."
- Part 4: "At 1 read + 1 write for the full softmax, this is the fused operation FlashAttention executes for every attention head."

---

### Gap 6 — Autotune shows the result but not the mechanism — Priority: Low

**Problem:** After the block-size sweep, the only explanation is "the winning block size depends on your GPU's SRAM and memory bandwidth." Autotuning feels like trial-and-error.

**Recommendation:** Add 4 sentences after the sweep result: "BLOCK=128 wins on A100 because a 128×128 fp16 tile = 32 KB — keeping all SMs busy without register spilling. BLOCK=256 creates a 128 KB tile exceeding the per-SM register file, so the compiler serializes warps. BLOCK=32 underutilizes tensor cores, which prefer tiles of at least 16×16. `@triton.autotune` runs this sweep at first call and caches the winner — no overhead on subsequent calls."

---

### Gap 7 — No mid-notebook vocabulary checkpoint — Priority: Low

**Problem:** By Part 4, learners have absorbed without pause: 1D and 2D `tl.program_id`, `tl.arange`, masked loads, 6 stride parameters, `tl.constexpr`, K-loop accumulation, `tl.dot`, `tl.math.tanh`. Part 3 and Part 4 add more on top.

**Recommendation:** Insert a markdown cell between Parts 2 and 3 — "Triton vocabulary checkpoint: `tl.program_id` = `blockIdx`, `tl.arange` = index range, `tl.load/store` = HBM↔register, `tl.dot` = tiled matmul, `tl.constexpr` = compile-time constant. Everything else in later kernels is a combination of these."

---

## Do NOT Change

- Opening 87% target thread and Part-by-Part "what it proves" TOC
- Prerequisite bridge table
- `TRITON_AVAILABLE` dual-path with annotated pseudocode
- "Predict first" in Parts 2 and 5
- Part 6 torch.compile → Triton loop-closure
- Running example `B=8, S=128, D=64` throughout
- Code comments that do real pedagogical work
- "When to Use What" and Tier 1/2/3 tables
