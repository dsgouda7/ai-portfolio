# Improvement Plan — FlashAttention Internals

**Audited:** 2026-07-26 | **Audience fit:** 6/10

## Overall Assessment

Technically correct, well-structured, and meaningfully connected to the Ch3 profiling callback. The Python tiling implementation passes `torch.allclose`, online softmax numerical stability is verified with extreme values, and the closing one-liner recommendation is exemplary. The primary failure is pedagogical: the reader is given the algorithm before being guided to feel the problem so acutely that tiling is the only remaining option. The "inevitable solution" arc — required by the authoring guide — is absent from both the tiling introduction (Part 2) and the online softmax introduction (Part 3). Additionally: the most mechanistically dense code block (the online softmax update inside the inner loop) is the most sparsely annotated.

---

## Strengths (preserve these)

- **Opening hook** — `> The 60% bottleneck` blockquote with specific metric from prior chapter earns trust immediately
- **Prerequisite bridge table** — maps HBM bandwidth, arithmetic intensity, and Ch3 profiler result to chapter premise
- **Running example threads consistently** — `B=8, D=64` used verbatim Parts 1–6
- **Predict-first exercises in Parts 1, 2, and 5** — three well-crafted prompts with named candidate outcomes
- **SRAM annotations on data-load lines** — "in SRAM" comments communicate the invariant correctly
- **Online softmax correctness verification** — `torch.allclose(..., atol=1e-5)` + extreme-values test
- **Closing decision cell** — Before/After one-liner, conditional dispatch string, four numbered recommendations
- **Three-tier coverage table**
- **Tiling diagram placement** — `flash-attention-tiling.png` appears before the code (correct order)

---

## Gaps & Recommended Changes

### Gap 1 — Tiling appears as an algorithm, not an inevitable derivation — Priority: High

**Problem:** Part 2 opens: "the S×S attention matrix is too large for SRAM, but we can process it in tiles." The reader is handed the solution. No moment where the reader is forced to confront: "SRAM is 228 KB; the S×S matrix is 8 MB; tiles of 32×64 fit; therefore..."

**Recommendation:** Add a 4–6 line markdown bridge between Parts 1 and 2:
> "The S×S score matrix at S=512 is 8 MB. On-chip SRAM on A100 is 228 KB per SM. You cannot hold the entire score matrix in SRAM. But you don't need all of it at once — you only need the slice of scores that contribute to one output tile. What is the smallest self-contained unit of computation? A block of Q rows against a block of K rows."
Then present tiling as the answer the reader just derived.

---

### Gap 2 — IO complexity stated before the reader feels the latency cost — Priority: High

**Problem:** Part 4 opens with "Standard attention requires O(S²) HBM reads." The reader sees the formula and a timing table. They do not feel why memory bandwidth starvation is a crisis worth an entirely new algorithm.

**Recommendation:** Add 3 lines before the IO complexity print block:
> "At 2 TB/s HBM bandwidth, reading 1 GB costs 0.5 ms. Standard attention at S=512 generates ~0.05 GB of S×S traffic (forward alone) — and six times that for a full fwd+bwd pass. On an A100 running at 312 TFLOPS, that bus time is longer than the compute time. The profiler in Ch3 measured exactly this gap."

---

### Gap 3 — Online softmax update block is the most important code and the most sparsely commented — Priority: High

**Problem:** The four lines computing `M_new`, `exp_S`, `L_new`, `O_new` share a single group comment: `# Online softmax update (Part 3)`. The expression `O_new = torch.exp(M_i - M_new).unsqueeze(-1) * O_i + torch.matmul(exp_S, V_j)` is the most important line in the notebook — which term rescales the old accumulator, and which adds this tile's contribution?

**Recommendation:** Replace the four bare lines with fully annotated versions:
```python
# new running max across all j-tiles seen so far for this Q_i block
M_new = torch.maximum(M_i, S_ij.max(dim=-1).values)
# numerically stable exponents: shift by new max to prevent fp overflow
exp_S = torch.exp(S_ij - M_new.unsqueeze(-1))
# correct old denominator for new max, then add this tile's contribution
L_new = torch.exp(M_i - M_new) * L_i + exp_S.sum(dim=-1)
# term 1: rescale old accumulator to account for updated max
# term 2: add this tile's weighted V contribution
O_new = (torch.exp(M_i - M_new).unsqueeze(-1) * O_i +
         torch.matmul(exp_S, V_j))
```
Then add a Code Walkthrough markdown cell explaining the correction factor `exp(M_old - M_new)` in prose.

---

### Gap 4 — Online softmax: no analogy for "running correction produces exact result" — Priority: Medium

**Problem:** Part 3 says "maintain running statistics (m, l) — update as each tile arrives." The reader is told it produces bit-identical results but not given an intuitive model for why a running correction can reconstruct an exact result without re-reading past data.

**Recommendation:** Add 2 sentences before the code: "Think of computing the average salary at a company you're joining one department at a time. When you see department 3 with higher salaries, you correct the running average using a formula rather than re-visiting departments 1 and 2. The correction factor `exp(old_max - new_max)` does exactly that for our running softmax."

---

### Gap 5 — SDPA dispatch test produces no useful signal on CPU — Priority: Medium

**Problem:** The Part 5 dispatch test uses `torch.backends.cuda.sdp_kernel(enable_flash=True, ...)` to check FlashAttention availability. On CPU, all three modes (flash, math, mem_efficient) are either unsupported or irrelevant. A CPU learner sees "CPU: no flash" for every test case and learns nothing about the actual dispatch conditions.

**Recommendation:** Add a note for CPU learners before the test:
> "On CPU: torch.backends.cuda.sdp_kernel is GPU-only. The dispatch conditions shown here (fp16/bf16 → FlashAttention, fp32 → standard) apply only when running on CUDA. Reference values from an A100 run are shown in the predict-first exercise above — those are the actual dispatch thresholds you'll encounter in production."

---

## Do NOT Change

- Opening `> The 60% bottleneck` hook
- Prerequisite bridge table
- Running example `B=8, D=64` throughout
- Predict-first exercises in Parts 1, 2, and 5
- `torch.allclose(..., atol=1e-5)` + extreme-values verification
- Closing decision cell format (Before/After one-liner + numbered recommendations)
- Three-tier coverage table
- `flash-attention-tiling.png` placement (before the code, not after)
