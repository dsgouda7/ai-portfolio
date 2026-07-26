# Improvement Plan — Distributed Training

**Audited:** 2026-07-26 | **Audience fit:** 5/10

## Overall Assessment

Strong skeleton: Riverside constraint drives every Part, the TOC maps a specific question to each section, pipeline bubble is triple-covered (formula + table + diagram), and the LLaMA-2-70B bridge validates that the simplified models are genuine simplifications. Two serious problems: (1) the FSDP memory math is numerically broken — the markdown claims 4× A100 80GB fits a 70B full fine-tuning job, but when executed the closing cell prints "✗ OOM" directly contradicting the static summary; (2) DDP and tensor parallelism both lack the mental-model analogies that prevent overwhelm for engineers new to distributed training.

---

## Strengths (preserve these)

- **Riverside thread end-to-end** — TOC table gives a specific question per Part; final summary maps each finding to a Riverside implication
- **DDP OOM → FSDP transition** — 840 GB derived step-by-step makes the move to FSDP feel inevitable
- **Pipeline bubble triple-coverage** — formula + efficiency table + matplotlib Gantt; engineer catches it in at least one medium
- **Part 5 (3D parallelism) appropriately narrow** — strategy table and four-line recommendation; no Megatron-LM rabbit hole
- **LLaMA-2-70B real-world bridge** — validates simplified models against Meta's actual training config

---

## Gaps & Recommended Changes

### Gap 1 — FSDP memory math is numerically broken — Priority: Critical

**Problem:** The Part 2 markdown states `(140+140+560) GB / 4 ≈ 52.5 GB — fits in 4× A100 80GB!` (dividing by 4 twice = wrong). The code variable `fsdp_4gpu` computes `218.75 GB`. The closing decision cell prints `✗ OOM, need TP too`. The static summary markdown says `✓ FSDP alone fits`. Engineers see a direct contradiction.

**Root cause:** Full fine-tuning of a 70B model requires 840 GB; FSDP across 4× A100 80GB gives 210 GB/GPU. 210 > 80. The scenario as stated is infeasible.

**Recommendation:** Reframe as LoRA fine-tuning. LoRA adapters for 70B are ~500M trainable parameters. The frozen model is loaded in bf16 (140 GB, with 4-bit NF4 further reducing this). Only adapter parameters need full gradients and optimizer states (~4 GB). This is what practitioners actually do on 4× A100 80GB and naturally links back to the Ch2 LoRA prerequisite. Update the Riverside scenario, re-derive memory math for LoRA, fix `fsdp_4gpu` to compute correctly, and verify the closing cell prints `✓ fits` when executed.

---

### Gap 2 — DDP gradient intuition has no analogy before the predict-first question — Priority: High

**Problem:** The predict-first question ("Are gradients identical, summed, or kept separate?") appears before any mental picture is established. A reader unfamiliar with all-reduce will guess rather than derive.

**Recommendation:** Add 3 sentences of analogy between the DDP description and the predict-first question:
> "Think of 4 engineers each reading a different section of Riverside's novel corpus. Each independently computes how to adjust the model (local gradient). Before anyone writes changes down, they compare notes and use the average of all four signals (all-reduce). Every engineer now updates from the same averaged gradient — the model stays in sync."

---

### Gap 3 — Tensor parallelism has no tiny concrete example before the abstract code — Priority: High

**Problem:** Tensor parallelism goes directly from "each GPU holds columns/rows of W" to a 256×1024 verification. An engineer who hasn't seen a 2×4 toy matrix split will not follow the stride arithmetic.

**Recommendation:** Add a 6-line worked example before the code:
> "Concretely: W is (4, 8). GPU 0 holds W[:, 0:4] (left 4 columns). GPU 1 holds W[:, 4:8] (right 4 columns). For input x (batch, 4): GPU 0 computes x @ W[:, 0:4] → partial output of shape (batch, 4). GPU 1 computes x @ W[:, 4:8] → another (batch, 4). Concatenate: (batch, 8) — identical to the full matmul."

---

### Gap 4 — No "🔮 Predict first" exercise for the FSDP memory comparison — Priority: Medium

**Problem:** The Part 2 prediction exercise is absent. The dramatic OOM→FSDP narrative moment has no learner engagement before the reveal.

**Recommendation:** Add a predict-first markdown cell before the FSDP memory analysis:
> "🔮 Before running: FSDP shards all parameters, gradients, and optimizer states across 4 GPUs. The per-GPU memory should be: (a) same as DDP (sharding doesn't help), (b) exactly 4× less than DDP, or (c) roughly 4× less but with ~20% communication overhead? Note your answer."

---

## Do NOT Change

- Riverside thread — specific question per Part; closing summary map
- DDP OOM → FSDP transition (840 GB derivation is correct)
- Pipeline bubble triple-coverage (formula + table + matplotlib Gantt)
- Part 5 three-strategy table (appropriate depth for this audience)
- LLaMA-2-70B training config mapping in Part 6
- "When to Use What" final table
