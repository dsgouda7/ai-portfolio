# AI Compute and Systems Plan

**What this covers:** The engineering that makes large language models practical at scale —
GPU hardware architecture, CUDA programming, training infrastructure, and inference systems.

**Target learner:** Someone who has finished `learning/genai/` and understands Transformers,
fine-tuning, RAG, and LLM gateways at a mechanistic level, and now wants to understand *why*
these systems behave the way they do at the hardware level and how to build or optimize them
in production.

**Directory to implement:** `learning/ai-infrastructure/`
(name rationale: mirrors `notes/07-ai-infrastructure/` which covers the same terrain; "ai-compute"
undersells the inference serving and systems chapters; "infrastructure" captures both the hardware
layer and the serving/orchestration layer above it.)

**Relationship to `notes/07-ai-infrastructure/`:** Most mechanistic content already exists as
rich markdown chapters (detailed narrative, equations, running scenario) and `notebook-supplement.ipynb`
files containing GPU-specific code. The work is to **promote the notes content to full gold-standard
notebooks**: add the pedagogical wrapper (challenge-before-title, roadmap table, threaded running
example, predict-first with candidate outcomes, closing decision) rather than rediscover the tech
content. See the notes coverage map at the end of this file.

---

## Why This Track Exists

After `learning/genai/`, a practitioner can fine-tune GPT-2, build a RAG pipeline, and reason
about LLM gateways. But they cannot answer questions like:

- "Why does LoRA training fit in 8 GB when full fine-tuning needs 40 GB?" (memory math)
- "What exactly does FlashAttention do differently, and why is it faster?" (GPU memory hierarchy)
- "Why does batch size 32 use less GPU memory per example than batch size 1?" (CUDA occupancy)
- "How do I profile a PyTorch model to find the actual bottleneck?" (profiling tools)
- "What does 'tensor parallel' mean, and why does it matter for >7B models?" (distributed training)
- "Why is GGUF faster than the original checkpoint for inference?" (quantization and runtime)

These are not niche questions — they arise the first time a practitioner tries to run a real model
on real hardware.

---

## Chapter Map

### Chapter 1 · GPU Hardware Foundations

**Why first:** Every optimization decision (mixed precision, batching, memory layout) only makes
sense with a mental model of the hardware. Most practitioners have no model at all.

**Narrative frame (Section 8):** *Your fine-tuning job from `04-llm` (30 minutes on a GPU-equipped
machine) takes 6 hours on your laptop. Marketing says they can rent one AWS A10G for an afternoon.
Why will that be 12× faster? This notebook answers that question from first principles.*
Every concept is introduced as answering one of these measurable questions. The closing decision
states: "Your laptop's CPU does {cpu_flops:.0f} GFLOPS. The A10G does {gpu_flops:.0f} GFLOPS.
But the real bottleneck is {actual_bottleneck} — measured below."

**Structural skeleton (Section 2):**
- Title cell: challenge-before-title (the laptop vs. A10G mystery, stated before the heading)
- Roadmap table: 6 Parts with Part → Concept → Key Idea columns
- Parts 1–6 each open with `#### Why does the A10G solve [specific sub-problem]?`
- Toy → real bridge in Part 6: `torch.profiler` trace showing *this notebook's own matmul* dispatching to the specific CUDA kernel
- Closing decision: filled-in CPU vs. GPU benchmark table for the `(B=8, S=128, D=256)` matmul

| Part | Topic | Key demonstration |
|---|---|---|
| 1 | CPU vs. GPU architecture: SIMD vs. SIMT; latency vs. throughput orientation | Matrix multiply timing: CPU (1 core), CPU (8 cores), GPU; log-scale bar chart |
| 2 | GPU memory hierarchy: HBM → L2 → shared memory → registers; sizes and latencies | Bandwidth benchmark: access patterns that hit L2 vs. HBM; print GB/s for each |
| 3 | Warp execution: 32 threads per warp, warp divergence, occupancy | Toy kernel that branches on thread ID; show divergence cost with nvml/PyTorch timer |
| 4 | CUDA grid/block/thread model: `blockIdx`, `threadIdx`, launching a kernel | Write a CUDA kernel for vector addition via `triton` (simpler than raw CUDA from Python); verify output matches CPU |
| 5 | Memory coalescing: why layout matters for throughput | Coalesced vs. strided access pattern; bandwidth comparison |
| 6 | Toy → real bridge: where does PyTorch's `matmul` live in this hierarchy? | `torch.profiler` trace showing `aten::mm` → CUDA kernel name |

**Gold-standard requirements:**
- **Running example:** the `(B=8, S=128, D=256)` attention-shaped matmul is established in Part 1 and appears in every timing cell through Part 6
- ` Predict first` before every timing experiment, with 3 candidate speedup ranges: e.g. *"How much faster is the GPU matmul vs. 8-CPU-core: (a) 2×, (b) 10×, (c) 50×?"*
- ` Your turn`: double batch size; predict whether GPU:CPU speedup ratio grows (it does — larger work fits better in the warp scheduler)
- Every speed claim measured, never asserted; `torch.cuda.synchronize()` before all stop times
- `#### What just happened — and what's missing` after Part 2 (bandwidth measured; missing: *when do we hit the bandwidth ceiling vs. compute ceiling?*) to plant Part 3
- Code Walkthrough cell (Section 9.1) after the `torch.profiler` setup cell in Part 6
- Closing tier-1/2/3 ledger: Tier 1 (CPU/GPU architecture, memory hierarchy, warp model, CUDA basics, coalescing), Tier 2 (Tensor Core operation — explained via diagram, not built), Tier 3 (NVLink, NVSwitch, MIG, multi-GPU topology — named with one-line reason)
- **Closing decision (Section 8.7):** "To explain the 12× speedup on the A10G: {measured breakdown by part} — the dominant factor is {X}. For your specific fine-tuning workload, the bottleneck is the {matmul/memory} operation at layer {N}."

**Subagent implementation task:**
> Create `learning/ai-infrastructure/01-gpu-hardware/gpu-hardware-foundations.ipynb`.
> **Primary source:** `notes/07-ai-infrastructure/ch01-gpu-architecture/gpu-architecture.md` ...
> [full task spec as above]
> Create `images-plan.md` in this chapter directory using the prompts below.

**images-plan.md for this chapter:**

| Asset | Placement | Teaching job |
|---|---|---|
| `gpu-memory-hierarchy.png` | Part 2 intro | HBM → L2 → SRAM → registers as a pyramid; bandwidth and size at each level |
| `roofline-model.png` | Part 3 intro | Arithmetic intensity vs. throughput on log-log axes; "LLM inference is memory-bound" region shaded |
| `warp-simt-execution.png` | Part 3 body | 32-thread warp executing in lock-step; branched threads grayed out to show divergence cost |

```text
[gpu-memory-hierarchy.png]
Flat vector technical diagram, wide 16:9, dark graphite background. A vertical pyramid
with four tiers: HBM at the base (largest, labeled "80 GB, 2 TB/s"), L2 cache above
(labeled "40 MB, 12 TB/s"), SRAM/shared memory above that (labeled "228 KB/SM"),
registers at the apex (labeled "256 KB/SM, fastest"). Each tier uses a different
muted color (teal base, amber middle, coral top). Ivory bandwidth labels on each tier.
No logos, no photorealism, no gradients, no tiny text.

[roofline-model.png]
Flat vector data visualization, wide 16:9, dark graphite background. Log-log axes:
x = "Arithmetic Intensity (FLOP/byte)", y = "Performance (TFLOP/s)". A teal roof-line
bends at the ridge point (label: "164 FLOP/byte, RTX 4090"). Left of ridge: sloped
"memory-bound" region shaded in coral. Right: flat "compute-bound" region in muted
teal. A labeled dot for "LLM single-token decode" sits far left in the memory-bound
region. Ivory labels. No logos, no photorealism, no gradients, no tiny text.

[warp-simt-execution.png]
Flat vector technical diagram, wide 16:9, dark graphite background. A grid of 32 small
ambrose squares (threads) grouped into one warp. A branch condition splits them: 20
threads take "if" path (teal), 12 threads are grayed out (disabled). Two serial passes
shown: pass 1 executes the "if" group, pass 2 executes the "else" group. Arrow shows
this doubles execution time. Ivory labels. No logos, no photorealism, no gradients.
```

---

### Chapter 2 · Mixed Precision and Memory Math

**Why here:** The "why does this fit in GPU memory" question is the single most common
confusion for practitioners moving from tutorial to production.

**Narrative frame (Section 8):** *The Riverside publishing firm's IT team has approved one AWS A10G
(24 GB VRAM) for a 3-day sprint. The team wants to fine-tune three models: GPT-2-Medium (355M),
LLaMA-3-1B, and LLaMA-3-8B. Which ones fit? At which precision? This notebook answers with
measured numbers, not rules of thumb.*
Every concept answers one of Riverside's questions: "Can we fit the 1B model for continued
pretraining?" (Part 1–2), "Will fp16 save us?" (Part 2), "What if we add gradient checkpointing?"
(Part 4). The closing decision names exactly which models Riverside can and cannot train, and why.

**Structural skeleton (Section 2):**
- Title cell: the three models and the 24 GB constraint stated before the heading
- Roadmap table: 6 Parts
- Parts 1–5 each answer one of Riverside's named questions
- Toy → real bridge: the GPT-2 model from `04-llm` is the "toy" reference; LLaMA-3-8B is "real"
- Closing decision: a filled-in Riverside model selection table scored against the 24 GB constraint

| Part | Topic | Key demonstration |
|---|---|---|
| 1 | Memory footprint math: `params × bytes_per_param`; optimizer states; activations; gradients | Print memory breakdown for GPT-2 (124M params) in fp32, fp16, bf16, int8 |
| 2 | fp32 vs fp16 vs bf16: exponent/mantissa trade-offs; overflow risk; why bf16 is preferred for training | Overflow demo: fp16 overflows on large LM gradients; bf16 does not |
| 3 | `torch.autocast` + `GradScaler`: the two-part recipe for stable fp16 training | Train a small model with and without `GradScaler`; show NaN loss without it |
| 4 | Gradient checkpointing / activation recomputation: trade compute for memory | Profile peak memory with and without `torch.utils.checkpoint`; show the trade-off curve |
| 5 | Memory profiling: `torch.cuda.memory_summary()`, `max_memory_allocated()`, `memory_snapshot()` | Step-by-step memory trace of a single forward+backward pass |
| 6 | Toy → real: memory estimate for fine-tuning LLaMA-3-8B in different precision regimes | Fill-in-the-blank calculator cell; connect to LoRA's memory advantage from `04-llm` |

**Gold-standard requirements:**
- **Named scenario threaded throughout** (Riverside; A10G; the three candidate models)
- ` Predict first` before the overflow experiment: *"Will fp16 overflow during a forward pass on GPT-2-Medium with a large batch: (a) never, (b) sometimes depending on the batch, (c) always at batch_size ≥ 32?"* (Answer: b — depends on activation magnitudes)
- ` Predict first` before gradient checkpointing: *"Checkpointing halves peak memory. Does it also halve training time: (a) yes, (b) no, it adds ~30% compute overhead, (c) it actually speeds things up?"* (Answer: b)
- Code Walkthrough cell (Section 9.1) after the `torch.autocast` + `GradScaler` training cell
- `#### What just happened — and what's missing` after Part 3 (stable fp16 training achieved; missing: *activation memory grows with sequence length* — plants Part 4)
- Cross-reference to LoRA section in `04-llm/02-llm-finetuning-parameter-techniques.ipynb`
- Closing tier-1/2/3 ledger: Tier 1 (fp32/fp16/bf16 precision, GradScaler, gradient checkpointing, memory profiling), Tier 2 (offloading to CPU RAM — explained as the trade-off, not built), Tier 3 (ZeRO-Offload, activation quantization — named)
- **Closing decision (Section 8.7):** "Riverside's A10G model selection: GPT-2-Medium at fp32 — {gpt2_fp32_gb:.1f} GB (fits); LLaMA-3-1B at bf16+checkpointing — {llama1b_gb:.1f} GB ({fits_str}); LLaMA-3-8B at bf16+LoRA — {llama8b_lora_gb:.1f} GB ({fits_str_8b}). Recommendation: {measured_recommendation}."

**Subagent implementation task:**
> Create `learning/ai-infrastructure/02-mixed-precision/mixed-precision-and-memory.ipynb`.
> **Primary source:** `notes/07-ai-infrastructure/ch02-memory-and-compute-budgets/memory-budgets.md`...
> [full spec as above]
> Create `images-plan.md` using the prompts below.

**images-plan.md for this chapter:**

| Asset | Teaching job |
|---|---|
| `memory-footprint-breakdown.png` | Stacked bar for GPT-2-Medium at fp32: parameters + gradients + optimizer states + activations; repeat for bf16 |
| `fp32-fp16-bf16-number-line.png` | Three number lines comparing exponent bits vs. mantissa bits; overflow threshold annotated |
| `gradient-checkpointing-tradeoff.png` | Two-axis chart: x=checkpointing frequency, y1=peak memory (falling), y2=compute overhead (rising); sweet spot marked |

```text
[memory-footprint-breakdown.png]
Flat vector stacked bar chart, wide 16:9, dark graphite background. Two side-by-side
bars: "fp32" and "bf16". Each bar stacked with segments: coral "parameters", amber
"gradients", teal "optimizer states", muted purple "activations". Ivory segment
labels showing GB. The bf16 bar is roughly 2x shorter. Title: "GPT-2-Medium (355M
params) memory footprint". No logos, no photorealism, no gradients, no tiny text.

[fp32-fp16-bf16-number-line.png]
Flat vector technical infographic, wide 16:9, dark graphite background. Three rows:
fp32, fp16, bf16. Each row shows a bit-layout diagram (32/16/16 boxes) split into
sign, exponent, mantissa. Exponent boxes in amber, mantissa in teal. A vertical coral
line marks the overflow threshold for fp16. A label shows bf16 has the same exponent
width as fp32. Ivory labels. No logos, no photorealism, no gradients, no tiny text.

[gradient-checkpointing-tradeoff.png]
Flat vector dual-axis line chart, wide 16:9, dark graphite background. X-axis:
checkpointing frequency (every 1 to every 8 layers). Left y-axis teal line: peak
memory (GB), falling from left to right. Right y-axis coral line: compute overhead
(%), rising from left to right. An amber vertical dashed line marks the sweet spot
where memory savings plateau and compute cost begins rising steeply. Ivory labels.
```

---

### Chapter 3 · PyTorch Profiling and Bottleneck Analysis

**Why here:** Without profiling tools, optimization is guesswork. This chapter teaches the
measurement habit *before* the optimization techniques — the same "prove, don't assert" principle
that runs through the entire genai track.

**Narrative frame (Section 8):** *Your fine-tuning loop from `04-llm` takes 45 seconds per epoch
on the machine you have. Someone tells you it should take 8 seconds. Who's right — and what
actually takes the time?* The notebook answers this by profiling a real `04-llm`-style fine-tuning
step end-to-end. Every profiling tool is introduced as a way to answer one specific question:
"Is the bottleneck the data loader, the forward pass, the backward pass, or the optimizer step?".
The closing decision names the actual bottleneck measured in this run and the one change with the
greatest expected speedup.

**Structural skeleton (Section 2):**
- Title cell: the 45-second vs. 8-second mystery stated before the heading
- Roadmap table with one row per profiling tool
- Each Part: "Before you instrument: what do you *expect* to be slow?" → profile → "What did you find?" → "What's still missing?"
- Closing decision: filled-in bottleneck table from this specific run

| Part | Topic | Key demonstration |
|---|---|---|
| 1 | `torch.profiler`: capturing a CPU+CUDA trace; reading the Chrome trace viewer | Profile a Transformer forward pass; identify the top-3 most expensive operators |
| 2 | `torch.autograd.profiler.profile` for targeted cell-level profiling | Compare profiling overhead: full profiler vs. targeted wrapper |
| 3 | Identifying bottlenecks: compute-bound vs. memory-bound operations | Benchmark two implementations of the same attention pattern; identify the memory-bound one |
| 4 | `torch.compile` (PyTorch 2.0): when it helps, when it doesn't | Time a Transformer block with and without `torch.compile`; show the warm-up cost |
| 5 | `nvtx.range_push/pop` for custom profiling regions | Instrument a training loop with custom regions; show them in the timeline |
| 6 | Toy → real: profile a LoRA fine-tuning step from `04-llm` | Show exactly where time is spent: data loading, forward, backward, optimizer step |

**Gold-standard requirements:**
- **Running example:** a LoRA fine-tuning step using the same model and adapter from `04-llm/02`, reloaded from disk at the top of this notebook (Section 13.2 save/reload pattern)
- ` Predict first` before Part 1 profiling result: *"Rank the four phases by time: (a) data_load > forward > backward > optimizer, (b) backward > forward > data_load > optimizer, (c) optimizer > backward > data_load > forward"* (correct answer varies by actual hardware — that's the lesson)
- ` Your turn`: disable `torch.compile`; measure whether the backward pass changes proportion
- `#### What just happened` cells after Parts 1, 3, and 6
- Code Walkthrough cell (Section 9.1) after the `torch.profiler` setup block (30+ lines)
- Two-sided health check (Section 14.2): full profiler overhead vs. no instrumentation
- Closing tier-1/2/3 ledger: Tier 1 (torch.profiler, autograd profiler, torch.compile, nvtx), Tier 2 (PyTorch Memory Snapshot — explained, not built here), Tier 3 (Nsight Systems, Perfetto, DCGM — named)
- **Closing decision (Section 8.7):** "For the 04-llm fine-tuning step: the actual bottleneck is {bottleneck} at {pct:.0f}% of wall time. The single change with the greatest expected speedup is {recommendation} — expected to save {expected_saving:.0f}%."

**Subagent implementation task:**
> Create `learning/ai-infrastructure/03-profiling/pytorch-profiling.ipynb`.
> **Notes coverage:** `notes/07-ai-infrastructure/ch01-gpu-architecture/notebook.ipynb` may
> have basic timing code; extract what exists. Remainder is fresh.
> Open with the 45-second mystery. Create `images-plan.md` using prompts below.
> Apply ALL authoring guide conventions.

**images-plan.md for this chapter:**

| Asset | Teaching job |
|---|---|
| `profiler-timeline-annotated.png` | Chrome trace timeline screenshot annotated with "data_load", "forward", "backward", "optimizer" bands |
| `compute-vs-memory-bound.png` | Two bar charts: compute-bound op (GPU util high, memory bandwidth low) vs. memory-bound op (reversed) |

```text
[profiler-timeline-annotated.png]
Flat vector technical diagram styled as a browser dev-tools timeline, wide 16:9, dark
graphite background. Four colored horizontal bands on a time axis: teal "data_load",
amber "forward pass", coral "backward pass", muted purple "optimizer step". Width
proportional to typical relative durations. Ivory time labels. Callout arrows pointing
to the widest band. Title: "Single fine-tuning step breakdown". No logos, no
photorealism, no gradients, no tiny text.

[compute-vs-memory-bound.png]
Flat vector side-by-side bar chart, wide 16:9, dark graphite background. Two panels.
Left "Compute-bound" (matmul): tall amber GPU-utilization bar, short teal memory-
bandwidth bar. Right "Memory-bound" (attention softmax): tall teal memory bar, short
amber compute bar. Labels show actual percentages. Subtitle: "The bottleneck determines
which hardware spec to optimize". Ivory labels. No logos, no photorealism.
```

---

### Chapter 4 · FlashAttention: Inside the Algorithm

**Why here:** FlashAttention is the single most impactful algorithmic improvement in practical
LLM inference and training, and almost no practitioners understand *why* it is faster (not an
approximation — it reorders computation to avoid HBM bandwidth bottlenecks).

**Narrative frame (Section 8):** *You just profiled your `02-transformers` attention function
(Chapter 3 of this track). The profiler says the `matmul(Q, K.T)` line accounts for 60% of
forward-pass time at S=512. PyTorch 2.0's `scaled_dot_product_attention` runs the same
calculation 3× faster. The only difference: it doesn't materialize the full S×S matrix in HBM.
This notebook explains exactly why that matters.*
Every Part answers one question: "What exactly is slow about standard attention?" (Part 1),
"How does tiling help?" (Part 2–3), "Is this actually faster on our machine?" (Parts 4–5),
"What else does tiling unlock?" (Part 6 — GQA/MQA).

**Structural skeleton (Section 2):**
- Title cell: the 60% bottleneck finding from Chapter 3, stated before the heading
- Roadmap table: 6 Parts
- Running example: the same `(B=8, S=128→512′2048, D=64)` attention head from Ch 1/3, scaling up across the chapter
- Toy → real bridge in Part 5: show `scaled_dot_product_attention`'s dispatch conditions; verify our tiling implementation produces identical output

| Part | Topic | Key demonstration |
|---|---|---|
| 1 | Standard attention's memory problem: the $O(S^2)$ intermediate matrix in HBM | Measure HBM traffic for standard attention vs. sequence length; show quadratic growth |
| 2 | Tiling insight: compute attention in blocks, keep intermediates in SRAM | Walk through Algorithm 1 of the Flash Attention paper in Python (readable pseudocode) |
| 3 | Online softmax trick: how to compute softmax without materializing the full $S×S$ matrix | Implement and verify against `torch.softmax` on random data |
| 4 | IO complexity: why FlashAttention is $O(S^2/M)$ vs. $O(S^2)$ HBM reads | Measure actual HBM traffic reduction using profiler; match to theory |
| 5 | `scaled_dot_product_attention` (PyTorch 2.0): when it dispatches to FlashAttention | Test conditions that enable vs. disable FlashAttention dispatch; time the difference |
| 6 | GQA (Grouped Query Attention) and MQA: reducing KV cache size without accuracy loss | Compare KV cache memory at S=2048 for MHA vs. GQA-8 vs. MQA |

**Gold-standard requirements:**
- ` Predict first` before IO measurement: *"Standard attention reads the S×S matrix N times during the backward pass (for grad of Q, K, V separately). Tiled FlashAttention avoids materializing it. Which reads more HBM: (a) standard 5× more, (b) tiled 3× more, (c) tiled 10× less?"* (Answer: a)
- ` Predict first` before `scaled_dot_product_attention` dispatch: *"At causal mask + fp32 + S=512: does PyTorch 2.0 dispatch to FlashAttention?"* (Answer: no — only fp16/bf16 triggers it; fp32 uses the standard path)
- ` Your turn`: double S from 512 to 1024; predict the IO traffic ratio change (it doubles — print confirms it)
- **Prove the tiling**: the online softmax implementation must produce output that passes `torch.allclose(..., atol=1e-5)` against `torch.nn.functional.scaled_dot_product_attention`
- `#### What just happened` after Part 3 (numerically equivalent but O(S²/M) reads; missing: *we still haven't measured the actual speedup* — plants Part 4)
- Code Walkthrough cell after the tiling algorithm implementation (Section 9.1)
- Closing tier-1/2/3 ledger: Tier 1 (IO complexity, tiling, online softmax, FlashAttention dispatch), Tier 2 (GQA/MQA mechanism — built at toy scale), Tier 3 (FlashAttention-3, Ring Attention, Sliding Window Attention — named)
- **Closing decision (Section 8.7):** "For the S=512 attention head from Chapter 3: standard attention read {std_hbm:.1f} GB from HBM; tiled FlashAttention read {flash_hbm:.1f} GB ({ratio:.1f}× less). The dispatch condition for `scaled_dot_product_attention` on this machine: {dispatch_conditions}. Recommendation: {recommendation}."

**Subagent implementation task:**
> Create `learning/ai-infrastructure/04-flash-attention/flash-attention-internals.ipynb`.
> **Primary source:** `notes/07-ai-infrastructure/ch05-inference-optimization/inference-optimization.md`
> covers FlashAttention IO complexity and tiling; draw the conceptual framing from there.
> Also check `notes/07-ai-infrastructure/ch05-inference-optimization/notebook-supplement.ipynb`
> for any existing attention benchmark code. The tiling algorithm implementation and online
> softmax are fresh content (the notes describe FlashAttention but don't walk through the
> Python tiling implementation).
> **What to keep from notes:** The IO complexity derivation, the roofline framing (compute-bound
> vs. memory-bound), and any existing benchmark cells. **What to add:** The tiling walkthrough
> in clean, readable Python that passes `torch.allclose`; the predict-first questions; the
> closing decision.
> Open with the 60%-bottleneck finding from Chapter 3 (or a reference to it).
> Apply ALL authoring guide conventions.

**images-plan.md for this chapter:**

| Asset | Teaching job |
|---|---|
| `standard-attention-io.png` | Standard attention HBM data flow; each roundtrip shown as a slow red arrow |
| `flash-attention-tiling.png` | Tiled attention keeping intermediates in SRAM; no N×N in HBM |
| `kv-cache-memory-gqa.png` | Side-by-side KV cache at S=2048: MHA vs. GQA-8 vs. MQA memory |

```text
[standard-attention-io.png]
Flat vector systems diagram, wide 16:9, dark graphite background. Q, K, V in HBM
(teal boxes) → large coral box "S = QKᵀ (N×N)" in HBM → P=softmax(S) in HBM →
O=PV in HBM. Fat slow red arrows between every HBM operation. Title: "Standard
attention: O(N²) HBM reads". Ivory labels. No logos, no photorealism.

[flash-attention-tiling.png]
Flat vector systems diagram, wide 16:9, dark graphite background. Q, K, V in HBM
(teal). Small amber SRAM block labeled "tile (BLOCK×d)". Fast teal looping arrow
inside SRAM; no large N×N matrix in HBM. Output O written directly to HBM.
Title: "FlashAttention: O(N²/M) HBM reads". Ivory labels. No logos.

[kv-cache-memory-gqa.png]
Flat vector bar chart, wide 16:9, dark graphite background. Three bars: MHA (coral,
tallest), GQA-8 (teal, 8× shorter), MQA (amber, shortest). Y-axis: KV cache GB at
S=2048 for a 7B model. Percentage reduction on each bar. Title: "KV cache size at
2048 tokens". Ivory labels. No logos, no photorealism.
```

**Why here:** Most LLMs over 7B parameters cannot be trained on a single GPU. A practitioner
needs to understand the three axes of parallelism (data, tensor, pipeline) to reason about
training recipes.

**Narrative frame (Section 8):** *The Riverside team's success with the 7-novel corpus lands them
a grant for a 70B model — but the full pre-training job requires 4 A100s. Your current single-GPU
code won't run it. The IT team asks: "Which parallelism strategy do we need, and how much of the
code changes?"* Every parallelism strategy is introduced as answering one of IT's questions:
"How do we use all 4 GPUs at once?" (DDP), "We still run out of memory even with 4 GPUs"
(FSDP), "The model itself is too large for one GPU's layers" (tensor/pipeline parallel).
The closing decision names the exact strategy for the 70B job and how much code it requires.

**Structural skeleton (Section 2):**
- Title cell: the 4-A100 + 70B constraint stated before the heading
- Roadmap table: 6 Parts
- Single running example: a toy 500M-parameter 2-layer Transformer (small enough to run on CPU process groups; large enough to show memory effects)
- Toy → real bridge in Part 6: map toy parallelism choices to the actual LLaMA-2-70B training config

| Part | Topic | Key demonstration |
|---|---|---|
| 1 | Data parallel (DDP): gradient all-reduce across GPUs | Run DDP on 2 GPUs (or simulate with CPU groups); verify gradients are identical after sync |
| 2 | FSDP (Fully Sharded Data Parallel): shard parameters, gradients, and optimizer states | Compare peak GPU memory: DDP vs. FSDP for a 1B parameter toy model |
| 3 | Tensor parallelism: split attention heads across GPUs | Toy example: split a 4-head MHA across 2 GPUs; verify output matches single-GPU |
| 4 | Pipeline parallelism: split layers across GPUs; micro-batching | Illustrate bubble overhead with a 4-stage pipeline and different micro-batch counts |
| 5 | 3D parallelism: data + tensor + pipeline combined (Megatron-LM style) | Table: which axis scales at which model/GPU count regime |
| 6 | Toy → real: what parallelism strategy does a 70B model training recipe use? | Walk through a public LLaMA-2 training config; map each setting to a parallelism axis |

**Gold-standard requirements:**
- ` Predict first` before DDP gradient sync: *"After DDP backward, are the gradients on GPU 0 and GPU 1: (a) identical, (b) summed (twice the magnitude), (c) averaged?"* (Answer: a — averaged by default; print `grad_gpu0 - grad_gpu1` to prove zero difference)
- ` Predict first` before FSDP memory comparison: *"FSDP with 4 GPUs shards parameters 4×. Peak memory per GPU should be: (a) same as DDP, (b) ~4× less than DDP, (c) ~2× less (because optimizer states are also sharded)?"* (Answer: c in practice, due to all-gather overhead)
- ` Your turn`: change DDP gradient bucket size (`bucket_cap_mb`); measure whether it changes throughput
- GPU gating: all multi-GPU cells skip gracefully with CPU process groups; print "Multi-GPU not available — simulating with CPU process groups (gradient math identical, timing not representative)"
- `#### What just happened` after Part 1 (DDP works; missing: *it still requires all parameters to fit on one GPU*) and after Part 2 (FSDP solves that; missing: *but what if a single layer's parameters don't fit?* — plants tensor parallelism)
- Prerequisite bridge cell (Section 13.1) from Chapter 2 (memory math is the foundation)
- Code Walkthrough cell (Section 9.1) after the FSDP configuration block
- Closing tier-1/2/3 ledger: Tier 1 (DDP, FSDP, basic tensor parallelism), Tier 2 (pipeline parallelism — explained with bubble diagram, not built end-to-end), Tier 3 (Megatron-LM, DeepSpeed ZeRO-3, sequence parallelism — named)
- **Closing decision (Section 8.7):** "For Riverside's 70B job on 4 A100s: {recommended_strategy}. Code change from the single-GPU baseline: {change_description}. Estimated time to first checkpoint: {time_estimate} (using reference training throughput from the LLaMA-2 paper)."

**Subagent implementation task:**
> Create `learning/ai-infrastructure/05-distributed-training/distributed-training.ipynb`.
> **Primary source:** `notes/07-ai-infrastructure/ch04-parallelism-and-distributed-training/parallelism.md`
> provides the full mechanistic content (DDP, FSDP, tensor/pipeline parallelism, 3D parallelism)
> and the training scenario. Draw from
> `notes/07-ai-infrastructure/ch04-parallelism-and-distributed-training/notebook-supplement.ipynb`
> for any GPU-specific distributed training code.
> **What to keep from notes:** The parallelism taxonomy (data/tensor/pipeline), the memory
> accounting for FSDP vs. DDP, and any LLaMA training config references.
> **What to replace/add:** Adapt the notes' scenario to the Riverside 70B grant scenario for
> narrative continuity; add the predict-first questions with candidate outcomes; add the closing
> decision. The notes' parallelism markdown is rich on mechanism but has no gold-standard
> pedagogical wrapper — that's the main addition.
> Open with the Riverside 70B scenario. CPU process group fallback prints the stated explanation.
> Apply ALL authoring guide conventions.

**images-plan.md for this chapter:**

| Asset | Teaching job |
|---|---|
| `ddp-gradient-allreduce.png` | 4-GPU ring; gradient tensors flowing around the ring to be averaged |
| `fsdp-vs-ddp-memory.png` | DDP (full params on each GPU) vs. FSDP (sharded params + all-gather) |
| `parallelism-strategy-matrix.png` | 2D grid: model size vs. GPU count; each cell shows recommended strategy |

```text
[ddp-gradient-allreduce.png]
Flat vector systems diagram, wide 16:9, dark graphite background. Four GPU boxes
in a ring connected by teal arrows. Each GPU box shows an amber gradient tensor.
A circular all-reduce arrow flows clockwise. After all-reduce, each GPU's gradient
is identical. Ivory labels. No logos, no photorealism, no gradients, no tiny text.

[fsdp-vs-ddp-memory.png]
Flat vector comparison diagram, wide 16:9, dark graphite background. Left "DDP":
4 GPU boxes each with a full amber model copy. Right "FSDP": 4 GPU boxes each
with a small teal shard; coral arrows show all-gather when a layer is needed.
Memory labels: DDP=4×, FSDP=1×+comm. Ivory labels. No logos, no photorealism.

[parallelism-strategy-matrix.png]
Flat vector heatmap grid, wide 16:9, dark graphite background. Y-axis: model size
(1B, 7B, 13B, 70B, 175B). X-axis: GPU count (1, 4, 8, 16, 64). Each cell: strategy
label ("DDP", "FSDP", "Tensor+Data", "3D") in ivory. Light teal (simple) to amber
(complex) coloring. No logos, no photorealism, no gradients, no tiny text.
```

**Why here:** Quantization is the primary lever for deploying large models on consumer hardware.
The `04-llm` chapter introduced dynamic quantization as a deployment trick; this chapter explains
it mechanistically and covers the production methods.

**Narrative frame (Section 8):** *Riverside wants to ship the editing assistant as a standalone
app that runs on authors' MacBooks (Apple Silicon, 16 GB unified memory, no data leaves the
machine). The 7B model in bf16 is 14 GB — barely fits, crushes the system, no memory for anything
else. The question: how far can you compress the model before the editing quality degrades past
acceptable?* Every technique is introduced as answering one of Riverside's questions: "Does int8
actually hurt quality?" (Parts 1–3), "int4 fits in 4 GB — is it still useful?" (Parts 4–5),
"What's the fastest format on Apple Silicon?" (Part 6), "Why does QLoRA use NF4 specifically?"
(Part 7).

**Structural skeleton (Section 2):**
- Title cell: the MacBook + 14 GB problem stated before the heading
- Roadmap table: 7 Parts
- Running example: the GPT-2-Medium model fine-tuned in `04-llm`, re-quantized in every Part
- Toy → real bridge: the same perplexity evaluation set from `04-llm/03` used as the quality metric throughout
- Closing decision: a Riverside model selection table (bf16 / int8 / GPTQ-int4 / GGUF-Q4_K_M) scored against quality threshold and memory footprint

| Part | Topic | Key demonstration |
|---|---|---|
| 1 | Quantization basics: map fp32 → int8 with scale+zero_point; rounding error | Show tensor before/after; print max absolute error; plot error distribution |
| 2 | Dynamic PTQ: quantize activations at inference time | Time and memory: fp32 model vs. dynamic int8 model on GPT-2 inference |
| 3 | Static PTQ with calibration: run calibration set, lock scale/zero_point | Calibrate on 50 sentences; compare perplexity to fp32 baseline |
| 4 | GPTQ: one-shot weight quantization with Hessian correction | Walk through the GPTQ algorithm conceptually; use a pre-quantized checkpoint via `auto-gptq` |
| 5 | AWQ (Activation-Aware Weight Quantization): protect salient channels | Compare AWQ vs. GPTQ perplexity at int4; show salient-channel identification |
| 6 | GGUF / llama.cpp: CPU-friendly quantization formats and runtime | Run a GGUF model locally; explain Q4_K_M format; compare tokens/second vs. fp16 |
| 7 | NF4 (normalized float 4): the data type used in QLoRA | Show NF4's non-uniform bucket distribution; compare to int4's uniform grid |

**Gold-standard requirements:**
- **Named scenario threaded throughout** (Riverside MacBook; 16 GB; the 7B editing assistant)
- **Running example:** GPT-2-Medium from `04-llm` fine-tuned checkpoint (or a fallback fresh GPT-2-Medium if checkpoint unavailable); Prerequisite Bridge cell (Section 13.1) noting this dependency
- ` Predict first` before dynamic int8 quality: *"Dynamic quantization halves the model size. Will held-out perplexity: (a) stay within 0.5 points of fp32, (b) increase by 2–3 points, (c) increase by 10+ points?"* (Answer: a for dynamic int8 on GPT-2)
- ` Predict first` before GPTQ vs. AWQ comparison: *"At int4, which method will show lower perplexity: (a) GPTQ, (b) AWQ, (c) within noise?"* (Answer: b in most benchmarks; print the actual measured difference)
- `#### What just happened` after Part 3 (PTQ static is good for int8; missing: *int4 static PTQ degrades much faster than int8 — the Hessian correction in GPTQ exists to fix this* — plants Part 4)
- Code Walkthrough cell after the GPTQ calibration block (Section 9.1)
- NF4 section explicitly connects to QLoRA in `04-llm/02-llm-finetuning-parameter-techniques.ipynb` with a forward reference
- Closing tier-1/2/3 ledger: Tier 1 (dynamic PTQ, static PTQ, GPTQ, GGUF runtime, NF4), Tier 2 (AWQ salient-channel mechanism — explained via diagram, not built from scratch), Tier 3 (SpQR, SmoothQuant, QuIP— named)
- **Closing decision (Section 8.7):** "For Riverside's MacBook (16 GB): dynamic int8 ({int8_gb:.1f} GB, perplexity {int8_ppl:.1f}) — fits, quality {int8_verdict}; GGUF Q4_K_M ({gguf_gb:.1f} GB, {gguf_tps:.0f} tok/s) — {gguf_verdict}. Recommendation: {recommendation_with_rationale}."

**Subagent implementation task:**
> Create `learning/ai-infrastructure/06-quantization/quantization-in-depth.ipynb`.
> **Primary source:** `notes/07-ai-infrastructure/ch03-quantization-and-precision/quantization.md`
> provides the full mechanistic content (PTQ, GPTQ, AWQ, GGUF, NF4 data type) and likely the
> InferenceBase deployment scenario. Draw from
> `notes/07-ai-infrastructure/ch03-quantization-and-precision/notebook-supplement.ipynb`
> for GPU-specific quantization code (int8/int4 benchmarks).
> **Secondary source:** `notes/02-bridging-to-transformers/ch10-pruning-mixed-precision/notebook.ipynb`
> has mixed precision + quantization-aware content that may complement Part 2 (fp16/bf16) and
> Part 3 (static PTQ calibration).
> **What to keep from notes:** The quantization math (scale/zero_point, rounding error analysis),
> the GPTQ/AWQ mechanism descriptions, the NF4 non-uniform bucket explanation.
> **What to replace/add:** Use Riverside MacBook scenario (instead of InferenceBase) for narrative
> continuity with `learning/genai/`; add predict-first questions; add closing decision that names
> which models fit in 16 GB and with what quality.
> Open with the Riverside MacBook scenario. Reload GPT-2 from `04-llm` checkpoint if available.
> Apply ALL authoring guide conventions.

**images-plan.md for this chapter:**

| Asset | Teaching job |
|---|---|
| `quantization-rounding-error.png` | fp32 value mapped to int8 bucket; scale, zero_point, and residual error annotated |
| `gptq-vs-awq-perplexity.png` | Bar chart: fp32, dynamic int8, GPTQ int4, AWQ int4 perplexity |
| `gguf-quantization-formats.png` | Table: Q4_K_M / Q5_K_M / Q8_0 / F16 with memory and quality bars |

```text
[quantization-rounding-error.png]
Flat vector technical diagram, wide 16:9, dark graphite background. A number line:
floating-point (continuous) above, int8 (discrete buckets) below. fp32 value 0.347
maps via teal arrow to int8 bucket 44. Coral gap shows rounding error. Scale and
zero_point in amber. Ivory labels. No logos, no photorealism, no gradients.

[gptq-vs-awq-perplexity.png]
Flat vector horizontal bar chart, wide 16:9, dark graphite background. Four bars:
fp32 (teal baseline), dynamic int8 (teal, +0.1), GPTQ int4 (amber, longer), AWQ
int4 (coral, between int8 and GPTQ). X-axis: perplexity. Delta labels vs. fp32.
Vertical baseline reference line. Ivory labels. No logos, no photorealism.

[gguf-quantization-formats.png]
Flat vector comparison table, wide 16:9, dark graphite background. Four rows:
Q4_K_M, Q5_K_M, Q8_0, F16. Columns: name, bits/weight (amber bar), GB for 7B
model, relative quality (teal bar). Q4_K_M row highlighted in coral. Ivory labels.
```

**Why here:** Understanding continuous batching, the KV cache, and speculative decoding is
essential for anyone deploying an LLM service — these topics explain performance characteristics
that are otherwise mysterious.

| Part | Topic | Key demonstration |
|---|---|---|
| 1 | KV cache: store past K and V projections; avoid recomputing attention on the prompt | Implement a KV-caching forward pass; measure tokens/second with and without cache |
| 2 | Continuous batching: how vLLM serves requests of varying length efficiently | Simulate a request queue with variable lengths; show static batching vs. continuous batching throughput |
| 3 | PagedAttention: the KV cache fragmentation problem and its solution | Illustrate page table for KV blocks; measure GPU utilization before/after |
| 4 | Speculative decoding: use a small draft model to propose tokens; verify with the main model | Implement a toy speculative decoding loop; measure speedup at different acceptance rates |
| 5 | Prefill vs. decode phases: why they have different optimal batch sizes | Profile a prompt prefill vs. a token-by-token decode on the same model |
| 6 | Toy → real: run a vLLM or TGI server; connect the architecture concepts to real metrics | Show how `vllm serve` metrics (throughput, TTFT, TPOT) map to the concepts above |

**Structural skeleton (Section 2):**
- Title cell: "Riverside's gateway is live. Marketing just told you: traffic launches next week at 100× today's load. What breaks first?" — stated before the heading
- Roadmap table: 6 Parts, each labelled with the metric it improves (throughput, latency, memory, latency at scale, GPU utilization, production)
- Running example: the `GPT2LMHeadModel` from `04-llm`, extended with a KV cache in Part 1, then scaled through Parts 2–5
- Toy → real bridge: connect each mechanism to the matching `vllm` metric (`throughput tokens/s`, `TTFT`, `TPOT`) in Part 6
- Closing decision: a 6-row capacity table showing which combination of optimizations gets from 10 req/min to 1000 req/min

**Gold-standard requirements:**
- **Named scenario threaded throughout** (Riverside's 100× traffic spike; GPT-2 as the toy model; vLLM as the production target)
- Prerequisite bridge cell (Section 13.1) from `04-llm/06-llm-gateway.ipynb` (the gateway is the starting point)
- ` Predict first` before KV cache speedup: *"Without caching, a 128-token prompt requires attention over 128 positions for every new token. With caching, each new token attends over how many new positions: (a) 128, (b) 1, (c) 1 + a cache lookup?"* (Answer: c — 1 new key/value pair computed, rest from cache)
- ` Predict first` before speculative decoding: *"A 7B verifier model and a 70M draft model: the draft proposes 5 tokens and the verifier accepts 3. Compared to 5 sequential verifier calls: (a) 2× faster, (b) 5× faster, (c) only marginally faster?"* (Answer: a–b depending on acceptance rate — print the actual measured speedup)
- ` Your turn`: change speculative decoding draft length from 4 to 8; predict whether mean acceptance rate goes up or down (it goes down — longer chains are harder to accept wholesale)
- KV cache section explicitly connects to `02-transformers` KV cache description (the "it was mentioned there as an optimization" payoff)
- `#### What just happened` after Part 1 (KV cache solves per-token latency; missing: *at 100× traffic, we're still processing one request at a time* — plants continuous batching)
- Code Walkthrough cell (Section 9.1) after the KV-caching forward pass implementation
- Closing tier-1/2/3 ledger: Tier 1 (KV cache, continuous batching, speculative decoding, prefill/decode profiling), Tier 2 (PagedAttention mechanism — explained via diagram, not built), Tier 3 (tensor-parallel inference, disaggregated prefill-decode, chunked prefill — named)
- **Closing decision (Section 8.7):** "For Riverside's 100× traffic target: naïve serving reaches {naive_tps:.0f} tokens/sec; KV cache alone: {kv_tps:.0f} tok/s ({kv_mult:.1f}×); continuous batching: {cb_tps:.0f} tok/s ({cb_mult:.1f}×); speculative decoding: {sd_tps:.0f} tok/s ({sd_mult:.1f}×). To reach {target_tps:.0f} tok/s for 1000 req/min: {recommendation}."

**Subagent implementation task:**
> Create `learning/ai-infrastructure/07-inference-systems/inference-systems.ipynb`.
> **Primary source:** `notes/07-ai-infrastructure/ch05-inference-optimization/` and `ch06/` and
> `ch12/`. Use Riverside 100× traffic scenario. KV cache must be real PyTorch forward pass.
> Create `images-plan.md` using prompts below. Apply ALL authoring guide conventions.

**images-plan.md for this chapter:**

| Asset | Teaching job |
|---|---|
| `kv-cache-mechanism.png` | Step-by-step: prompt tokens run once (prefill); each new token appends one K/V pair (decode); cached K/Vs shown as a growing teal list |
| `continuous-batching-vs-static.png` | Timeline: static batching idles while short requests wait for long ones; continuous batching slots in new requests as soon as a slot frees |
| `speculative-decoding-accept-reject.png` | Draft model proposes 5 tokens; verifier accepts 3, rejects 2; one verifier call replaces 3 sequential decode calls |

```text
[kv-cache-mechanism.png]
Flat vector sequence diagram, wide 16:9, dark graphite background. Left: "Prefill"
phase showing all prompt tokens processed in parallel (teal arrows to K/V cache).
Right: "Decode" phase showing one new token generated per step, appending one K/V
pair (amber arrow) to the growing teal cache list. The cache grows step by step.
Ivory labels. No logos, no photorealism, no gradients, no tiny text.

[continuous-batching-vs-static.png]
Flat vector timeline diagram, wide 16:9, dark graphite background. Top half "Static
batching": a long request and a short request start together; the slot for the short
request sits empty (coral hatching) after it finishes, waiting for the long one.
Bottom half "Continuous batching": as soon as the short request finishes, a new
request (amber) fills the slot immediately. Efficiency labels on right. Ivory text.

[speculative-decoding-accept-reject.png]
Flat vector pipeline diagram, wide 16:9, dark graphite background. Left: a small teal
"Draft" model proposes 5 tokens in sequence. Right: a large amber "Verifier" model
checks all 5 in one parallel forward pass. Tokens 1-3 are accepted (teal check), 4-5
are rejected (coral cross). The accepted 3 tokens are output. Time comparison shows
3x draft calls replaced 1 verifier call. Ivory labels.
```

---

### Chapter 8 · Custom Kernels with Triton

**Why here:** Triton bridges the gap between "I understand what CUDA does" (Chapter 1) and
"I can write a fast custom operator" without requiring raw C++ CUDA. A practitioner who can
write a Triton kernel can implement FlashAttention variants, fused activations, and custom
quantization ops.

**Narrative frame (Section 8):** *Your Chapter 4 tiling implementation proved the FlashAttention
mechanism. Now you want it to actually run fast. The Python/PyTorch version is correct but slow;
the production FlashAttention-2 Triton kernel runs at 87% of peak HBM bandwidth. This notebook
shows you how to write GPU kernels in Python using the same tiling patterns — starting from vector
addition and building up to a fused softmax that's 3× faster than the naive version.*
Every kernel is introduced as answering one question: "Why isn't standard Python/PyTorch fast
enough for this?" The closing decision: "For the fused softmax kernel: {speedup:.1f}× faster
than the two-step baseline, with {memory_savings:.0f}% fewer HBM reads. Production Triton kernels
can reach {theoretical_peak:.0f}% of hardware peak bandwidth."

**Structural skeleton (Section 2):**
- Title cell: the 87%-peak-bandwidth target stated before the heading
- Roadmap table: 6 Parts, each a kernel that builds on the previous
- Running example: the `(B=8, S=128, D=64)` attention head from Ch 1, ported progressively from Python to Triton
- Toy → real bridge in Part 6: show Triton kernel names in a `torch.compile`'d forward pass

| Part | Topic | Key demonstration |
|---|---|---|
| 1 | Triton programming model: `@triton.jit`, `tl.load`, `tl.store`, `tl.dot` | Port the vector addition kernel from Chapter 1 to Triton; verify output |
| 2 | Tiled matrix multiplication: block-level parallelism in Triton | Implement a tiled matmul; benchmark against `torch.matmul` |
| 3 | Fused activation functions: combine GELU + bias into one kernel | Implement fused GELU+bias; profile memory savings vs. two separate calls |
| 4 | Fused attention (toy FlashAttention in Triton) | Implement a simplified tiled softmax in Triton; verify against standard attention |
| 5 | Autotuning: `@triton.autotune` for block size selection | Autotune the matmul kernel; show how block size affects throughput |
| 6 | Toy → real: where are the Triton kernels in `torch.compile`'d code? | Trace `torch.compile` output to show Triton kernel names |

**Gold-standard requirements:**
- Prerequisite bridge cell (Section 13.1) from Chapters 1 and 4 (CUDA model + FlashAttention tiling)
- ` Predict first` before tiled matmul benchmark: *"A tiled matmul in Triton vs. `torch.matmul`: (a) Triton is 2× faster, (b) `torch.matmul` is faster (it uses cuBLAS), (c) within 5%?"* (Answer: b for square matrices; Triton wins on irregular shapes — print confirms which)
- ` Predict first` before the autotune sweep: *"Which block size will win for the `(1024, 1024)` matmul: (a) 32, (b) 64, (c) 128?"* (Answer: hardware-dependent; the autotune result is the lesson)
- Every kernel output verified against the PyTorch reference with `torch.allclose(..., atol=1e-3)` (Triton uses fp16 internally — note this in the tolerance comment)
- `#### What just happened` after Part 1 (kernel works; missing: *it's no faster than PyTorch yet — we need tiling and coalesced memory* — plants Part 2)
- Code Walkthrough cell after the `@triton.jit` kernel definition in Part 1 (first time a learner sees this syntax)
- `TRITON_AVAILABLE` flag gates all Triton cells; CPU fallback prints the kernel logic as readable pseudocode
- Closing tier-1/2/3 ledger: Tier 1 (Triton kernel basics, tiled matmul, fused ops, autotuning), Tier 2 (the FlashAttention-2 Triton kernel from the official repo — explained structurally, not rewritten), Tier 3 (cutlass, pallas/XLA, cuBLAS — named)
- **Closing decision (Section 8.7):** "For the fused softmax: {speedup:.1f}× faster than the two-step baseline, {memory_savings:.0f}% fewer HBM reads (measured). For production use: the official FlashAttention-2 Triton kernel already exists — use it via `torch.nn.functional.scaled_dot_product_attention`. Write custom Triton kernels when (a) your attention variant isn't covered, or (b) you need a fused op that reduces memory bandwidth for a specific layer pattern."

**Subagent implementation task:**
> Create `learning/ai-infrastructure/08-triton-kernels/triton-kernels.ipynb`.
> **Primary source:** `notes/07-ai-infrastructure/ch01-gpu-architecture/` for CUDA model.
> All kernel code is fresh. Create `images-plan.md` using prompts below.
> Apply ALL authoring guide conventions.

**images-plan.md for this chapter:**

| Asset | Teaching job |
|---|---|
| `triton-grid-block-thread.png` | CUDA grid → thread blocks → individual threads hierarchy; same layout used by Triton `@triton.jit` |
| `fused-vs-unfused-gelu.png` | Two-step unfused (read activations from HBM, write bias, read again, apply GELU, write) vs. one-step fused (single HBM read/write) |
| `autotune-block-size-sweep.png` | Bar chart: throughput (TFLOP/s) vs. block size (16, 32, 64, 128, 256) for the tiled matmul; optimal block size highlighted |

```text
[triton-grid-block-thread.png]
Flat vector hierarchy diagram, wide 16:9, dark graphite background. Three levels:
Top: a large teal box labeled "CUDA Grid" containing a 4×4 grid of smaller amber
boxes labeled "Thread Blocks". Each block contains a 4×4 grid of tiny ivory squares
labeled "Threads". Bracket annotations show "gridDim", "blockDim". Ivory labels.
No logos, no photorealism, no gradients, no tiny text.

[fused-vs-unfused-gelu.png]
Flat vector pipeline diagram, wide 16:9, dark graphite background. Top half "Unfused":
HBM box → load activations (coral slow arrow) → add bias (amber) → write to HBM
(coral) → load again (coral) → apply GELU (amber) → write to HBM (coral). 4 HBM
accesses total. Bottom half "Fused Triton kernel": HBM box → load (teal) → add bias
+ GELU (amber, in SRAM) → write output (teal). 2 HBM accesses. Ivory labels.

[autotune-block-size-sweep.png]
Flat vector bar chart, wide 16:9, dark graphite background. X-axis: block size (16,
32, 64, 128, 256). Y-axis: throughput in TFLOP/s. Bars in muted teal rising to a
peak at 128, then falling at 256. The 128 bar is amber and labeled "autotune winner".
A horizontal dashed line shows torch.matmul reference throughput. Ivory labels.
```

---

## AI Infrastructure Directory Structure (after implementation)

```
learning/
  ai-infrastructure/
    README.md                        # who this is for; prerequisites; how to set up CUDA env
    requirements.txt                 # torch >= 2.1, triton, auto-gptq, llama-cpp-python
    01-gpu-hardware/
      gpu-hardware-foundations.ipynb  # source: notes/07-ai-infrastructure/ch01
      images/
    02-mixed-precision/
      mixed-precision-and-memory.ipynb  # source: notes/07-ai-infrastructure/ch02 + notes/02/ch10
      images/
    03-profiling/
      pytorch-profiling.ipynb         # mostly fresh; some code from notes/07-ai-infrastructure/ch01
      images/
    04-flash-attention/
      flash-attention-internals.ipynb # source: notes/07-ai-infrastructure/ch05 (concepts)
      images/
    05-distributed-training/
      distributed-training.ipynb      # source: notes/07-ai-infrastructure/ch04
      images/
    06-quantization/
      quantization-in-depth.ipynb     # source: notes/07-ai-infrastructure/ch03 + notes/02/ch10
      images/
    07-inference-systems/
      inference-systems.ipynb         # source: notes/07-ai-infrastructure/ch05 + ch06 + ch12
      images/
    08-triton-kernels/
      triton-kernels.ipynb            # mostly fresh; CUDA model from notes/07-ai-infrastructure/ch01
      images/
```

### Notes coverage map

| `ai-infrastructure` chapter | Primary notes source | Source has | What's fresh |
|---|---|---|---|
| Ch1 GPU Hardware | `notes/07-ai-infrastructure/ch01-gpu-architecture/` | Full notebook + markdown + InferenceBase scenario | Riverside-linked scenario variant, predict-first Qs, closing decision |
| Ch2 Mixed Precision | `notes/07-ai-infrastructure/ch02-memory-and-compute-budgets/` + `notes/02-bridging/ch10` | Markdown + supplement; mixed precision notebook in notes/02 | Riverside A10G scenario, GradScaler NaN demo, closing decision |
| Ch3 Profiling | `notes/07-ai-infrastructure/ch01` (partial) | Basic timing code only | Mostly fresh: profiler setup, bottleneck identification, `torch.compile` timing |
| Ch4 FlashAttention | `notes/07-ai-infrastructure/ch05-inference-optimization/` | IO complexity and tiling concepts in markdown | Fresh: Python tiling walkthrough, online softmax implementation |
| Ch5 Distributed | `notes/07-ai-infrastructure/ch04-parallelism-and-distributed-training/` | Full markdown + supplement | Riverside 70B scenario, predict-first Qs, CPU fallback |
| Ch6 Quantization | `notes/07-ai-infrastructure/ch03-quantization-and-precision/` + `notes/02-bridging/ch10` | Full markdown (PTQ/GPTQ/AWQ/GGUF/NF4) + supplement | Riverside MacBook scenario, perplexity eval harness |
| Ch7 Inference Systems | `notes/07-ai-infrastructure/ch05 + ch06 + ch12` | Markdown + two full notebooks | Riverside 100× traffic scenario, real KV cache PyTorch pass |
| Ch8 Triton Kernels | `notes/07-ai-infrastructure/ch01` (CUDA model only) | GPU execution model description | All Triton kernel code is fresh |

### Notes chapters with no current `ai-infrastructure` mapping

These chapters in `notes/07-ai-infrastructure/` are not covered by the current 8-chapter plan:
- `ch07-ai-specific-networking` (RDMA, NVLink topology)
- `ch08-feature-stores`
- `ch09-ml-experiment-tracking`
- `ch10-production-ml-monitoring`
- `ch11-end-to-end-deployment`

They are Tier-3 out-of-scope for the current `ai-infrastructure` track. If the track is extended,
these chapters provide ready-made source material for a Ch9+ deployment track.

---

## Implementation Parallelization Map

All 8 chapters are independent of each other and can be implemented by separate subagents
simultaneously. Each chapter should apply all conventions from `learning/genai/authoring-guide.md`.

| Agent | Chapter | Key dependency |
|---|---|---|
| Agent A | 01-gpu-hardware (`learning/ai-infrastructure/01-gpu-hardware/`) | Source: `notes/07-ai-infrastructure/ch01`; `torch`, `triton` (optional) |
| Agent B | 02-mixed-precision (`learning/ai-infrastructure/02-mixed-precision/`) | Source: `notes/07-ai-infrastructure/ch02` + `notes/02/ch10`; `torch` |
| Agent C | 03-profiling (`learning/ai-infrastructure/03-profiling/`) | Source: partial `notes/07/ch01`; mostly fresh; `torch` with CUDA |
| Agent D | 04-flash-attention (`learning/ai-infrastructure/04-flash-attention/`) | Source: `notes/07-ai-infrastructure/ch05` (concepts); `torch >= 2.0` |
| Agent E | 05-distributed-training (`learning/ai-infrastructure/05-distributed-training/`) | Source: `notes/07-ai-infrastructure/ch04`; `torch.distributed`, `accelerate` |
| Agent F | 06-quantization (`learning/ai-infrastructure/06-quantization/`) | Source: `notes/07-ai-infrastructure/ch03` + `notes/02/ch10`; `auto-gptq`, `transformers` |
| Agent G | 07-inference-systems (`learning/ai-infrastructure/07-inference-systems/`) | Source: `notes/07-ai-infrastructure/ch05+ch06+ch12`; `transformers`, `vllm` (gated) |
| Agent H | 08-triton-kernels (`learning/ai-infrastructure/08-triton-kernels/`) | Source: `notes/07/ch01` (CUDA model); all kernel code fresh; `triton` |

---

## Cross-Arc Connections

This track explicitly connects back to `learning/genai/`:

| `ai-compute` chapter | `genai` chapter it explains |
|---|---|
| Ch 1 GPU Hardware | Why QLoRA and gradient checkpointing are needed (Ch 2 of this track explains the math) |
| Ch 2 Mixed Precision | Why LoRA training fits in 8 GB (`04-llm/02-llm-finetuning-parameter-techniques.ipynb`) |
| Ch 3 Profiling | Where time is actually spent in a fine-tuning step from `04-llm` |
| Ch 4 FlashAttention | The `scaled_dot_product_attention` call in `02-transformers` |
| Ch 5 Distributed | What "tensor parallelism" means in the `04-llm` Tier 3 scaling notes |
| Ch 6 Quantization | The dynamic quantization section in `04-llm/02` |
| Ch 7 Inference Systems | The KV cache asides in `02-transformers` and `06-llm-gateway` |
| Ch 8 Triton Kernels | The FlashAttention and custom-op references throughout the track |
