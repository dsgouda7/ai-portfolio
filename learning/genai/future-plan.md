# AI Compute and Systems Plan

**What this covers:** The engineering that makes large language models practical at scale —
GPU hardware architecture, CUDA programming, training infrastructure, and inference systems.

**Target learner:** Someone who has finished `learning/genai/` and understands Transformers,
fine-tuning, RAG, and LLM gateways at a mechanistic level, and now wants to understand *why*
these systems behave the way they do at the hardware level and how to build or optimize them
in production.

**Directory to implement:** `learning/ai-compute/`
(name rationale: covers both compute hardware — GPUs, accelerators — and the systems that
use that hardware for training and inference; "systems" is too ambiguous, "gpu" undersells
the inference and serving content, "ai-compute" is the intersection)

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
- `🔮 Predict first` before every timing experiment, with 3 candidate speedup ranges: e.g. *"How much faster is the GPU matmul vs. 8-CPU-core: (a) 2×, (b) 10×, (c) 50×?"*
- `🧪 Your turn`: double batch size; predict whether GPU:CPU speedup ratio grows (it does — larger work fits better in the warp scheduler)
- Every speed claim measured, never asserted; `torch.cuda.synchronize()` before all stop times
- `#### What just happened — and what's missing` after Part 2 (bandwidth measured; missing: *when do we hit the bandwidth ceiling vs. compute ceiling?*) to plant Part 3
- Code Walkthrough cell (Section 9.1) after the `torch.profiler` setup cell in Part 6
- Closing tier-1/2/3 ledger: Tier 1 (CPU/GPU architecture, memory hierarchy, warp model, CUDA basics, coalescing), Tier 2 (Tensor Core operation — explained via diagram, not built), Tier 3 (NVLink, NVSwitch, MIG, multi-GPU topology — named with one-line reason)
- **Closing decision (Section 8.7):** "To explain the 12× speedup on the A10G: {measured breakdown by part} — the dominant factor is {X}. For your specific fine-tuning workload, the bottleneck is the {matmul/memory} operation at layer {N}."

**Subagent implementation task:**
> Create `learning/ai-compute/01-gpu-hardware/gpu-hardware-foundations.ipynb`.
> Open with the laptop-vs-A10G question *before* the title cell, following the 04-llm pattern.
> The `(B=8, S=128, D=256)` matmul must be established as a named constant and referenced in
> every subsequent timing cell. All timing cells use `torch.cuda.synchronize()` before stop.
> Use `torch.cuda.is_available()` to gate GPU cells with a graceful CPU fallback that prints
> "GPU not available — showing timing ratios from a reference run instead."
> Import `triton` for the kernel section (skip gracefully with pseudocode if not installed).
> Apply ALL conventions from `learning/genai/authoring-guide.md` including Code Walkthrough
> cells after dense blocks, math-mirroring variable names, and the closing tier-1/2/3 ledger.

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
- `🔮 Predict first` before the overflow experiment: *"Will fp16 overflow during a forward pass on GPT-2-Medium with a large batch: (a) never, (b) sometimes depending on the batch, (c) always at batch_size ≥ 32?"* (Answer: b — depends on activation magnitudes)
- `🔮 Predict first` before gradient checkpointing: *"Checkpointing halves peak memory. Does it also halve training time: (a) yes, (b) no, it adds ~30% compute overhead, (c) it actually speeds things up?"* (Answer: b)
- Code Walkthrough cell (Section 9.1) after the `torch.autocast` + `GradScaler` training cell
- `#### What just happened — and what's missing` after Part 3 (stable fp16 training achieved; missing: *activation memory grows with sequence length* — plants Part 4)
- Cross-reference to LoRA section in `04-llm/02-llm-finetuning-parameter-techniques.ipynb`
- Closing tier-1/2/3 ledger: Tier 1 (fp32/fp16/bf16 precision, GradScaler, gradient checkpointing, memory profiling), Tier 2 (offloading to CPU RAM — explained as the trade-off, not built), Tier 3 (ZeRO-Offload, activation quantization — named)
- **Closing decision (Section 8.7):** "Riverside's A10G model selection: GPT-2-Medium at fp32 — {gpt2_fp32_gb:.1f} GB (fits); LLaMA-3-1B at bf16+checkpointing — {llama1b_gb:.1f} GB ({fits_str}); LLaMA-3-8B at bf16+LoRA — {llama8b_lora_gb:.1f} GB ({fits_str_8b}). Recommendation: {measured_recommendation}."

**Subagent implementation task:**
> Create `learning/ai-compute/02-mixed-precision/mixed-precision-and-memory.ipynb`.
> Open with the Riverside A10G scenario before the title. The memory math section must use
> real `model.parameters()` to measure actual footprints, not hardcoded estimates.
> The `GradScaler` demo must show an actual NaN loss without it (use a model small enough to run
> on CPU but with fp16 arithmetic that overflows). Apply ALL authoring guide conventions.

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
- `🔮 Predict first` before Part 1 profiling result: *"Rank the four phases by time: (a) data_load > forward > backward > optimizer, (b) backward > forward > data_load > optimizer, (c) optimizer > backward > data_load > forward"* (correct answer varies by actual hardware — that's the lesson)
- `🧪 Your turn`: disable `torch.compile`; measure whether the backward pass changes proportion
- `#### What just happened` cells after Parts 1, 3, and 6
- Code Walkthrough cell (Section 9.1) after the `torch.profiler` setup block (30+ lines)
- Two-sided health check (Section 14.2): full profiler overhead vs. no instrumentation
- Closing tier-1/2/3 ledger: Tier 1 (torch.profiler, autograd profiler, torch.compile, nvtx), Tier 2 (PyTorch Memory Snapshot — explained, not built here), Tier 3 (Nsight Systems, Perfetto, DCGM — named)
- **Closing decision (Section 8.7):** "For the 04-llm fine-tuning step: the actual bottleneck is {bottleneck} at {pct:.0f}% of wall time. The single change with the greatest expected speedup is {recommendation} — expected to save {expected_saving:.0f}%."

**Subagent implementation task:**
> Create `learning/ai-compute/03-profiling/pytorch-profiling.ipynb`.
> Open with the 45-second mystery. Reload the `04-llm` LoRA adapter from disk at the top
> using the Section 13.2 save/reload pattern (add a note: "if you haven't run 04-llm notebooks,
> the setup cell creates an equivalent toy model"). All profiling cells must produce real output.
> The Chrome trace viewer instructions must include a code cell that prints the trace file path.
> Apply ALL authoring guide conventions.

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
- `🔮 Predict first` before IO measurement: *"Standard attention reads the S×S matrix N times during the backward pass (for grad of Q, K, V separately). Tiled FlashAttention avoids materializing it. Which reads more HBM: (a) standard 5× more, (b) tiled 3× more, (c) tiled 10× less?"* (Answer: a)
- `🔮 Predict first` before `scaled_dot_product_attention` dispatch: *"At causal mask + fp32 + S=512: does PyTorch 2.0 dispatch to FlashAttention?"* (Answer: no — only fp16/bf16 triggers it; fp32 uses the standard path)
- `🧪 Your turn`: double S from 512 to 1024; predict the IO traffic ratio change (it doubles — print confirms it)
- **Prove the tiling**: the online softmax implementation must produce output that passes `torch.allclose(..., atol=1e-5)` against `torch.nn.functional.scaled_dot_product_attention`
- `#### What just happened` after Part 3 (numerically equivalent but O(S²/M) reads; missing: *we still haven't measured the actual speedup* — plants Part 4)
- Code Walkthrough cell after the tiling algorithm implementation (Section 9.1)
- Closing tier-1/2/3 ledger: Tier 1 (IO complexity, tiling, online softmax, FlashAttention dispatch), Tier 2 (GQA/MQA mechanism — built at toy scale), Tier 3 (FlashAttention-3, Ring Attention, Sliding Window Attention — named)
- **Closing decision (Section 8.7):** "For the S=512 attention head from Chapter 3: standard attention read {std_hbm:.1f} GB from HBM; tiled FlashAttention read {flash_hbm:.1f} GB ({ratio:.1f}× less). The dispatch condition for `scaled_dot_product_attention` on this machine: {dispatch_conditions}. Recommendation: {recommendation}."

**Subagent implementation task:**
> Create `learning/ai-compute/04-flash-attention/flash-attention-internals.ipynb`.
> Open with the 60%-bottleneck finding from Chapter 3 (or a reference to it). The tiling walkthrough
> must be clean, readable Python (not pseudocode comments) that passes `torch.allclose` against
> the reference implementation. IO complexity measurements must use `torch.profiler` rather than
> wall-clock time. Apply ALL authoring guide conventions.

---

### Chapter 5 · Distributed Training

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
- `🔮 Predict first` before DDP gradient sync: *"After DDP backward, are the gradients on GPU 0 and GPU 1: (a) identical, (b) summed (twice the magnitude), (c) averaged?"* (Answer: a — averaged by default; print `grad_gpu0 - grad_gpu1` to prove zero difference)
- `🔮 Predict first` before FSDP memory comparison: *"FSDP with 4 GPUs shards parameters 4×. Peak memory per GPU should be: (a) same as DDP, (b) ~4× less than DDP, (c) ~2× less (because optimizer states are also sharded)?"* (Answer: c in practice, due to all-gather overhead)
- `🧪 Your turn`: change DDP gradient bucket size (`bucket_cap_mb`); measure whether it changes throughput
- GPU gating: all multi-GPU cells skip gracefully with CPU process groups; print "Multi-GPU not available — simulating with CPU process groups (gradient math identical, timing not representative)"
- `#### What just happened` after Part 1 (DDP works; missing: *it still requires all parameters to fit on one GPU*) and after Part 2 (FSDP solves that; missing: *but what if a single layer's parameters don't fit?* — plants tensor parallelism)
- Prerequisite bridge cell (Section 13.1) from Chapter 2 (memory math is the foundation)
- Code Walkthrough cell (Section 9.1) after the FSDP configuration block
- Closing tier-1/2/3 ledger: Tier 1 (DDP, FSDP, basic tensor parallelism), Tier 2 (pipeline parallelism — explained with bubble diagram, not built end-to-end), Tier 3 (Megatron-LM, DeepSpeed ZeRO-3, sequence parallelism — named)
- **Closing decision (Section 8.7):** "For Riverside's 70B job on 4 A100s: {recommended_strategy}. Code change from the single-GPU baseline: {change_description}. Estimated time to first checkpoint: {time_estimate} (using reference training throughput from the LLaMA-2 paper)."

**Subagent implementation task:**
> Create `learning/ai-compute/05-distributed-training/distributed-training.ipynb`.
> Open with the Riverside 70B scenario. The toy model must be a real Transformer (not a dummy
> linear stack) so the memory comparisons are proportionally realistic. CPU process group fallback
> must print the explanation stated above. Apply ALL authoring guide conventions.

---

### Chapter 6 · Quantization in Depth

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
- `🔮 Predict first` before dynamic int8 quality: *"Dynamic quantization halves the model size. Will held-out perplexity: (a) stay within 0.5 points of fp32, (b) increase by 2–3 points, (c) increase by 10+ points?"* (Answer: a for dynamic int8 on GPT-2)
- `🔮 Predict first` before GPTQ vs. AWQ comparison: *"At int4, which method will show lower perplexity: (a) GPTQ, (b) AWQ, (c) within noise?"* (Answer: b in most benchmarks; print the actual measured difference)
- `#### What just happened` after Part 3 (PTQ static is good for int8; missing: *int4 static PTQ degrades much faster than int8 — the Hessian correction in GPTQ exists to fix this* — plants Part 4)
- Code Walkthrough cell after the GPTQ calibration block (Section 9.1)
- NF4 section explicitly connects to QLoRA in `04-llm/02-llm-finetuning-parameter-techniques.ipynb` with a forward reference
- Closing tier-1/2/3 ledger: Tier 1 (dynamic PTQ, static PTQ, GPTQ, GGUF runtime, NF4), Tier 2 (AWQ salient-channel mechanism — explained via diagram, not built from scratch), Tier 3 (SpQR, SmoothQuant, QuIP— named)
- **Closing decision (Section 8.7):** "For Riverside's MacBook (16 GB): dynamic int8 ({int8_gb:.1f} GB, perplexity {int8_ppl:.1f}) — fits, quality {int8_verdict}; GGUF Q4_K_M ({gguf_gb:.1f} GB, {gguf_tps:.0f} tok/s) — {gguf_verdict}. Recommendation: {recommendation_with_rationale}."

**Subagent implementation task:**
> Create `learning/ai-compute/06-quantization/quantization-in-depth.ipynb`.
> Open with the Riverside MacBook scenario. The GPT-2 model must be reloaded from the `04-llm`
> checkpoint directory if it exists, with a fallback to a fresh `GPT2LMHeadModel`. All perplexity
> measurements must use the same 50-sentence evaluation set for fair comparison. The GGUF section
> uses `llama-cpp-python`; skip gracefully if not installed. Apply ALL authoring guide conventions.

---

### Chapter 7 · Inference Systems and Serving

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
- `🔮 Predict first` before KV cache speedup: *"Without caching, a 128-token prompt requires attention over 128 positions for every new token. With caching, each new token attends over how many new positions: (a) 128, (b) 1, (c) 1 + a cache lookup?"* (Answer: c — 1 new key/value pair computed, rest from cache)
- `🔮 Predict first` before speculative decoding: *"A 7B verifier model and a 70M draft model: the draft proposes 5 tokens and the verifier accepts 3. Compared to 5 sequential verifier calls: (a) 2× faster, (b) 5× faster, (c) only marginally faster?"* (Answer: a–b depending on acceptance rate — print the actual measured speedup)
- `🧪 Your turn`: change speculative decoding draft length from 4 to 8; predict whether mean acceptance rate goes up or down (it goes down — longer chains are harder to accept wholesale)
- KV cache section explicitly connects to `02-transformers` KV cache description (the "it was mentioned there as an optimization" payoff)
- `#### What just happened` after Part 1 (KV cache solves per-token latency; missing: *at 100× traffic, we're still processing one request at a time* — plants continuous batching)
- Code Walkthrough cell (Section 9.1) after the KV-caching forward pass implementation
- Closing tier-1/2/3 ledger: Tier 1 (KV cache, continuous batching, speculative decoding, prefill/decode profiling), Tier 2 (PagedAttention mechanism — explained via diagram, not built), Tier 3 (tensor-parallel inference, disaggregated prefill-decode, chunked prefill — named)
- **Closing decision (Section 8.7):** "For Riverside's 100× traffic target: naïve serving reaches {naive_tps:.0f} tokens/sec; KV cache alone: {kv_tps:.0f} tok/s ({kv_mult:.1f}×); continuous batching: {cb_tps:.0f} tok/s ({cb_mult:.1f}×); speculative decoding: {sd_tps:.0f} tok/s ({sd_mult:.1f}×). To reach {target_tps:.0f} tok/s for 1000 req/min: {recommendation}."

**Subagent implementation task:**
> Create `learning/ai-compute/07-inference-systems/inference-systems.ipynb`.
> Open with the Riverside 100× traffic scenario. The KV cache must be a real PyTorch
> forward pass on GPT-2, not pseudocode — implement as a `past_key_values` cache list and
> verify output matches the uncached forward pass with `torch.allclose`.
> The vLLM section gates on a running server — include a code cell that prints setup instructions
> and checks connectivity before running benchmarks.
> Apply ALL authoring guide conventions.

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
- `🔮 Predict first` before tiled matmul benchmark: *"A tiled matmul in Triton vs. `torch.matmul`: (a) Triton is 2× faster, (b) `torch.matmul` is faster (it uses cuBLAS), (c) within 5%?"* (Answer: b for square matrices; Triton wins on irregular shapes — print confirms which)
- `🔮 Predict first` before the autotune sweep: *"Which block size will win for the `(1024, 1024)` matmul: (a) 32, (b) 64, (c) 128?"* (Answer: hardware-dependent; the autotune result is the lesson)
- Every kernel output verified against the PyTorch reference with `torch.allclose(..., atol=1e-3)` (Triton uses fp16 internally — note this in the tolerance comment)
- `#### What just happened` after Part 1 (kernel works; missing: *it's no faster than PyTorch yet — we need tiling and coalesced memory* — plants Part 2)
- Code Walkthrough cell after the `@triton.jit` kernel definition in Part 1 (first time a learner sees this syntax)
- `TRITON_AVAILABLE` flag gates all Triton cells; CPU fallback prints the kernel logic as readable pseudocode
- Closing tier-1/2/3 ledger: Tier 1 (Triton kernel basics, tiled matmul, fused ops, autotuning), Tier 2 (the FlashAttention-2 Triton kernel from the official repo — explained structurally, not rewritten), Tier 3 (cutlass, pallas/XLA, cuBLAS — named)
- **Closing decision (Section 8.7):** "For the fused softmax: {speedup:.1f}× faster than the two-step baseline, {memory_savings:.0f}% fewer HBM reads (measured). For production use: the official FlashAttention-2 Triton kernel already exists — use it via `torch.nn.functional.scaled_dot_product_attention`. Write custom Triton kernels when (a) your attention variant isn't covered, or (b) you need a fused op that reduces memory bandwidth for a specific layer pattern."

**Subagent implementation task:**
> Create `learning/ai-compute/08-triton-kernels/triton-kernels.ipynb`.
> Open with the 87%-peak-bandwidth target. All Triton cells require a CUDA GPU; include
> explicit `TRITON_AVAILABLE = torch.cuda.is_available()` flag and CPU fallback that prints
> the kernel logic as annotated pseudocode. Every kernel must pass `torch.allclose` against
> the PyTorch reference. Apply ALL authoring guide conventions.

---

## AI Compute Directory Structure (after implementation)

```
learning/
  ai-compute/
    README.md                        # who this is for; prerequisites; how to set up CUDA env
    requirements.txt                 # torch >= 2.1, triton, auto-gptq, llama-cpp-python
    01-gpu-hardware/
      gpu-hardware-foundations.ipynb
      images/
    02-mixed-precision/
      mixed-precision-and-memory.ipynb
      images/
    03-profiling/
      pytorch-profiling.ipynb
      images/
    04-flash-attention/
      flash-attention-internals.ipynb
      images/
    05-distributed-training/
      distributed-training.ipynb
      images/
    06-quantization/
      quantization-in-depth.ipynb
      images/
    07-inference-systems/
      inference-systems.ipynb
      images/
    08-triton-kernels/
      triton-kernels.ipynb
      images/
```

---

## Implementation Parallelization Map

All 8 chapters are independent of each other and can be implemented by separate subagents
simultaneously. Each chapter should apply all conventions from `learning/genai/authoring-guide.md`.

| Agent | Chapter | Key dependency |
|---|---|---|
| Agent A | 01-gpu-hardware | `torch`, `triton` (optional), `nvml` |
| Agent B | 02-mixed-precision | `torch` |
| Agent C | 03-profiling | `torch` with CUDA (CPU fallback) |
| Agent D | 04-flash-attention | `torch >= 2.0` |
| Agent E | 05-distributed-training | `torch.distributed`, `accelerate` (CPU fallback for single-GPU) |
| Agent F | 06-quantization | `auto-gptq`, `llama-cpp-python`, `transformers` |
| Agent G | 07-inference-systems | `transformers`, `vllm` (gated) |
| Agent H | 08-triton-kernels | `triton` (CPU fallback for non-CUDA) |

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
