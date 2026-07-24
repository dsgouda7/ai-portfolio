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

| Part | Topic | Key demonstration |
|---|---|---|
| 1 | CPU vs. GPU architecture: SIMD vs. SIMT; latency vs. throughput orientation | Matrix multiply timing: CPU (1 core), CPU (8 cores), GPU; log-scale bar chart |
| 2 | GPU memory hierarchy: HBM → L2 → shared memory → registers; sizes and latencies | Bandwidth benchmark: access patterns that hit L2 vs. HBM; print GB/s for each |
| 3 | Warp execution: 32 threads per warp, warp divergence, occupancy | Toy kernel that branches on thread ID; show divergence cost with nvml/PyTorch timer |
| 4 | CUDA grid/block/thread model: `blockIdx`, `threadIdx`, launching a kernel | Write a CUDA kernel for vector addition via `triton` (simpler than raw CUDA from Python); verify output matches CPU |
| 5 | Memory coalescing: why layout matters for throughput | Coalesced vs. strided access pattern; bandwidth comparison |
| 6 | Toy → real bridge: where does PyTorch's `matmul` live in this hierarchy? | `torch.profiler` trace showing `aten::mm` → CUDA kernel name |

**Gold-standard requirements:**
- Running example: a `(B, S, D)` attention-shaped matmul (B=8, S=128, D=256) used throughout
- `🔮 Predict first` before every timing experiment (learner guesses speedup before measurement)
- `🧪 Your turn`: change batch size or sequence length; predict whether GPU speedup grows or shrinks
- Every claim about speed is measured, not asserted
- Closing tier-1/2/3 ledger (Tensor Cores, NVLink, MIG are Tier 3)

**Subagent implementation task:**
> Create `learning/ai-compute/01-gpu-hardware/gpu-hardware-foundations.ipynb`.
> Use `torch.cuda.is_available()` to gate GPU cells with a graceful CPU fallback.
> All timing cells use `torch.cuda.synchronize()` before recording stop times.
> Apply all conventions from `learning/genai/authoring-guide.md`.
> Import `triton` for the kernel writing section (skip gracefully if not installed).

---

### Chapter 2 · Mixed Precision and Memory Math

**Why here:** The "why does this fit in GPU memory" question is the single most common
confusion for practitioners moving from tutorial to production.

| Part | Topic | Key demonstration |
|---|---|---|
| 1 | Memory footprint math: `params × bytes_per_param`; optimizer states; activations; gradients | Print memory breakdown for GPT-2 (124M params) in fp32, fp16, bf16, int8 |
| 2 | fp32 vs fp16 vs bf16: exponent/mantissa trade-offs; overflow risk; why bf16 is preferred for training | Overflow demo: fp16 overflows on large LM gradients; bf16 does not |
| 3 | `torch.autocast` + `GradScaler`: the two-part recipe for stable fp16 training | Train a small model with and without `GradScaler`; show NaN loss without it |
| 4 | Gradient checkpointing / activation recomputation: trade compute for memory | Profile peak memory with and without `torch.utils.checkpoint`; show the trade-off curve |
| 5 | Memory profiling: `torch.cuda.memory_summary()`, `max_memory_allocated()`, `memory_snapshot()` | Step-by-step memory trace of a single forward+backward pass |
| 6 | Toy → real: memory estimate for fine-tuning LLaMA-3-8B in different precision regimes | Fill-in-the-blank calculator cell; connect to LoRA's memory advantage from `04-llm` |

**Gold-standard requirements:**
- Narrative framing: "Your GPU has 24 GB. LLaMA-3-8B has 8 billion parameters. Will it fit?" —
  answered definitively by the end of the chapter
- `🔮 Predict first` before the overflow experiment
- Closing decision: "for Riverside's laptop CPU scenario from `04-llm`, what is the maximum
  model that fits, and at what precision?"
- Cross-reference to LoRA section in `04-llm` (LoRA saves memory by keeping the base model frozen)

**Subagent implementation task:**
> Create `learning/ai-compute/02-mixed-precision/mixed-precision-and-memory.ipynb`.
> The memory math section must use real `model.parameters()` to measure actual footprints,
> not hardcoded estimates. The `GradScaler` demo must show an actual NaN loss without it.

---

### Chapter 3 · PyTorch Profiling and Bottleneck Analysis

**Why here:** Without profiling tools, optimization is guesswork. This chapter teaches the
habit before the techniques.

| Part | Topic | Key demonstration |
|---|---|---|
| 1 | `torch.profiler`: capturing a CPU+CUDA trace; reading the Chrome trace viewer | Profile a Transformer forward pass; identify the top-3 most expensive operators |
| 2 | `torch.autograd.profiler.profile` for targeted cell-level profiling | Compare profiling overhead: full profiler vs. targeted wrapper |
| 3 | Identifying bottlenecks: compute-bound vs. memory-bound operations | Benchmark two implementations of the same attention pattern; identify the memory-bound one |
| 4 | `torch.compile` (PyTorch 2.0): when it helps, when it doesn't | Time a Transformer block with and without `torch.compile`; show the warm-up cost |
| 5 | `nvtx.range_push/pop` for custom profiling regions | Instrument a training loop with custom regions; show them in the timeline |
| 6 | Toy → real: profile a LoRA fine-tuning step from `04-llm` | Show exactly where time is spent: data loading, forward, backward, optimizer step |

**Gold-standard requirements:**
- "Prove the bottleneck, don't guess it" — every section produces a profiling artifact
- `🧪 Your turn`: change batch size; observe how the bottleneck shifts
- Images: profiling timeline screenshot embedded as a static PNG (generated locally, not Perchance)

**Subagent implementation task:**
> Create `learning/ai-compute/03-profiling/pytorch-profiling.ipynb`.
> All profiling cells must produce real output when run; the Chrome trace viewer instructions
> should include a code cell that opens the trace file path.

---

### Chapter 4 · FlashAttention: Inside the Algorithm

**Why here:** FlashAttention is the single most impactful algorithmic improvement in practical
LLM inference and training, and almost no practitioners understand *why* it is faster (not an
approximation — it reorders computation to avoid HBM bandwidth bottlenecks).

| Part | Topic | Key demonstration |
|---|---|---|
| 1 | Standard attention's memory problem: the $O(S^2)$ intermediate matrix in HBM | Measure HBM traffic for standard attention vs. sequence length; show quadratic growth |
| 2 | Tiling insight: compute attention in blocks, keep intermediates in SRAM | Walk through Algorithm 1 of the Flash Attention paper in Python (readable pseudocode) |
| 3 | Online softmax trick: how to compute softmax without materializing the full $S×S$ matrix | Implement and verify against `torch.softmax` on random data |
| 4 | IO complexity: why FlashAttention is $O(S^2/M)$ vs. $O(S^2)$ HBM reads | Measure actual HBM traffic reduction using profiler; match to theory |
| 5 | `scaled_dot_product_attention` (PyTorch 2.0): when it dispatches to FlashAttention | Test conditions that enable vs. disable FlashAttention dispatch; time the difference |
| 6 | GQA (Grouped Query Attention) and MQA: reducing KV cache size without accuracy loss | Compare KV cache memory at S=2048 for MHA vs. GQA-8 vs. MQA |

**Gold-standard requirements:**
- Running example: the same `(B, S, D)` matmul from Chapter 1 — scaled to S=512, then S=2048
- `🔮 Predict first` before the IO measurement: "which implementation reads more from HBM?"
- Every algorithm claim verified against `torch.nn.functional.scaled_dot_product_attention`
- Tier-3 ledger: FlashAttention-3, Ring Attention, Sliding Window Attention

**Subagent implementation task:**
> Create `learning/ai-compute/04-flash-attention/flash-attention-internals.ipynb`.
> The tiling walkthrough must be clean, readable Python (not pseudocode comments) that
> produces bit-identical output to `torch.softmax` on the test case. All IO complexity
> measurements must use `torch.profiler` rather than wall-clock time.

---

### Chapter 5 · Distributed Training

**Why here:** Most LLMs over 7B parameters cannot be trained on a single GPU. A practitioner
needs to understand the three axes of parallelism (data, tensor, pipeline) to reason about
training recipes.

| Part | Topic | Key demonstration |
|---|---|---|
| 1 | Data parallel (DDP): gradient all-reduce across GPUs | Run DDP on 2 GPUs (or simulate with CPU groups); verify gradients are identical after sync |
| 2 | FSDP (Fully Sharded Data Parallel): shard parameters, gradients, and optimizer states | Compare peak GPU memory: DDP vs. FSDP for a 1B parameter toy model |
| 3 | Tensor parallelism: split attention heads across GPUs | Toy example: split a 4-head MHA across 2 GPUs; verify output matches single-GPU |
| 4 | Pipeline parallelism: split layers across GPUs; micro-batching | Illustrate bubble overhead with a 4-stage pipeline and different micro-batch counts |
| 5 | 3D parallelism: data + tensor + pipeline combined (Megatron-LM style) | Table: which axis scales at which model/GPU count regime |
| 6 | Toy → real: what parallelism strategy does a 70B model training recipe use? | Walk through a public LLaMA-2 training config; map each setting to a parallelism axis |

**Gold-standard requirements:**
- GPU gating: all multi-GPU cells skip gracefully with a printed explanation if only 1 GPU present
- `🔮 Predict first` before the FSDP memory comparison
- Narrative frame: "You have a budget of four 80GB A100s. Which strategy fits a 13B model?"
- Closing decision with a filled-in parallelism-selection matrix

**Subagent implementation task:**
> Create `learning/ai-compute/05-distributed-training/distributed-training.ipynb`.
> Single-GPU fallback must print: "Multi-GPU sections require ≥2 GPUs; skipping DDP demo.
> The concepts and measurements are still shown on CPU process groups." All distributed
> primitives use `torch.distributed` or HuggingFace `accelerate`.

---

### Chapter 6 · Quantization in Depth

**Why here:** Quantization is the primary lever for deploying large models on consumer hardware.
The `04-llm` chapter introduced dynamic quantization as a deployment trick; this chapter explains
it mechanistically and covers the production methods.

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
- Running example: the same GPT-2 model from `04-llm` chapters, so the learner sees their
  previously-trained model re-quantized
- Every quantization method includes a perplexity measurement vs. fp32 baseline
- `🔮 Predict first` before GPTQ vs. AWQ perplexity comparison
- Cross-reference: NF4 section explicitly connects to QLoRA in `04-llm/02-llm-finetuning-parameter-techniques.ipynb`
- Closing tier-1/2/3 ledger (SpQR, SmoothQuant, QuIP are Tier 3)

**Subagent implementation task:**
> Create `learning/ai-compute/06-quantization/quantization-in-depth.ipynb`.
> All perplexity measurements must use the same evaluation set for fair comparison.
> The GGUF section should use `llama-cpp-python`; skip gracefully if not installed.

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

**Gold-standard requirements:**
- Narrative frame: "Riverside's gateway (`06-llm-gateway.ipynb`) is running. Now make it 10×
  faster." — each section contributes one lever toward that goal
- KV cache section cross-references `02-transformers` KV cache explanation (now making the
  mechanism they saw described there concrete in running code)
- `🔮 Predict first` before speculative decoding acceptance rate measurement
- Closing tier-1/2/3 ledger (tensor-parallel inference, disaggregated prefill-decode are Tier 3)

**Subagent implementation task:**
> Create `learning/ai-compute/07-inference-systems/inference-systems.ipynb`.
> The KV cache implementation must be a real PyTorch forward pass (not pseudocode).
> The vLLM section requires a running server — gate it with a try/except and a clear
> "start vLLM server with `vllm serve model_name`" instruction before the cells.

---

### Chapter 8 · Custom Kernels with Triton

**Why here:** Triton bridges the gap between "I understand what CUDA does" (Chapter 1) and
"I can write a fast custom operator" without requiring raw C++ CUDA. A practitioner who can
write a Triton kernel can implement FlashAttention variants, fused activations, and custom
quantization ops.

| Part | Topic | Key demonstration |
|---|---|---|
| 1 | Triton programming model: `@triton.jit`, `tl.load`, `tl.store`, `tl.dot` | Port the vector addition kernel from Chapter 1 to Triton; verify output |
| 2 | Tiled matrix multiplication: block-level parallelism in Triton | Implement a tiled matmul; benchmark against `torch.matmul` |
| 3 | Fused activation functions: combine GELU + bias into one kernel | Implement fused GELU+bias; profile memory savings vs. two separate calls |
| 4 | Fused attention (toy FlashAttention in Triton) | Implement a simplified tiled softmax in Triton; verify against standard attention |
| 5 | Autotuning: `@triton.autotune` for block size selection | Autotune the matmul kernel; show how block size affects throughput |
| 6 | Toy → real: where are the Triton kernels in `torch.compile`'d code? | Trace `torch.compile` output to show Triton kernel names |

**Gold-standard requirements:**
- Every kernel output verified against the equivalent PyTorch reference implementation
- `🔮 Predict first` before the autotune sweep
- Closing tier-1/2/3 ledger (cutlass, cuBLAS, pallas/XLA are Tier 3)

**Subagent implementation task:**
> Create `learning/ai-compute/08-triton-kernels/triton-kernels.ipynb`.
> All Triton cells require a CUDA GPU; include explicit `TRITON_AVAILABLE` flag and
> graceful CPU fallback that prints kernel logic as readable pseudocode when Triton is absent.

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
