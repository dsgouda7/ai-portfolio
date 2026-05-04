# Ch.6 — Model Serving Frameworks

> **The story.** In **2022** the LLM serving bottleneck wasn't model quality — GPT-3 had already proved that scale works. The bottleneck was **throughput under budget**. A single A100 GPU ($1.50/hour on AWS) could serve only 10 requests/second with naive HuggingFace inference, while users were sending 1,000 req/s. The math didn't work: you'd need 100 GPUs ($150/hour) to handle load, burning $108,000/month just on compute. Then in **June 2023**, researchers at UC Berkeley released **vLLM** — an LLM inference engine that used **continuous batching** (pack requests as they arrive, don't wait) and **PagedAttention** (manage GPU memory like OS virtual memory, zero waste). The same A100 now served **160 requests/second** — a **16× throughput improvement**. Suddenly self-hosting Llama-2-70B was cheaper than OpenAI API calls. Within six months, every major AI lab (Anyscale, Together AI, Fireworks AI) switched to vLLM or built their own version. The lesson: **inference engineering is as important as model training** — the right serving framework turns a $100k/month API bill into a $10k/month self-hosted cluster.
>
> **Where you are in the curriculum.** You've just finished [Ch.5: Inference Optimization](../ch05_inference_optimization) where you learned **how** a single GPU speeds up inference (KV cache, Flash Attention, quantization). Now you need to scale that single GPU to **production workloads**: thousands of concurrent users, 99.9% uptime, sub-200ms latency. This chapter teaches the three production-grade serving frameworks that actually run in the wild: **vLLM** (LLM-specific, PagedAttention), **ONNX Runtime** (cross-platform, quantization-first), and **TensorRT** (NVIDIA-only, maximum throughput). You'll deploy Llama-2-7B with each, measure throughput/latency/memory, and learn the decision matrix for choosing the right tool.
>
> **Notation in this chapter.** `throughput` — requests per second (req/s) served by one GPU; `latency` — time from request arrival to first token (ms); `continuous batching` — dynamic batching that starts inference before the full batch is filled; `KV cache` — cached key/value tensors from previous tokens (memory bottleneck); `PagedAttention` — vLLM's paged memory manager for KV cache (like OS virtual memory); `ONNX` — Open Neural Network Exchange (cross-platform model format); `TensorRT` — NVIDIA's optimizing compiler for inference (fuses ops, auto-tunes kernels).
<!-- notation: key variables defined here -->

---

## 0 · The Challenge — Where We Are

> 💡 **The mission**: You're the founding Platform Engineer at **InferenceBase** (the AI startup from Ch.1-5). The product is a document intelligence API: users upload PDFs, your API returns structured JSON from Llama-2-7B. You're currently using **HuggingFace Transformers** for inference — simple to code, but slow. Current metrics:
> - ✅ Works correctly (API functional)
> - ✅ Runs locally (one A100 GPU)
> - ❌ **300ms latency** per request (users want <200ms)
> - ❌ **10 requests/second max throughput** (need to scale to 100 req/s for launch)
> - ❌ **OOM errors at batch size >4** (KV cache fills 40GB VRAM)

**What's blocking us:**
The CEO forwards a spreadsheet comparing self-hosting vs OpenAI API:

| Approach | Cost/month | Latency | Throughput | Status |
|---|---|---|---|---|
| OpenAI API | $80,000 | ~200ms | Unlimited* | Current production |
| Self-hosted (HF) | $1,080 (1 A100) | 300ms | 10 req/s | Can't scale |
| Self-hosted (optimized) | $5,400 (5 A100s) | <200ms | 100+ req/s | **Need to prove feasibility** |

You need to prove that **self-hosting with the right framework** hits both cost and performance targets. The question: **which serving framework?**

**What this chapter unlocks:**
The **model serving framework** decision matrix:
1. **vLLM** — Best for LLMs (continuous batching, PagedAttention, 10-20× throughput vs HF)
2. **ONNX Runtime** — Best for cross-platform deployment (CPU, mobile, edge devices)
3. **TensorRT** — Best for absolute maximum throughput (NVIDIA GPUs only, complex setup)

✅ **After this chapter**: You'll deploy Llama-2-7B with vLLM, achieve **150 req/s throughput** on one A100 (15× improvement), measure **180ms latency** (40% faster), and stay under **20GB memory usage** (KV cache paging). The math now works: 5 GPUs handle 750 req/s for $5,400/month.

---

## Animation

![Chapter animation](img/ch06-serving-throughput-needle.gif)

*Throughput: 10 req/s → 150 req/s with vLLM*

---

## 1 · The Core Idea — Serving Frameworks Are Inference Compilers + Schedulers

Training frameworks (PyTorch, TensorFlow) optimize for **fast iteration** — you want to modify model architecture, add debug prints, and re-run in seconds. Inference frameworks optimize for **fast execution** — you freeze the model once, then serve millions of requests.

### The Three Optimizations Every Serving Framework Implements

| Optimization | What It Does | How It Helps | Example |
|---|---|---|---|
| **Operator fusion** | Merge multiple ops into one kernel | Fewer GPU launches, less memory movement | Merge LayerNorm + GELU into one fused kernel |
| **Dynamic batching** | Batch multiple requests into one forward pass | Higher GPU utilization (more FLOPS per second) | Batch 16 requests → 1 forward pass instead of 16 |
| **Memory optimization** | Reduce memory footprint | Fit larger batches, avoid OOM | KV cache paging (vLLM), quantization (ONNX) |

**The key insight:** Training frameworks treat every batch as independent (no state carried between batches). Inference frameworks are **stateful** — they track which requests are in-flight, which tokens have been generated, and how much KV cache each request owns.

---

## 1.5 · The Practitioner Workflow — Your 4-Phase Framework Selection

**Before diving into benchmarks, understand the decision workflow you'll follow for every model deployment:**

> 📊 **What you'll build by the end:** A performance comparison matrix (throughput, latency, memory) across 3 frameworks + a deployment architecture with monitoring — answering "which framework wins for my workload?"

```
Phase 1: REQUIREMENTS        Phase 2: BENCHMARK           Phase 3: COMPARE             Phase 4: DEPLOY
────────────────────────────────────────────────────────────────────────────────────────────────────────
Define success criteria:     Test frameworks:             Analyze metrics:             Production rollout:

• Target latency (ms)        • vLLM (LLM-optimized)      • Throughput (req/s)         • FastAPI wrapper
• Target throughput (req/s)  • ONNX Runtime (portable)   • Latency (P50/P95/P99)      • Load balancer + replicas
• Hardware constraints       • TensorRT (NVIDIA-max)     • Memory usage (VRAM)        • Prometheus monitoring
• Deployment platform        • Baseline (HF)             • Ease of integration        • Health checks + metrics

→ DECISION:                  → DECISION:                 → DECISION:                  → DECISION:
  What are we optimizing?      Run all 3 on sample load    Which framework wins?        Ready for production?
  • Latency? Throughput?       • Same model (Llama-2-7B)   • vLLM: Best for LLMs       • Monitoring in place?
  • GPU/CPU? Cloud/edge?       • Same workload (100 req)   • ONNX: Best cross-platform • Autoscaling configured?
  • Budget? Scale?             • Measure all 3 metrics     • TensorRT: Max throughput  • Rollback plan ready?
```

**The workflow maps to this chapter:**
- **Phase 1 (REQUIREMENTS)** → §1.5.1 Define Success Criteria (below)
- **Phase 2 (BENCHMARK)** → §2 Running Example (Steps 1-4: HF, vLLM, ONNX, TensorRT benchmarks)
- **Phase 3 (COMPARE)** → §6 Hyperparameter Dials (Framework Selection Matrix)
- **Phase 4 (DEPLOY)** → §4 Step-by-Step (vLLM + FastAPI production deployment)

> 💡 **Usage note:** Complete Phase 1 requirements gathering before benchmarking (avoids wasted work testing irrelevant frameworks). Run Phase 2 benchmarks in parallel (all frameworks tested with identical workload for fair comparison). Phase 3 decision is typically clear from metrics, but edge cases exist (see Decision Checkpoint 3).

---

### 1.5.1 Phase 1: REQUIREMENTS — Define Success Criteria **[Phase 1: REQUIREMENTS]**

**Before touching any code, document your constraints.** Most framework selection mistakes come from optimizing the wrong metric (e.g., maximizing throughput when latency was the actual requirement).

#### Requirements Specification Template

Copy this table and fill in your values BEFORE benchmarking:

```python
# File: requirements.yaml (document before benchmarking)

model:
  name: "meta-llama/Llama-2-7b-chat-hf"
  size_gb: 14.2  # FP16 model weights
  architecture: "decoder-only"  # LLM vs encoder vs vision

workload:
  avg_prompt_length: 200  # tokens
  avg_output_length: 50   # tokens
  requests_per_second: 100  # peak load
  concurrent_users: 1000    # max simultaneous

constraints:
  latency_p95_target_ms: 200   # 95th percentile latency requirement
  throughput_target_rps: 100   # requests per second target
  budget_gpu_count: 5          # max GPUs available (or budget in $/month)
  budget_gpu_type: "A100-40GB" # available hardware

deployment:
  platform: "self-hosted"  # vs "cloud API" vs "edge device"
  os: "linux"              # vs "windows" vs "mobile"
  gpu_vendor: "nvidia"     # vs "amd" vs "cpu-only"

priorities:  # Rank 1-5 (1 = most important)
  latency: 1       # Time to first token
  throughput: 2    # Requests per second per GPU
  memory: 3        # VRAM efficiency (enables larger batch sizes)
  ease_of_setup: 4 # Developer time to deploy
  portability: 5   # Cross-platform compatibility
```

**Example: The InferenceBase scenario (from §0)**
```yaml
model:
  name: "meta-llama/Llama-2-7b-chat-hf"
  size_gb: 14.2
  architecture: "decoder-only"

workload:
  avg_prompt_length: 180
  avg_output_length: 50
  requests_per_second: 100
  concurrent_users: 1000

constraints:
  latency_p95_target_ms: 200
  throughput_target_rps: 100
  budget_gpu_count: 5
  budget_gpu_type: "A100-40GB"

deployment:
  platform: "self-hosted"
  os: "linux"
  gpu_vendor: "nvidia"

priorities:
  latency: 1       # CEO requirement: <200ms
  throughput: 2    # Need 100 req/s for launch
  memory: 3        # Want to maximize batch size
  ease_of_setup: 4 # Team has ML experience
  portability: 5   # NVIDIA-only is fine
```

> 💡 **Industry Standard:** `requirements.yaml` or similar specifications
>
> Real production teams document these constraints before any engineering work. This YAML becomes the **acceptance criteria** for framework selection — if a framework hits all targets, it wins regardless of hype.
>
> **When to use:** Always. Even for "quick prototypes" — defining requirements takes 15 minutes and saves hours of false starts.
> **Common mistake:** Skipping this step and defaulting to HuggingFace Transformers ("it's easy") without measuring if it meets performance targets.

---

#### 1.5.1.1 DECISION CHECKPOINT — Phase 1 Complete

**What you just documented:**
- Latency target: <200ms P95 (95% of requests must respond within 200ms)
- Throughput target: 100 req/s (total across all GPUs)
- Hardware: 5× A100 40GB GPUs (budget constraint)
- Platform: Self-hosted Linux with NVIDIA GPUs (rules out edge/mobile/AMD)

**What it means:**
- **NVIDIA-only is acceptable** → TensorRT and vLLM are candidates (both NVIDIA-exclusive)
- **Latency is Priority #1** → Rules out frameworks with >200ms P95 (e.g., CPU-only ONNX)
- **Throughput target is 100 req/s** → With 5 GPUs, need 20 req/s per GPU minimum (any framework achieving 20+ req/s per GPU satisfies this)
- **Self-hosted** → Rules out managed API services (OpenAI, Anthropic) — we control the stack

**What to do next:**
→ **Benchmark all NVIDIA-compatible frameworks** (vLLM, TensorRT, ONNX GPU, HuggingFace baseline)
→ **Test with representative workload** (180-token prompts, 50-token outputs, 100 concurrent requests)
→ **Measure all 3 metrics** (latency P50/P95/P99, throughput req/s, memory GB)
→ **For InferenceBase:** Proceed to Phase 2 with vLLM (LLM-optimized) and TensorRT (max performance) as top candidates — ONNX is backup if cross-platform becomes required later

---

### Why HuggingFace Transformers Is Slow for Production

```python
# Naive HuggingFace inference (what you're doing now)
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Process ONE request at a time
for request in incoming_requests:
    inputs = tokenizer(request.text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(outputs[0])
    send_response(response)
```

**Problems with this code:**
1. **No batching** — Each request gets its own forward pass (GPU sits idle between requests)
2. **Synchronous** — Can't start request #2 until request #1 finishes all 100 tokens
3. **Memory waste** — KV cache for finished requests stays in VRAM until batch ends
4. **No op fusion** — Every PyTorch op is a separate kernel launch

**Result:** 10 req/s throughput, 300ms latency, 40GB memory usage for batch size 4.

---

> 💡 **Industry Standard:** **vLLM** — The LLM Serving Standard (UC Berkeley, 2023)
>
> **Adoption:** Used by Anyscale, Together AI, Fireworks AI, Modal, Replicate, and dozens of self-hosted deployments serving billions of tokens daily. vLLM has become the **de facto standard** for production LLM serving.
>
> **Key innovations:**
> 1. **PagedAttention** — KV cache allocated in pages (4MB blocks) like OS virtual memory, eliminates fragmentation
> 2. **Continuous batching** — starts inference on new requests immediately (don't wait for full batch), keeps GPU busy
> 3. **Prefix caching** — shares KV cache for common prompt prefixes (e.g., system messages), saves memory
> 4. **Automatic parallelism** — tensor parallelism across multiple GPUs with single config parameter
>
> **Performance gains vs HuggingFace Transformers:**
> - **10-20× throughput** (requests/second per GPU)
> - **30-50% lower latency** (time to first token)
> - **2-4× higher batch sizes** (more concurrent users per GPU)
>
> **When to use:** Default choice for **any** LLM serving on NVIDIA GPUs (GPT, Llama, Mistral, Falcon, etc.). Only deviate if you have specific constraints (CPU-only, AMD GPUs, absolute minimum latency requiring TensorRT).
>
> **Installation:** `pip install vllm` (requires CUDA 11.8+, works on A100/H100/L40S/RTX 4090)
>
> **See also:** [vLLM paper](https://arxiv.org/abs/2309.06180), [vLLM docs](https://vllm.readthedocs.io/)

---

### What vLLM Does Differently (Continuous Batching + PagedAttention)

```python
# vLLM inference (what you'll switch to)
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf", tensor_parallel_size=1)

# Process requests as they arrive (continuous batching)
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

# vLLM automatically batches and schedules
outputs = llm.generate([req.text for req in incoming_requests], sampling_params)
```

**What vLLM optimizes under the hood:**
1. **Continuous batching** — Start inference on request #2 before request #1 finishes (don't wait for full batch)
2. **PagedAttention** — Allocate KV cache in pages (like OS virtual memory), free finished requests immediately
3. **Operator fusion** — Fuse common transformer patterns (attention + softmax + dropout)
4. **Prefix caching** — Share KV cache for common prompt prefixes (e.g., system messages)

**Result:** 150 req/s throughput (15× faster), 180ms latency (40% faster), 18GB memory usage (KV cache paged efficiently).

---

## 2 · Running Example — Deploy Llama-2-7B with Three Frameworks **[Phase 2: BENCHMARK]**

You'll deploy the same model (**Llama-2-7B-chat-hf**) with three frameworks and measure the differences. The test workload: 100 concurrent requests, each generating 50 tokens.

> 💡 **Benchmarking principle:** Use **identical** workload across all frameworks (same model, same prompts, same output length) for fair comparison. Any workload variation invalidates the comparison.

### Step 1: Baseline (HuggingFace Transformers)

**Code:**
```python
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Warm up
prompt = "Explain quantum computing in one sentence."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
_ = model.generate(**inputs, max_new_tokens=10)

# Measure single request latency
start = time.time()
outputs = model.generate(**inputs, max_new_tokens=50)
latency = (time.time() - start) * 1000  # Convert to ms
print(f"Latency: {latency:.1f} ms")

# Measure throughput (sequential requests)
start = time.time()
for i in range(100):
    _ = model.generate(**inputs, max_new_tokens=50)
duration = time.time() - start
throughput = 100 / duration
print(f"Throughput: {throughput:.1f} req/s")
```

**Results (A100 40GB):**
- **Latency:** 312 ms (time to generate 50 tokens for one request)
- **Throughput:** 9.8 req/s (100 sequential requests)
- **Memory usage:** 14.2 GB (model weights) + 8.5 GB (KV cache for 1 request) = 22.7 GB

**Why it's slow:**
- No batching → GPU utilization ~40% (measured with `nvidia-smi dmon`)
- No op fusion → 156 kernel launches per forward pass (measured with Nsight Systems)
- KV cache allocated statically → Can't fit >4 requests in 40GB VRAM

### Step 2: vLLM (Continuous Batching + PagedAttention)

**Installation:**
```bash
pip install vllm
```

**Code:**
```python
from vllm import LLM, SamplingParams
import time

# Initialize vLLM engine
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    tensor_parallel_size=1,  # Single GPU
    dtype="float16",
    gpu_memory_utilization=0.9  # Use 90% of VRAM for KV cache pages
)

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=50
)

# Warm up
_ = llm.generate(["Test prompt"], sampling_params)

# Measure latency (single request)
start = time.time()
outputs = llm.generate(["Explain quantum computing in one sentence."], sampling_params)
latency = (time.time() - start) * 1000
print(f"Latency: {latency:.1f} ms")

# Measure throughput (100 concurrent requests)
prompts = ["Explain quantum computing in one sentence."] * 100
start = time.time()
outputs = llm.generate(prompts, sampling_params)
duration = time.time() - start
throughput = 100 / duration
print(f"Throughput: {throughput:.1f} req/s")
```

**Results (A100 40GB):**
- **Latency:** 184 ms (41% faster than HF)
- **Throughput:** 147 req/s (15× faster than HF)
- **Memory usage:** 14.2 GB (weights) + 18.3 GB (KV cache pool) = 32.5 GB

**Why it's faster:**
- **Continuous batching:** vLLM processes all 100 requests in parallel, starting inference as each arrives
- **PagedAttention:** KV cache allocated in 4MB pages, freed as requests finish (no fragmentation)
- **Operator fusion:** Attention kernels fused into single CUDA kernel (85% GPU utilization vs 40%)

**Memory breakdown (vLLM-specific):**
```
vLLM memory allocation:
- Model weights (FP16):     14.2 GB  (7B params × 2 bytes)
- KV cache (paged):         18.3 GB  (pool of 512 pages × 36 MB/page)
- Activation buffers:        2.8 GB  (temporary tensors during forward pass)
- CUDA context overhead:     1.2 GB
Total:                      36.5 GB / 40 GB (91% utilization)
```

**KV cache page allocation example:**
```
Request 1: [Page 0, Page 1, Page 2]        ← 3 pages (152 tokens generated)
Request 2: [Page 3, Page 4]                ← 2 pages (98 tokens generated)
Request 3: [Page 5, Page 6, Page 7]        ← 3 pages (145 tokens)
...

When Request 1 finishes:
- Pages [0, 1, 2] immediately freed
- New requests can allocate those pages
- No waiting for full batch to complete
```

### Step 3: ONNX Runtime (Quantization + Graph Optimization)

**Use case:** Cross-platform deployment (works on CPU, ARM, NVIDIA, AMD, mobile). Not LLM-specific but excellent for quantized models.

**Convert model to ONNX:**
```bash
pip install optimum[exporters]

# Export to ONNX format with INT8 quantization
optimum-cli export onnx \
    --model meta-llama/Llama-2-7b-chat-hf \
    --task text-generation-with-past \
    --quantize int8 \
    llama2-7b-onnx/
```

**Code:**
```python
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer
import time

model = ORTModelForCausalLM.from_pretrained(
    "llama2-7b-onnx",
    provider="CUDAExecutionProvider"  # Use GPU
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Warm up
inputs = tokenizer("Test", return_tensors="pt").to("cuda")
_ = model.generate(**inputs, max_new_tokens=10)

# Measure latency
start = time.time()
outputs = model.generate(**inputs, max_new_tokens=50)
latency = (time.time() - start) * 1000
print(f"Latency: {latency:.1f} ms")

# Measure throughput (sequential, ONNX doesn't batch automatically)
start = time.time()
for _ in range(100):
    _ = model.generate(**inputs, max_new_tokens=50)
throughput = 100 / (time.time() - start)
print(f"Throughput: {throughput:.1f} req/s")
```

**Results (A100 40GB):**
- **Latency:** 198 ms (36% faster than HF, but 8% slower than vLLM)
- **Throughput:** 24 req/s (2.4× faster than HF, but 6× slower than vLLM)
- **Memory usage:** 7.8 GB (INT8 quantized weights) + 4.2 GB (KV cache) = 12.0 GB

**Why ONNX is slower than vLLM:**
- No continuous batching (processes requests sequentially)
- Graph optimization helps single-request latency but doesn't scale to concurrent load
- KV cache not paged (static allocation)

**Why ONNX is valuable:**
- **Memory efficiency:** 12 GB vs 36 GB (3× less VRAM, can run on RTX 3090 24GB)
- **Cross-platform:** Same model runs on CPU, ARM, AMD GPUs (vLLM is NVIDIA-only)
- **Quantization:** INT8 is free (built into ONNX export), minimal accuracy loss

**When to use ONNX:**
- Edge deployment (mobile, Raspberry Pi, AWS Lambda with CPU)
- CPU inference (no GPU available)
- Multi-vendor deployment (AMD Instinct, Intel ARC, Apple Silicon)

---

> 💡 **Industry Standard:** **ONNX Runtime** — Cross-Platform Inference (Microsoft, 2018)
>
> **Adoption:** Powers Microsoft's production ML (Bing, Office 365, Azure Cognitive Services), used by Meta's PyTorch Mobile, runs in TensorFlow Lite alternative workflows. ONNX Runtime is the **only truly cross-platform** inference engine with production-grade performance.
>
> **Key strengths:**
> 1. **Universal format** — Export from PyTorch/TensorFlow/JAX → ONNX → deploy anywhere (CPU/GPU/mobile/edge)
> 2. **Built-in quantization** — INT8/INT4 quantization with minimal accuracy loss (2-4× memory reduction)
> 3. **Execution providers** — Same model code runs on NVIDIA GPU (CUDA), AMD GPU (ROCm), Intel (OpenVINO), ARM (CoreML), or CPU
> 4. **Graph optimizations** — Operator fusion, constant folding, layout transformations (10-30% speedup without code changes)
>
> **Performance profile:**
> - **Throughput:** 2-3× faster than naive PyTorch (but 5-8× slower than vLLM for LLMs)
> - **Latency:** 30-40% faster than PyTorch baseline
> - **Memory:** INT8 quantization achieves 2× reduction (vs FP16)
>
> **When to use:**
> - **Cross-platform requirement** — need same model on NVIDIA + AMD + CPU + mobile
> - **Edge deployment** — limited hardware (Raspberry Pi, Jetson Nano, mobile devices)
> - **Memory-constrained GPUs** — can't fit model in VRAM without quantization
> - **Non-LLM models** — vision models (ResNet, YOLO), encoders (BERT), or custom architectures
>
> **Installation:** `pip install onnxruntime-gpu` (CUDA) or `onnxruntime` (CPU-only)
>
> **Trade-off:** Slower than vLLM/TensorRT for LLMs (lacks continuous batching) but works **everywhere**. Choose ONNX when portability > absolute speed.
>
> **See also:** [ONNX Runtime docs](https://onnxruntime.ai/), [ONNX format spec](https://onnx.ai/)

---

### Step 4: TensorRT (Maximum Throughput, NVIDIA-Only)

**Use case:** Absolute maximum performance on NVIDIA GPUs. More complex setup than vLLM, but 20-30% faster for latency-critical applications.

**Convert model to TensorRT:**
```bash
pip install nvidia-tensorrt

# Build TensorRT engine (one-time compilation)
trtllm-build \
    --checkpoint_dir llama2-7b-fp16 \
    --output_dir llama2-7b-trt \
    --max_batch_size 128 \
    --max_input_len 512 \
    --max_output_len 200 \
    --gemm_plugin float16
```

**Code:**
```python
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner
import time

# Load TensorRT engine
runner = ModelRunner.from_dir("llama2-7b-trt")

# Warm up
_ = runner.generate(["Test prompt"], max_new_tokens=10)

# Measure latency
start = time.time()
outputs = runner.generate(["Explain quantum computing in one sentence."], max_new_tokens=50)
latency = (time.time() - start) * 1000
print(f"Latency: {latency:.1f} ms")

# Measure throughput (batch inference)
prompts = ["Explain quantum computing in one sentence."] * 100
start = time.time()
outputs = runner.generate(prompts, max_new_tokens=50)
throughput = 100 / (time.time() - start)
print(f"Throughput: {throughput:.1f} req/s")
```

**Results (A100 40GB):**
- **Latency:** 142 ms (55% faster than HF, 23% faster than vLLM)
- **Throughput:** 189 req/s (19× faster than HF, 28% faster than vLLM)
- **Memory usage:** 14.2 GB (weights) + 16.8 GB (KV cache) = 31.0 GB

**Why TensorRT is fastest:**
- **Kernel auto-tuning:** TensorRT benchmarks all CUDA kernels at build time, picks fastest
- **Layer fusion:** More aggressive than vLLM (fuses attention + MLP into single kernel where possible)
- **FP16 accumulation:** Uses Tensor Cores for maximum TFLOPS (vLLM uses FP32 accumulation by default)

**Why TensorRT is harder to use:**
- **Build time:** 10-15 minutes to compile engine (vs instant with vLLM)
- **Static shapes:** Must specify max batch size, seq length at build time (can't change without rebuild)
- **NVIDIA-only:** No AMD, no CPU fallback
- **Complex debugging:** Engine is binary blob (can't inspect like PyTorch model)

**When to use TensorRT:**
- Latency-critical applications (<150ms requirement)
- High-throughput serving (maximizing req/s per GPU)
- Production deployment with static workload (batch size, seq length don't vary much)

---

> 💡 **Industry Standard:** **TensorRT** — NVIDIA's Maximum Performance Compiler (2016)
>
> **Adoption:** Powers NVIDIA Triton Inference Server (used by AWS SageMaker, Azure ML, Google Vertex AI), runs in all major cloud GPU offerings. TensorRT is the **fastest** inference engine for NVIDIA hardware — period.
>
> **Key optimizations:**
> 1. **Kernel fusion** — Merges multiple ops into single CUDA kernels (e.g., LayerNorm + GELU + Dropout → 1 kernel instead of 3)
> 2. **Auto-tuning** — Benchmarks all kernel implementations at build time, selects fastest variant for your GPU
> 3. **Precision calibration** — INT8/FP16 quantization with automatic accuracy recovery (mixed precision)
> 4. **Tensor Core utilization** — Maximizes usage of specialized matrix-multiply units (19.5 TFLOPS on A100)
>
> **Performance profile:**
> - **Throughput:** 20-30× faster than PyTorch baseline (19× in our Llama-2-7B benchmark)
> - **Latency:** 40-55% faster than vLLM (142 ms vs 184 ms in benchmark)
> - **Memory:** Similar to vLLM (31 GB vs 32.5 GB)
>
> **When to use:**
> - **Latency is top priority** — need absolute minimum time-to-first-token (<150ms)
> - **Static workloads** — batch size, sequence length, input shapes are predictable
> - **Maximum GPU utilization** — want every TFLOP the hardware can deliver
> - **NVIDIA Triton deployment** — TensorRT is first-class citizen in Triton ecosystem
>
> **When NOT to use:**
> - **Dynamic workloads** — if batch size/seq length vary widely, rebuild overhead kills performance
> - **Rapid iteration** — 10-15 min build time per model change (vs instant with vLLM)
> - **Cross-platform** — NVIDIA-only, no AMD/CPU fallback
>
> **Installation:** `pip install nvidia-tensorrt` (requires CUDA 11.8+, driver 525+)
>
> **Trade-off:** Absolute maximum speed but highest complexity. Choose TensorRT when latency requirements are **so strict** that vLLM's 184 ms isn't good enough and you need 142 ms.
>
> **See also:** [TensorRT docs](https://docs.nvidia.com/deeplearning/tensorrt/), [TensorRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM), [NVIDIA Triton](https://github.com/triton-inference-server/server)

---

### 2.1 DECISION CHECKPOINT — Phase 2 Complete (Benchmarks Measured)

**What you just saw:**
- **HuggingFace:** 10 req/s throughput, 312 ms latency, 22.7 GB memory — baseline is too slow
- **vLLM:** 147 req/s (15× faster), 184 ms latency (41% faster), 32.5 GB memory — winner for LLMs
- **ONNX Runtime:** 24 req/s (2.4× faster), 198 ms latency (36% faster), 12.0 GB memory — best memory efficiency
- **TensorRT:** 189 req/s (19× faster), 142 ms latency (55% faster), 31.0 GB memory — absolute maximum performance

**What it means:**
- **vLLM wins for general LLM serving:** 15× throughput gain with continuous batching + PagedAttention, near-instant setup (pip install), production-ready (used by Anyscale, Together AI)
- **TensorRT wins for latency-critical apps:** 142 ms vs vLLM's 184 ms (23% faster), but requires 10-15 min build time and complex static shape configuration
- **ONNX wins for cross-platform:** 12 GB memory (2.7× less than vLLM) enables deployment on smaller GPUs (RTX 3090), works on AMD/CPU/mobile
- **HuggingFace should never be used for production serving:** 10 req/s is 15× too slow — only acceptable for research prototypes

**What to do next:**
→ **For InferenceBase (our scenario):** Choose **vLLM** — it hits latency target (<200ms), exceeds throughput target (147 req/s per GPU = 735 req/s with 5 GPUs >> 100 req/s requirement), and setup is simple
→ **If latency requirement was <150ms:** Choose **TensorRT** instead (only framework that achieves 142 ms)
→ **If deploying to edge devices or AMD GPUs:** Choose **ONNX Runtime** (cross-platform portability)
→ **Next:** Proceed to Phase 3 (framework comparison matrix) to validate the decision against other criteria

> ⚠️ **Edge case — Multi-framework deployment:** Some teams deploy vLLM for cloud (NVIDIA A100s) and ONNX for edge (Raspberry Pi 5 with CPU) — same model, two frameworks. This adds operational complexity but may be justified if edge deployment is a hard requirement.

---

## 3 · The Math — Throughput Modeling

How do we predict throughput before deploying? Model it as a **queuing system**.

### Throughput Formula

$$
\text{Throughput} = \frac{\text{Batch Size}}{\text{Latency per Batch}}
$$

**Example (vLLM):**
- Batch size: 16 requests (max concurrent)
- Latency per batch: 184 ms per request (measured in Step 2)
- Throughput: $\frac{16}{0.184 \text{ s}} = 87 \text{ req/s}$

But vLLM does **continuous batching** — it doesn't wait for 16 requests to arrive before starting inference. Instead:

$$
\text{Throughput}_{\text{continuous}} = \frac{N_{\text{concurrent}}}{L_{\text{avg}} + \frac{1}{\lambda}}
$$

Where:
- $N_{\text{concurrent}}$ = max concurrent requests (limited by VRAM for KV cache)
- $L_{\text{avg}}$ = average latency per request (seconds)
- $\lambda$ = request arrival rate (req/s)

**Plugging in vLLM numbers:**
- $N_{\text{concurrent}} = 64$ (VRAM can fit 64 KV caches)
- $L_{\text{avg}} = 0.184$ s (measured)
- $\lambda = 100$ req/s (target load)

$$
\text{Throughput} = \frac{64}{0.184 + \frac{1}{100}} = \frac{64}{0.194} = 330 \text{ req/s}
$$

This is the **theoretical max**. In practice, vLLM achieves ~147 req/s (45% of theoretical) due to:
- KV cache fragmentation (pages not perfectly packed)
- GPU kernel launch overhead (~0.05ms per batch)
- Request length variance (some requests generate 20 tokens, others 200)

### Memory-Bound vs Compute-Bound

When does memory limit throughput?

**KV cache memory per request:**
$$
M_{\text{KV}} = 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times \text{seq\_len} \times \text{bytes\_per\_element}
$$

For Llama-2-7B (FP16):
- $n_{\text{layers}} = 32$
- $n_{\text{heads}} = 32$
- $d_{\text{head}} = 128$
- $\text{seq\_len} = 200$ (average generated length)
- $\text{bytes} = 2$ (FP16)

$$
M_{\text{KV}} = 2 \times 32 \times 32 \times 128 \times 200 \times 2 = 104.9 \text{ MB per request}
$$

**Max concurrent requests on A100 40GB:**
Available VRAM after loading model: $40 - 14.2 = 25.8$ GB

$$
N_{\text{max}} = \frac{25.8 \times 1024}{104.9} = 252 \text{ requests}
$$

But vLLM reserves overhead (activation buffers, CUDA context), so practical limit is ~64 concurrent requests.

**Decision rule:**
- If $N_{\text{concurrent}} < 32$: **Memory-bound** (not enough VRAM for KV cache, need quantization or larger GPU)
- If $N_{\text{concurrent}} > 100$: **Compute-bound** (GPU can't process requests fast enough, need faster kernels or more GPUs)

---

## 4 · Step-by-Step — Deploy vLLM with Monitoring **[Phase 4: DEPLOY]**

Let's deploy vLLM as a production service with logging and metrics.

> 💡 **Production deployment checklist:** (1) API wrapper (FastAPI), (2) Health checks (`/health` endpoint), (3) Metrics export (Prometheus `/metrics`), (4) Load testing (verify throughput target), (5) Monitoring dashboard (Grafana or similar). This section walks through all 5.

### Step 1: Install vLLM and Dependencies

```bash
# Install vLLM (requires CUDA 11.8+)
pip install vllm

# Install FastAPI for REST API
pip install fastapi uvicorn

# Install monitoring tools
pip install prometheus-client
```

### Step 2: Create API Server (vLLM + FastAPI)

**File: `serve.py`**
```python
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from prometheus_client import Counter, Histogram, generate_latest
import time

app = FastAPI()

# Initialize vLLM
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9
)

# Prometheus metrics
request_counter = Counter('vllm_requests_total', 'Total requests')
latency_histogram = Histogram('vllm_latency_seconds', 'Request latency')
throughput_counter = Counter('vllm_tokens_generated', 'Tokens generated')

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.8

@app.post("/generate")
async def generate(request: GenerateRequest):
    request_counter.inc()
    start_time = time.time()

    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )

    outputs = llm.generate([request.prompt], sampling_params)
    generated_text = outputs[0].outputs[0].text

    latency = time.time() - start_time
    latency_histogram.observe(latency)
    throughput_counter.inc(len(outputs[0].outputs[0].token_ids))

    return {
        "generated_text": generated_text,
        "latency_ms": latency * 1000,
        "tokens_generated": len(outputs[0].outputs[0].token_ids)
    }

@app.get("/metrics")
async def metrics():
    return generate_latest().decode("utf-8")

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

### Step 3: Run Server

```bash
# Start server (listens on http://localhost:8000)
uvicorn serve:app --host 0.0.0.0 --port 8000 --workers 1
```

**Why `--workers 1`?** vLLM manages its own batching. Multiple workers would compete for GPU resources.

### Step 4: Test API

```bash
# Single request
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum computing", "max_tokens": 50}'

# Response:
# {
#   "generated_text": "in simple terms.\n\nQuantum computing is...",
#   "latency_ms": 184.2,
#   "tokens_generated": 50
# }
```

### Step 5: Load Test (Measure Throughput)

```bash
# Install load testing tool
pip install locust

# Create load test script (locustfile.py)
from locust import HttpUser, task, between

class VLLMUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def generate(self):
        self.client.post("/generate", json={
            "prompt": "Explain machine learning in one sentence.",
            "max_tokens": 50
        })

# Run load test (100 concurrent users)
locust -f locustfile.py --host http://localhost:8000 --users 100 --spawn-rate 10
```

**Results (from Locust dashboard at http://localhost:8089):**
- **Total requests:** 10,000 (over 2 minutes)
- **Throughput:** 147 req/s (average)
- **P50 latency:** 182 ms
- **P95 latency:** 245 ms
- **P99 latency:** 312 ms

### Step 6: Monitor with Prometheus

**File: `prometheus.yml`**
```yaml
scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

**Start Prometheus:**
```bash
# Run Prometheus in Docker
docker run -p 9090:9090 \
  -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus
```

**Query metrics:**
- Go to http://localhost:9090
- Query `rate(vllm_requests_total[1m])` to see requests per second
- Query `histogram_quantile(0.95, vllm_latency_seconds)` to see P95 latency

---

### 4.1 DECISION CHECKPOINT — Phase 4 Complete (Production Deployment)

**What you just built:**
- **FastAPI wrapper** with `/generate` endpoint (handles JSON requests/responses)
- **Health checks** at `/health` (load balancer can detect failures)
- **Prometheus metrics** at `/metrics` (request count, latency histogram, token count)
- **Load test verification** with Locust (100 concurrent users → 147 req/s sustained)
- **Monitoring dashboard** with Prometheus (query P50/P95/P99 latency in real-time)

**What it means:**
- **Production-ready:** The deployment now has all standard observability (health, metrics, logs) — can be integrated with any enterprise monitoring stack (Datadog, New Relic, Grafana)
- **Performance validated:** Load test confirmed 147 req/s throughput (meets InferenceBase's 100 req/s requirement with headroom)
- **Scalability path clear:** Single GPU serves 147 req/s; 5 GPUs behind NGINX load balancer serve 735 req/s (7× over requirement)
- **Cost target achieved:** 5× A100 GPUs = $5,400/month (vs $80,000/month OpenAI API) — 93% cost reduction

**What to do next:**
→ **Horizontal scaling:** Add more vLLM replicas (one per GPU) behind an NGINX load balancer (see §5 Diagram 1 for architecture)
→ **Autoscaling:** Configure Kubernetes HPA (Horizontal Pod Autoscaler) to add/remove replicas based on `vllm_requests_total` rate
→ **Disaster recovery:** Set up health checks with automatic replica replacement if a GPU OOMs or crashes
→ **A/B testing:** Deploy two vLLM versions (e.g., Llama-2-7B vs Llama-2-13B) and route 10% of traffic to the larger model for quality comparison

> 💡 **Industry Standard:** Production LLM serving stacks (Anyscale Endpoints, Together AI, Fireworks AI)
>
> All three major LLM API providers use this exact stack: vLLM inference engine + FastAPI wrapper + Prometheus metrics + Kubernetes orchestration. The architecture you just built is **production-grade** — the same patterns used to serve billions of tokens per day at scale.
>
> **When to use:** Always for LLM production deployments. This is not a prototype pattern — this IS the production pattern.
> **Next evolution:** Add request routing (based on model size), multi-region deployment (latency optimization), and continuous deployment (blue/green rollout for model updates).

---

## 5 · Key Diagrams

### Diagram 1: Serving Architecture (Load Balancer → vLLM Replicas)

```
┌──────────────────────────────────────────────────────────────┐
│                    Load Balancer (NGINX)                     │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Health checks: GET /health every 10s                    │ │
│  │ Routing: Round-robin across healthy replicas           │ │
│  └─────────────────────────────────────────────────────────┘ │
└────┬──────────────────────┬──────────────────────┬──────────┘
     │                      │                      │
     ▼                      ▼                      ▼
┌──────────┐          ┌──────────┐          ┌──────────┐
│ vLLM #1  │          │ vLLM #2  │          │ vLLM #3  │
│ A100 GPU │          │ A100 GPU │          │ A100 GPU │
│          │          │          │          │          │
│ 150 r/s  │          │ 150 r/s  │          │ 150 r/s  │
└──────────┘          └──────────┘          └──────────┘
     │                      │                      │
     └──────────────────────┴──────────────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │   Prometheus    │
                   │   Monitoring    │
                   └─────────────────┘
```

**Key points:**
- Each vLLM replica handles 150 req/s
- Load balancer distributes requests across 3 replicas → 450 req/s total
- Health checks detect if a replica OOMs or crashes → route around it
- Prometheus scrapes `/metrics` from each replica every 15s

### Diagram 2: Throughput Comparison (vLLM vs Baseline)

*See `gen_scripts/gen_ch06_throughput_comparison.py` for code*

Bar chart comparing:
- HuggingFace: 10 req/s
- ONNX Runtime: 24 req/s
- vLLM: 147 req/s
- TensorRT: 189 req/s

### Diagram 3: KV Cache Visualization (Memory Usage Over Time)

*See `gen_scripts/gen_ch06_kv_cache_usage.py` for code*

Line chart showing:
- X-axis: Time (seconds)
- Y-axis: VRAM usage (GB)
- Three lines:
  - HuggingFace (static allocation, peaks at 22 GB, stays flat)
  - vLLM without paging (peaks at 35 GB, OOM at 40 GB)
  - vLLM with PagedAttention (peaks at 32 GB, pages freed as requests finish)

### Diagram 4: Framework Decision Tree

*See `gen_scripts/gen_ch06_decision_tree.py` for code*

```
                  ┌─────────────────────────┐
                  │ Choose Serving Framework│
                  └───────────┬─────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
         Is it an LLM?                    Not an LLM
         (decoder-only)                  (encoder, vision, etc.)
              │                               │
        ┌─────┴─────┐                   ┌────┴────┐
        │           │                   │         │
      Yes          No                 ONNX    TensorRT
        │                             (cross-  (NVIDIA
   ┌────┴────┐                      platform)   only)
   │         │
 vLLM    TensorRT
(best     (max
choice)  perf)
```

---

## 6 · Hyperparameter Dials **[Phase 3: COMPARE]**

### Framework Selection Matrix

| Requirement | vLLM | ONNX Runtime | TensorRT | HuggingFace |
|---|---|---|---|---|
| **LLM inference (GPT, Llama, Mistral)** | ✅ Excellent | ⚠️ Works but slow | ✅ Fastest | ❌ Too slow |
| **Cross-platform (CPU, ARM, AMD)** | ❌ NVIDIA only | ✅ Excellent | ❌ NVIDIA only | ✅ Works everywhere |
| **Throughput (req/s per GPU)** | ✅ 10-20× baseline | ⚠️ 2-3× baseline | ✅ 20-30× baseline | ❌ Baseline |
| **Latency (ms per request)** | ✅ 40% faster | ⚠️ 30% faster | ✅ 55% faster | ❌ Baseline |
| **Memory efficiency** | ✅ Paged KV cache | ✅ INT8 quantization | ⚠️ Same as vLLM | ❌ Static allocation |
| **Ease of setup** | ✅ Pip install | ✅ Pip install | ⚠️ Complex build | ✅ Pip install |
| **Dynamic batching** | ✅ Continuous | ❌ Manual | ✅ Static batch size | ❌ Manual |
| **Production readiness** | ✅ Used by Anyscale, Together | ✅ Microsoft production | ✅ NVIDIA Triton | ⚠️ Research only |

---

### 6.1 DECISION CHECKPOINT — Phase 3 Complete (Framework Comparison)

**What you just saw:**
- **Matrix row "LLM inference":** vLLM and TensorRT both score ✅ Excellent, ONNX scores ⚠️ Works but slow, HuggingFace scores ❌ Too slow
- **Matrix row "Cross-platform":** ONNX scores ✅ Excellent (CPU/ARM/AMD), vLLM and TensorRT score ❌ NVIDIA only
- **Matrix row "Throughput":** TensorRT 20-30× (highest), vLLM 10-20× (second), ONNX 2-3× (third), HuggingFace 1× (baseline)
- **Matrix row "Ease of setup":** vLLM and ONNX both ✅ Pip install, TensorRT ⚠️ Complex build (10-15 min compile)

**What it means:**
- **No single framework dominates all criteria** — each excels in different scenarios (vLLM for LLMs, ONNX for portability, TensorRT for max speed)
- **Your priorities (from Phase 1) determine the winner** — if latency is Priority #1 and you're on NVIDIA hardware → TensorRT; if ease of setup and throughput matter → vLLM; if cross-platform is required → ONNX
- **HuggingFace Transformers should never win this comparison** — it loses on all performance metrics, only acceptable for research prototypes or when framework compatibility is blocking (rare)

**What to do next:**
→ **Match your Phase 1 priorities to matrix rows:**
  - Latency Priority #1 + NVIDIA hardware → **TensorRT** (142 ms P50, fastest)
  - Throughput Priority #1 + LLM workload → **vLLM** or **TensorRT** (147-189 req/s)
  - Cross-platform Priority #1 → **ONNX Runtime** (only option for AMD/CPU/mobile)
  - Memory efficiency Priority #1 → **ONNX INT8** (12 GB vs 32 GB for vLLM FP16)

→ **For InferenceBase (our scenario):**
  - Priorities: Latency #1 (200ms), Throughput #2 (100 req/s), Memory #3
  - vLLM satisfies: Latency ✅ (184 ms < 200 ms target), Throughput ✅ (147 req/s >> 20 req/s per GPU needed)
  - **Decision: vLLM wins** — meets all requirements, easiest setup (pip install), production-proven

→ **Next:** Proceed to Phase 4 (production deployment with FastAPI + monitoring)

> 💡 **Industry Standard:** Framework selection matrices like this appear in every major AI lab's internal docs (OpenAI, Anthropic, Hugging Face all maintain similar tables). The key is **matching your row priorities to your Phase 1 requirements** — don't default to "most popular" without measuring.

---

### vLLM Configuration Dials

| Parameter | Default | What It Does | When to Tune |
|---|---|---|---|
| `gpu_memory_utilization` | 0.9 | % of VRAM allocated to KV cache pool | Lower to 0.7 if OOM, raise to 0.95 if memory is idle |
| `max_num_seqs` | 256 | Max concurrent requests | Lower if OOM (each request needs KV cache), raise if GPU is idle |
| `max_model_len` | 4096 | Max sequence length (input + output) | Lower for longer outputs (saves KV cache memory) |
| `tensor_parallel_size` | 1 | Number of GPUs for tensor parallelism | Set to 2, 4, or 8 for models >30B params (e.g., Llama-70B) |
| `enforce_eager` | False | Disable CUDA graph optimization | Enable (`True`) if seeing CUDA errors (trades speed for stability) |
| `dtype` | `"auto"` | Model precision (FP16, BF16, FP32) | Use `"bfloat16"` on A100/H100 (better numeric stability than FP16) |

**Example config for production (Llama-2-7B on A100):**
```python
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    tensor_parallel_size=1,
    dtype="bfloat16",
    gpu_memory_utilization=0.85,  # Leave 15% buffer for spikes
    max_num_seqs=128,              # Balance throughput vs latency
    max_model_len=2048             # Cap at 2048 tokens (system + user + response)
)
```

---

> 💡 **Industry Standard:** **BentoML** for Model Deployment Orchestration
>
> ```python
> # File: service.py (BentoML wrapper for vLLM)
> import bentoml
> from vllm import LLM, SamplingParams
>
> @bentoml.service(
>     resources={"gpu": 1, "gpu_type": "nvidia-a100"},
>     traffic={"timeout": 60}
> )
> class LlamaService:
>     def __init__(self):
>         self.llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")
>
>     @bentoml.api
>     def generate(self, prompt: str, max_tokens: int = 100) -> str:
>         outputs = self.llm.generate([prompt], SamplingParams(max_tokens=max_tokens))
>         return outputs[0].outputs[0].text
>
> # Deploy with: bentoml serve service:LlamaService
> # Containerize with: bentoml containerize llamaservice:latest
> ```
>
> **What BentoML adds over raw vLLM:**
> - **Automatic API generation** — decorators (`@bentoml.api`) generate REST/gRPC endpoints with OpenAPI docs
> - **Containerization** — `bentoml containerize` builds optimized Docker images with all dependencies
> - **Model versioning** — track model versions, A/B test different models, rollback on failures
> - **Adaptive batching** — automatically batch concurrent requests (complements vLLM's continuous batching)
> - **Multi-model serving** — serve multiple models (e.g., Llama-7B + Llama-13B) in one service
>
> **When to use:** Teams deploying multiple models or needing standardized ML deployment patterns. BentoML is the "Docker for ML models" — handles packaging, versioning, and deployment. Wraps vLLM (or ONNX, TensorRT) as the inference engine.
>
> **Trade-off:** Adds abstraction layer (slight latency overhead ~5-10ms) but massively simplifies DevOps (no manual Dockerfile writing, automatic health checks, built-in monitoring).
>
> **See also:** [BentoML documentation](https://docs.bentoml.com/), [vLLM + BentoML guide](https://docs.bentoml.com/en/latest/use-cases/large-language-models/vllm.html)

---

## 7 · What Can Go Wrong

### Problem 1: OOM (Out of Memory) with Large Batches

**Symptom:**
```
RuntimeError: CUDA out of memory. Tried to allocate 3.2 GB (GPU 0; 39.4 GB total capacity)
```

**Why it happens:**
- KV cache grows linearly with batch size and sequence length
- Formula: `KV_cache = batch_size × seq_len × hidden_dim × num_layers × 4 bytes`
- Example: 64 requests × 500 tokens × 4096 hidden × 32 layers × 4 = 16.8 GB just for KV cache

**Solutions:**
1. **Lower `max_num_seqs`** (reduce concurrent requests):
   ```python
   llm = LLM(model=..., max_num_seqs=32)  # Default is 256
   ```

2. **Lower `max_model_len`** (cap sequence length):
   ```python
   llm = LLM(model=..., max_model_len=1024)  # Default is 4096
   ```

3. **Reduce `gpu_memory_utilization`** (leave more VRAM buffer):
   ```python
   llm = LLM(model=..., gpu_memory_utilization=0.7)  # Default is 0.9
   ```

4. **Use quantization** (INT8 or INT4 reduces memory 2-4×):
   ```python
   llm = LLM(model=..., quantization="awq")  # Requires AWQ-quantized checkpoint
   ```

### Problem 2: KV Cache Eviction (PagedAttention Thrashing)

**Symptom:**
- High throughput initially (150 req/s)
- Sudden drop to 50 req/s after 1 minute
- Logs show `WARNING: KV cache eviction triggered`

**Why it happens:**
- Too many long-running requests fill all KV cache pages
- New requests trigger eviction (free oldest pages, recompute KV from scratch)
- Recomputation is expensive (defeats the purpose of KV cache)

**Diagnosis:**
Check vLLM metrics:
```python
# In your monitoring dashboard
vllm_cache_hit_rate = cache_hits / (cache_hits + cache_misses)
```
If hit rate <80%, you're evicting too often.

**Solutions:**
1. **Lower `max_tokens`** (shorter responses = faster request completion):
   ```python
   sampling_params = SamplingParams(max_tokens=50)  # Instead of 200
   ```

2. **Increase GPU memory** (upgrade from A100 40GB to A100 80GB or H100 80GB)

3. **Enable prefix caching** (share KV cache for common prompt prefixes):
   ```python
   llm = LLM(model=..., enable_prefix_caching=True)
   ```
   This helps if many requests share the same system prompt.

### Problem 3: GPU Underutilization (Low Throughput Despite Available VRAM)

**Symptom:**
- `nvidia-smi` shows only 40% GPU utilization
- VRAM usage is only 20 GB / 40 GB (half full)
- Throughput is 50 req/s (expected 150 req/s)

**Why it happens:**
- Not enough concurrent requests to fill a batch
- Requests arrive slowly (low arrival rate)
- vLLM is waiting for more requests to batch together

**Diagnosis:**
Check request arrival rate:
```python
arrival_rate = total_requests / total_time  # Should be >100 req/s for good batching
```

**Solutions:**
1. **Increase load** (add more clients sending requests)

2. **Lower latency target** (accept smaller batches):
   ```python
   # This is internal vLLM config (not exposed via API)
   # But you can reduce max_num_seqs to force smaller batches to start sooner
   llm = LLM(model=..., max_num_seqs=16)
   ```

3. **Use burst traffic patterns** (requests come in waves, not uniformly)

### Problem 4: First Request is Slow (Cold Start)

**Symptom:**
- First request takes 2-3 seconds
- Subsequent requests take 180 ms (normal)

**Why it happens:**
- CUDA kernels are compiled on first use (JIT compilation)
- Model weights are lazy-loaded into VRAM on first forward pass

**Solutions:**
1. **Warm up the model** at server startup:
   ```python
   # After loading LLM
   _ = llm.generate(["Warmup prompt"], SamplingParams(max_tokens=1))
   logger.info("Model warmed up")
   ```

2. **Use CUDA graphs** (eliminates kernel launch overhead after warmup):
   ```python
   llm = LLM(model=..., enforce_eager=False)  # Enable CUDA graphs (default)
   ```

---

## Progress Check — Can You Answer These?

Before moving to the next chapter, verify you can answer:

1. **Why is vLLM 15× faster than HuggingFace Transformers for serving Llama-2-7B?**
   <details><summary>Answer</summary>

   vLLM implements three key optimizations: (1) **Continuous batching** — starts inference on new requests immediately instead of waiting for a full batch, keeping GPU busy; (2) **PagedAttention** — manages KV cache in pages like OS virtual memory, eliminating fragmentation and freeing memory as requests finish; (3) **Operator fusion** — merges multiple CUDA kernels into single fused kernels, reducing launch overhead. Together these achieve ~85% GPU utilization vs ~40% with naive PyTorch.
   </details>

2. **What is PagedAttention and why does it help with memory?**
   <details><summary>Answer</summary>

   PagedAttention is vLLM's memory manager that allocates KV cache in fixed-size pages (e.g., 4 MB blocks) instead of contiguous buffers. Benefits: (1) **No fragmentation** — pages can be allocated anywhere in VRAM; (2) **Dynamic allocation** — pages freed immediately when requests finish (no waiting for batch end); (3) **Sharing** — pages can be shared across requests with common prompt prefixes (prefix caching). This allows 2-4× more concurrent requests vs static allocation.
   </details>

3. **When should you use ONNX Runtime instead of vLLM?**
   <details><summary>Answer</summary>

   Use ONNX Runtime when: (1) **Cross-platform deployment** — need to run on CPU, ARM, AMD GPUs, or mobile devices (vLLM is NVIDIA-only); (2) **Memory constraints** — ONNX INT8 quantization uses 2× less VRAM than vLLM FP16; (3) **Non-LLM models** — ONNX supports vision, encoder, and custom architectures (vLLM is LLM-specific). Trade-off: ONNX is 2-6× slower than vLLM for LLMs because it lacks continuous batching.
   </details>

4. **How do you calculate the max concurrent requests for a given GPU?**
   <details><summary>Answer</summary>

   Formula: $N_{\text{max}} = \frac{V_{\text{available}}}{M_{\text{KV per request}}}$

   Where $V_{\text{available}} = V_{\text{total}} - M_{\text{model}}$ (VRAM after loading model)

   And $M_{\text{KV}} = 2 \times n_{\text{layers}} \times d_{\text{model}} \times \text{seq\_len} \times \text{bytes}$

   Example (Llama-2-7B, A100 40GB, 200 tokens/request):
   - Available VRAM: 40 - 14.2 = 25.8 GB
   - KV per request: 2 × 32 × 4096 × 200 × 2 bytes = 105 MB
   - Max concurrent: 25.8 GB / 105 MB = 246 requests

   In practice, vLLM achieves ~60% of theoretical due to activation buffers and overhead.
   </details>

5. **What is continuous batching and why does it improve throughput?**
   <details><summary>Answer</summary>

   Continuous batching (aka iteration-level batching) starts inference on new requests as soon as they arrive, without waiting for a full batch. Traditional batching waits until N requests accumulate before starting the forward pass. Benefits: (1) **Lower latency** — first request doesn't wait for others; (2) **Higher GPU utilization** — GPU stays busy even if requests arrive slowly; (3) **Better throughput** — can process variable-length requests efficiently (finished requests are replaced immediately). Implemented by vLLM and TensorRT-LLM.
   </details>

---

## Where This Reappears

<!-- TODO: add forward pointer table -->

---

## Bridge to Next Chapter

✅ **What you learned:** How to deploy LLMs with production-grade serving frameworks (vLLM, ONNX, TensorRT), measure throughput/latency/memory, and choose the right tool for your workload.

🎯 **Your new capability:** You can serve Llama-2-7B at 150 req/s on one GPU (vs 10 req/s with baseline), handle 1000+ concurrent users with 3-5 GPUs, and deploy with health checks, metrics, and load balancing.

➡️ **Next chapter (Ch.7 — AI-Specific Networking):** Now that you can max out one GPU, the next bottleneck is **GPU-to-GPU communication**. When you split Llama-70B across 4 GPUs with tensor parallelism, how does GPU #1 send activations to GPU #2? Why does PCIe (~32 GB/s) become the bottleneck? How do NVLink (~600 GB/s) and InfiniBand (~400 Gbps) solve this? Ch.7 teaches the networking layer that makes multi-GPU inference practical — and explains why NVIDIA's DGX systems are 10× faster than cobbled-together GPU clusters.

**Cross-chapter connections:**
- [Ch.1: GPU Architecture](../ch01_gpu_architecture) — Tensor Cores, memory bandwidth (hardware limits vLLM throughput)
- [Ch.2: Memory & Compute Budgets](../ch02_memory_and_compute_budgets) — KV cache sizing (determines max concurrent requests)
- [Ch.4: Parallelism](../ch04_parallelism_and_distributed_training) — Tensor parallelism (how vLLM splits large models across GPUs)
- [Ch.5: Inference Optimization](../ch05_inference_optimization) — Flash Attention, quantization (techniques vLLM uses internally)
- [Ch.10: Production ML Monitoring](../ch10_production_ml_monitoring) — A/B testing, drift detection (monitoring deployed vLLM endpoints)

**Prerequisites for future learning:**
- DevOps → Docker: You'll containerize vLLM servers
- DevOps → Kubernetes: You'll orchestrate vLLM replicas across a cluster
- DevOps → Load Balancing: You'll distribute requests with NGINX or Envoy

---

## Further Reading

**Official docs:**
- [vLLM documentation](https://vllm.readthedocs.io/) — API reference, performance tuning, deployment guides
- [ONNX Runtime docs](https://onnxruntime.ai/) — Quantization, execution providers, model zoo
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) — Advanced optimizations, multi-GPU configs

**Papers:**
- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) (2022) — Continuous batching
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (2023) — vLLM paper
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (2023) — 2-3× speedup for autoregressive models

**Benchmarks:**
- [AnyScale LLM Performance](https://www.anyscale.com/blog/continuous-batching-llm-inference) — vLLM vs HuggingFace vs TensorRT benchmarks
- [Microsoft ONNX Runtime benchmarks](https://onnxruntime.ai/docs/performance/benchmarks.html) — Cross-platform performance data

**Tutorials:**
- [Deploying vLLM on Kubernetes](https://docs.vllm.ai/en/latest/serving/deploying_with_kubernetes.html)
- [Quantizing models for ONNX Runtime](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [TensorRT optimization best practices](https://docs.nvidia.com/deeplearning/tensorrt/best-practices/)
