# Ch.5 — Inference Optimization

> **The story.** For the first wave of LLM inference (GPT-2/3 era, 2019–2022), the standard approach was "generate one token, concatenate, repeat" — no batching, no caching optimizations. Every request was independent. **Continuous batching** (Orca, Yu et al., Microsoft, **2022**) changed this: instead of waiting for the slowest request in a batch to finish, dynamically add new requests as soon as a slot frees up → 10× higher throughput. **PagedAttention** (Kwon et al., vLLM, **2023**) solved KV cache fragmentation by treating it like OS virtual memory — page in/out blocks as needed → 24× higher batch sizes. **Speculative decoding** (Leviathan et al., Google, **2023**) used a small "draft" model (1B params) to predict multiple tokens, then verified with the large model in parallel → 2–3× speedup. By 2024, these techniques were bundled into serving frameworks (vLLM, TGI, TensorRT-LLM), making them transparent to users.
>
> **Where you are in the curriculum.** Ch.3 validated INT4 quantization → 12,000 req/day throughput at batch=4. But that was a synthetic benchmark (uniform request arrival). Real production traffic has spiky arrival patterns, variable sequence lengths, and different request priorities. This chapter optimizes the *inference loop* for real-world conditions: continuous batching, PagedAttention, speculative decoding, and KV cache management. The InferenceBase question: *Can we maintain <2s latency at 12,000 req/day under realistic load?*
>
> **Notation.** `TTFT` = time to first token (prefill latency). `TPOT` = time per output token (decode step latency). `KV cache` = key-value attention tensors cached from the prefill pass to avoid recomputing prior-token attention. `req/s` = requests per second. `tok/s` = tokens per second. `batch` = number of sequences processed concurrently in a single forward pass.
<!-- notation: key variables defined here -->

---

## 0 · The Challenge — Where We Are

> 🎯 **The mission**: Self-host Llama-3-8B for <$15k/month, replacing $80k OpenAI API costs
>
> **6 Constraints**: #1 Cost (<$15k/mo) • #2 Latency (≤2s) • #3 Throughput (≥10k req/day) • #4 Memory (fit in VRAM) • #5 Quality (≥95% accuracy) • #6 Reliability (>99% uptime)

> 🎬 *Animation placeholder — needle-builder agent will generate this showing constraint #2 (Latency) and #3 (Throughput) moving toward target during this chapter.*

**What we know so far**:
- ✅ Ch.1-3: RTX 4090 + INT4 quantization → 12,000 req/day throughput ✅
- ✅ Ch.3: 96.2% accuracy maintained ✅, 1.2s p95 latency ✅

**What's blocking us**:

🚨 **Synthetic benchmarks ≠ production reality. Real traffic patterns may break latency target.**

**Current situation**: Engineer stress-testing INT4 model with realistic load patterns

```
Load test day 1 (uniform traffic, 10 req/sec constant):
✅ Throughput: 12,000 req/day
✅ Latency p95: 1.2s

Load test day 2 (realistic spiky traffic):
- Normal: 5 req/sec (9am–5pm)
- Spike: 40 req/sec (lunch rush 12pm–1pm)

Result at 12:15pm:
❌ Latency p95: 8.7s (misses 2s target by 4×!)
❌ Queue depth: 180 requests waiting
❌ Some requests timeout after 30s

Engineer: "Our batch=4 config works fine at steady load, but when 40 requests
          arrive simultaneously, the queue explodes. By the time we process
          batch 1, batch 10 is still waiting 8 seconds."

CEO: "So we can't actually handle lunch rush? That's 30% of daily revenue!"

Engineer: "The problem is **static batching**. We wait until batch=4 fills,
          then process all 4 together. But during spikes:
          - First 4 requests: start processing immediately
          - Next 36 requests: sit in queue for 8+ seconds

          We need **continuous batching** — add requests dynamically as soon
          as any slot frees up. Also, our KV cache is fragmented (wasting
          2GB VRAM on padding). If we use PagedAttention, we can fit batch=8
          instead of batch=4 → cut queue wait time in half."
```

**Problems**:
1. ❌ **Static batching kills tail latency**: Wait for batch=4 to fill → new requests queue for seconds
2. ❌ **KV cache fragmentation**: Padding short sequences to max length wastes VRAM → limits batch size
3. ❌ **No request prioritization**: High-priority requests wait behind low-priority bulk jobs
4. ❌ **Generation speed bottleneck**: 280ms per request × 4 = 1,120ms total → can we go faster?
5. ❌ **Unknown headroom**: Can current setup handle 2× traffic (20k req/day) if business grows?

**Business impact**:
- **Lunch rush failures = lost revenue**: 30% of daily traffic experiences 8s latency → users abandon, revenue drops
- **Cannot scale to growth**: If Q2 traffic hits 20k req/day, system collapses
- **No SLA confidence**: Cannot promise <2s latency under realistic load patterns

**What this chapter unlocks**:

🚀 **Production-grade inference optimizations**:
1. **Continuous batching**: Add requests dynamically as slots free → eliminate queue wait spikes
2. **PagedAttention**: Eliminate KV cache fragmentation → batch=4 → batch=8 (2× throughput)
3. **Speculative decoding**: Draft with Llama-3-1B, verify with Llama-3-8B → 30% faster generation
4. **Request prioritization**: High-priority requests skip queue (premium tier customers)
5. **Autoscaling**: Spin up 2nd GPU during lunch rush, shut down at night → cost-optimized

⚡ **Expected outcomes**:
- **Throughput**: 12,000 → 22,000 req/day (batch=8 + speculative decoding) ✅
- **Latency under spike**: 8.7s p95 → 1.8s p95 (continuous batching eliminates queue) ✅
- **Latency normal load**: 1.2s → 680ms p95 (continuous batching + PagedAttention + speculative decoding) ✅
- **VRAM headroom**: 10GB → 8GB (PagedAttention eliminates padding waste)
- **Cost maintained**: $1,095/month (no additional GPUs needed)

**Constraint status after Ch.5**:
- #1 (Cost): ✅ **MAINTAINED** ($1,095/month)
- #2 (Latency): ✅ **VALIDATED UNDER LOAD** (680ms p95, 66% better than 2s target)
- #3 (Throughput): ✅ **EXCEEDED** (22,000 req/day, 220% of target)
- #4 (Memory): ✅ **OPTIMIZED** (8GB / 24GB = 33% utilization)
- #5 (Quality): ✅ **MAINTAINED** (96.2% accuracy)
- #6 (Reliability): ⚡ **IMPROVED** (handles 2× traffic spikes without degradation)

**Critical breakthrough**: **Production-ready inference** — handles real-world traffic patterns, not just synthetic benchmarks. Ready for serving framework selection (Ch.6).

---

## Animation

<!-- TODO: add animation GIF here -->

---

## 1 · Core Idea

Standard inference loop processes requests one-by-one:
```python
while True:
    request = queue.pop()
    response = model.generate(request.prompt)
    send_response(response)
```

**Optimized inference loop** uses:
1. **Continuous batching**: Dynamic batch formation (add requests as slots free)
2. **PagedAttention**: KV cache paging (eliminate fragmentation waste)
3. **Speculative decoding**: Draft model predicts, large model verifies

Result: 3–5× higher throughput, 50% lower tail latency, same VRAM budget.

---

## 1.5 · The Practitioner Workflow — Your 4-Phase Optimization Loop

> ⚠️ **Two ways to read this chapter:**
> - **Theory-first (recommended for learning):** Read §0→§11 sequentially to understand continuous batching, PagedAttention, and speculative decoding concepts, then use this workflow as your reference
> - **Workflow-first (practitioners debugging production):** Use this diagram as a jump-to guide when diagnosing latency spikes or throughput bottlenecks
>
> **Note:** Section numbers don't follow phase order because the chapter teaches concepts pedagogically (theory before application). The workflow below shows how to APPLY those concepts.

**What you'll build by the end:** A production inference configuration with continuous batching, PagedAttention, and speculative decoding — validated under realistic spiky traffic to ensure <2s p95 latency at 10k+ req/day throughput. This is the benchmark table from §9 showing 22,000 req/day at 680ms p95 latency.

**Before diving into theory, understand the workflow you'll follow with every inference deployment:**

```
Phase 1: PROFILE              Phase 2: IDENTIFY             Phase 3: OPTIMIZE             Phase 4: VALIDATE
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
Baseline measurement          Bottleneck analysis           Apply techniques              Performance regression testing

• PyTorch Profiler            • Component timing            • Continuous batching         • A/B test harness
• CUDA event timing           • CPU vs memory vs I/O        • PagedAttention              • Accuracy delta check
• Breakdown:                  • Queue depth analysis        • Speculative decoding        • Latency SLA validation
  - Tokenization (ms)         • VRAM utilization            • KV cache tuning             • Throughput gains
  - Prefill (ms)              • Batch size limits           • Operator fusion             • Load test under spike
  - Decode per token (ms)
  - Detokenization (ms)       → DECISION:                   → DECISION:                   → DECISION:
                                Which bottleneck?             Which optimization?           Accept or iterate?
→ DECISION:                     • CPU-bound:                  • Queue wait: cont. batch     • Accuracy drop <0.5pp:
  Establish baseline             → operator fusion            • VRAM limit: PagedAttn         accept ✅
  • TTFT (prefill latency)      • Memory-bound:              • Decode speed: speculative   • Latency miss target:
  • TPOT (per-token latency)      → quantize (Ch.3)          • All three: stack them!        iterate Phase 2→3 ↻
  • End-to-end latency          • I/O-bound:                                               • Throughput < target:
  • Throughput (req/s)            → continuous batching                                       iterate Phase 2→3 ↻

  → Target: <2s p95, ≥10k/day   → Target: Identify primary   → Target: Implement fix      → Target: Validate under
                                   component >50% of latency     without accuracy loss          realistic load pattern
```

**Loop structure**: After Phase 4, if SLA not met or ROI diminishing, return to Phase 2 with new profile → re-identify next bottleneck → re-optimize → re-validate. Typically converges in 2–3 iterations.

**The workflow maps to these sections:**
- **Phase 1 (PROFILE)** → §3.12 Profiling Baseline, Code Snippet 1
- **Phase 2 (IDENTIFY)** → §3.13 Bottleneck Analysis, Code Snippet 2
- **Phase 3 (OPTIMIZE)** → §3 Continuous Batching, §4 PagedAttention, §5 Speculative Decoding, Code Snippet 3
- **Phase 4 (VALIDATE)** → §9 Benchmarking, Code Snippet 4

> 💡 **How to use this workflow:** Run Phase 1 once to establish baseline metrics. Then loop Phase 2→3→4 until latency SLA is met or throughput target is achieved. The sections below teach WHY each technique works; refer back here for WHAT to do.

**Typical iteration example (InferenceBase):**

```
Iteration 0 (Baseline):
Profile → 8.7s p95 latency under lunch rush spike (40 req/sec)

Iteration 1:
Identify → Queue wait = 7.2s (82% of latency) → CPU/memory fine, just queue explosion
Optimize → Apply continuous batching (vLLM)
Validate → 3.1s p95 latency ⚠️ (still misses 2s target, but 64% improvement)

Iteration 2:
Identify → Queue still backing up (batch=4 too small), VRAM headroom = 14GB unused
Optimize → Apply PagedAttention → batch=8 instead of batch=4
Validate → 1.9s p95 latency ✅ (within target! 78% improvement from baseline)

Iteration 3 (optional headroom):
Identify → Decode speed = 140ms per token (could be faster)
Optimize → Add speculative decoding (Llama-3-1B draft model)
Validate → 1.3s p95 latency ✅ (extra 31% improvement, 35% better than target)

Final config: Continuous batching + PagedAttention + speculative decoding
Result: 1.3s p95 at 40 req/sec spike (680ms p95 under normal load)
```

> ⚠️ **When to stop iterating:** Stop when (a) latency SLA is met with 20% margin, (b) throughput exceeds target by 2×, or (c) next optimization's ROI < 10% improvement. Don't over-optimize — focus on shipping.

**Industry reality check:**

| Company | Typical iterations | Primary bottleneck | Final optimization stack |
|---------|-------------------|-------------------|-------------------------|
| **OpenAI (GPT-4)** | 4+ iterations | Queue management + decode speed | Continuous batching + speculative sampling + custom CUDA kernels |
| **Anthropic (Claude)** | 3 iterations | Memory fragmentation | PagedAttention + INT4 + multi-GPU pipeline parallelism |
| **Llama.cpp (local)** | 2 iterations | CPU-bound on consumer hardware | Quantization (Q4) + operator fusion (no batching needed for single-user) |
| **HuggingFace TGI** | 2–3 iterations | VRAM limits + queue spikes | PagedAttention + continuous batching (speculative decoding optional) |
| **InferenceBase** | 2 iterations | Queue spikes under lunch rush | Continuous batching + PagedAttention (speculative added for headroom) |

> 📖 **Optional depth — When to parallelize vs batch:** Batching (Phase 3) increases throughput by processing multiple requests simultaneously within one GPU. Parallelism (Ch.4) increases throughput by splitting one large model across multiple GPUs. Use batching first (cheaper, no inter-GPU communication overhead). Use parallelism when model size exceeds single-GPU VRAM or when batch=1 latency must be <100ms (real-time systems).

---

## 2 · Running Example

**Problem**: Lunch rush (40 req/sec spike) causes 8.7s p95 latency → misses 2s SLA.

**Root cause**: Static batch=4 → when 40 requests arrive, first batch starts immediately, but request #37 waits for 9 batches to complete (9 × 1.2s = 10.8s queue wait).

**Solution**:
1. **Continuous batching**: Start processing request #5 as soon as request #1 finishes generating (even if #2, #3, #4 still running)
2. **PagedAttention**: Eliminate padding waste → batch=8 instead of batch=4 → cut queue wait by 50%
3. **Speculative decoding**: Llama-3-1B drafts 3 tokens, Llama-3-8B verifies in parallel → 30% faster

**Result**: 8.7s p95 → 1.8s p95 under spike load ✅ (within 2s target!).

---

## 3 · **[Phase 1: PROFILE]** Continuous Batching

**Static batching** (naive approach):
```
Batch formation:
  Wait until batch_size=4 requests arrive
  Process all 4 together (1,200ms)
  Repeat

Timeline:
  t=0ms:    Request 1,2,3,4 arrive  → start batch
  t=1200ms: Batch completes         → start next batch
  t=1200ms: Request 5,6,7,8 arrive  → start batch
  t=2400ms: Batch completes

Request 8 latency: 2,400ms (waited 1,200ms in queue)
```

**Continuous batching** (vLLM approach):
```
Batch formation:
  Maintain active batch of up to batch_size=4 requests
  As soon as any request finishes, add next request from queue

Timeline:
  t=0ms:    Request 1,2,3,4 arrive → start batch (all 4 active)
  t=800ms:  Request 1 finishes    → add Request 5 (now 2,3,4,5 active)
  t=1000ms: Request 2 finishes    → add Request 6 (now 3,4,5,6 active)
  t=1200ms: Request 3 finishes    → add Request 7 (now 4,5,6,7 active)

Request 5 latency: 1,300ms (vs 2,400ms static batching)
Improvement: 46% faster!
```

**Why it works**: Requests have variable generation lengths (50–500 tokens). Continuous batching exploits this variability to keep GPU saturated.

---

### 3.12 · Profiling Baseline Performance **[Phase 1: PROFILE]**

Before optimizing, measure where time is actually spent. Inference has 4 components:

| Component | What it does | Typical % of latency |
|-----------|--------------|---------------------|
| **Tokenization** | Convert text → token IDs | 5–10% |
| **Prefill (TTFT)** | First forward pass (prompt encoding) | 20–30% |
| **Decode (TPOT)** | Autoregressive generation (one token per step) | 50–70% |
| **Detokenization** | Convert token IDs → text | 1–3% |

**InferenceBase baseline profile:**

```
Single request (150 output tokens):
- Tokenization:    42ms  (5%)
- Prefill (TTFT):  180ms (22%)
- Decode (TPOT):   140ms × 150 = 21,000ms (70%)  ← bottleneck!
- Detokenization:  28ms  (3%)
Total: 780ms per request

Batch=4 (same 150 tokens each):
- Tokenization:    42ms × 4 = 168ms  (still sequential, not batched)
- Prefill:         220ms (batched! only 22% slower despite 4× work)
- Decode:          180ms × 150 = 27,000ms (70% of total)
- Detokenization:  28ms × 4 = 112ms
Total: 1,200ms for 4 requests (300ms per request amortized)

Throughput: 4 requests / 1.2s = 3.33 req/sec = 12,000 req/day ✅
```

**Key insight**: Decode step is 70% of latency. Continuous batching + speculative decoding must target this.

> 💡 **Industry Standard: PyTorch Profiler**
> ```python
> import torch.profiler as profiler
>
> with profiler.profile(
>     activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
>     record_shapes=True
> ) as prof:
>     output = model.generate(input_ids, max_new_tokens=150)
>
> print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
> ```
> **When to use:** Always profile before optimizing. Measure wall-clock time, CUDA kernel time, and memory allocations.
> **Common alternatives:** NVIDIA Nsight Systems (full system trace), `torch.cuda.Event()` (manual timing), vLLM built-in stats API.

---

#### Code Snippet 1: Profiling with PyTorch Profiler + CUDA Events **[Phase 1: PROFILE]**

```python
# Phase 1: PROFILE — Baseline performance measurement
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cuda:0"
)

prompt = "Summarize the Q3 earnings report in 150 words:"

# CUDA event timing (most accurate for GPU workloads)
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# Component 1: Tokenization
start_tok = time.perf_counter()
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
tok_time = (time.perf_counter() - start_tok) * 1000  # ms

# Component 2+3: Prefill + Decode (combined in generate())
start_event.record()
with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_new_tokens=150,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
end_event.record()
torch.cuda.synchronize()
gen_time = start_event.elapsed_time(end_event)  # ms

# Component 4: Detokenization
start_detok = time.perf_counter()
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
detok_time = (time.perf_counter() - start_detok) * 1000  # ms

# PROFILE OUTPUT
print(f"=== BASELINE PROFILE ===")
print(f"Tokenization:    {tok_time:.1f}ms  ({tok_time/(tok_time+gen_time+detok_time)*100:.1f}%)")
print(f"Generate (TTFT+decode): {gen_time:.1f}ms  ({gen_time/(tok_time+gen_time+detok_time)*100:.1f}%)")
print(f"Detokenization:  {detok_time:.1f}ms  ({detok_time/(tok_time+gen_time+detok_time)*100:.1f}%)")
print(f"Total latency:   {tok_time+gen_time+detok_time:.1f}ms")
print(f"\nOutput length: {len(output_ids[0]) - len(input_ids[0])} tokens")
print(f"TPOT (time per output token): {gen_time / (len(output_ids[0]) - len(input_ids[0])):.1f}ms")

# Expected output:
# === BASELINE PROFILE ===
# Tokenization:    42.3ms  (5.1%)
# Generate (TTFT+decode): 738.2ms  (89.4%)
# Detokenization:  45.1ms  (5.5%)
# Total latency:   825.6ms
#
# Output length: 150 tokens
# TPOT (time per output token): 4.9ms
```

> 💡 **What the numbers tell you:**
> - **Tokenization >10%** → Switch to faster tokenizer (SentencePiece, tiktoken)
> - **TTFT (first-token) >500ms** → Prompt too long, or prefill not optimized (try FlashAttention)
> - **TPOT >5ms** → Decode bottleneck → Apply continuous batching + speculative decoding
> - **Detokenization >5%** → Batch detokenization or use streaming response (return tokens before decoding)

---

### 3.13 · Bottleneck Identification **[Phase 2: IDENTIFY]**

Profile breakdown from Phase 1 reveals where to optimize. Three bottleneck categories:

| Bottleneck type | Symptom | Root cause | Fix (this chapter) |
|----------------|---------|------------|-------------------|
| **CPU-bound** | High CPU util, low GPU util | Tokenization slow, or Python overhead | Batch tokenization, operator fusion (§6) |
| **Memory-bound** | VRAM at 95%+, batch size limited | KV cache fragmentation | PagedAttention (§4) |
| **I/O-bound** | Queue depth >10, requests waiting seconds | Static batching, no dynamic scheduling | Continuous batching (§3) |

**InferenceBase diagnosis (from Phase 1 profile):**

```
Load test under lunch rush (40 req/sec):
CPU utilization: 35% (plenty of headroom)
GPU utilization: 78% (decent, but not saturated)
VRAM usage: 10GB / 24GB (41% utilization)
Queue depth: 180 requests waiting

BOTTLENECK: I/O-bound (queue explosion from static batching)

Analysis:
- GPU is not maxed out (78%) → not compute-bound
- VRAM has 14GB free → not memory-bound
- Queue depth 180 → requests spending 8+ seconds in queue before processing

Root cause: Static batch=4 waits for batch to fill before processing.
            When 40 requests arrive simultaneously:
            - Batch 1-4: start immediately
            - Batch 5-8: wait 1.2s
            - Batch 9-12: wait 2.4s
            - ...
            - Batch 37-40: wait 10.8s ← misses 2s SLA by 5×!

Fix: Continuous batching (§3) — add requests as soon as any slot frees.
```

> ⚠️ **Common misdiagnosis:** "High latency = slow model" → assumes compute-bound. But InferenceBase's GPU is only 78% utilized — the bottleneck is *queue management*, not model speed. Always profile before assuming the bottleneck type.

---

#### Code Snippet 2: Bottleneck Analysis Script **[Phase 2: IDENTIFY]**

```python
# Phase 2: IDENTIFY — Component timing breakdown and bottleneck classification
import torch
import time
import psutil
import numpy as np
from collections import deque

class InferenceProfiler:
    """
    Tracks component-level timing and resource utilization to identify bottlenecks.
    """
    def __init__(self, model, tokenizer, device="cuda:0"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.metrics = {
            "tokenization_ms": [],
            "prefill_ms": [],
            "decode_ms": [],
            "detokenization_ms": [],
            "queue_wait_ms": [],
            "vram_used_gb": [],
            "gpu_util_pct": []
        }

    def profile_request(self, prompt: str, max_tokens: int = 150):
        """Profile a single request through all components."""
        # Tokenization
        t0 = time.perf_counter()
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        tok_time = (time.perf_counter() - t0) * 1000

        # Prefill (first forward pass)
        start_prefill = torch.cuda.Event(enable_timing=True)
        end_prefill = torch.cuda.Event(enable_timing=True)

        start_prefill.record()
        with torch.no_grad():
            # Run one forward pass to get first token
            outputs = self.model(input_ids)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        end_prefill.record()
        torch.cuda.synchronize()
        prefill_time = start_prefill.elapsed_time(end_prefill)

        # Decode (autoregressive generation)
        start_decode = torch.cuda.Event(enable_timing=True)
        end_decode = torch.cuda.Event(enable_timing=True)

        generated = input_ids
        start_decode.record()
        for _ in range(max_tokens - 1):
            with torch.no_grad():
                outputs = self.model(generated)
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        end_decode.record()
        torch.cuda.synchronize()
        decode_time = start_decode.elapsed_time(end_decode)

        # Detokenization
        t0 = time.perf_counter()
        output_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        detok_time = (time.perf_counter() - t0) * 1000

        # Resource utilization
        vram_used = torch.cuda.memory_allocated(self.device) / 1e9  # GB

        # Store metrics
        self.metrics["tokenization_ms"].append(tok_time)
        self.metrics["prefill_ms"].append(prefill_time)
        self.metrics["decode_ms"].append(decode_time)
        self.metrics["detokenization_ms"].append(detok_time)
        self.metrics["vram_used_gb"].append(vram_used)

        return {
            "total_ms": tok_time + prefill_time + decode_time + detok_time,
            "output_tokens": len(generated[0]) - len(input_ids[0])
        }

    def diagnose_bottleneck(self):
        """Analyze metrics and classify bottleneck type."""
        avg_tok = np.mean(self.metrics["tokenization_ms"])
        avg_prefill = np.mean(self.metrics["prefill_ms"])
        avg_decode = np.mean(self.metrics["decode_ms"])
        avg_detok = np.mean(self.metrics["detokenization_ms"])
        total = avg_tok + avg_prefill + avg_decode + avg_detok

        # Component breakdown
        print("\n=== BOTTLENECK DIAGNOSIS ===")
        print(f"Tokenization:    {avg_tok:.1f}ms  ({avg_tok/total*100:.1f}%)")
        print(f"Prefill (TTFT):  {avg_prefill:.1f}ms  ({avg_prefill/total*100:.1f}%)")
        print(f"Decode (TPOT):   {avg_decode:.1f}ms  ({avg_decode/total*100:.1f}%)  ← Primary focus")
        print(f"Detokenization:  {avg_detok:.1f}ms  ({avg_detok/total*100:.1f}%)")
        print(f"Total:           {total:.1f}ms")

        # VRAM analysis
        avg_vram = np.mean(self.metrics["vram_used_gb"])
        vram_total = torch.cuda.get_device_properties(self.device).total_memory / 1e9
        print(f"\nVRAM: {avg_vram:.1f}GB / {vram_total:.1f}GB ({avg_vram/vram_total*100:.1f}% utilized)")

        # DECISION LOGIC (inline annotation)
        print("\n→ BOTTLENECK CLASSIFICATION:")
        if avg_tok / total > 0.15:
            print("  ❌ CPU-BOUND (tokenization): Switch to faster tokenizer or batch tokenization")
        elif avg_vram / vram_total > 0.85:
            print("  ❌ MEMORY-BOUND: Apply PagedAttention (§4) to reduce KV cache fragmentation")
        elif avg_decode / total > 0.60:
            if len(self.metrics["queue_wait_ms"]) > 0 and np.mean(self.metrics["queue_wait_ms"]) > 1000:
                print("  ❌ I/O-BOUND (queue explosion): Apply continuous batching (§3)")
            else:
                print("  ❌ COMPUTE-BOUND (decode): Apply speculative decoding (§5) + operator fusion")
        else:
            print("  ✅ BALANCED: No single bottleneck >60%. Consider profiling under load.")

        return {
            "tokenization_pct": avg_tok / total * 100,
            "decode_pct": avg_decode / total * 100,
            "vram_util_pct": avg_vram / vram_total * 100
        }

# Usage example
profiler = InferenceProfiler(model, tokenizer)

# Run 10 sample requests
prompts = [
    "Summarize the Q3 earnings report:",
    "What are the key risks mentioned?",
    # ... 8 more prompts
]

for prompt in prompts:
    profiler.profile_request(prompt, max_tokens=150)

# Diagnose bottleneck
diagnosis = profiler.diagnose_bottleneck()

# Expected output:
# === BOTTLENECK DIAGNOSIS ===
# Tokenization:    42.3ms  (5.1%)
# Prefill (TTFT):  180.2ms (21.8%)
# Decode (TPOT):   580.5ms (70.2%)  ← Primary focus
# Detokenization:  23.8ms  (2.9%)
# Total:           826.8ms
#
# VRAM: 10.2GB / 24.0GB (42.5% utilized)
#
# → BOTTLENECK CLASSIFICATION:
#   ❌ COMPUTE-BOUND (decode): Apply speculative decoding (§5) + operator fusion
```

> 💡 **Industry Standard: vLLM Stats API**
> ```python
> from vllm import LLM, SamplingParams
>
> llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")
> stats = llm.get_stats()  # Returns queue depth, batch size, TPOT, TTFT
>
> print(f"Queue depth: {stats.num_waiting_requests}")
> print(f"Active batch size: {stats.num_running_requests}")
> print(f"Avg TPOT: {stats.avg_time_per_output_token_ms:.1f}ms")
> ```
> **When to use:** Production monitoring. vLLM tracks stats automatically; export to Prometheus/Grafana for dashboards.
> **Common alternatives:** TensorRT-LLM profiler, HuggingFace TGI `/metrics` endpoint, custom logging with `torch.profiler`.

---

### 3.13.1 DECISION CHECKPOINT — Phase 2 Complete **[Phase 2: IDENTIFY]**

**What you just saw:**
- Baseline profile: 826ms total latency, 70% decode (580ms), 22% prefill (180ms), 8% tokenization+detok
- VRAM utilization: 42.5% (10.2GB / 24GB) — plenty of headroom
- Under load: Queue depth explodes to 180 requests, p95 latency 8.7s

**What it means:**
- **Primary bottleneck: I/O-bound (queue management)** — GPU only 78% utilized, but requests wait 8+ seconds in queue
- **Secondary opportunity: Decode speed** — 70% of latency is autoregressive generation; can be accelerated with speculative decoding
- **Not memory-bound yet** — 58% VRAM headroom means we can increase batch size safely

**What to do next:**
→ **Phase 3 optimization priority:**
  1. **First:** Apply continuous batching (§3) — eliminates queue wait spikes (expect 70% latency reduction under load)
  2. **Second:** Apply PagedAttention (§4) — enables batch=8 instead of batch=4 (expect 2× throughput)
  3. **Third:** Apply speculative decoding (§5) — accelerates decode step (expect 30% further speedup)

→ For **CPU-bound** scenarios (tokenization >15%): Batch tokenization preprocessing or switch to `tiktoken` (10× faster than HuggingFace)
→ For **memory-bound** scenarios (VRAM >85%): Must apply PagedAttention first, then assess if continuous batching fits

**InferenceBase decision:** Proceed with all three optimizations in sequence — bottleneck analysis confirms we have the resources (VRAM, GPU headroom) to stack them without conflicts.

---

## 4 · **[Phase 3: OPTIMIZE]** PagedAttention

**Problem**: KV cache is allocated as contiguous memory blocks. Short sequences (50 tokens) are padded to max length (2048 tokens) → 97% waste!

```
Without PagedAttention (contiguous allocation):
  Request 1: 50 tokens   → allocate 2048-token block (waste 1998 slots)
  Request 2: 200 tokens  → allocate 2048-token block (waste 1848 slots)
  Request 3: 1500 tokens → allocate 2048-token block (waste 548 slots)
  Request 4: 80 tokens   → allocate 2048-token block (waste 1968 slots)

  Total: 4 × 4GB = 16GB KV cache
         → Only 1,830 tokens actually used (10% efficiency!)
```

**PagedAttention** (vLLM innovation):
- Divide KV cache into 64-token **pages**
- Allocate pages on-demand as sequences grow
- Page table maps logical sequence positions → physical pages (like OS virtual memory)

```
With PagedAttention (paged allocation):
  Request 1: 50 tokens   → 1 page (64 slots, 14 wasted)
  Request 2: 200 tokens  → 4 pages (256 slots, 56 wasted)
  Request 3: 1500 tokens → 24 pages (1536 slots, 36 wasted)
  Request 4: 80 tokens   → 2 pages (128 slots, 48 wasted)

  Total: 31 pages × 128MB = 3.97GB KV cache
         → 1,830 tokens used (92% efficiency! ✅)
```

**Impact**: Same 24GB VRAM budget → batch=4 (16GB KV) → batch=8 (8GB KV with PagedAttention).

![PagedAttention memory layout — Paged allocation eliminates fragmentation, enabling 2× larger batch size](img/pagedattention-memory-layout.png)

> 💡 **Industry Standard: vLLM (PagedAttention built-in)**
> ```python
> from vllm import LLM, SamplingParams
>
> # PagedAttention is enabled by default in vLLM
> llm = LLM(
>     model="meta-llama/Meta-Llama-3-8B-Instruct",
>     max_num_seqs=8,        # batch size (was 4 before PagedAttention)
>     block_size=16,         # page size in tokens (default: 16)
>     gpu_memory_utilization=0.90  # use 90% VRAM, leave 10% headroom
> )
> ```
> **When to use:** Always in production. PagedAttention is vLLM's core innovation — zero code changes needed, just set `max_num_seqs` higher than your VRAM would normally allow.
> **Common alternatives:** TensorRT-LLM (similar paging), TGI (uses continuous batching but not PagedAttention — less efficient), llama.cpp (no paging, designed for single-user edge deployment).

---

## 5 · **[Phase 3: OPTIMIZE]** Speculative Decoding

**Problem**: LLM generation is autoregressive — must generate one token at a time. 100 tokens = 100 forward passes → slow.

**Insight**: Most tokens are "easy" (high probability). Can a small model draft multiple tokens, then verify with large model in parallel?

```
Standard decoding (Llama-3-8B only):
  t=0ms:   Generate token 1  (140ms)
  t=140ms: Generate token 2  (140ms)
  t=280ms: Generate token 3  (140ms)
  Total: 420ms for 3 tokens

Speculative decoding (Llama-3-1B draft + Llama-3-8B verify):
  t=0ms:   Draft model generates 3 tokens (30ms × 3 = 90ms)
  t=90ms:  Large model verifies all 3 in parallel (180ms)
  t=270ms: Accept correct tokens, reject wrong ones

  Total: 270ms for 2.5 tokens (avg) → 35% speedup!
```

**When it works**:
- Draft model = distilled version (Llama-3-1B) of large model (Llama-3-8B)
- Task has high token predictability (code, structured output, translations)
- Acceptance rate >70% (if <50%, overhead dominates)

**InferenceBase scenario**: Document extraction has structured output (JSON) → 75% acceptance rate → 30% speedup ✅

![Speculative decoding flow — Llama-3-1B draft model predicts 3 tokens, Llama-3-8B verifies in parallel](img/speculative-decoding-flow.png)

> 💡 **Industry Standard: vLLM + Speculative Decoding**
> ```python
> from vllm import LLM, SamplingParams
>
> llm = LLM(
>     model="meta-llama/Meta-Llama-3-8B-Instruct",
>     speculative_model="meta-llama/Meta-Llama-3-1B",  # draft model
>     num_speculative_tokens=3,                         # tokens to draft ahead
>     speculative_disable_by_batch_size=8               # disable if batch >8 (overhead)
> )
>
> # Usage same as before — speculative decoding is transparent
> outputs = llm.generate(prompts, SamplingParams(temperature=0.7, max_tokens=150))
> ```
> **When to use:** Structured output tasks (JSON, code, translations). Measure acceptance rate first — if <60%, skip it.
> **Common alternatives:** TensorRT-LLM (supports speculative), TGI (experimental support), Medusa (multi-head speculative, 2024).

---

### 5.1 · Operator Fusion (Bonus Optimization) **[Phase 3: OPTIMIZE]**

**Problem**: Standard PyTorch/Transformers executes each operation (matmul, softmax, layernorm) as a separate CUDA kernel → kernel launch overhead + memory round-trips.

**Solution**: Fuse multiple operations into a single CUDA kernel → fewer launches, data stays in GPU cache.

Example fused operations:
- **FlashAttention**: Fuses softmax + matmul + dropout into one kernel → 2–4× faster attention
- **LayerNorm + Linear**: Fuses normalization + weight multiply → 20% faster feedforward
- **RMSNorm + RoPE**: Common in Llama models, fused in vLLM/TRT-LLM

**InferenceBase impact**: FlashAttention reduces prefill latency by 40% (180ms → 108ms), but decode latency unchanged (already memory-bound, not compute-bound).

> ⚠️ **When operator fusion helps:** Prefill (attention over long prompts). Decode step is memory-bound (fetching weights from VRAM), so fusion has minimal impact there. Focus on continuous batching + speculative decoding for decode speedup.

> 💡 **Industry Standard: FlashAttention-2 (automatic in vLLM)**
> ```python
> # FlashAttention is enabled by default in vLLM if installed
> llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")  # FlashAttention auto-detected
> ```
> **When to use:** Always. FlashAttention-2 is a drop-in replacement for standard attention with no downsides.
> **Common alternatives:** xFormers (Facebook), CUTLASS (NVIDIA), custom CUDA kernels for non-attention ops.

---

#### Code Snippet 3: INT8 Quantization with bitsandbytes **[Phase 3: OPTIMIZE]**

```python
# Phase 3: OPTIMIZE — Apply INT8 quantization (bridges to Ch.3 content)
# Note: This snippet demonstrates quantization as an optimization technique;
#       InferenceBase already uses INT4 from Ch.3, so this is for reference.

from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# INT8 quantization config
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,               # Enable 8-bit quantization
    llm_int8_threshold=6.0,          # Outlier threshold (default 6.0)
    llm_int8_skip_modules=["lm_head"]  # Skip final layer (quality preservation)
)

# Load model with INT8 quantization
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    quantization_config=quant_config,
    device_map="cuda:0"
)

# Memory savings: FP16 (16GB) → INT8 (8GB) → 50% reduction
print(f"Model memory: {model.get_memory_footprint() / 1e9:.1f} GB")

# Inference same as before (quantization is transparent)
input_ids = tokenizer("Summarize:", return_tensors="pt").input_ids.to("cuda")
output_ids = model.generate(input_ids, max_new_tokens=150)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Performance impact:
# - Latency: ~10-15% slower per token (INT8 math slightly slower than FP16 on RTX 4090)
# - Throughput: ~1.8× higher (2× memory savings allows larger batch size)
# - Accuracy: ~0.5pp drop (95.2% → 94.7% on InferenceBase benchmark)

# When to use INT8 instead of INT4:
# - INT4 (Ch.3): 4× memory reduction, ~1-2% accuracy drop, best for throughput
# - INT8: 2× memory reduction, ~0.5% accuracy drop, best when quality > throughput
# - FP16: No savings, highest quality, use only if VRAM unlimited
```

> ⚠️ **InferenceBase already uses INT4 (Ch.3)**, which is more aggressive than INT8. This snippet is included for practitioners working with other models where INT8 is the quality-throughput sweet spot. For InferenceBase, INT4 + PagedAttention + continuous batching is the optimal stack.

---

## 6 · KV Cache Management Strategies

| Strategy | Pros | Cons | When to use |
|----------|------|------|-------------|
| **Recompute** | Zero KV cache memory | 50% slower (recompute attention) | Extreme memory pressure |
| **Cache all** | Fastest (no recomputation) | High memory (4GB per request) | Batch=1, plenty of VRAM |
| **Rolling window** | Cap KV cache at 512 tokens | Loses context beyond window | Streaming tasks (chatbots) |
| **PagedAttention** | 90% memory efficiency | Slightly slower (paging overhead) | Production (default choice) |

**InferenceBase choice**: PagedAttention (batch=8 vs batch=4, worth 5% latency overhead).

---

## 7 · Implementing Continuous Batching (Pseudocode)

```python
class ContinuousBatchEngine:
    def __init__(self, model, max_batch_size=8):
        self.model = model
        self.max_batch_size = max_batch_size
        self.active_requests = []  # Currently generating
        self.queue = Queue()       # Waiting requests

    def add_request(self, prompt):
        request = Request(prompt, status="queued")
        self.queue.put(request)
        return request

    def step(self):
        # Remove finished requests
        self.active_requests = [r for r in self.active_requests if not r.finished]

        # Fill empty slots from queue
        while len(self.active_requests) < self.max_batch_size and not self.queue.empty():
            self.active_requests.append(self.queue.get())

        if not self.active_requests:
            return

        # Generate one token for all active requests
        prompts = [r.get_context() for r in self.active_requests]
        next_tokens = self.model.generate_batch(prompts, max_new_tokens=1)

        # Update each request
        for request, token in zip(self.active_requests, next_tokens):
            request.append_token(token)
            if token == EOS or request.length >= request.max_tokens:
                request.finish()

    def run(self):
        while True:
            self.step()
```

**Key insight**: `step()` generates 1 token per request → finished requests free slots immediately → new requests join mid-generation.

---

## 8 · Latency Analysis: Static vs Continuous Batching

**Scenario**: 12 requests arrive simultaneously, batch_size=4, 300ms per batch.

**Static batching**:
```
Batch 1 (req 1-4):   t=0     → t=300   (latency: 300ms)
Batch 2 (req 5-8):   t=300   → t=600   (latency: 600ms for req 5)
Batch 3 (req 9-12):  t=600   → t=900   (latency: 900ms for req 9)

Average latency: (300 + 600 + 900) / 3 = 600ms
P95 latency: 900ms
```

**Continuous batching** (assume 100ms per request finishes staggered):
```
t=0:   Start req 1,2,3,4
t=100: Req 1 done → add req 5  (latency: 100ms)
t=200: Req 2 done → add req 6  (latency: 200ms)
t=300: Req 3 done → add req 7  (latency: 300ms)
t=400: Req 4 done → add req 8  (latency: 400ms for req 4)
t=500: Req 5 done → add req 9  (latency: 500ms for req 5)
t=600: Req 6 done → add req 10 (latency: 600ms for req 6)
...

Average latency: ~350ms (vs 600ms static)
P95 latency: 500ms (vs 900ms static)

Improvement: 42% better tail latency! ✅
```

---

## 9 · **[Phase 4: VALIDATE]** Benchmarking Inference Optimizations

**Test setup**: Llama-3-8B INT4, 1000 requests (avg 150 tokens), spiky arrival (λ=20 req/sec for 50s)

**Test document**: ACME Corp Q3 earnings report (47 pages, 28,000 tokens) — InferenceBase's canonical benchmark document for document extraction workloads

| Configuration | Throughput | p50 Latency | p95 Latency | VRAM |
|---------------|------------|-------------|-------------|------|
| Baseline (batch=1, no opt) | 3,800 req/day | 280ms | 450ms | 10GB |
| Static batch=4 | 12,000 req/day | 420ms | 1,200ms | 10GB |
| Continuous batch=4 | 13,500 req/day | 380ms | 720ms | 10GB |
| Continuous + PagedAttention (batch=8) | 18,000 req/day ✅ | 450ms | 950ms | 8GB |
| + Speculative decoding | 22,000 req/day ✅ | 320ms | 680ms ✅ | 10GB |

**Winner**: Continuous batching + PagedAttention + speculative decoding → **22,000 req/day, 680ms p95** ✅

![Benchmark results — Throughput and latency improvements from baseline to fully optimized configuration](img/optimization-benchmark-results.png)

> 💡 **Industry Standard: Locust (load testing framework)**
> ```python
> from locust import HttpUser, task, between
>
> class InferenceLoadTest(HttpUser):
>     wait_time = between(0.1, 0.5)  # Spiky traffic: 2-10 req/sec per user
>
>     @task
>     def generate(self):
>         self.client.post("/generate", json={
>             "prompt": "Summarize Q3 earnings:",
>             "max_tokens": 150
>         })
>
> # Run: locust -f load_test.py --users 40 --spawn-rate 10 --host http://localhost:8000
> ```
> **When to use:** Always load-test before production. Simulate realistic traffic patterns (spikes, long-tail distributions).
> **Common alternatives:** Apache JMeter, Gatling, custom Python scripts with `asyncio` + `aiohttp`.

---

#### Code Snippet 4: A/B Test Harness (Accuracy + Latency Regression) **[Phase 4: VALIDATE]**

```python
# Phase 4: VALIDATE — Performance regression testing after optimization
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class BenchmarkResult:
    config_name: str
    throughput_req_per_day: float
    latency_p50_ms: float
    latency_p95_ms: float
    accuracy_pct: float
    vram_gb: float

class ABTestHarness:
    """
    Compare baseline vs optimized configurations on accuracy and latency.
    """
    def __init__(self, baseline_model, optimized_model, test_dataset):
        self.baseline = baseline_model
        self.optimized = optimized_model
        self.test_dataset = test_dataset  # List of (prompt, expected_output) tuples

    def measure_accuracy(self, model, name: str) -> float:
        """Measure exact-match accuracy on test dataset."""
        correct = 0
        for prompt, expected in self.test_dataset:
            output = model.generate(prompt, max_tokens=150)
            if self._fuzzy_match(output, expected):
                correct += 1
        accuracy = correct / len(self.test_dataset) * 100
        print(f"{name} accuracy: {accuracy:.1f}% ({correct}/{len(self.test_dataset)})")
        return accuracy

    def measure_latency(self, model, name: str, num_requests: int = 100) -> Tuple[float, float]:
        """Measure p50 and p95 latency over N requests."""
        latencies = []
        for i in range(num_requests):
            prompt = self.test_dataset[i % len(self.test_dataset)][0]

            start = time.perf_counter()
            _ = model.generate(prompt, max_tokens=150)
            latency_ms = (time.perf_counter() - start) * 1000

            latencies.append(latency_ms)

        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        print(f"{name} latency: p50={p50:.1f}ms, p95={p95:.1f}ms")
        return p50, p95

    def run_ab_test(self) -> Tuple[BenchmarkResult, BenchmarkResult]:
        """Run full A/B test: accuracy + latency for baseline vs optimized."""
        print("=== A/B TEST: BASELINE VS OPTIMIZED ===\n")

        # Phase 4.1: Accuracy check
        print("[Phase 4.1] Accuracy validation...")
        baseline_acc = self.measure_accuracy(self.baseline, "Baseline")
        optimized_acc = self.measure_accuracy(self.optimized, "Optimized")
        accuracy_delta = optimized_acc - baseline_acc

        # DECISION LOGIC
        if accuracy_delta < -0.5:
            print(f"\n⚠️  ACCURACY REGRESSION: {accuracy_delta:.2f}pp drop (threshold: -0.5pp)")
            print("   → Reject optimization. Investigate quantization settings or draft model quality.")
        else:
            print(f"\n✅ ACCURACY MAINTAINED: {accuracy_delta:+.2f}pp delta (within ±0.5pp threshold)")

        # Phase 4.2: Latency check
        print("\n[Phase 4.2] Latency validation...")
        baseline_p50, baseline_p95 = self.measure_latency(self.baseline, "Baseline", num_requests=100)
        optimized_p50, optimized_p95 = self.measure_latency(self.optimized, "Optimized", num_requests=100)

        # DECISION LOGIC
        latency_improvement = (baseline_p95 - optimized_p95) / baseline_p95 * 100
        sla_target_ms = 2000

        if optimized_p95 > sla_target_ms:
            print(f"\n⚠️  LATENCY SLA MISS: {optimized_p95:.1f}ms p95 (target: <{sla_target_ms}ms)")
            print("   → Iterate Phase 2→3: Re-profile and apply additional optimizations.")
        elif latency_improvement < 10:
            print(f"\n⚠️  LOW ROI: Only {latency_improvement:.1f}% improvement (threshold: >10%)")
            print("   → Consider skipping this optimization (overhead may not be worth complexity).")
        else:
            print(f"\n✅ LATENCY IMPROVED: {latency_improvement:.1f}% faster p95 (target: <{sla_target_ms}ms)")

        # Phase 4.3: Throughput check
        print("\n[Phase 4.3] Throughput calculation...")
        baseline_throughput = (1000 / baseline_p50) * 86400  # req/day
        optimized_throughput = (1000 / optimized_p50) * 86400
        throughput_gain = (optimized_throughput - baseline_throughput) / baseline_throughput * 100

        print(f"Baseline:  {baseline_throughput:,.0f} req/day")
        print(f"Optimized: {optimized_throughput:,.0f} req/day (+{throughput_gain:.1f}%)")

        # DECISION LOGIC
        throughput_target = 10000  # req/day
        if optimized_throughput < throughput_target:
            print(f"\n⚠️  THROUGHPUT BELOW TARGET: {optimized_throughput:,.0f} req/day (target: {throughput_target:,})")
            print("   → Iterate Phase 2→3: Increase batch size or add speculative decoding.")
        else:
            print(f"\n✅ THROUGHPUT TARGET MET: {optimized_throughput:,.0f} req/day (target: {throughput_target:,})")

        # Phase 4.4: Final decision
        print("\n=== FINAL DECISION ===")
        if accuracy_delta >= -0.5 and optimized_p95 <= sla_target_ms and optimized_throughput >= throughput_target:
            print("✅ ACCEPT OPTIMIZATION: All criteria met (accuracy, latency SLA, throughput target)")
            print("   → Deploy to production with 20% traffic canary rollout.")
        else:
            print("❌ REJECT OR ITERATE: One or more criteria failed.")
            print("   → Return to Phase 2 (IDENTIFY) with new profile data.")

        return (
            BenchmarkResult("Baseline", baseline_throughput, baseline_p50, baseline_p95, baseline_acc, 10.0),
            BenchmarkResult("Optimized", optimized_throughput, optimized_p50, optimized_p95, optimized_acc, 8.0)
        )

    def _fuzzy_match(self, output: str, expected: str, threshold: float = 0.85) -> bool:
        """Fuzzy string match (allows minor wording differences)."""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, output, expected).ratio() >= threshold

# Usage example
# test_dataset = [
#     ("Summarize Q3 earnings:", "Q3 revenue $2.1B, net income $450M..."),
#     # ... 99 more test cases
# ]
#
# harness = ABTestHarness(baseline_model, optimized_model, test_dataset)
# baseline_result, optimized_result = harness.run_ab_test()

# Expected output:
# === A/B TEST: BASELINE VS OPTIMIZED ===
#
# [Phase 4.1] Accuracy validation...
# Baseline accuracy: 96.2% (962/1000)
# Optimized accuracy: 95.8% (958/1000)
#
# ✅ ACCURACY MAINTAINED: -0.4pp delta (within ±0.5pp threshold)
#
# [Phase 4.2] Latency validation...
# Baseline latency: p50=420ms, p95=1200ms
# Optimized latency: p50=320ms, p95=680ms
#
# ✅ LATENCY IMPROVED: 43.3% faster p95 (target: <2000ms)
#
# [Phase 4.3] Throughput calculation...
# Baseline:  12,000 req/day
# Optimized: 22,000 req/day (+83.3%)
#
# ✅ THROUGHPUT TARGET MET: 22,000 req/day (target: 10,000)
#
# === FINAL DECISION ===
# ✅ ACCEPT OPTIMIZATION: All criteria met (accuracy, latency SLA, throughput target)
#    → Deploy to production with 20% traffic canary rollout.
```

> 💡 **What the A/B test validates:**
> - **Accuracy regression**: Optimized model must stay within 0.5pp of baseline (avoids quality degradation from quantization/speculative errors)
> - **Latency SLA**: p95 latency must be <2s under load (business requirement)
> - **Throughput target**: Must exceed 10k req/day (capacity planning requirement)
> - **ROI threshold**: Latency improvement must be >10% to justify deployment complexity

---

### 9.1 DECISION CHECKPOINT — Phase 4 Complete **[Phase 4: VALIDATE]**

**What you just saw:**
- Baseline: 12,000 req/day, 1,200ms p95, 96.2% accuracy
- After continuous batching: 13,500 req/day (+12.5%), 720ms p95 (40% improvement)
- After + PagedAttention (batch=8): 18,000 req/day (+50%), 950ms p95
- After + speculative decoding: 22,000 req/day (+83%), 680ms p95 (43% improvement), 95.8% accuracy

**What it means:**
- **Accuracy maintained**: 96.2% → 95.8% (-0.4pp) — within acceptable threshold (±0.5pp)
- **Latency SLA met**: 680ms p95 << 2,000ms target (66% better than target!)
- **Throughput exceeded**: 22,000 req/day >> 10,000 target (220% of requirement)
- **VRAM optimized**: 10GB → 8GB (PagedAttention eliminated 2GB fragmentation waste)

**What to do next:**
→ **Accept optimization stack** ✅ — all criteria met (accuracy, latency, throughput)
→ **Deploy to production** with 20% traffic canary rollout (validate under real user load)
→ **Monitor for 48 hours**: Track p95 latency, accuracy on live traffic, queue depth during lunch rush
→ **Full rollout** if canary shows no regressions (expect 100% rollout by week 2)

→ For scenarios where **latency still misses SLA**: Return to Phase 2 (IDENTIFY) → profile with new configuration → find remaining bottleneck (e.g., network I/O, detokenization) → apply targeted fix
→ For scenarios where **accuracy drops >0.5pp**: Investigate speculative decoding acceptance rate (may need to tune draft depth or switch draft model) or quantization settings (INT4 → INT8 if quality critical)

**InferenceBase final decision:** Deploy continuous batching + PagedAttention + speculative decoding to production. Lunch rush validated at 40 req/sec with 1.8s p95 latency (within 2s SLA). Ready for Ch.6 serving framework selection (vLLM is the leading candidate given built-in PagedAttention support).

---

## 10 · Key Diagrams

### Continuous Batching vs Static Batching

```
Static Batching (naive):
  Queue: [1][2][3][4] | [5][6][7][8] | [9][10][11][12]
         ↓ wait      ↓ wait         ↓ wait
  Batch: [1,2,3,4]──300ms──>[done]
                            [5,6,7,8]──300ms──>[done]
                                              [9,10,11,12]──300ms──>[done]

  Request 9 latency: 900ms (waited through 2 full batches)

Continuous Batching (vLLM):
  Queue: [1][2][3][4][5][6][7][8][9][10][11][12]
  Batch: [1,2,3,4]
         ↓ (1 finishes at t=100ms)
         [2,3,4,5] ← immediately add 5
         ↓ (2 finishes at t=200ms)
         [3,4,5,6] ← immediately add 6
         ...

  Request 5 latency: 350ms (joined as soon as slot freed)

Improvement: 2.5× lower tail latency! ✅
```

---

## 11 · What Can Go Wrong

- **Implementing continuous batching from scratch** — extremely complex (attention mask management, KV cache indexing); use vLLM/TGI instead
- **Assuming speculative decoding always helps** — if draft model acceptance rate <50%, overhead dominates → measure on your workload first
- **Not tuning batch size** — too large → OOM, too small → underutilized GPU; profile to find sweet spot
- **Forgetting request timeouts** — long-running requests can block batch slots; set max_tokens limits
- **PagedAttention without framework support** — requires custom CUDA kernels; only available in vLLM (2023+)

---

## The Hyperparameter Dial

Three knobs govern the throughput-latency-memory trade-off in inference optimization.

### Dial 1 — Max Batch Size

| Max batch | KV cache VRAM | Throughput (tok/s) | p95 Latency | Notes |
|-----------|--------------|---------------------|-------------|-------|
| 4 | ~4 GB | ~130 tok/s | ~600 ms | Low-traffic baseline |
| 8 | ~8 GB | ~220 tok/s | ~900 ms | InferenceBase target (12k req/day) |
| 16 | ~16 GB | ~350 tok/s | ~1.8 s | Requires INT4; approaching OOM at BF16 |
| 32 | ~32 GB ❌ | — | — | OOM on single RTX 4090 |

> 💡 Use Little's Law: $L = \lambda W$. If arrival rate $\lambda = 10$ req/s and service time $W = 0.8$ s, queue depth $L = 8$. Match max_batch_size to expected $L$ under peak traffic.

### Dial 2 — PagedAttention Block Size

PagedAttention divides KV cache into fixed-size pages. Block size controls the granularity:

| Block size (tokens) | Fragmentation | Overhead | Best for |
|--------------------|--------------|----------|----------|
| 16 | Low (fine-grained) | High (many page table entries) | Short/variable requests |
| 32 | Medium | Medium | General-purpose (vLLM default) |
| 64 | Higher | Low | Long-context serving ($S > 2048$) |
| 128 | Highest | Lowest | Offline batch jobs, max throughput |

> ⚠️ High fragmentation (block=16) wastes memory even though pages are small — each pre-allocated block can be half-empty on short requests. Block=32 is the empirically validated sweet spot.

### Dial 3 — Speculative Decoding Draft Depth

| Draft depth (tokens ahead) | Acceptance rate (typical) | Speedup | Memory overhead |
|---------------------------|--------------------------|---------|-----------------|
| 1 | ~85% | ~1.1× | +draft model VRAM |
| 2 | ~75% | ~1.2× | +draft model VRAM |
| 3 | ~65% | ~1.25× | +draft model VRAM |
| 4 | ~55% | ~1.2× (plateaus) | +draft model VRAM |
| 5+ | <50% | <1.1× (overhead dominates) | Diminishing returns |

> ⚠️ Draft depth must be tuned on your workload. On code generation, acceptance rate is ~75% at depth=3 (code is predictable). On open-ended chat, it drops to ~55% at depth=2.

---

## Code Skeleton

```python
# Educational: simplified continuous batching queue simulation
from collections import deque
import time

class SimpleBatchQueue:
    """
    Demonstrates the core logic of continuous batching:
    process requests in dynamic batches without waiting for full batch.
    """
    def __init__(self, max_batch_size: int = 8):
        self.queue = deque()
        self.max_batch_size = max_batch_size

    def add_request(self, request_id: str, tokens: int):
        self.queue.append({"id": request_id, "tokens": tokens, "arrived": time.time()})

    def get_next_batch(self) -> list:
        """Return up to max_batch_size ready requests."""
        batch, total_tokens = [], 0
        while self.queue and len(batch) < self.max_batch_size:
            req = self.queue.popleft()
            if total_tokens + req["tokens"] <= 2048 * self.max_batch_size:
                batch.append(req)
                total_tokens += req["tokens"]
            else:
                self.queue.appendleft(req)  # put it back
                break
        return batch
```

```python
# Production: vLLM AsyncLLMEngine with continuous batching and PagedAttention
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
import asyncio

engine_args = AsyncEngineArgs(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    quantization="gptq",                    # INT4 from Ch.3
    max_model_len=4096,
    max_num_seqs=8,                         # max concurrent requests (Dial 1)
    block_size=32,                          # PagedAttention block size (Dial 2)
    speculative_model="meta-llama/Meta-Llama-3-1B",  # draft model (Dial 3)
    num_speculative_tokens=3,               # draft depth
    gpu_memory_utilization=0.90,            # 90% VRAM, 10% headroom
    tensor_parallel_size=1,                 # single RTX 4090
)

engine = AsyncLLMEngine.from_engine_args(engine_args)

async def generate(prompt: str, max_tokens: int = 256) -> str:
    sampling = SamplingParams(temperature=0.7, max_tokens=max_tokens)
    request_id = f"req-{id(prompt)}"
    async for output in engine.generate(prompt, sampling, request_id):
        if output.finished:
            return output.outputs[0].text
```

---

## Where This Reappears

| Chapter | How inference optimization concepts appear |
|---------|--------------------------------------------|
| **Ch.2 — Memory Budgets** | PagedAttention page pool size is calculated from the VRAM budget formulas in Ch.2; pages must fit within remaining KV cache headroom |
| **Ch.6 — vLLM & Serving** | vLLM is the production implementation of continuous batching + PagedAttention + speculative decoding — all concepts from this chapter |
| **AI track — Cost & Latency** | Throughput (req/s) under continuous batching directly sets the cost-per-request; the relationship is cost = hourly_rate / throughput |
| **Multi-agent AI track** | Agentic multi-turn conversations consume many KV cache tokens per session; the PagedAttention block reclamation pattern here applies to session memory management |
| **AI Infrastructure Ch.1** | GPU bandwidth (TB/s) from Ch.1 determines the ceiling for single-token generation latency regardless of batching strategy |

---

## 12 · Progress Check — What We've Accomplished

🎉 **PRODUCTION-READY INFERENCE! Handles spiky traffic patterns within latency targets ✅**

**Unlocked capabilities**:
- ✅ **Continuous batching**: Dynamic request scheduling eliminates queue wait spikes
- ✅ **PagedAttention**: Batch=4 → batch=8 (2× throughput) via KV cache paging
- ✅ **Speculative decoding**: 30% faster generation with Llama-3-1B draft model
- ✅ **Load-tested**: 22,000 req/day, 680ms p95 latency under realistic spiky traffic ✅

**Progress toward constraints**:

| Constraint | Status | Current State |
|------------|--------|---------------|
| #1 COST | ✅ **MAINTAINED** | $1,095/month RTX 4090 (no changes) |
| #2 LATENCY | ✅ **VALIDATED UNDER LOAD** | 680ms p95 (66% better than 2s target) |
| #3 THROUGHPUT | ✅ **EXCEEDED** | 22,000 req/day (220% of 10k target) |
| #4 MEMORY | ✅ **OPTIMIZED** | 8GB / 24GB = 33% utilization |
| #5 QUALITY | ✅ **MAINTAINED** | 96.2% accuracy (no degradation from optimizations) |
| #6 RELIABILITY | ⚡ **IMPROVED** | Handles 2× traffic spikes gracefully |

**What we can solve now**:

✅ **Handle lunch rush without latency spikes**:
```
Before Ch.5:
Lunch rush (40 req/sec spike):
❌ p95 latency: 8.7s (missed 2s target by 4×!)
❌ Queue depth: 180 requests

After Ch.5 (continuous batching + PagedAttention):
Lunch rush (40 req/sec spike):
✅ p95 latency: 1.8s (within 2s target!)
✅ Queue depth: max 12 requests (clears within seconds)

CEO: "So we can handle lunch rush now?"
Engineer: "Yes. Continuous batching eliminates the queue explosion.
          When traffic spikes, new requests start processing within
          100ms instead of waiting for full batches to complete."

Result: ✅ Lunch rush revenue protected! (30% of daily transactions)
```

✅ **2× throughput headroom for growth**:
```
CEO: "What if Q2 traffic doubles to 20k req/day?"

Engineer: "Current capacity with optimizations:
 - Continuous batching: 13,500 req/day (vs 12k static)
 - + PagedAttention (batch=8): 18,000 req/day
 - + Speculative decoding: 22,000 req/day ✅

 We have 2× headroom over target (22k vs 10k). Can handle 2× growth
 without buying additional GPUs."

Result: ✅ Cost-efficient scaling path validated!
```

✅ **Latency budget for monitoring/logging**:
```
Before Ch.5:
p95 latency: 1.2s (60% of 2s budget)
→ Only 800ms left for network + monitoring + logging

After Ch.5:
p95 latency: 680ms (34% of 2s budget)
→ 1,320ms left for network + monitoring + logging ✅

Result: ✅ Comfortable latency margin for production observability!
```

**What's still blocking**:

- ⚡ **No serving framework selected**: Raw Python + custom continuous batching → **Need Ch.6 to pick vLLM vs TGI vs TensorRT-LLM**
- ⚡ **Single GPU = single point of failure**: If RTX 4090 crashes, entire service down → **Need Ch.7 for multi-GPU redundancy**
- ⚡ **No autoscaling**: Running 24/7 even during low-traffic nights → **Need Ch.8 for cost-optimized cloud deployment**
- ⚡ **No production monitoring**: How do we detect latency regressions? → **Need Ch.9 for observability**

**Next chapter**: [Serving Frameworks](../ServingFrameworks) compares vLLM, TensorRT-LLM, TGI:
- Benchmarks: throughput, latency, memory efficiency
- Framework selection criteria (ease of deployment, community support, feature completeness)
- **Final decision: vLLM for production deployment**

**Key interview concepts from this chapter**:

| Must know | Likely asked | Trap to avoid |
|---|---|---|
| Continuous batching adds requests dynamically as slots free (vs static batch waits to fill) | Why does continuous batching improve tail latency? | Thinking it improves single-request speed — it reduces queue wait time |
| PagedAttention treats KV cache like virtual memory (64-token pages) → eliminates fragmentation | How does PagedAttention increase batch size? | Saying it "compresses KV cache" — it eliminates padding waste, not actual compression |
| Speculative decoding: draft model (1B) predicts N tokens, large model (8B) verifies in parallel → 2–3× speedup | When does speculative decoding NOT help? | Not knowing acceptance rate matters — if <50%, overhead dominates |
| KV cache grows linearly with sequence length AND batch size (most variable component) | Why can't we just increase batch size to 32? | Forgetting KV cache scales quadratically with load (batch × seq_len) |
| Autoscaling: spin up GPUs during traffic spikes, shut down during low traffic → cost savings | How do you handle traffic spikes cost-efficiently? | Suggesting "just buy more GPUs" — autoscaling is cheaper for spiky workloads |

---

## 13 · Bridge to Chapter 6

Ch.5 validated that continuous batching, PagedAttention, and speculative decoding can hit throughput and latency targets. But implementing these from scratch requires months of CUDA kernel engineering. Ch.6 (Serving Frameworks) evaluates production-ready frameworks that bundle these optimizations: **vLLM** (easiest deployment, PagedAttention built-in), **TensorRT-LLM** (fastest, but complex setup), **TGI** (Hugging Face, good Python integration), and **llama.cpp** (CPU/edge inference). The question: which framework best fits InferenceBase's needs?

## Illustrations

![Inference optimization — Continuous batching timeline, PagedAttention memory savings, speculative decoding acceleration](img/Inference%20Optimization.png)
