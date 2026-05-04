# Ch.5 — Inference Optimization

> **The story.** For the first wave of LLM inference (GPT-2/3 era, 2019–2022), the standard approach was "generate one token, concatenate, repeat" — no batching, no caching optimizations. Every request was independent. **Continuous batching** (Orca, Yu et al., Microsoft, **2022**) changed this: instead of waiting for the slowest request in a batch to finish, dynamically add new requests as soon as a slot frees up → 10× higher throughput. **PagedAttention** (Kwon et al., vLLM, **2023**) solved KV cache fragmentation by treating it like OS virtual memory — page in/out blocks as needed → 24× higher batch sizes. **Speculative decoding** (Leviathan et al., Google, **2023**) used a small "draft" model (1B params) to predict multiple tokens, then verified with the large model in parallel → 2–3× speedup. By 2024, these techniques were bundled into serving frameworks (vLLM, TGI, TensorRT-LLM), making them transparent to users.
>
> **Where you are in the curriculum.** Ch.3 validated INT4 quantization → 12,000 req/day throughput at batch=4. But that was a synthetic benchmark (uniform request arrival). Real production traffic has spiky arrival patterns, variable sequence lengths, and different request priorities. This chapter optimizes the *inference loop* for real-world conditions: continuous batching, PagedAttention, speculative decoding, and KV cache management. The InferenceBase question: *Can we maintain <2s latency at 12,000 req/day under realistic load?*
> **Notation.** `TTFT` = time to first token (prefill latency). `TPOT` = time per output token (decode step latency). `KV cache` = key-value attention tensors cached from the prefill pass to avoid recomputing prior-token attention. `req/s` = requests per second. `tok/s` = tokens per second. `batch` = number of sequences processed concurrently in a single forward pass.

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

## 2 · Running Example

**Problem**: Lunch rush (40 req/sec spike) causes 8.7s p95 latency → misses 2s SLA.

**Root cause**: Static batch=4 → when 40 requests arrive, first batch starts immediately, but request #37 waits for 9 batches to complete (9 × 1.2s = 10.8s queue wait).

**Solution**:
1. **Continuous batching**: Start processing request #5 as soon as request #1 finishes generating (even if #2, #3, #4 still running)
2. **PagedAttention**: Eliminate padding waste → batch=8 instead of batch=4 → cut queue wait by 50%
3. **Speculative decoding**: Llama-3-1B drafts 3 tokens, Llama-3-8B verifies in parallel → 30% faster

**Result**: 8.7s p95 → 1.8s p95 under spike load ✅ (within 2s target!).

---

## 3 · Continuous Batching

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

## 4 · PagedAttention

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

---

## 5 · Speculative Decoding

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

## 9 · Benchmarking Inference Optimizations

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

---

## 10 · The Key Diagram

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

