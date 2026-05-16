# AI Infrastructure — Interview Primer

← Back to learning chapter: [AI Infrastructure](../07-ai-infrastructure/README.md)

> This guide prepares you to answer senior-level infrastructure questions at AI companies (Google, Meta, OpenAI, Anthropic). The distinction: junior candidates quote GPU specs; senior candidates explain why an H100 with 3× the compute of an A100 is only 1.7× faster for LLM inference. You'll learn the roofline model, VRAM budgeting (weights + KV cache + activations), quantization tradeoffs (INT8 vs INT4 quality loss), vLLM's PagedAttention innovation, and how to architect serving for <100ms P99 TTFT at <$20k/year. Every answer grounds in the InferenceBase system: Llama-3-8B self-hosting that went from $80k/year (naive deployment) to $15k/year (production-optimized).

---

> **How to use the junior/senior answer comparisons** — Each question below includes a junior-level answer and a senior-level answer. Junior answers are technically correct but surface-level. Senior answers demonstrate production experience, failure awareness, and trade-off reasoning. Hiring managers at FAANG and growth-stage AI companies distinguish these instantly. Study the DIFFERENCE between the two, not just the senior answer.

## 1 · Concept Map — The 10 Questions That Matter

| # | Cluster | What the interviewer is testing |
|---|---------|----------------------------------|
| 1 | **GPU Roofline Model** | Do you know arithmetic intensity? Can you place LLM decode on the roofline? |
| 2 | **KV Cache Sizing** | Can you compute KV cache memory per token/request? Know growth with batch × seq_len? |
| 3 | **Quantization Tradeoffs** | Know the memory savings per dtype? Understand when perplexity loss matters? |
| 4 | **TP vs DP vs ZeRO** | Can you explain tensor parallel vs data parallel and when communication dominates? |
| 5 | **Prefill vs. Decode Bottleneck** | Do you know which phase is compute-bound vs memory-bound and why? |
| 6 | **PagedAttention & Continuous Batching** | Can you explain vLLM's core innovations and the throughput gain they produce? |
| 7 | **Production Serving Stack** | Do you know vLLM vs Ollama vs TGI and when to use each? |
| 8 | **Cloud vs. Self-Host Economics** | Know $/token at SLA vs $/GPU-hour? Spot vs. reserved tradeoffs? |
| 9 | **MLOps & Checkpointing** | Can you explain gradient checkpointing (memory) vs checkpoint saving (fault tolerance)? |
| 10 | **End-to-End TTFT + P99** | Do you distinguish TTFT from throughput? Know which matters for users vs. cost? |

---

## 2 · Section-by-Section Deep Dives

### GPU Architecture & The Roofline Model — What They're Testing

Do you understand why LLM inference is fundamentally memory-bound, not compute-bound? Can you place a workload on the roofline and explain what happens when you add more compute vs more bandwidth? The interviewer wants to know if you've moved beyond "bigger GPU = faster" thinking into the real bottleneck analysis that drives production decisions.

**The Junior Answer vs Senior Answer**

**Q: "Why is LLM inference slow on a high-TFLOP GPU?"**
**Junior**: "The model is large and there are many parameters to compute."
*Why it signals junior:* Focuses on size without understanding the bottleneck. Doesn't distinguish compute from memory bandwidth.
**Senior**: "LLM decode at batch=1 has arithmetic intensity around 2–5 FLOP/byte — far left of the roofline ridge point at ~156 FLOP/byte for an A100. The GPU spends 80–90% of time waiting for weights to arrive from HBM, not computing. Adding more TFLOP/s doesn't help; bandwidth is the limit. This is why an H100 at 3.35 TB/s HBM bandwidth delivers 1.7× faster inference than an A100 at 2 TB/s — despite having 3× the compute."
*Why it signals senior:* Names arithmetic intensity, references the roofline ridge point, explains the 80–90% memory stall time, quantifies bandwidth gain vs compute gain.
**Key insight**: In the InferenceBase system (Llama-3-8B self-hosting), moving from an A100 to H100 gave us 1.7× throughput improvement — exactly matching the HBM bandwidth ratio, not the 3× compute ratio. Bandwidth is the real cost driver.

**The Key Tradeoffs**

| Approach | When It Wins | When It Loses | Decision Criterion |
|----------|--------------|---------------|--------------------|
| Optimize for HBM bandwidth (H100, A100) | LLM decode, batch ≤16, memory-bound workloads | Small models that fit in L2 cache, batch ≥64 | If arithmetic intensity <50 FLOP/byte (most inference), bandwidth wins |
| Optimize for TFLOP/s (more CUDA/Tensor cores) | Training, large batch sizes (≥64), compute-bound | Small batch decode, memory-bound inference | If arithmetic intensity >156 FLOP/byte (training), compute wins |
| Consumer GPU (RTX 4090: 1 TB/s, $1600) | Development, low-concurrency serving (<5 req/s) | Production serving (no ECC, no NVLink, driver instability) | If cost <$2k and no SLA requirements, consumer wins |
| Data center GPU (A100: 2 TB/s, $10k) | Production serving, multi-GPU scaling, SLA-critical | Budget-constrained development | If uptime >99.9% required or multi-GPU, data center wins |

**Failure Mode Gotchas**

**Warning — Gotcha Q:** "We upgraded from 4×A100 to 4×H100 but only got 1.5× speedup, not 3×. Why?"
**Why it's hard:** Candidates assume more compute = proportional speedup without checking the actual bottleneck.
**What to say:** "Your workload is still memory-bound. Check arithmetic intensity — if it's <50 FLOP/byte, you're paying for 3× compute you can't use. The 1.5× gain matches the bandwidth increase minus inter-GPU communication overhead. To get more speedup: increase batch size (raises arithmetic intensity) or use speculative decoding (reduces weight reads per token)."

**Warning — Gotcha Q:** "Why does a 7B model run slower than a 1B model even though we have 312 TFLOP/s available?"
**Why it's hard:** The candidate forgets that model size directly impacts memory reads per token.
**What to say:** "7B model = 14 GB weights in BF16. Each token generation reads all 14 GB from HBM. At 2 TB/s bandwidth, that's 7ms per token just for weight reads — before any compute. The 1B model only reads 2 GB (1ms read time). You're 7× slower because you're reading 7× more data, and bandwidth is the bottleneck, not the 312 TFLOP/s compute."

**The Production Angle**

**At InferenceBase scale (Llama-3-8B, 10k requests/day):**
- **Development**: RTX 4090 (1 TB/s) handles batch=1 decode at ~25 tokens/sec — adequate for prototyping
- **Production**: A100 (2 TB/s) with continuous batching (batch=16–32) delivers ~400 tokens/sec aggregate throughput
- **Why the jump matters**: Production needs <100ms P99 TTFT. RTX 4090 can't maintain this under load spikes (no ECC, driver crashes). A100 gives 2× bandwidth + ECC + NVLink for multi-GPU failover.
- **Cost tradeoff**: $10k A100 vs $1600 RTX 4090 — but one A100 replaces 2–3 RTX 4090s once you account for batch efficiency and uptime SLA
**Production trap:** "We'll save money with consumer GPUs" — until the first driver crash during peak traffic costs you $50k in lost revenue. The $8k premium for data center GPUs is insurance, not overhead.

---

### Memory & Compute Budgets — What They're Testing

Can you size VRAM requirements for inference vs training? Do you account for KV cache growth as sequence length and batch size increase? The interviewer wants to see if you can do back-of-the-envelope calculations that distinguish "this fits" from "this explodes" — the difference between a working production deployment and a $50k GPU bill.

**The Junior Answer vs Senior Answer**

**Q: "How much VRAM do you need to serve Llama-3-8B at batch=16, max context 4096 tokens?"**
**Junior**: "8 billion parameters, so probably around 8–10 GB."
*Why it signals junior:* Only considers parameter count, ignores precision, KV cache, and activations.
**Senior**: "Weights: 8B × 2 bytes (BF16) = 16 GB. KV cache per request: 2 × 32 layers × 4096 hidden_dim × 2 bytes = 512 KB/token. At 4096 tokens max: 2 GB per request. Batch=16: 32 GB KV cache. Activations: ~2 GB. Total: 16 + 32 + 2 = 50 GB. You need an A100-80GB or 2×A100-40GB with tensor parallelism. At batch=8, you'd fit on a single A100-40GB with 8 GB headroom."
*Why it signals senior:* Breaks down all three memory components, computes KV cache growth with batch size, gives specific deployment options with headroom analysis.
**Key insight**: At InferenceBase, we initially tried batch=32 on A100-40GB and hit OOM after the first user with a 3k-token context. Calculating KV cache pressure up front would have saved 4 hours of debugging.

**The Key Tradeoffs**

| Quantization | VRAM Savings | Quality Loss | When to Use |
|--------------|--------------|--------------|-------------|
| BF16 (baseline) | 1× (16 GB for 8B model) | 0% loss | Always use for training; use for inference if VRAM available |
| INT8 | 2× (8 GB for 8B model) | <1% perplexity increase | Production inference when VRAM tight but quality critical |
| INT4 | 4× (4 GB for 8B model) | 5–10% perplexity loss | Development, non-critical serving, or when VRAM <8 GB |
| NF4 (QLoRA) | 4× + reduced activation memory | 3–7% loss | Fine-tuning large models on consumer GPUs |

**Decision criterion:** If output quality measured by human eval drops >5%, don't use INT4 in production. If VRAM cost >$X/month, try INT8 + A/B test against BF16 baseline.

**Failure Mode Gotchas**

**Warning — Gotcha Q:** "We sized VRAM for the model weights but we're still getting OOM. Why?"
**Why it's hard:** Forgetting KV cache is a dynamic data structure that grows with context and batch size.
**What to say:** "KV cache is the most common OOM culprit in production. For Llama-2-7B, each token adds 512 KB — a 2048-token context at batch=16 uses 16 GB KV cache alone. You need to budget: weights + (KV_per_token × max_seq_len × batch_size) + activation memory. Enable PagedAttention (vLLM) to reduce KV cache fragmentation and fit 2× larger batches."

**Warning — Gotcha Q:** "Why does our INT4 quantized model give terrible results on reasoning tasks but fine on summarization?"
**Why it's hard:** Not all tasks have the same quantization tolerance.
**What to say:** "Reasoning tasks (math, logic, chain-of-thought) rely on precise intermediate activations. INT4 quantization loses 5–10% accuracy on these tasks because rounding errors compound across layers. Summarization is more robust because it's pattern-matching, not computation. Test quantization per task — use INT8 for reasoning, INT4 for summarization if VRAM is tight."

**The Production Angle**

**At InferenceBase scale (Llama-3-8B self-hosting):**
- **Initial plan**: BF16 weights (16 GB) + batch=16 + 4k context = 50 GB → 2×A100-40GB via tensor parallelism
- **Cost reality**: $3.60/hr for 2×A100-40GB spot vs $2.00/hr for 1×A100-80GB reserved (3-year commit)
- **Actual deployment**: INT8 quantization (8 GB weights) + batch=24 + 4k context = 8 + 48 + 2 = 58 GB → fits on 1×A100-80GB with headroom
- **Quality check**: Human eval on 200-sample test set showed 0.8% quality drop (acceptable for our SLA)
- **Budget impact**: Reduced from $80k/year (2×A100-40GB on-demand) to $15k/year (1×A100-80GB reserved + INT8) — 81% cost reduction
**Production rule:** Always budget VRAM as: `weights + (KV_per_token × max_seq × batch) + 20%_headroom`. The 20% accounts for activation spikes and PyTorch memory allocator fragmentation.

---

### Production Serving & Inference Optimization — What They're Testing

Do you know the difference between TTFT (user-visible latency) and throughput (cost metric)? Can you explain vLLM's PagedAttention and continuous batching innovations? The interviewer wants to see if you understand production serving at scale — not just "run a model locally with Ollama."

**The Junior Answer vs Senior Answer**

**Q: "How would you serve Llama-3-8B at 10k requests/day with <100ms P99 TTFT?"**
**Junior**: "I'd use Ollama or Hugging Face Transformers with a GPU. Maybe add a load balancer."
*Why it signals junior:* Ollama has no concurrency primitives for production serving. Missing TTFT vs throughput distinction. No mention of batching or KV cache optimization.
**Senior**: "Use vLLM for continuous batching and PagedAttention — these give 2–3× throughput vs naive serving. Deploy on A100-80GB (2 TB/s HBM) to hit TTFT target. Configure max batch size based on VRAM budget: at 4k context, batch=24 fits in 58 GB with INT8 quantization. Monitor P99 TTFT and throughput separately — optimize batch size to maximize throughput while keeping P99 <100ms. If load spikes exceed single-GPU capacity, add tensor parallelism across 2×A100 with NVLink (avoids inter-node communication overhead)."
*Why it signals senior:* Names vLLM-specific optimizations, distinguishes TTFT from throughput, gives VRAM calculation, explains batch size tuning strategy, addresses scale-out with communication-aware architecture.
**Key insight**: At InferenceBase, switching from Hugging Face Transformers to vLLM gave us 2.4× throughput improvement (170 tok/s → 410 tok/s) on the same A100. PagedAttention eliminated KV cache fragmentation and let us run batch=24 instead of batch=12.

**The Key Tradeoffs**

| Approach | When It Wins | When It Loses | Decision Criterion |
|----------|--------------|---------------|--------------------|
| vLLM (PagedAttention + continuous batching) | High-concurrency serving (>50 req/s), need max throughput | Single-user inference, exotic models not in vLLM model registry | If serving >50 req/s or need <100ms P99, use vLLM |
| Ollama (CPU + GPU fallback, local dev focus) | Local development, single-user testing, prototyping | Production serving with concurrency >5, SLA requirements | If development/demo only, Ollama works; production needs vLLM |
| TGI (Hugging Face Text Generation Inference) | Hugging Face ecosystem integration, official model support | Custom model architectures, need latest optimizations | If using Hugging Face Hub models + need support SLA, use TGI |
| Ray Serve (general ML serving framework) | Multi-model serving, custom business logic, non-LLM models | Pure LLM serving (overhead vs vLLM) | If serving multiple model types or need custom routing, use Ray Serve + vLLM backend |

**Decision criterion:** If serving a single LLM at high concurrency (>50 req/s), vLLM wins. If prototyping locally, Ollama wins. If serving multiple models with custom orchestration, Ray Serve + vLLM wins.

**Failure Mode Gotchas**

**Warning — Gotcha Q:** "We're hitting 500 tokens/sec throughput but users complain about slowness. Why?"
**Why it's hard:** Optimizing the wrong metric — throughput vs P99 latency.
**What to say:** "High aggregate throughput doesn't guarantee low P99 latency. You might have batch=64 for max throughput, but users in a large batch wait for all 64 requests to complete before seeing their first token. Reduce batch size to 16–24 — throughput drops 20% but P99 TTFT improves 3×. Users notice latency, not your GPU utilization."

**Warning — Gotcha Q:** "Continuous batching sounds great but our P99 latency got worse. What happened?"
**Why it's hard:** Continuous batching can introduce head-of-line blocking if not configured correctly.
**What to say:** "Check your max sequence length limit. If one request generates 2048 tokens while others need 128 tokens, the long request blocks the queue. Set per-request timeouts and max_tokens limits. Use priority queuing — short requests (≤512 tokens) in high-priority queue, long requests (>512) in low-priority. This is what we did at InferenceBase to keep P99 <100ms while still serving 4k-context research queries."

**The Production Angle**

**At InferenceBase scale (10k requests/day, target: <100ms P99 TTFT, <$20k/year):**
- **Initial deployment**: Hugging Face Transformers on A100-80GB → 170 tokens/sec aggregate, P99 TTFT = 250ms (missed SLA)
- **Migration to vLLM**: Enabled PagedAttention + continuous batching → 410 tokens/sec, P99 TTFT = 85ms (hit SLA)
- **Batch size tuning**: Started at batch=32 (max throughput) but P99 = 140ms. Reduced to batch=24 → throughput dropped 15% but P99 = 85ms (acceptable tradeoff)
- **Cost optimization**: Reserved 1×A100-80GB at $1.25/hr (3-year commit) = $10,950/year base + $4k/year for INT8 model serving license = $15k/year total
- **Monitoring**: Track TTFT and throughput separately; alert if P99 TTFT >95ms or throughput <350 tok/s
**Production rule:** TTFT is what users complain about; throughput is what accountants complain about. Optimize TTFT first (keep users happy), then maximize throughput within the TTFT constraint.

---

> **GPU sizing verdict:** TTFT is what users complain about; throughput is what accountants complain about. Optimize TTFT first.
> ➡ The Rapid-Fire Round at the end covers the most common follow-up questions interviewers use to probe depth.

---

## 3 · The Rapid-Fire Round

> 20 Q&A pairs. Each answer: ≤ 3 sentences.

**1. What is arithmetic intensity?**
FLOPs performed per byte read from memory. LLM decode at batch=1 has ~1–5 FLOP/byte — far left of the roofline, memory-bound. Adding compute does not help; bandwidth is the limit.

**2. Where does LLM decode sit on the roofline?**
Left of the ridge point — memory-bound. The GPU spends most time waiting for weights to arrive from HBM, not computing. This is why HBM bandwidth is the key inference spec.

**3. How much VRAM does a 7B BF16 model need?**
~14 GB for weights (2 bytes × 7B params), plus KV cache and activations. Total for serving at moderate sequence length: ~18–20 GB.

**4. How does batching improve GPU utilisation?**
Batch=32 reads weights once and produces 32 outputs — arithmetic intensity rises from ~2 to ~50+ FLOP/byte. This moves the workload toward the ridge point, better utilizing compute units.

**5. What is the KV cache size per token for LLaMA-2-7B?**
2 × 32 layers × 4096 hidden_dim × 2 bytes = 512 KB per token. A 2048-token sequence uses ~1 GB KV cache per request.

**6. What does INT4 quantization buy you?**
~8× VRAM reduction vs FP32 (0.5 bytes vs 4 bytes per param). A 7B model fits in ~3.5 GB. Tradeoff: 5–10% perplexity loss, especially for reasoning tasks.

**7. Prefill vs. decode — what's the bottleneck for each?**
Prefill (processing the prompt) is compute-bound — benefits from large batches. Decode (generating each token) is memory-bound — bottlenecked by HBM bandwidth.

**8. What is PagedAttention?**
KV cache stored in non-contiguous memory pages, like OS virtual memory. Eliminates KV cache fragmentation. Enables 2× larger batch sizes → 2× throughput — the core vLLM innovation.

**9. What is continuous batching?**
Adding new requests to a batch as others complete, rather than waiting for the whole batch. Keeps GPU utilization high. Achieves 2–3× throughput improvement vs static batching.

**10. vLLM vs. Ollama — when to use each?**
vLLM for high-concurrency production serving (continuous batching, PagedAttention, tensor parallelism). Ollama for local development and testing. Ollama has no production concurrency primitives.

**11. What is gradient checkpointing?**
Recomputing activations during the backward pass instead of storing them. Saves ~30–40% of activation memory at the cost of ~30% slower training. Essential for large models in limited VRAM.

**12. Why does training need ~4× more memory than inference?**
Training stores weights + optimizer states (2× weights for Adam) + gradients (1× weights) + activations. Inference stores only weights + KV cache + activations.

**13. TP vs. DP — what is the key difference?**
Tensor Parallelism (TP) shards the model weights across GPUs — each GPU holds part of each layer, and all-reduce happens per layer forward/backward. Data Parallelism (DP) replicates the full model on each GPU, syncing gradients after the backward pass.

**14. Why does communication dominate at large GPU counts in DP?**
All-reduce cost grows with batch size and model size. Beyond 8–16 GPUs, inter-node InfiniBand bandwidth (25 GB/s) vs intra-node NVLink (600+ GB/s) creates a communication wall.

**15. What is ZeRO and which stage should you use?**
ZeRO (Zero Redundancy Optimizer) partitions optimizer states (Stage 1), gradients (Stage 2), and parameters (Stage 3) across GPUs. Use Stage 2 for most training; Stage 3 for models that don't fit on a single GPU.

**16. $/GPU-hour vs. $/token — which metric matters?**
$/token at your SLA latency is the right metric. A cheap GPU that misses P99 latency targets has zero value in production. $/GPU-hour is a procurement metric, not a serving metric.

**17. When should you use spot instances?**
For training, where jobs are long and checkpointing is cheap. Never for production inference (users need availability). Spot can save 60–80% vs on-demand.

**18. What is TTFT and why does it matter?**
Time to First Token — latency from request submission to the first generated token. This is the user-perceived "responsiveness" metric. Throughput (tokens/sec overall) is the cost metric; TTFT is the UX metric.

**19. How does speculative decoding work?**
A smaller draft model proposes K tokens cheaply; the large target model verifies them in one forward pass. If all K are accepted, you get K tokens for the cost of 1 large-model step. Works best when the draft model is accurate (same family, quantized version).

**20. What is the difference between BF16 and FP16?**
Both use 16 bits total but different mantissa/exponent splits. BF16 has the same exponent range as FP32 (8 bits) with a shorter mantissa — much more stable during training. FP16 has a smaller exponent range and is more prone to loss spikes (overflow/underflow). Modern GPUs and TPUs prefer BF16 for training.

---

## 4 · Signal Words That Distinguish Answers

** Senior signals:**
- "Arithmetic intensity of 2–5 FLOP/byte places decode left of the ridge point" (shows roofline understanding)
- "HBM bandwidth is the bottleneck; TFLOP/s doesn't matter until you cross the ridge" (distinguishes bandwidth from compute)
- "KV cache pressure grows as batch × seq_len — budget 512 KB/token for Llama-2-7B" (quantifies memory growth)
- "I'd instrument TTFT and throughput separately — P99 TTFT for user experience, tokens/sec for cost" (distinguishes metrics)
- "PagedAttention eliminates KV cache fragmentation, enabling 2× larger batch sizes" (names specific vLLM innovation)
- "Continuous batching keeps GPU utilization high by adding requests as others complete" (explains throughput gain mechanism)
- "INT8 quantization gives 2× VRAM savings with <1% perplexity loss; INT4 gives 4× but loses 5–10% on reasoning tasks" (specific tradeoffs)
- "Tensor parallelism shards weights across GPUs with all-reduce per layer; data parallelism replicates model and syncs gradients" (distinguishes TP from DP)
- "At InferenceBase, we budget VRAM as: weights + (KV_per_token × max_seq × batch) + 20% headroom" (shows production awareness)

** Junior signals:**
- "Just use a bigger GPU" → missing bandwidth vs compute analysis
- "The model is large" → vague; doesn't quantify or identify bottleneck
- "Quantization is free" → ignores quality loss, especially for reasoning tasks
- "It depends on your use case" → without specifying decision criteria
- "More GPUs = linear speedup" → ignores communication overhead
- "Use Ollama in production" → Ollama lacks concurrency primitives for high-traffic serving
- "Average latency is 50ms" → ignoring P99 latency (what users actually experience during load spikes)
- "I'd monitor it" → without naming specific metrics (TTFT, throughput, P99)

**Interview framing patterns that signal senior thinking:**
- "At InferenceBase scale (10k req/day), we found that..." (grounds answer in production system)
- "The tradeoff is: if VRAM <40 GB, use INT8; if quality critical, stay BF16" (specific decision criteria)
- "Here's how I'd debug this: check arithmetic intensity first, then compare to ridge point" (systematic approach)
- "I'd A/B test INT8 vs BF16 on a 200-sample eval set before deploying" (empirical validation)
- "The naive approach reads weights once per token; speculative decoding amortizes this across K tokens" (explains mechanism)

---

<details>
<summary> 5-Minute Crammer — last-resort prep</summary>

## 5 · The 5-Minute Concept Cram

> For topics you're shaky on — ultra-dense explanations that give enough vocabulary and structure to answer basic questions without embarrassment.

### Roofline Model (2 minutes)

**What it is:** A plot of achievable FLOP/s (y-axis) vs arithmetic intensity (x-axis, FLOP/byte).

**Why it matters:** Shows whether your workload is memory-bound (left of ridge) or compute-bound (right of ridge). LLM decode is always left of ridge — bandwidth-limited.

**Key numbers for A100:**
- Peak compute: 312 TFLOP/s (BF16)
- Peak bandwidth: 2 TB/s (HBM)
- Ridge point: 312 TFLOP/s ÷ 2 TB/s = 156 FLOP/byte
- LLM decode at batch=1: ~2–5 FLOP/byte (far left of ridge)

**Interview answer:** "For an A100, the ridge point is 156 FLOP/byte. LLM decode at batch=1 has arithmetic intensity around 2–5 FLOP/byte, so it's memory-bound. Adding more compute doesn't help — we need more bandwidth or higher batch size to move right on the roofline."

### KV Cache Sizing (1 minute)

**Formula:** `KV_size = 2 × num_layers × hidden_dim × 2 bytes × num_tokens`

**Example (Llama-2-7B):** `2 × 32 × 4096 × 2 = 512 KB per token`
- 2048 tokens = 1 GB KV cache per request
- Batch=16 at 2048 tokens = 16 GB KV cache

**Interview answer:** "For Llama-2-7B, each token adds 512 KB to KV cache. At 2048-token context and batch=16, that's 16 GB KV cache alone — half the VRAM on an A100-40GB."

### Quantization Quick Reference (1 minute)

| Precision | Bytes/param | 8B model size | Quality loss | When to use |
|-----------|-------------|---------------|--------------|-------------|
| BF16 | 2 | 16 GB | 0% (baseline) | Training, inference if VRAM available |
| INT8 | 1 | 8 GB | <1% | Production inference, VRAM tight |
| INT4 | 0.5 | 4 GB | 5–10% | Non-critical serving, dev/test |

**Interview answer:** "INT8 halves VRAM usage with <1% quality loss. INT4 quarters it but loses 5–10% on reasoning tasks. Always A/B test quantized models before production deployment."

### vLLM vs Ollama (1 minute)

**vLLM:** Production serving framework with PagedAttention (eliminates KV cache fragmentation) and continuous batching (adds requests as others complete). Built for high-concurrency serving (>50 req/s).

**Ollama:** Local development tool with simple CLI. No concurrency primitives, no PagedAttention, no request queuing. Great for prototyping, terrible for production.

**Interview answer:** "Use vLLM for production serving — it gives 2–3× throughput improvement via continuous batching and PagedAttention. Ollama is for local dev and demos only."

</details>

---

## Related Topics

> ➡ **Forward links** — these build on AI Infrastructure concepts:
- [Agentic AI Interview Guide](agentic-ai.md) — Cost & Latency section covers the application-layer view of the same GPU/inference concepts
- [Multimodal AI / Local Diffusion Lab](../05-multimodal_ai/ch13_local_diffusion_lab) — same serving patterns apply to diffusion models

> 📖 **Optional depth** — academic rigor for mathematical proofs:
- [AI / Fine-tuning](.03-ai/ch10_fine_tuning) — QLoRA is quantization + LoRA combined
- [AI / Cost & Latency](.03-ai/ch09_cost_and_latency) — VRAM side of the same cost model

