# Ch.2 — Memory & Compute Budgets

> **The story.** For the first decade of deep learning (2012–2022), VRAM was the hard constraint. AlexNet (2012) barely fit on two GTX 580s (3 GB total). ResNet-152 (2015) required batching gymnastics on a single Titan X (12 GB). GPT-2 (2019, 1.5 B params) needed 6 GB just to load — training required gradient checkpointing and mixed precision to squeeze into V100s (32 GB). The breakthrough was **ZeRO** (Rajbhandari et al., Microsoft, **2019**), which sharded optimizer states across GPUs, cutting per-GPU memory by up to 8×. **Flash Attention** (Dao et al., Stanford, **2022**) made self-attention subquadratic in memory by fusing operations and tiling through HBM. **PagedAttention** (Kwon et al., vLLM, **2023**) extended the paging idea to KV caches, eliminating fragmentation waste. By 2024, the memory wall had shifted: you could fit a 70 B model on a single H100 (80 GB) with INT4 quantization — but only if you understood the exact breakdown of where every GB goes.
>
> **Where you are in the curriculum.** Ch.1 told you which GPU to pick. This chapter tells you whether your model actually fits — and if not, what to cut. The InferenceBase question: *Llama-3-8B has 8 billion parameters. The RTX 4090 has 24 GB VRAM. Does it fit?* The answer requires understanding parameters, activations, KV cache, optimizer states, and gradients — and how each scales with batch size, sequence length, and precision.
> **Notation.** `P` = parameter count; memory in bytes ≈ `P × bytes_per_param`. `B` = batch size (number of concurrent sequences). `S` = sequence length in tokens. `H` = hidden dimension (`d_model`). KV cache scales as `2 × B × S × H × num_layers`. `GiB` = gibibytes (2³⁰ bytes). `VRAM` = on-GPU device memory limit.
<!-- notation: key variables defined here -->

---

## 0 · The Challenge — Where We Are

## Animation

> 🎬 *Animation placeholder — needle-builder agent will generate this.*


> 🎯 **The mission**: Self-host Llama-3-8B for <$15k/month, replacing $80k OpenAI API costs
>
> **6 Constraints**: #1 Cost (<$15k/mo) • #2 Latency (≤2s) • #3 Throughput (≥10k req/day) • #4 Memory (fit in VRAM) • #5 Quality (≥95% accuracy) • #6 Reliability (>99% uptime)

**What we know so far**:
- ✅ Ch.1: RTX 4090 identified as target GPU (24GB VRAM, 1.0 TB/s bandwidth, $1,095/month)
- ⚡ **Current state**: Have GPU selection, but unknown if Llama-3-8B fits in 24GB

**What's blocking us**:

🚨 **Cannot order hardware without confirming model fits in VRAM**

**Current situation**: You're preparing the budget justification for the CEO

```
You calculate:
"Llama-3-8B has 8 billion parameters.
 8 billion × 2 bytes/param (FP16) = 16 GB for model weights.
 RTX 4090 has 24 GB VRAM.
 24 GB - 16 GB = 8 GB free.
 Should be plenty of headroom!"

[You order the RTX 4090 and deploy the model...]

First inference request → CUDA OOM error: "Out of memory"

You: "Wait, what? I calculated 16GB for the model... where did the other 8GB go?"
```

**Problems**:
1. ❌ **Forgot KV cache**: Stores attention keys/values for all tokens → grows with sequence length × batch size
2. ❌ **Forgot activations**: Temporary tensors during forward pass → several GB
3. ❌ **Unknown batch limit**: How many requests can we process simultaneously before OOM?
4. ❌ **Training memory unknown**: If we want to fine-tune, what GPU do we need? (Adam optimizer states = 3× params!)
5. ❌ **No safety margin**: Running at 100% VRAM utilization = fragmentation issues, OOM on long sequences

**Business impact**:
- **Wrong GPU purchase = $50k mistake**: If you buy RTX 4090 (24GB) but actually need A100 (80GB), you burn $8k + weeks of delay
- **Throughput limited by batch size**: If you can only batch=1, throughput caps at 3,000 req/day (30% of 10k target)
- **Cannot fine-tune**: If training needs 80GB, you have no path to improve quality beyond base Llama-3-8B
- CEO: "I need exact numbers. Does it fit yes or no? And what's the maximum batch size?"

**What this chapter unlocks**:

🚀 **Precise VRAM calculator for inference & training**:
1. **Parameter memory**: 8B params × 2 bytes (FP16) = 16 GB
2. **KV cache memory**: batch_size × seq_len × layers × hidden_dim × 2 (K+V) × 2 bytes
3. **Activation memory**: Temporary tensors during forward/backward pass
4. **Optimizer states** (training): Adam = params + momentum + variance = 3× param memory
5. **Gradient memory** (training): Same shape as parameters = 1× param memory

⚡ **Expected outcomes**:
- **Inference VRAM**: 16GB params + 4GB KV cache (batch=1, seq=2048) + 2GB activations = **22GB total**
- **Fits in RTX 4090**: 22GB / 24GB = 92% utilization ✅ (2GB headroom)
- **Max batch size**: (24GB - 16GB - 2GB) / 4GB per batch = **batch=1.5** → can only do batch=1 ❌
- **Training VRAM**: 16GB params + 48GB optimizer states + 16GB gradients = **80GB** → need A100 80GB for fine-tuning
- **Quantization motivation**: INT4 → 4GB params (vs 16GB) → frees 12GB for KV cache → enables batch=4

**Constraint status after Ch.2**:
- #1 (Cost): ✅ **MAINTAINED** ($1,095/month RTX 4090 confirmed)
- #2 (Latency): ⚡ **BLOCKED** (batch=1 → sequential processing → high latency under load)
- #3 (Throughput): ❌ **SHORTFALL** (batch=1 → max 3,000 req/day vs 10k target)
- #4 (Memory): ✅ **TARGET HIT!** (22GB / 24GB = fits, but zero batch headroom)
- #5 (Quality): ⚡ **ON TRACK** (Llama-3-8B baseline)
- #6 (Reliability): ⚡ **UNKNOWN**

**Critical realization**: Model fits, but **batch=1 limit kills throughput**. Need Ch.3 quantization to free VRAM for batching.

---

## 1 · Core Idea

VRAM consumption in deep learning breaks into five categories:

1. **Parameters** — the model weights themselves (static, loaded once)
2. **KV cache** — attention keys and values stored for all previous tokens (grows with seq_len × batch_size)
3. **Activations** — intermediate tensors computed during forward pass (cleared after backward pass)
4. **Optimizer states** — momentum, variance for Adam/AdamW (training only, 2× parameter memory)
5. **Gradients** — same shape as parameters, computed during backprop (training only, 1× parameter memory)

For **inference**, you only pay for #1, #2, #3. For **training**, you pay for all five.

---

## 1.5 · The Practitioner Workflow — Your 3-Phase Memory Audit

**Before diving into formulas, understand the workflow you'll follow with every model deployment:**

> 📊 **What you'll build by the end:** A VRAM budget spreadsheet that tells you (1) exact memory breakdown, (2) whether model fits in your GPU, and (3) optimization levers to pull if it doesn't fit.

```
Phase 1: CALCULATE              Phase 2: CHECK                  Phase 3: OPTIMIZE
────────────────────────────────────────────────────────────────────────────
Calculate exact memory          Does total fit in GPU VRAM?     If no → adjust batch size,
for each component:                                            quantization, or gradient
                                                               checkpointing
• Parameters (model weights)    Compare:
• KV cache (attention state)    Total Memory ≤ GPU VRAM?       Decision tree:
• Activations (forward pass)                                   • Reduce batch → cut KV cache
• Optimizer states (training)   If YES → proceed to deploy     • INT8/INT4 → cut params 2-4×
• Gradients (backprop)          If NO → go to Phase 3          • Gradient checkpoint → cut
                                                                 activations 90%

→ DECISION:                     → DECISION:                     → DECISION:
  For inference: skip optimizer   Leave 2-4GB headroom for       Each optimization has quality
  and gradients                   fragmentation + safety         tradeoff — validate after!
```

> ⚠️ **Two ways to read this chapter:**
>
> **Option A (Workflow-first)**: Read §1.5 → follow phase markers → run code snippets → hit decision checkpoints as they appear. Best for practitioners deploying models NOW.
>
> **Option B (Theory-first)**: Read §1–13 linearly for complete theory, then return to §1.5 workflow summary. Best for interviews or foundational understanding.
>
> **Both paths teach the same content** — workflow just reorganizes it by practitioner decision sequence.

> 💡 **Usage note:** Phases 1–2 are always sequential (can't check fit without calculating first). Phase 3 is iterative — you may cycle through multiple optimizations (quantize → check → reduce batch → check) until model fits.

> 💡 **Memory verdict:** 16GB params + 4GB KV cache + 2GB activations = 22GB — fits RTX 4090 24GB with 2GB headroom (batch=1 only; quantization needed for batch growth) ✅.

---

## 2 · Running Example

Llama-3-8B at FP16 precision:
- Parameters: 8B × 2 bytes = **16 GB**
- KV cache (batch=1, seq=2048): **4 GB**
- Activations (forward pass): **2 GB**
- **Total: 22 GB** → fits in RTX 4090 (24 GB) with 2 GB margin

But: 2 GB headroom is not enough for batch=2 (would need 4 GB more for KV cache). This means **batch=1 only** → throughput capped at ~3,000 requests/day (30% of target).

The tension: we need batching to hit throughput target, but VRAM is exhausted. Solution: Ch.3 quantization (16 GB params → 4 GB params) frees 12 GB for KV cache, enabling batch=4.

---

## 3 · [Phase 1: CALCULATE] Parameter Memory

For a transformer model with $N$ parameters stored in precision $P$ bytes per parameter:

$$\text{Parameter Memory} = N \times P$$

| Precision | Bytes per parameter | 8B model | 70B model |
|-----------|---------------------|----------|-----------|
| FP32 | 4 | 32 GB | 280 GB |
| FP16 / BF16 | 2 | 16 GB | 140 GB |
| INT8 | 1 | 8 GB | 70 GB |
| INT4 | 0.5 | 4 GB | 35 GB |

Llama-3-8B at FP16: **8,030,000,000 params × 2 bytes = 16,060 MB ≈ 16 GB**

**Code snippet — Calculate parameter memory:**

```python
# Phase 1: Calculate parameter memory for Llama-3-8B
num_params = 8_000_000_000  # 8 billion parameters
bytes_per_param = 2  # FP16 precision

param_memory_bytes = num_params * bytes_per_param
param_memory_gb = param_memory_bytes / (1024**3)

print(f"Parameter memory: {param_memory_gb:.2f} GB")
print(f"Precision: FP16 ({bytes_per_param} bytes/param)")

# Compare precisions:
for precision, bytes_val in [("FP32", 4), ("FP16", 2), ("INT8", 1), ("INT4", 0.5)]:
    memory_gb = (num_params * bytes_val) / (1024**3)
    print(f"{precision}: {memory_gb:.1f} GB")

# Output:
# Parameter memory: 14.90 GB  (Note: actual = 16GB accounting for embedding layers)
# Precision: FP16 (2 bytes/param)
# FP32: 29.8 GB
# FP16: 14.9 GB
# INT8: 7.5 GB
# INT4: 3.7 GB
```

> 💡 **Industry Standard:** `transformers` library automatically handles precision
>
> ```python
> from transformers import AutoModelForCausalLM
> import torch
>
> # Load model directly in FP16
> model = AutoModelForCausalLM.from_pretrained(
>     "meta-llama/Llama-2-7b-hf",
>     torch_dtype=torch.float16,  # FP16 precision
>     device_map="auto"  # Automatically map to GPU
> )
>
> # Check actual memory usage
> param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
> print(f"Loaded model uses {param_memory / 1e9:.2f} GB")
> ```
>
> **When to use:** Always in production for automatic device placement and precision handling.
> **Common alternatives:** `torch.bfloat16` (better numerical stability), `torch.int8` (quantized)
> **See also:** [Transformers precision guide](https://huggingface.co/docs/transformers/main_classes/model#torch-dtype)

---

## 4 · [Phase 1: CALCULATE] KV Cache Memory

For each token generated, the model stores the key (K) and value (V) tensors for all layers. These are reused in subsequent decoding steps to avoid recomputing attention over the entire history.

$$\text{KV Cache} = 2 \times L \times H \times S \times B \times P$$

Where:
- $L$ = number of layers
- $H$ = hidden dimension
- $S$ = sequence length
- $B$ = batch size
- $P$ = bytes per element (2 for FP16)
- Factor of 2 = one tensor for keys, one for values

**Llama-3-8B example** (32 layers, 4096 hidden dim, FP16):

| Batch | Seq Length | KV Cache Size |
|-------|------------|---------------|
| 1 | 512 | 1 GB |
| 1 | 2048 | 4 GB |
| 4 | 2048 | 16 GB |
| 8 | 2048 | 32 GB |

**The bottleneck**: KV cache scales linearly with batch size. At batch=4, it consumes 16 GB alone — as much as the entire model!

> 💡 **Industry Standard:** Flash Attention for memory-efficient attention
>
> ```python
> # Standard attention: O(N²) memory for attention matrix
> # Flash Attention: Fused kernel with tiling → 3-4× less memory
>
> from transformers import AutoModelForCausalLM
> import torch
>
> model = AutoModelForCausalLM.from_pretrained(
>     "meta-llama/Llama-2-7b-hf",
>     torch_dtype=torch.float16,
>     attn_implementation="flash_attention_2",  # Enable Flash Attention
>     device_map="auto",
> )
>
> # Memory savings:
> # - Standard attention: stores full attention matrix (seq_len × seq_len)
> # - Flash Attention: tiles through HBM, never materializes full matrix
> # - Result: ~20-30% less KV cache memory, 2-3× faster training
> ```
>
> **When to use:** Always for training long sequences (>2048 tokens) or when memory-constrained. Flash Attention 2 is production-ready in Transformers 4.36+.
> **Common alternatives:** Memory-efficient attention (xFormers), Triton FlashAttention, Paged Attention (vLLM)
> **See also:** [Flash Attention paper](https://arxiv.org/abs/2205.14135), [Flash Attention 2 (2023)](https://arxiv.org/abs/2307.08691)

**Code snippet — Calculate KV cache with batch scaling:**

```python
# Phase 1: Calculate KV cache memory for different batch sizes
num_layers = 32  # Llama-3-8B
hidden_size = 4096
seq_len = 2048
bytes_per_param = 2  # FP16

def calculate_kv_cache(batch_size):
    """Calculate KV cache memory in GB."""
    # 2× for keys + values
    kv_cache_bytes = 2 * batch_size * seq_len * num_layers * hidden_size * bytes_per_param
    return kv_cache_bytes / (1024**3)

print("Batch Size | KV Cache Memory | Use Case")
print("-" * 50)
for batch in [1, 2, 4, 8, 16]:
    kv_gb = calculate_kv_cache(batch)

    # DECISION LOGIC: Can we fit this batch size?
    if kv_gb < 4:
        status = "✅ FITS - Low memory"
    elif kv_gb < 10:
        status = "⚠️ MODERATE - Monitor closely"
    else:
        status = "❌ HIGH - Likely OOM on 24GB GPU"

    print(f"batch={batch:2d}  | {kv_gb:5.2f} GB      | {status}")

# Output:
# Batch Size | KV Cache Memory | Use Case
# --------------------------------------------------
# batch= 1  |  1.00 GB      | ✅ FITS - Low memory
# batch= 2  |  2.00 GB      | ✅ FITS - Low memory
# batch= 4  |  4.00 GB      | ⚠️ MODERATE - Monitor closely
# batch= 8  |  8.00 GB      | ⚠️ MODERATE - Monitor closely
# batch=16  | 16.00 GB      | ❌ HIGH - Likely OOM on 24GB GPU
```

### 4.1 ✓ DECISION CHECKPOINT — Phase 1 Parameter & KV Cache Complete

**What you just saw:**
- **Parameter memory (FP16):** 16 GB fixed — doesn't change with batch size
- **KV cache scaling:** 1 GB (batch=1) → 4 GB (batch=4) → 16 GB (batch=16)
- **Critical insight:** KV cache grows linearly with batch; at batch=16, KV cache equals entire model size!

**What it means:**
- **Batch size is the primary memory lever** — doubling batch doubles KV cache (but parameters stay constant)
- **High batch = high throughput BUT high memory** — need to balance based on GPU VRAM
- **Sequence length multiplier:** Doubling seq_len from 2048→4096 also doubles KV cache

**What to do next:**
→ **Add activations & optimizer states** (Phase 1 continuation): §5-6 calculate remaining memory components
→ **Then move to Phase 2**: §7 CHECK if total memory fits in your GPU VRAM
→ **For our Llama-3-8B scenario:** So far 16GB params + 4GB KV (batch=4) = 20GB. Need to add activations (~2GB) to get total, then check against RTX 4090's 24GB.

---

## 5 · [Phase 1: CALCULATE] Activation Memory

Activations are the intermediate tensors computed during the forward pass (attention scores, FFN outputs, layer norm results). They are kept in VRAM until the backward pass (training) or discarded immediately (inference).

**Inference**: ~2–4 GB for Llama-3-8B (varies by batch size and sequence length)
**Training**: ~8–16 GB (must keep activations for backward pass)

**Gradient checkpointing** (training optimization): recompute activations during backward pass instead of storing them → cuts activation memory by 90%, at the cost of 30% slower training.

> 💡 **Industry Standard:** PyTorch `torch.utils.checkpoint`
>
> ```python
> import torch
> from torch.utils.checkpoint import checkpoint
>
> class TransformerBlock(torch.nn.Module):
>     def __init__(self, ...):
>         super().__init__()
>         self.attention = MultiHeadAttention(...)
>         self.ffn = FeedForward(...)
>
>     def forward(self, x):
>         # Standard forward: stores all activations (~8GB for Llama-3-8B)
>         # x = self.attention(x)
>         # x = self.ffn(x)
>
>         # Gradient checkpointing: recompute activations during backward
>         # Memory: ~800MB (90% reduction), Speed: 30% slower
>         x = checkpoint(self.attention, x, use_reentrant=False)
>         x = checkpoint(self.ffn, x, use_reentrant=False)
>         return x
> ```
>
> **When to use:** When training OOMs but you can tolerate 30% longer training time.
> **Common alternatives:** Selective checkpointing (only checkpoint every Nth layer), DeepSpeed activation checkpointing
> **See also:** [PyTorch checkpoint docs](https://pytorch.org/docs/stable/checkpoint.html)

---

## 6 · [Phase 1: CALCULATE] Optimizer States (Training Only)

**Adam/AdamW optimizer** maintains two state tensors per parameter:
- **Momentum** (first moment): same shape as parameters
- **Variance** (second moment): same shape as parameters

$$\text{Optimizer Memory} = 2 \times N \times P$$

For Llama-3-8B at FP32 optimizer states (standard):
- Parameters: 8B × 4 bytes = 32 GB
- Momentum: 8B × 4 bytes = 32 GB
- Variance: 8B × 4 bytes = 32 GB
- **Total optimizer memory: 64 GB**

**Full training memory** = 16 GB (FP16 params) + 64 GB (FP32 optimizer) + 16 GB (FP16 gradients) + 8 GB (activations) = **104 GB**

→ Requires A100 80GB × 2 GPUs or ZeRO-2 sharding (Ch.4)

> 💡 **Industry Standard:** DeepSpeed ZeRO for distributed training
>
> ```python
> # ZeRO Stage 2: Shard optimizer states + gradients across GPUs
> # Reduces per-GPU memory from 104GB → 28GB on 4× GPUs
>
> from transformers import TrainingArguments, Trainer
>
> training_args = TrainingArguments(
>     output_dir="./results",
>     per_device_train_batch_size=1,
>     gradient_accumulation_steps=4,
>     fp16=True,  # FP16 training
>     deepspeed="ds_config.json",  # Enable DeepSpeed
> )
>
> # ds_config.json:
> # {
> #   "zero_optimization": {
> #     "stage": 2,  # Shard optimizer + gradients
> #     "offload_optimizer": {"device": "cpu"}  # Optional: offload to CPU RAM
> #   },
> #   "fp16": {"enabled": true}
> # }
>
> trainer = Trainer(
>     model=model,
>     args=training_args,
>     train_dataset=train_dataset,
> )
> trainer.train()
> ```
>
> **When to use:** Multi-GPU training when single GPU OOMs. ZeRO-2 is the sweet spot for most use cases.
> **Common alternatives:** ZeRO-3 (shard everything including params, highest memory savings), FSDP (PyTorch native alternative)
> **See also:** [DeepSpeed ZeRO paper](https://arxiv.org/abs/1910.02054), [Hugging Face DeepSpeed integration](https://huggingface.co/docs/transformers/main_classes/deepspeed)

---

## 7 · [Phase 2: CHECK] VRAM Budget Calculator — Inference

**Formula**:
$$\text{VRAM}_{\text{inference}} = \text{Params} + \text{KV Cache} + \text{Activations}$$

**Llama-3-8B on RTX 4090 (24 GB):**

| Component | FP16 | INT8 | INT4 |
|-----------|------|------|------|
| Parameters | 16 GB | 8 GB | 4 GB |
| KV Cache (batch=1, seq=2048) | 4 GB | 4 GB | 4 GB |
| Activations | 2 GB | 2 GB | 2 GB |
| **Total** | **22 GB** | **14 GB** | **10 GB** |
| **Free VRAM** | **2 GB** | **10 GB** | **14 GB** |
| **Max batch size** | **1** | **3** | **4** |

**Critical insight**: INT4 quantization frees 12 GB, enabling batch=4 → 4× throughput!

**Code snippet — Phase 2 CHECK: Does model fit?**

```python
# Phase 2: Calculate total memory and check against GPU VRAM
def check_vram_fit(precision="FP16", batch_size=1, seq_len=2048, gpu_vram_gb=24):
    """Check if Llama-3-8B fits in GPU memory."""

    # Phase 1 calculations (from above)
    num_params = 8_000_000_000
    precision_map = {"FP32": 4, "FP16": 2, "INT8": 1, "INT4": 0.5}
    bytes_per_param = precision_map[precision]

    # Component breakdown
    param_memory_gb = (num_params * bytes_per_param) / (1024**3)

    # KV cache: 2 × layers × hidden × seq × batch × bytes
    kv_cache_gb = (2 * 32 * 4096 * seq_len * batch_size * bytes_per_param) / (1024**3)

    # Activations: rough estimate
    activations_gb = 2.0 if batch_size <= 2 else 4.0

    total_memory_gb = param_memory_gb + kv_cache_gb + activations_gb

    # DECISION LOGIC: Does it fit with safety margin?
    safety_margin_gb = 2.0  # Leave 2GB headroom for fragmentation
    fits = total_memory_gb <= (gpu_vram_gb - safety_margin_gb)
    utilization_pct = (total_memory_gb / gpu_vram_gb) * 100

    print(f"\n=== VRAM CHECK: {precision} | batch={batch_size} | seq={seq_len} ===")
    print(f"Parameter memory:  {param_memory_gb:6.2f} GB")
    print(f"KV cache:          {kv_cache_gb:6.2f} GB")
    print(f"Activations:       {activations_gb:6.2f} GB")
    print(f"{'='*50}")
    print(f"Total required:    {total_memory_gb:6.2f} GB")
    print(f"GPU VRAM:          {gpu_vram_gb:6.2f} GB")
    print(f"Utilization:       {utilization_pct:5.1f}%")
    print(f"\nVerdict: {'\u2705 FITS' if fits else '\u274c OOM - DOES NOT FIT'}")

    if not fits:
        print(f"\n⚠️ Need {total_memory_gb - (gpu_vram_gb - safety_margin_gb):.1f} GB more VRAM")
        print("  → Options: reduce batch, use quantization, or upgrade GPU")

    return fits, total_memory_gb

# Test scenarios
check_vram_fit(precision="FP16", batch_size=1, seq_len=2048, gpu_vram_gb=24)
check_vram_fit(precision="FP16", batch_size=4, seq_len=2048, gpu_vram_gb=24)  # Will OOM
check_vram_fit(precision="INT4", batch_size=4, seq_len=2048, gpu_vram_gb=24)  # Will fit!

# Output:
# === VRAM CHECK: FP16 | batch=1 | seq=2048 ===
# Parameter memory:   14.90 GB
# KV cache:            1.00 GB
# Activations:         2.00 GB
# ==================================================
# Total required:     17.90 GB
# GPU VRAM:           24.00 GB
# Utilization:        74.6%
#
# Verdict: ✅ FITS
```

### 7.1 ✓ DECISION CHECKPOINT — Phase 2 Complete

**What you just saw:**
- **FP16, batch=1:** Total = 18GB, fits in 24GB RTX 4090 with 6GB headroom ✅
- **FP16, batch=4:** Total = 34GB, **exceeds 24GB** → OOM error ❌
- **INT4, batch=4:** Total = 10GB, fits with 14GB headroom ✅ (4× throughput unlocked!)

**What it means:**
- **Model fits at batch=1** but no room for higher throughput
- **Batch scaling hits wall fast:** Going from batch=1→4 adds 12GB of KV cache (exceeds budget)
- **Quantization is mandatory for batching:** INT4 shrinks params 16GB→4GB, freeing 12GB for KV cache

**What to do next:**
→ **If model fits (✅):** Proceed to deployment — skip Phase 3
→ **If model OOMs (❌):** Go to Phase 3 OPTIMIZE — apply quantization, reduce batch, or enable gradient checkpointing
→ **For our scenario:** Model fits at batch=1 but **throughput target requires batch=4**. We MUST apply INT4 quantization (Ch.3) to free VRAM.

---

## 8 · [Phase 2: CHECK] VRAM Budget Calculator — Training

**Formula**:
$$\text{VRAM}_{\text{training}} = \text{Params} + \text{Optimizer States} + \text{Gradients} + \text{Activations}$$

**Llama-3-8B training (full fine-tuning, Adam, no checkpointing):**

| Component | FP16 weights, FP32 optimizer |
|-----------|------------------------------|
| Parameters (FP16) | 16 GB |
| Optimizer states (FP32) | 64 GB |
| Gradients (FP16) | 16 GB |
| Activations (batch=1) | 8 GB |
| **Total** | **104 GB** |

**Cannot fit on single RTX 4090 (24 GB)!**

**Solutions**:
- A100 80GB × 2 with ZeRO-2 sharding (Ch.4)
- Gradient checkpointing: 104 GB → 30 GB (fits on A100 40GB)
- LoRA fine-tuning: only train adapter weights → 16 GB params + 2 GB adapter = 18 GB (fits on RTX 4090!)

### 8.1 ✓ DECISION CHECKPOINT — Phase 2 Training Memory Check

**What you just saw:**
- **Training memory:** 104GB total (16GB params + 64GB optimizer + 16GB gradients + 8GB activations)
- **RTX 4090 has 24GB** → training **does not fit** ❌
- **A100 80GB × 2 required** for full fine-tuning ($6,000/month vs $1,095/month for RTX 4090)

**What it means:**
- **Full fine-tuning is 4–5× more expensive** than inference (optimizer states dominate)
- **Budget constraint conflict:** Training needs $6k/month but budget is $15k total (includes inference)
- **Gradient checkpointing helps** but still needs 30GB (A100 40GB at $3,000/month)

**What to do next:**
→ **Option 1 (Full fine-tune):** Upgrade to A100 80GB × 2 with ZeRO-2 sharding — **$6,000/month** ⚠️
→ **Option 2 (Gradient checkpoint):** A100 40GB — **$3,000/month**, 30% slower training
→ **Option 3 (LoRA fine-tune):** Train only adapters (2GB), keep base frozen — **fits on RTX 4090 ($1,095/month)** ✅
→ **For our scenario:** Budget is tight. **Choose LoRA (Ch.4)** to fine-tune on RTX 4090 without exceeding $15k/month total budget.

---

## 9 · Step by Step — Calculating VRAM for Your Model

**Example**: You want to run Llama-2-70B at INT4 quantization, batch=2, seq_len=4096.

```
Step 1: Parameter memory
  70B params × 0.5 bytes (INT4) = 35 GB

Step 2: KV cache
  L = 80 layers, H = 8192 hidden dim, S = 4096 seq, B = 2
  KV = 2 × 80 × 8192 × 4096 × 2 × 2 bytes
     = 2 × 80 × 8192 × 4096 × 2 × 2
     = 21,474,836,480 bytes ≈ 21.5 GB

Step 3: Activations (estimate)
  ~8 GB (scales with batch × seq)

Total: 35 + 21.5 + 8 = 64.5 GB

Conclusion: Fits on A100 80GB (80 - 64.5 = 15.5 GB margin)
            Does NOT fit on A100 40GB
```

---

## 10 · Key Diagrams

### VRAM Breakdown: Llama-3-8B Inference vs Training

```
INFERENCE (FP16)                      INFERENCE (INT4)                      TRAINING (FP16/FP32)
─────────────────                     ─────────────────                     ────────────────────
RTX 4090 24GB │                       RTX 4090 24GB │                       A100 80GB × 2 │
              │                                     │                                       │
Params:     16GB ████████████████      Params:     4GB ████                Params:     16GB (FP16)
KV Cache:    4GB ████ (batch=1)       KV Cache:  16GB ████████████████    Optimizer:  64GB (FP32)
Activations: 2GB ██                    Activations: 2GB ██                  Gradients:  16GB (FP16)
             ─────                                 ─────                   Activations:  8GB
Total:      22GB (92% utilization)    Total:     22GB (92% utilization)                ─────
Free:        2GB ✅                    Free:       2GB ✅                   Total:     104GB ❌
             ↓                                      ↓                                    ↓
Batch max:    1 (limited!)            Batch max:   4 (4× throughput!)     Requires: 2× A100 80GB
Throughput: 3k req/day ❌             Throughput: 12k req/day ✅          or gradient checkpointing
```

---

## 11 · What Can Go Wrong

- **Forgetting KV cache** — it is not part of the model file, but grows during inference and can OOM unexpectedly on long sequences
- **Ignoring batch size scaling** — doubling batch size does NOT double total VRAM; KV cache scales linearly, but params stay constant
- **Using FP32 optimizer states on a 24 GB GPU** — Adam needs 8× parameter memory; always use FP16 params + FP32 optimizer
- **Not accounting for fragmentation** — running at 95%+ VRAM utilization causes CUDA memory allocator failures; leave 2–4 GB headroom
- **Assuming training = 3× inference** — it is closer to 5–8× due to optimizer states, especially with Adam/AdamW

---

---

## 11.5 · [Phase 3: OPTIMIZE] The Hyperparameter Dial

> 🎯 **You are here because:** Your model OOMed in Phase 2 CHECK, OR you need higher throughput (larger batch) but don't have VRAM headroom.

Three knobs control VRAM. Each can be turned independently, but they interact through the total budget constraint.

### Dial 1 — Batch Size

$$\text{VRAM}_\text{KV} = 2 \times L \times H \times S \times B \times P$$

| Batch size $B$ | KV cache VRAM (Llama-3-8B, $S=2048$, BF16) | Total inference VRAM | Throughput (tok/s est.) |
|----------------|---------------------------------------------|----------------------|-------------------------|
| 1 | ~0.9 GB | ~17 GB | ~40 tok/s |
| 2 | ~1.8 GB | ~18 GB | ~75 tok/s |
| 4 | ~3.5 GB | ~19.5 GB | ~130 tok/s |
| 8 | ~7.0 GB | ~23 GB | ~200 tok/s |
| 16 | ~14 GB | ~30 GB ❌ OOM on 24 GB | — |

> ⚠️ Never push batch=16 on a 24 GB GPU with Llama-3-8B at BF16. Use INT8/INT4 quantization (Ch.3) to free ~8 GB first.

### Dial 2 — Sequence Length

Sequence length $S$ scales KV cache linearly. Halving sequence length halves KV cache:

| Sequence length | KV cache (batch=4, BF16) | Notes |
|-----------------|--------------------------|-------|
| 512 tokens | ~0.9 GB | Customer support queries |
| 2,048 tokens | ~3.5 GB | Standard context window |
| 8,192 tokens | ~14 GB | Near-OOM at batch=4; need INT8 |
| 32,768 tokens | ~56 GB ❌ | Multi-GPU only |

### Dial 3 — Precision

Precision affects parameter memory and KV cache simultaneously:

| Precision | Bytes per param $P$ | Llama-3-8B params | KV cache per unit | Total VRAM (batch=4, $S=2048$) |
|-----------|--------------------|--------------------|-------------------|-------------------------------|
| FP32 | 4 | 32 GB | 4 bytes | ~50 GB ❌ |
| BF16 | 2 | 16 GB | 2 bytes | ~19.5 GB ✅ |
| INT8 | 1 | 8 GB | 1 byte | ~9.8 GB ✅✅ |
| INT4 | 0.5 | 4 GB | 0.5 byte | ~5 GB — headroom for batch=16 |

> 💡 KV cache grows with batch × seq_len. Parameters are **fixed** at load time. Quantization shrinks both; smaller batch reduces only the KV cache.

**Code snippet — Phase 3 OPTIMIZE: Apply INT8 quantization**

```python
# Phase 3: Apply quantization to free VRAM for higher batch size
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Original FP16 model: 16GB parameters
# Target: Reduce to 8GB (INT8) to free 8GB for KV cache (batch=1 → batch=4)

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,  # INT8 quantization
    llm_int8_threshold=6.0,  # Keep outlier features in FP16
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=quantization_config,
    device_map="auto",  # Automatically place on GPU
)

# Verify memory reduction
param_memory_bytes = sum(
    p.numel() * p.element_size() for p in model.parameters()
)
param_memory_gb = param_memory_bytes / (1024**3)

print(f"Quantized model memory: {param_memory_gb:.2f} GB")
print(f"Memory saved: {16 - param_memory_gb:.2f} GB")
print(f"Now have headroom for batch=4 KV cache (4GB)")

# Output:
# Quantized model memory: 7.85 GB  (50% reduction from 16GB)
# Memory saved: 8.15 GB
# Now have headroom for batch=4 KV cache (4GB)
```

> 💡 **Industry Standard:** `bitsandbytes` library for production quantization
>
> ```python
> # INT4 quantization with QLoRA (best memory efficiency)
> from transformers import AutoModelForCausalLM, BitsAndBytesConfig
>
> bnb_config = BitsAndBytesConfig(
>     load_in_4bit=True,  # INT4 quantization
>     bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in BF16 for stability
>     bnb_4bit_use_double_quant=True,  # Double quantization for extra compression
>     bnb_4bit_quant_type="nf4",  # NormalFloat4 (better for LLMs)
> )
>
> model = AutoModelForCausalLM.from_pretrained(
>     "meta-llama/Llama-2-7b-hf",
>     quantization_config=bnb_config,
>     device_map="auto",
> )
> # Memory: ~4GB (75% reduction from 16GB FP16)
> # Enables batch=8 on RTX 4090 (24GB)
> ```
>
> **When to use:** When you need higher batch size but GPU VRAM is constrained. INT8 is the safe default; INT4 requires quality validation (Ch.3).
> **Common alternatives:** GPTQ (faster INT4 inference), AWQ (activation-aware quantization), SmoothQuant
> **See also:** [bitsandbytes docs](https://github.com/TimDettmers/bitsandbytes), [QLoRA paper](https://arxiv.org/abs/2305.14314)

### 11.5.4 ✓ DECISION CHECKPOINT — Phase 3 Optimization Complete

**What you just saw:**
- **Batch size lever:** Doubling batch doubles KV cache but leaves params unchanged → choose based on throughput needs
- **Sequence length lever:** Halving seq_len halves KV cache → use when prompts are predictably short
- **Precision lever:** INT8 cuts memory 50%, INT4 cuts 75% → most powerful optimization but requires quality validation

**What it means:**
- **Quantization unlocks batch scaling:** FP16 @ batch=1 (18GB) → INT4 @ batch=4 (10GB) = 4× throughput in same GPU
- **Each lever has tradeoffs:**
  - Reduce batch → lower throughput (bad for production load)
  - Reduce seq_len → can't handle long documents (limits use cases)
  - Quantize → potential quality loss (must validate perplexity/accuracy in Ch.3)
- **Gradient checkpointing (training only):** 90% activation memory reduction, 30% speed penalty

**What to do next:**
→ **After optimization:** Re-run Phase 2 CHECK with new parameters to confirm model now fits
→ **Validate quality:** Especially after quantization — measure perplexity, run test prompts (Ch.3 benchmarks)
→ **For our scenario:** Applying INT4 quantization freed 12GB. Re-check: 4GB params + 4GB KV (batch=4) + 2GB activations = **10GB total ✅ fits!** Proceed to Ch.3 to validate quality.

---

## 11.6 · Code Skeleton

> 📦 **Complete workflow implementation** — use these functions to implement the 3-phase workflow (§1.5) in your deployment pipeline.

```python
# Educational: VRAM budget calculator from scratch (Phase 1 + Phase 2)
def vram_budget_inference(
    n_params: int,           # total model parameters (e.g., 8_000_000_000)
    bytes_per_param: float,  # precision: FP32=4, BF16=2, INT8=1, INT4=0.5
    n_layers: int,           # transformer layers (e.g., 32 for Llama-3-8B)
    n_heads: int,            # attention heads (e.g., 32)
    seq_len: int,            # max sequence length (e.g., 2048)
    batch_size: int,         # concurrent requests
    activation_overhead_gb: float = 2.0,  # typical activation memory
) -> dict:
    """Calculate inference VRAM breakdown in GB.

    Implements Phase 1 (CALCULATE) + Phase 2 (CHECK) of the workflow.
    """
    params_gb = (n_params * bytes_per_param) / 1e9
    head_dim = 128  # typical; hidden_dim / n_heads
    kv_cache_gb = (2 * n_layers * n_heads * head_dim * seq_len * batch_size * bytes_per_param) / 1e9
    total_gb = params_gb + kv_cache_gb + activation_overhead_gb
    return {
        "params_gb": round(params_gb, 2),
        "kv_cache_gb": round(kv_cache_gb, 2),
        "activations_gb": activation_overhead_gb,
        "total_gb": round(total_gb, 2),
        "fits_rtx4090": total_gb <= 22.0,  # leave 2 GB headroom
    }

# Llama-3-8B, BF16, batch=4
print(vram_budget_inference(8_000_000_000, 2.0, 32, 32, 2048, 4))
# → {'params_gb': 16.0, 'kv_cache_gb': 3.44, 'activations_gb': 2.0, 'total_gb': 21.44, 'fits_rtx4090': True}
```

```python
# Production: pre-flight VRAM check before model load (Phase 2 CHECK)
import subprocess, json

def check_gpu_vram_available() -> float:
    """Return available VRAM in GB using nvidia-smi."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    return int(result.stdout.strip()) / 1024  # MiB → GB

def preflight_vram_check(required_gb: float, safety_margin_gb: float = 2.0) -> bool:
    """Abort if not enough VRAM. Call before loading model."""
    available = check_gpu_vram_available()
    if available < required_gb + safety_margin_gb:
        raise RuntimeError(
            f"Insufficient VRAM: {available:.1f} GB available, "
            f"{required_gb + safety_margin_gb:.1f} GB needed (incl. {safety_margin_gb} GB margin)"
        )
    return True
```

---

## 11.7 · Where This Reappears

| Chapter | How memory budget concepts appear |
|---------|------------------------------------|
| **Ch.3 — Quantization** | INT8/INT4 reduces bytes-per-param $P$ in the VRAM formula; the same formulas here predict post-quantization savings. **Apply Phase 3 OPTIMIZE workflow** to choose quantization method. |
| **Ch.5 — Inference Optimization** | PagedAttention manages KV cache pages dynamically — the same KV cache VRAM formula here determines page pool size. **Phase 2 CHECK** validates total memory fits. |
| **Ch.6 — vLLM & Serving** | vLLM's `gpu_memory_utilization` parameter is directly the `1 - headroom` fraction from the VRAM budget. **Phase 1 CALCULATE** determines safe utilization threshold. |
| **AI Infrastructure Ch.4** | LoRA fine-tuning needs parameter + optimizer + gradient memory; the optimizer state formula (8× for Adam) comes from the training budget section here. **Phase 2 CHECK** shows LoRA fits on RTX 4090. |
| **Cost & Latency (AI track)** | Cost-per-token = (hourly rate) / (tokens/sec); tokens/sec depends on batch size, which is capped by VRAM budget derived here. **Phase 3 OPTIMIZE** determines max batch for throughput. |

**🔄 Workflow integration:** Every chapter above uses the **3-phase workflow** (CALCULATE → CHECK → OPTIMIZE) from §1.5. When you encounter memory constraints in later chapters, return to this workflow to diagnose bottlenecks.

---

## 12 · Progress Check — What We've Accomplished

🎉 **VRAM BUDGET CONFIRMED! Llama-3-8B fits in RTX 4090 at FP16**

**Unlocked capabilities**:
- ✅ **3-phase workflow mastered**: CALCULATE (§3-6) → CHECK (§7-8) → OPTIMIZE (§11.5) — reusable for any model deployment
- ✅ **Precise VRAM calculator**: Know exact memory breakdown for any model/batch/seq_len
- ✅ **Inference budget**: 16GB params + 4GB KV + 2GB activations = 22GB (fits in 24GB) ✅
- ✅ **Training budget**: 104GB total → need A100 80GB × 2 or gradient checkpointing
- ✅ **Batch size limits**: batch=1 max at FP16 → need quantization (Ch.3) for batch=4
- ✅ **Decision checkpoints**: 3 critical go/no-go points for deployment confidence

**Progress toward constraints**:

| Constraint | Status | Current State |
|------------|--------|---------------|
| #1 COST | ✅ **MAINTAINED** | $1,095/month RTX 4090 confirmed |
| #2 LATENCY | ❌ **BLOCKED** | batch=1 → sequential processing → high latency under load |
| #3 THROUGHPUT | ❌ **SHORTFALL** | batch=1 → 3,000 req/day (30% of 10k target) |
| #4 MEMORY | ✅ **TARGET HIT!** | 22GB / 24GB = fits! (but zero batch headroom) |
| #5 QUALITY | ⚡ **ON TRACK** | Llama-3-8B baseline (assumed >95%) |
| #6 RELIABILITY | ⚡ **UNKNOWN** | Need Ch.9-10 for production deployment |

**What we can solve now**:

✅ **Confirm hardware purchase with exact VRAM breakdown**:
```
Before Ch.2:
You: "8B params × 2 bytes = 16GB. RTX 4090 has 24GB. Should fit!"
[Order GPU, deploy → OOM error]

After Ch.2:
You calculate:
"Parameters: 16GB
 KV cache (batch=1, seq=2048): 4GB
 Activations: 2GB
 Total: 22GB / 24GB = 92% utilization ✅
 Headroom: 2GB (not enough for batch=2)

 Conclusion: Fits for batch=1 only. Need quantization (Ch.3) to enable batching."

CEO: "So we CAN use RTX 4090, but throughput will be limited?"
You: "Correct. We'll hit 3,000 req/day at batch=1. To reach 10k target,
      we need INT4 quantization (Ch.3) to free VRAM for batch=4."

Result: ✅ Confident hardware purchase + clear roadmap to hit throughput target!
```

✅ **Understand training infrastructure needs**:
```
Before Ch.2:
"Can we fine-tune Llama-3-8B on RTX 4090?"

After Ch.2:
"Training memory: 16GB params + 64GB optimizer + 16GB gradients = 96GB
 RTX 4090 only has 24GB → cannot fit!

 Options:
 1. A100 80GB × 2 with ZeRO-2 sharding ($6,000/month)
 2. Gradient checkpointing → 30GB (fits on A100 40GB, $3,000/month)
 3. LoRA fine-tuning → only 18GB (fits on RTX 4090, $1,095/month!) ✅

 Decision: Use LoRA (Ch.4) to fine-tune on RTX 4090 without budget increase."

Result: ✅ Can fine-tune without exceeding $15k/month budget!
```

✅ **Set realistic batch size expectations**:
```
CEO: "Can we batch 10 requests at once for efficiency?"

You: "Let me calculate:
 batch=10, seq=2048:
 KV cache = 2 × 32 layers × 4096 dim × 2048 seq × 10 batch × 2 bytes
          = 40 GB for KV cache alone!

 RTX 4090 only has 24GB total → cannot fit batch=10.

 At FP16: max batch=1 (2GB free VRAM)
 At INT4: max batch=4 (14GB free VRAM, 4GB per batch KV cache)

 To reach batch=10, need A100 80GB or multi-GPU setup (Ch.7)."

Result: ✅ Realistic throughput expectations set!
```

**🔄 How the workflow helped:**

**Before workflow approach:**
- "Model OOMs — no idea why or how to fix"
- Guess-and-check with GPU purchases ($50k mistakes)

**After workflow approach:**
- "Phase 2 CHECK shows 34GB required but 24GB available → OOM expected"
- "Phase 3 OPTIMIZE: apply INT4 quantization → re-CHECK: 10GB required ✅ fits!"
- Confident deployment without trial-and-error

**What's still blocking**:

- ❌ **Throughput target unreachable**: batch=1 → 3,000 req/day (need 10k) → **Need Ch.3 quantization to enable batch=4**
- ❌ **Latency under load**: Sequential processing → queuing delays when traffic spikes
- ❌ **No fine-tuning path on current budget**: 104GB training needs expensive GPUs → **Need Ch.4 LoRA for RTX 4090 fine-tuning**
- ❌ **No serving framework selected**: Raw inference loop is slow → **Need Ch.5-6 for optimized serving**

**Next chapter**: [Quantization & Precision](../ch03_quantization_and_precision) shrinks model from 16GB → 4GB:
- INT4 quantization (GPTQ/AWQ)
- Quality validation (perplexity benchmarks)
- **Unlocks batch=4 → 12,000 req/day throughput ✅ (hits target!)**

**Key interview concepts from this chapter**:

| Must know | Likely asked | Trap to avoid |
|---|---|---|
| VRAM = params + KV cache + activations (inference); add optimizer + gradients (training) | How much VRAM does Llama-2-70B need for inference at batch=4, seq=4096? | Forgetting KV cache scales linearly with batch × seq_len |
| KV cache formula: $2 \times L \times H \times S \times B \times P$ | Why does batch=1 use so little VRAM compared to batch=8? | Thinking activations dominate — KV cache is usually the biggest variable component |
| Adam optimizer = 2× param memory (momentum + variance in FP32) | Can you fine-tune a 13B model on a single A100 40GB? | Not accounting for optimizer states (65% of training VRAM) |
| Gradient checkpointing trades 30% speed for 90% less activation memory | What is the memory breakdown for training vs inference? | Confusing inference (no optimizer/gradients) with training (5–8× more VRAM) |
| INT4 quantization: 16GB → 4GB params (75% reduction) enables 4× batch size | How does quantization help throughput? | Saying "quantization speeds up inference" — it speeds up throughput via batching, not single-request latency |

---

## 13 · Bridge to Chapter 3

Ch.2 confirmed the model fits — but revealed a critical bottleneck: **batch=1 limits throughput to 3,000 req/day** (30% of target). The 2 GB of free VRAM is not enough to batch multiple requests.

**The workflow diagnosis:**
- **Phase 1 CALCULATE**: 16GB params + 4GB KV (batch=1) + 2GB activations = 22GB
- **Phase 2 CHECK**: Fits in 24GB ✅, but only 2GB headroom (insufficient for batch=2)
- **Phase 3 OPTIMIZE**: Need to free 12GB to enable batch=4 → **quantization is the only viable lever**

Ch.3 (Quantization & Precision) attacks this problem directly: by shrinking the model from 16 GB to 4 GB via INT4 quantization, we free 12 GB for KV cache, enabling batch=4 and 4× throughput.

The question: **does INT4 quantization destroy quality?** That is what Ch.3 answers. You'll apply the same **Phase 3 OPTIMIZE** workflow, then re-run **Phase 2 CHECK** to validate the quantized model fits at batch=4.

## Illustrations

![Memory budgets — VRAM breakdown for inference vs training, KV cache scaling, batch size limits](img/Memory%20Budgets.png)

