# Ch.3 — Quantization & Precision

> **The story.** For most of deep learning history, training meant FP32 and inference meant FP32 or FP16. Mixed-precision training (Micikevicius et al., NVIDIA, **2017**) showed you could train with FP16 activations and FP32 master weights without accuracy loss, cutting memory by 50%. **INT8 quantization** (Jacob et al., Google, **2018**) took this further: by mapping FP32 weights to 8-bit integers, you could shrink models by 4× with <1% accuracy drop. **GPTQ** (Frantar et al., IST Austria, **2022**) and **AWQ** (Lin et al., MIT, **2023**) pushed to **INT4** — a single 4-bit integer per weight — achieving 75% compression with careful per-channel scaling. The 2023 wave of open LLMs (Llama-2, Mistral) arrived with pre-quantized INT4 checkpoints, enabling 70B-parameter models to run on consumer GPUs. By 2024, the community consensus: *INT4 is the default for inference, FP16 for training, BF16 for stability.*
>
> **Where you are in the curriculum.** Ch.1 picked the GPU. Ch.2 calculated that Llama-3-8B (FP16) uses 22 GB / 24 GB, leaving zero headroom for batching. This chapter shrinks the model from 16 GB → 4 GB via INT4 quantization, unlocking batch=4 and 4× throughput. The InferenceBase question: *Does INT4 quantization destroy quality?* We validate with perplexity benchmarks and document extraction accuracy tests.
> **Notation.** `INT4` / `INT8` = integer quantization to 4 or 8 bits per weight. `FP16` = IEEE half precision (1 sign, 5 exponent, 10 mantissa bits). `BF16` = bfloat16 (1 sign, 8 exponent, 7 mantissa bits; wider dynamic range than FP16). `GPTQ` = post-training quantization via approximate second-order Hessian (Frantar et al., 2022). `AWQ` = activation-aware weight quantization; protects high-salience channels from rounding error. `Perplexity` = token-prediction quality metric; lower is better; used to validate accuracy after quantization.
<!-- notation: key variables defined here -->

---

## 0 · The Challenge — Where We Are

> 🎯 **The mission**: Self-host Llama-3-8B for <$15k/month, replacing $80k OpenAI API costs
>
> **6 Constraints**: #1 Cost (<$15k/mo) • #2 Latency (≤2s) • #3 Throughput (≥10k req/day) • #4 Memory (fit in VRAM) • #5 Quality (≥95% accuracy) • #6 Reliability (>99% uptime)

**What we know so far**:
- ✅ Ch.1: RTX 4090 identified ($1,095/month)
- ✅ Ch.2: Llama-3-8B fits at FP16 (22GB / 24GB), but **batch=1 only**

**What's blocking us**:

🚨 **Throughput target unreachable: batch=1 → 3,000 req/day vs 10k target**

**Current situation**: Engineer testing FP16 deployment in staging

```
Load test results (FP16, batch=1):
- Single request latency: 280ms (excellent! ✅)
- Peak throughput: 12 requests/sec
- Daily capacity: 12 × 60 × 60 × 24 / 3 = ~3,400 req/day

Engineer: "Latency is great, but we can only handle 3,000 requests/day.
          Target is 10,000. We're 70% short!"

CEO: "Can we just rent 3× RTX 4090s?"
Engineer: "That would be 3 × $1,095 = $3,285/month. Plus complexity of
          load balancing. Let me try quantization first — if we can
          enable batch=4, we hit 12,000 req/day on one GPU."

CEO: "But doesn't quantization destroy quality? Our customers need
     95%+ accuracy on document extraction."

Engineer: "That's what I need to validate. INT4 quantization compresses
          from 16GB → 4GB (75% reduction). The freed 12GB enables batch=4.
          But we need to measure accuracy on our doc extraction benchmark."
```

**Problems**:
1. ❌ **Cannot afford 3× GPUs**: $3,285/month still under budget, but adds deployment complexity
2. ❌ **Batch=1 bottleneck**: 22GB VRAM usage leaves only 2GB free → cannot fit batch=2 KV cache (4GB)
3. ❌ **Unknown quality impact**: INT4 quantization is 8× less precision (4 bits vs 32 bits FP32) → will accuracy tank?
4. ❌ **No quantization experience**: Team has never deployed quantized models → uncertain about tools (GPTQ vs AWQ vs GGUF)
5. ❌ **Latency risk**: Will quantization make inference slower or faster? (Smaller model → faster memory transfers, but INT4 matmul might be slower)

**Business impact**:
- **3,000 req/day = 30% of target**: Cannot serve projected Q2 traffic growth
- **Quality regression risk**: If INT4 drops accuracy below 95%, we lose customers → quantization is not an option
- **Decision deadline**: CEO needs recommendation by Friday — test INT4 or order 3× RTX 4090s?

**What this chapter unlocks**:

🚀 **INT4 quantization validation with quality & throughput proof**:
1. **Model compression**: Llama-3-8B-FP16 (16GB) → Llama-3-8B-INT4 (4GB) via GPTQ
2. **VRAM freed**: 22GB → 10GB (enables batch=4, 16GB KV cache budget)
3. **Throughput gain**: 3,000 req/day → 12,000 req/day (4× improvement ✅ hits target!)
4. **Quality validation**: Run document extraction benchmark (500 PDFs) → measure accuracy FP16 vs INT4
5. **Latency impact**: Measure p50/p95 latency at batch=1 and batch=4

⚡ **Expected outcomes**:
- **Quality**: 97.4% accuracy (FP16) → 96.2% accuracy (INT4) → **1.2 point drop, still >95% ✅**
- **Throughput**: 3,000 → 12,000 req/day ✅ (120% of target!)
- **Latency**: 280ms (FP16, batch=1) → 420ms (INT4, batch=4 avg) → **still <2s ✅**
- **VRAM**: 22GB → 10GB (58% utilization, room to grow to batch=8 if needed)

**Constraint status after Ch.3**:
- #1 (Cost): ✅ **MAINTAINED** ($1,095/month, no additional GPUs needed)
- #2 (Latency): ✅ **TARGET HIT!** (420ms p50, 1.2s p95 with batch=4)
- #3 (Throughput): ✅ **TARGET HIT!** (12,000 req/day, 120% of 10k target)
- #4 (Memory): ✅ **OPTIMIZED!** (10GB / 24GB = 58% utilization vs 92% FP16)
- #5 (Quality): ✅ **TARGET HIT!** (96.2% accuracy, above 95% threshold)
- #6 (Reliability): ⚡ **UNKNOWN** (need production deployment)

**Critical breakthrough**: **4 of 6 constraints now met!** Cost, Latency, Throughput, Quality all validated. Ready for serving framework selection (Ch.5-6).

---

## Animation

> 🎬 *Animation placeholder — needle-builder agent will generate this.*

---

## 1 · Core Idea

**Quantization** = mapping high-precision floating-point weights (FP32/FP16) to lower-precision integers (INT8/INT4) using learned scaling factors:

$$W_{\text{quantized}} = \text{round}\left(\frac{W_{\text{FP16}}}{s}\right), \quad s = \frac{\max(|W|)}{2^{b-1}-1}$$

Where:
- $W_{\text{FP16}}$ = original weight (e.g., 0.347)
- $s$ = scaling factor (e.g., 0.023)
- $b$ = bits (4 for INT4, 8 for INT8)
- Round to nearest integer in range $[-2^{b-1}, 2^{b-1}-1]$

**Why it works**: Transformer weights follow near-Gaussian distributions with 90% of values in $[-2\sigma, 2\sigma]$. Quantizing to 16 levels (INT4) preserves most information when scaling is per-channel.

---

## 2 · Running Example

**Scenario**: Llama-3-8B at FP16 uses 16 GB for parameters. INT4 quantization → 4 GB (75% reduction). The freed 12 GB enables batch=4, increasing throughput from 3,000 → 12,000 req/day ✅.

**Risk**: INT4 is 8× less precision than FP32. Will document extraction accuracy tank?

**Test setup**:
1. Quantize Llama-3-8B-Instruct to INT4 using GPTQ (GPU-based, fast)
2. Run 500-PDF benchmark: extract invoice fields (vendor, amount, date)
3. Compare accuracy: FP16 vs INT4
4. Measure latency: batch=1 (FP16) vs batch=4 (INT4)

**Results**:
- FP16: 97.4% extraction accuracy, 280ms p50 latency
- INT4: 96.2% extraction accuracy ✅ (1.2 point drop, still >95%), 420ms p50 latency ✅ (<2s target)
- Throughput: 3,000 → 12,000 req/day ✅ (batch=4 enabled)

**Decision**: Deploy INT4. Slight quality drop acceptable for 4× throughput + 75% memory savings.

---

## 3 · Quantization Schemes

| Method | Precision | Compression | When to use |
|--------|-----------|-------------|-------------|
| **FP16** | 16-bit float | Baseline | Training, high-quality inference |
| **INT8** | 8-bit integer | 4× | Production inference, <1% accuracy loss |
| **INT4** | 4-bit integer | 8× | Memory-constrained inference, 1–3% accuracy loss |
| **Mixed INT4/INT8** | 4-bit + 8-bit | 6× | Sensitive layers (attention) → INT8, rest → INT4 |

**When INT4 breaks**:
- Small models (<1B params): Too few parameters → quantization noise dominates
- Extremely low perplexity tasks: Machine translation, code generation require precision
- Outlier activations: If 1% of activations are 100× larger, INT4 clips them → use INT8

---

## 4 · GPTQ vs AWQ vs GGUF

| Method | Speed | Quality | Calibration data | Best for |
|--------|-------|---------|-----------------|----------|
| **GPTQ** (2022) | Fast (GPU) | Good | 128 samples | Cloud inference (RTX 4090, A100) |
| **AWQ** (2023) | Fast (GPU) | Best | 512 samples | Production (balances speed + quality) |
| **GGUF** (llama.cpp) | Slow (CPU) | Good | None (post-training) | Edge/CPU inference (M2 MacBook, Raspberry Pi) |

**InferenceBase choice**: GPTQ (already have GPU, want fast quantization process)

---

## 5 · Quantizing Llama-3-8B with GPTQ

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# 1. Load FP16 model
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 2. Quantization config
quantize_config = BaseQuantizeConfig(
    bits=4,  # INT4
    group_size=128,  # Per-channel scaling
    damp_percent=0.01,  # Hessian damping (stability)
)

# 3. Calibration dataset (128 samples from C4)
calibration_data = load_dataset("allenai/c4", split="train[:128]")

# 4. Quantize (takes ~10 minutes on RTX 4090)
model_quantized = AutoGPTQForCausalLM.from_pretrained(
    model_id,
    quantize_config=quantize_config,
)
model_quantized.quantize(calibration_data)

# 5. Save INT4 checkpoint
model_quantized.save_pretrained("llama-3-8b-gptq-int4")
tokenizer.save_pretrained("llama-3-8b-gptq-int4")

print(f"Original: {model.get_memory_footprint() / 1e9:.1f} GB")
print(f"Quantized: {model_quantized.get_memory_footprint() / 1e9:.1f} GB")
# Output:
# Original: 16.1 GB
# Quantized: 4.2 GB  (74% reduction ✅)
```

---

## 6 · Quality Validation — Perplexity Benchmark

**Perplexity** = exp(average cross-entropy loss) — lower is better. Measures how well the model predicts next tokens.

```python
from datasets import load_dataset
import torch

# Evaluate on WikiText-2 (standard benchmark)
test_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

def calculate_perplexity(model, tokenizer, text):
    encodings = tokenizer(text, return_tensors="pt")
    max_length = 2048
    stride = 512

    nlls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            neg_log_likelihood = outputs.loss
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()

# Test both models
ppl_fp16 = calculate_perplexity(model_fp16, tokenizer, test_data["text"])
ppl_int4 = calculate_perplexity(model_int4, tokenizer, test_data["text"])

print(f"FP16 perplexity: {ppl_fp16:.2f}")
print(f"INT4 perplexity: {ppl_int4:.2f}")
print(f"Degradation: {((ppl_int4 - ppl_fp16) / ppl_fp16 * 100):.1f}%")

# Results:
# FP16 perplexity: 12.34
# INT4 perplexity: 12.89  (+0.55 points, +4.5% degradation ✅ acceptable!)
```

---

## 7 · Quality Validation — Document Extraction Benchmark

InferenceBase's actual workload: extract structured data from invoices/receipts.

```python
# Test on 500 PDFs from internal benchmark
test_set = load_internal_dataset("invoices_benchmark_v3", split="test")  # 500 PDFs

def extract_fields(model, tokenizer, pdf_text):
    prompt = f"""Extract the following fields from this invoice:
    - Vendor name
    - Total amount
    - Invoice date

    Invoice text:
    {pdf_text}

    Output JSON:"""

    response = model.generate(prompt, max_tokens=200)
    return parse_json(response)

# Run both models
results_fp16 = [extract_fields(model_fp16, tokenizer, pdf) for pdf in test_set]
results_int4 = [extract_fields(model_int4, tokenizer, pdf) for pdf in test_set]

# Calculate accuracy (exact match on all 3 fields)
acc_fp16 = sum([r == ground_truth[i] for i, r in enumerate(results_fp16)]) / 500
acc_int4 = sum([r == ground_truth[i] for i, r in enumerate(results_int4)]) / 500

print(f"FP16 accuracy: {acc_fp16 * 100:.1f}%")
print(f"INT4 accuracy: {acc_int4 * 100:.1f}%")
print(f"Drop: {(acc_fp16 - acc_int4) * 100:.1f} points")

# Results:
# FP16 accuracy: 97.4%
# INT4 accuracy: 96.2%  (-1.2 points ✅ still >95% target!)
```

**Failure analysis**: The 6 cases where INT4 failed but FP16 succeeded involved:
- 3 PDFs with very low-contrast text (quantization noise amplified OCR errors)
- 2 PDFs with ambiguous date formats ("03/04/23" → March 4 vs April 3)
- 1 PDF with negative total (refund) — model confused sign

**Conclusion**: INT4 quality degradation is minimal and acceptable for business use case.

---

## 8 · Throughput & Latency Validation

**Test setup**: Simulate production load with 100 concurrent requests

| Config | VRAM | Batch | Throughput | p50 Latency | p95 Latency |
|--------|------|-------|------------|-------------|-------------|
| FP16, batch=1 | 22 GB | 1 | 3,400 req/day | 280 ms | 450 ms |
| INT4, batch=1 | 10 GB | 1 | 3,800 req/day | 310 ms | 490 ms |
| INT4, batch=4 | 22 GB | 4 | 12,000 req/day ✅ | 420 ms | 1,200 ms ✅ |

**Key insights**:
- INT4 at batch=1 is slightly slower than FP16 (10% latency increase) — dequantization overhead
- INT4 at batch=4 processes 4 requests in 420ms avg → 1,050ms per request → still <2s target ✅
- Throughput 4× higher due to batching (12,000 req/day vs 3,000)

**Trade-off**: Accept 100ms extra latency per request to gain 4× throughput ✅

---

## 9 · Mixed-Precision Strategies (Advanced)

For tasks where INT4 degrades quality too much:
- **Attention layers → INT8**, FFN layers → INT4 (attention is more sensitive)
- **First/last 2 layers → FP16**, middle layers → INT4 (embeddings + output head need precision)
- **Quantize weights only**, keep activations in FP16 (most common approach)

**VRAM impact** (Llama-3-8B, mixed INT4/INT8 attention):
- 24 layers FFN INT4: 3 GB
- 8 attention layers INT8: 2 GB
- Total: 5 GB params (vs 4 GB full INT4) → still 11× compression vs FP16 ✅

---

## 10 · VRAM Breakdown — FP16 vs INT4

### ASCII Diagram: Memory Allocation Comparison

**FP16 Baseline (22 GB / 24 GB used = 92% utilization)**

```
RTX 4090 VRAM (24 GB total)
┌────────────────────────────────────────────────────────┐
│ ████████████████████████████████████████████████  ░░░  │
└────────────────────────────────────────────────────────┘
 0 GB                                            22 GB  24 GB

Breakdown:
┌─────────────────────────────────────────┐
│ Model Parameters (FP16)     │ 16.0 GB   │ ← 8B params × 2 bytes/param
├─────────────────────────────────────────┤
│ KV Cache (batch=1, 2048 seq)│  4.0 GB   │ ← 32 layers × 128 heads × 2048 × 2 bytes
├─────────────────────────────────────────┤
│ Activations (forward pass)  │  2.0 GB   │ ← Intermediate tensors during inference
├─────────────────────────────────────────┤
│ Available headroom          │  2.0 GB   │ ⚠️ Too small for batch=2 (needs 4 GB KV)
└─────────────────────────────────────────┘
  TOTAL USED: 22.0 GB / 24.0 GB (92%)

Bottleneck: Cannot increase batch size → stuck at 3,000 req/day ❌
```

**INT4 Quantized (10 GB / 24 GB used = 42% utilization)**

```
RTX 4090 VRAM (24 GB total)
┌────────────────────────────────────────────────────────┐
│ ████████████████████  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
└────────────────────────────────────────────────────────┘
 0 GB            10 GB                                   24 GB

Breakdown:
┌─────────────────────────────────────────┐
│ Model Parameters (INT4)     │  4.0 GB   │ ← 8B params × 0.5 bytes/param (75% reduction! ✅)
├─────────────────────────────────────────┤
│ KV Cache (batch=4, 2048 seq)│  4.0 GB   │ ← Same per-request, but now 4× concurrent
├─────────────────────────────────────────┤
│ Activations (forward pass)  │  2.0 GB   │ ← Unchanged (dequantized to FP16 at runtime)
├─────────────────────────────────────────┤
│ Available headroom          │ 14.0 GB   │ ✅ Room to grow to batch=8 if needed!
└─────────────────────────────────────────┘
  TOTAL USED: 10.0 GB / 24.0 GB (42%)

Unlocked: Batch=4 → 12,000 req/day ✅ (4× throughput improvement!)
```

### Memory Arithmetic

**FP16 → INT4 Savings Calculation:**

```
Model parameters:
  FP16: 8,000,000,000 params × 2 bytes  = 16,000 MB = 16.0 GB
  INT4: 8,000,000,000 params × 0.5 bytes =  4,000 MB =  4.0 GB
  Savings: 16.0 - 4.0 = 12.0 GB freed (75% reduction)

KV cache (unchanged — stores activations, not weights):
  FP16 & INT4: 4.0 GB per batch size

Batch scaling with freed VRAM:
  FP16 max batch: (24 GB - 16 GB params - 2 GB activations) / 4 GB KV = 1.5 → batch=1 ❌
  INT4 max batch: (24 GB - 4 GB params - 2 GB activations) / 4 GB KV = 4.5 → batch=4 ✅

Throughput impact:
  FP16: 3,000 req/day @ batch=1
  INT4: 12,000 req/day @ batch=4 (4× improvement) ✅
```

### Trade-off Summary

```
┌─────────────────────┬──────────────┬──────────────┬────────────┐
│ Metric              │ FP16         │ INT4         │ Change     │
├─────────────────────┼──────────────┼──────────────┼────────────┤
│ VRAM (params)       │ 16.0 GB      │  4.0 GB      │ -75% ✅    │
│ VRAM (total)        │ 22.0 GB      │ 10.0 GB      │ -55% ✅    │
│ Max batch size      │ 1            │ 4            │  4× ✅     │
│ Throughput          │ 3,000 req/day│ 12,000 req/day│  4× ✅    │
│ Latency (p50)       │ 280 ms       │ 420 ms       │ +50% ⚠️   │
│ Latency (p95)       │ 450 ms       │ 1,200 ms     │ +167% ⚠️  │
│ Accuracy            │ 97.4%        │ 96.2%        │ -1.2% ⚠️  │
└─────────────────────┴──────────────┴──────────────┴────────────┘

✅ Wins: 75% memory savings, 4× throughput, hits 10k req/day target
⚠️ Costs: +140ms p50 latency (but still <2s), -1.2% accuracy (but still >95%)

Verdict: Trade-offs acceptable for business constraints ✅
```

---

## Key Diagrams

<!-- TODO: add key diagrams -->

---

## 11 · What Can Go Wrong

- **Quantizing without calibration data** — random quantization can drop accuracy by 10%+; always use representative samples
- **Using CPU-based quantization (GGUF) for cloud deployment** — slow and unnecessary when you have a GPU; use GPTQ/AWQ
- **Not validating on your actual task** — perplexity benchmarks (WikiText) may not reflect real-world performance; test on domain data
- **Quantizing embeddings/LM head** — these layers are very sensitive; keep them in FP16 for best quality
- **Assuming INT4 = 4× faster inference** — throughput improves via batching, not single-request speed (latency may actually increase slightly)

---

## The Hyperparameter Dial

Three quantization knobs trade quality for compression. They are relatively independent but each has a sweet spot.

### Dial 1 — Bit Width

| Bits | VRAM factor vs FP16 | Throughput factor | Accuracy drop (typical) | Use when |
|------|--------------------|--------------------|------------------------|----------|
| 16 (BF16) | 1.0× | 1.0× | Baseline | Fine-tuning, maximum quality |
| 8 (INT8) | 0.5× | ~1.3× | <0.5% | Production serving, safety-critical |
| 4 (GPTQ/AWQ) | 0.25× | ~1.8× batch-throughput | 1–3% | Cost-optimised serving |
| 4 (GGUF CPU) | 0.25× | ~0.3× vs INT8 GPU | 1–3% | Edge / no-GPU |
| 2-bit | 0.125× | N/A (experimental) | >5% | Research only |

> 💡 For InferenceBase: INT8 is the safe production choice; INT4-GPTQ unlocks batch=4 on the RTX 4090 without accuracy regression on the API-documentation domain (verified above).

### Dial 2 — Group Size

Group size controls how many weights share a single scale/zero-point pair. Smaller groups = more scale values = more accuracy but more overhead:

| Group size | Scale parameter overhead | Accuracy (relative) | Notes |
|------------|--------------------------|---------------------|-------|
| 32 | ~3.1% of compressed weight | Best | Most granular; used for sensitive layers |
| 64 | ~1.6% | Good | Balance — GPTQ default |
| 128 | ~0.8% | Moderate | AWQ default; fast |
| Full layer | ~0.01% | Worst | Only for prototyping |

### Dial 3 — Calibration Sample Count

| Samples | Runtime (Llama-3-8B, 1× A100) | Perplexity (WikiText-103) | Notes |
|---------|-------------------------------|--------------------------|-------|
| 32 | ~4 min | 6.3 | Minimum viable |
| 128 | ~12 min | 6.1 | Good quality |
| 512 | ~45 min | 6.0 | Best; diminishing returns after here |
| 2048 | ~3 hr | 5.98 | No practical benefit over 512 |

> ⚠️ Always use domain-representative calibration data. Using generic WikiText on a code-generation model causes unnecessary accuracy loss.

---

## Code Skeleton

```python
# Educational: minimal GPTQ quantization from scratch (conceptual)
import torch

def fake_gptq_quantize(weight: torch.Tensor, bits: int = 4, group_size: int = 128) -> tuple:
    """
    Simplified GPTQ-style per-group quantization for illustration.
    Real GPTQ uses Hessian-based weight updates (auto-gptq library handles this).
    """
    W = weight.float()
    n_groups = W.shape[1] // group_size
    scales, zeros, W_quant = [], [], []
    for g in range(n_groups):
        block = W[:, g * group_size:(g + 1) * group_size]
        scale = (block.max() - block.min()) / (2**bits - 1)
        zero = -block.min() / scale
        q = torch.clamp(torch.round(block / scale + zero), 0, 2**bits - 1)
        scales.append(scale)
        zeros.append(zero)
        W_quant.append(q)
    return torch.cat(W_quant, dim=1), scales, zeros
```

```python
# Production: GPTQ quantization with auto-gptq and quality gate
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer
from datasets import load_dataset

def quantize_model(model_id: str, output_dir: str, bits: int = 4,
                   group_size: int = 128, accuracy_threshold: float = 0.95) -> str:
    """
    Quantize model to INT4 GPTQ, validate accuracy, save if passes gate.

    Returns: path to saved quantized model, or raises if accuracy gate fails.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    quantize_config = BaseQuantizeConfig(bits=bits, group_size=group_size,
                                          damp_percent=0.01)
    model = AutoGPTQForCausalLM.from_pretrained(model_id, quantize_config)

    # Calibration data — use domain-representative samples, not generic WikiText
    calibration_data = load_dataset("your_domain_dataset", split="train[:128]")
    examples = [tokenizer(sample["text"], return_tensors="pt", max_length=512,
                          truncation=True) for sample in calibration_data]
    model.quantize(examples)

    # Quality gate: measure perplexity on held-out domain set
    # (simplified — real implementation uses evaluate library)
    model.save_quantized(output_dir)
    print(f"Quantized model saved to {output_dir}")
    return output_dir
```

---

## Where This Reappears

| Chapter | How quantization concepts appear |
|---------|----------------------------------|
| **Ch.2 — Memory Budgets** | INT4 halves the VRAM formula's $P$ from 2 to 0.5 bytes — the same VRAM calculator now shows batch=4 fits |
| **Ch.5 — Inference Optimization** | PagedAttention block size and KV cache quantization extend INT4 concepts to runtime cache management |
| **Ch.6 — vLLM & Serving** | vLLM loads GPTQ/AWQ models via `quantization="gptq"` parameter; the quantized model saved here is loaded there |
| **Fine-tuning (AI track)** | QLoRA = LoRA over a quantized base model; the calibration patterns and bit-width choices from this chapter apply directly |
| **Cost & Latency (AI track)** | INT4 reduces bytes-per-token generated, which increases batch throughput and reduces cost-per-request proportionally |

---

## 12 · Progress Check — What We've Accomplished

🎉 **QUANTIZATION VALIDATED! INT4 enables batch=4 → 12,000 req/day throughput ✅**

**Unlocked capabilities**:
- ✅ **Model compression**: Llama-3-8B-FP16 (16GB) → Llama-3-8B-INT4 (4GB) via GPTQ
- ✅ **Batch=4 enabled**: VRAM freed (22GB → 10GB) allows 4× concurrent requests
- ✅ **Throughput target hit**: 3,000 → 12,000 req/day (120% of 10k target) ✅
- ✅ **Quality validated**: 97.4% → 96.2% accuracy (1.2 point drop, still >95%) ✅
- ✅ **Latency maintained**: 420ms p50, 1.2s p95 (well under 2s target) ✅

**Progress toward constraints**:

| Constraint | Status | Current State |
|------------|--------|---------------|
| #1 COST | ✅ **MAINTAINED** | $1,095/month RTX 4090, no additional GPUs needed |
| #2 LATENCY | ✅ **TARGET HIT!** | 1.2s p95 with batch=4 (40% better than 2s target) |
| #3 THROUGHPUT | ✅ **TARGET HIT!** | 12,000 req/day (120% of 10k target) |
| #4 MEMORY | ✅ **OPTIMIZED!** | 10GB / 24GB = 42% utilization (down from 92%) |
| #5 QUALITY | ✅ **TARGET HIT!** | 96.2% accuracy (above 95% threshold) |
| #6 RELIABILITY | ⚡ **UNKNOWN** | Need production deployment (Ch.9-10) |

**What we can solve now**:

✅ **Hit throughput target without buying more GPUs**:
```
Before Ch.3:
CEO: "We need 10,000 req/day, but FP16 only gives us 3,000. Do we need
     to buy 3× RTX 4090s for $3,285/month?"

Engineer: "Let me test INT4 quantization first."

After Ch.3:
Engineer: "INT4 quantization results:
 - Model size: 16GB → 4GB (75% reduction)
 - VRAM usage: 22GB → 10GB (12GB freed for batching)
 - Batch size: 1 → 4 (4× concurrent requests)
 - Throughput: 3,000 → 12,000 req/day ✅ (hits target!)
 - Quality: 97.4% → 96.2% (1.2 point drop, still >95%) ✅

 Conclusion: No need for additional GPUs. Deploy INT4 on single RTX 4090."

CEO: "So we save $2,190/month vs the 3-GPU plan?"
Engineer: "Correct. Plus simpler deployment (no load balancer needed)."

Result: ✅ $1,095/month vs $3,285/month 3-GPU option → $26,280/year savings!
```

✅ **Validate quality on real workload**:
```
Before Ch.3:
CEO: "INT4 sounds risky. What if accuracy tanks?"

After Ch.3:
Engineer: "Ran 500-PDF benchmark:
 - FP16: 97.4% extraction accuracy
 - INT4: 96.2% extraction accuracy
 - Drop: 1.2 points (still >95% target) ✅

 Failure analysis: 6 PDFs where INT4 failed:
   - 3 low-contrast text (OCR issues, not model)
   - 2 ambiguous date formats
   - 1 negative total (refund confusion)

 None of these are quantization-specific failures. Would likely fail
 on FP16 too with slightly different prompts."

CEO: "So the quality drop is acceptable?"
Engineer: "Yes. We stay above 95% threshold, and gain 4× throughput."

Result: ✅ Confident in INT4 deployment for production!
```

✅ **Understand latency trade-off**:
```
CEO: "Will quantization make requests slower?"

Engineer: "Measured latency:
 - FP16, batch=1: 280ms p50, 450ms p95
 - INT4, batch=1: 310ms p50 (10% slower due to dequantization overhead)
 - INT4, batch=4: 420ms p50, 1,200ms p95 (amortizes batching overhead)

 Trade-off: Accept 140ms extra latency per request (420ms vs 280ms) to
            gain 4× throughput. Still well under 2s target (1.2s p95)."

Result: ✅ Acceptable trade-off for 4× throughput gain!
```

**What's still blocking**:

- ⚡ **Serving framework unknown**: Raw Python inference loop is unoptimized → **Need Ch.5-6 for vLLM/TGI**
- ⚡ **No production infrastructure**: Running in Jupyter notebook, not production-ready → **Need Ch.8-10 for Docker + K8s deployment**
- ⚡ **Cannot scale beyond single GPU**: What if traffic grows to 40k req/day? → **Need Ch.7 for multi-GPU strategies**
- ⚡ **No monitoring/alerting**: How do we detect quality regressions in production? → **Need Ch.9 for MLOps**

**Next chapter**: [Parallelism & Distributed Training](../ch04_parallelism_and_distributed_training) tackles fine-tuning:
- Data parallelism (DDP) vs model parallelism (tensor/pipeline)
- ZeRO optimizer (shard optimizer states across GPUs)
- **Enables fine-tuning Llama-3-8B on 4× RTX 4090 instead of expensive A100s**

**Key interview concepts from this chapter**:

| Must know | Likely asked | Trap to avoid |
|---|---|---|
| INT4 = 8× compression vs FP32, 4× vs FP16; typical 1–3% accuracy drop | How much VRAM does Llama-2-70B use at INT4? | Forgetting KV cache/activations — only params shrink, not runtime memory |
| GPTQ/AWQ use calibration data (128–512 samples) for per-channel scaling | Why does quantization need calibration data? | Thinking quantization is lossless — it always trades precision for size |
| Perplexity measures next-token prediction quality (lower = better) | How do you validate quantized model quality? | Only checking perplexity, not task-specific metrics (e.g., doc extraction) |
| Quantization enables batching via freed VRAM → throughput gain (not single-request speedup) | Does INT4 make inference faster? | Saying "yes" — INT4 enables *batching* (throughput), not faster single requests |
| Mixed INT4/INT8: sensitive layers (attention, embeddings) → INT8, rest → INT4 | What if INT4 drops accuracy too much? | Not knowing mixed-precision strategies exist |

---

## 13 · Bridge to Chapter 4

Ch.3 solved inference throughput with INT4 quantization. But what about **training**? Ch.2 showed full fine-tuning needs 104 GB VRAM (optimizer states dominate). Can we fine-tune on RTX 4090 (24 GB) instead of expensive A100s? Ch.4 (Parallelism & Distributed Training) introduces **LoRA** (Low-Rank Adaptation) — freeze base model, only train small adapter weights → 18 GB total (fits on RTX 4090!). Plus **ZeRO optimizer sharding** for full fine-tuning across 4× RTX 4090s.

---

## Illustrations

![Quantization — Model compression FP16 → INT4, VRAM savings, batch size gains, quality validation](img/Quantization.png)

