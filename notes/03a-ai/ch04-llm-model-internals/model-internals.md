# Ch.4 · LLM Model Internals — Parameters, Memory, Quantization

> **The story.** In May 2020, OpenAI released GPT-3 and watched in astonishment as it performed few-shot learning—a capability they never explicitly trained. By 2021, the question haunting AI labs wasn't "can we build bigger models?" but "do we understand what we built?" Anthropic researchers published *Toy Models of Superposition* (2022), revealing that models compress hundreds of features into dozens of neurons—like storing a library in a shoebox. The race was on: reverse-engineer the black box before it gets too powerful to control.
>
> **Why this matters:** You can prompt the smartest model in the world, but if you don't understand parameter counts, VRAM constraints, and quantization trade-offs, you'll ship applications that crash on deployment or cost 10× more than necessary. This chapter gives you the engineering vocabulary to make informed decisions about model selection, inference optimization, and deployment architecture.

---

## 6 · Model Size & Mixture of Experts (MoE)
**Quick Estimate:**
> **Model size → VRAM needed (fp16):**
> - 7B model: ~14 GB VRAM minimum
> - 13B model: ~26 GB
> - 70B model: ~140 GB (needs 2× A100 80GB)
> - GPT-4: estimated 1.8T parameters (but with MoE magic — see below)
>
> **Rule of thumb:** Parameter count (in billions) × 2 bytes = VRAM in GB (for fp16)

### Mixture of Experts (MoE)

**The intuition:** Standard models activate ALL parameters for every token (everyone does all the work). MoE models have **specialists** — only the relevant experts activate for each token (delegation).

**How it works:**
1. Replace dense FFN layers with N "expert" sub-networks (e.g., 8 experts)
2. Add a lightweight **router** that picks the top k experts per token (e.g., top-2)
3. Only those k experts compute — the rest stay idle

**Visual metaphor:** Think of a hospital:
- **Standard model:** Every doctor sees every patient (exhausting, slow)
- **MoE model:** A triage nurse (router) sends patients to the right specialists — cardiologist, neurologist, orthopedist, etc. (efficient, specialized)

The router learns during training: "Python code? Send to Expert 0 (code specialist). Philosophy question? Send to Expert 2 (reasoning specialist)."

**Routing Example: "Python" vs "philosophy"**

Suppose an MoE layer has 8 experts and activates top-2 per token. Watch who gets called:

**Token: "Python"**
```
Router scores (like confidence levels):
✓ Expert 0 (code): 42% ← Selected! High confidence
✓ Expert 1 (syntax): 31% ← Selected! Medium confidence
 Expert 2 (philosophy): 8%
 Expert 3 (literature): 5%
 Expert 4 (math): 4%
 (others): 10%

Result: Blend 42% of Expert 0's output + 31% of Expert 1's output
```

**Token: "philosophy"**
```
Router scores:
 Expert 0 (code): 2%
 Expert 1 (syntax): 1%
✓ Expert 2 (philosophy): 53% ← Selected! Obvious choice
✓ Expert 3 (literature): 24% ← Selected! Related field
 Expert 4 (math): 7%
 (others): 13%

Result: Blend 53% of Expert 2's output + 24% of Expert 3's output
```

**Key insight:** Nobody programmed Expert 0 to handle code or Expert 2 to handle philosophy. **The model learned this routing pattern** during training by trying to predict the next token accurately. Specialization emerges naturally.

**Why MoE matters:**
- **Scale without pain:** GPT-4's 1.8T parameters → only ~200-400B active per token (~10-20% of total). You get a 1.8T model's capacity at roughly a 200B model's compute cost per inference
- **Specialization for free:** Different experts naturally specialize during training — some light up for code, others for natural language, others for structured data
- **Training efficiency:** Model capacity scales with total experts; compute cost scales with active experts per token
**Quick Estimate:**
> **MoE memory trap:** Mixtral-8×7B has 8 experts × 7B each = **~93 GB VRAM needed** (fp16) to load all experts, but only **~12.9B parameters active** per token (so inference feels like a 13B model).
>
> **Translation:** You pay full memory cost, but only partial compute cost. Still need the huge GPU, but inference is much faster than a dense 93B model.
**Quick Estimate:**
> **MoE memory trap:** Mixtral-8×7B has 8 experts × 7B each = **~93 GB VRAM needed** (fp16) to load all experts, but only **~12.9B parameters active** per token (so inference feels like a 13B model).
>
> **Translation:** You pay full memory cost, but only partial compute cost. Still need the huge GPU, but inference is much faster than a dense 93B model.

**Inference cost factors:** Cost scales with (parameter count) × (context length) × (batch size). A 70B model at 128k context costs **~50× more** to run than a 7B model at 4k context. This is why production systems use smaller models wherever possible.

> **Model selection strategy:**
> **Cheap experiments:** GPT-4o-mini for factual retrieval, structured output, testing (fast, low cost)
> **Complex reasoning:** GPT-4o when accuracy matters more than cost
> **Self-hosted:** LoRA-adapted 7B can match GPT-4o-mini quality at ~$0.0003/1k tokens (6× cheaper)

---

## 6A · Token Flow & Parameters

You've seen the transformer block (§2A), attention mechanisms (§2A), and inference loops (§3A). This section answers: **Where are the 7 billion parameters in a 7B model? What happens when a token enters the model?**

### The Data Flow — Token to Logits

```
Input: Token ID (integer) e.g., 5812
 ↓
1. Token Embedding: 5812 → [0.21, -0.45, 0.67, ..., 0.12] (4096-dim vector)
 ↓
2. Positional Encoding: Add position information to embedding
 ↓
3. Transformer Blocks (L layers): Attention + FFN + residuals + layer norm (repeated L times)
 ↓
4. Final Layer Norm: Normalize final hidden state
 ↓
5. Output Projection (LM Head): (4096-dim) → (vocab_size-dim) logits
 ↓
Output: Logits over vocabulary [2.3, -1.1, 4.5, ..., -0.8] (50,000-dim for 50k vocab)
```

Each stage has **learnable parameters**. Their sum is the model's parameter count.

### Parameter Breakdown — Where the 7B Lives

**Think of a 7B model as a $7 billion budget** distributed across different "departments." Here's where the money goes:

| Component | LLaMA 7B Count | Percentage | Budget Analogy |
|-----------|----------------|------------|----------------|
| **Token Embeddings** (input dictionary) | 131M | 2% | Employee directory — just a lookup table |
| **Positional Embeddings** | 0 | 0% | (Built into the system, no extra cost) |
| **Per Transformer Block:** | | | |
| • Attention ($W_Q, W_K, W_V, W_O$) | ~67M | 1% | Search & retrieve operations |
| • Feed-Forward Networks (FFN) | ~90M | 1.4% | **Heavy computation department** |
| • Layer norms | negligible | ~0% | Quality control checkpoints |
| **Total per block** | **~158M** | 2.4% | One complete processing unit |
| **All 32 blocks** | **5,050M** | **76%** | **Most of your budget** |
| **Output LM head** (prediction layer) | 131M | 2% | Final decision-making board |
| **Total** | **6.7B** | 100% | Full model budget |
**Quick Estimate:**
> **Rule of thumb:** ~75% of parameters live in the FFN layers. If you see "7B parameters," think "~5B are in the FFN layers, doing the heavy lifting."
>
> **Why this matters:** When optimizing, focus on FFN layers first — they're where the compute actually happens.

**Key insights:**
- **The FFN layers are the heavy lifters:** ~57% of your budget goes here — if you optimize anywhere, optimize FFN operations
- **Embeddings are just lookup tables:** Input (131M) + output (131M) = 2% each. They're conceptually important but computationally cheap
- **Attention gets the glory, FFN does the work:** Attention is only ~20% of parameters. The real compute happens in those massive feed-forward layers

### The Embedding Layer — Token ID to Vector

**The dictionary lookup:** Every token has a learned vector. Think of it as **looking up a word in a dictionary** — instant, not calculated.

```python
# Not a formula — just a lookup!
token_id = 5812
embedding = embedding_matrix[token_id] # Grab row 5812 → (4096 numbers)
```
**Quick Estimate:**
> **Cost:** Essentially free (~0.01% of total inference time)
> **Why:** Just grabbing a row from a table, not doing matrix math
>
> **Size:** For LLaMA 7B with 32k vocab and 4096 dimensions:
> 32,000 tokens × 4,096 numbers = 131 million parameters (~2% of model)

**What the embeddings encode:** During training, the model learns to **place similar words near each other** in this 4096-dimensional space. It's not programmed to do this — it discovers it naturally while learning to predict the next token.

**Clustering examples:**
- "King" and "queen" → close together (royalty cluster)
- "Jump," "leap," "hop" → close together (motion cluster)
- "1," "2," "3" → arranged by magnitude (number line)
- Python code tokens → cluster separately from natural language

**Visual intuition:** If you squash those 4096 dimensions down to 2D (like looking at a map from above), you'd see **neighborhoods**: nouns in one area, verbs in another, code in a corner, numbers forming a gradient.

Nobody told the model to organize this way. It emerged from billions of "predict the next token" exercises.

> **Checkpoint:** Your 7B budget is mostly spent on FFN layers (~75% total across all blocks). Attention gets ~20%, embeddings are ~4%. Token embeddings are just a lookup table — no heavy math, just "find token 5812, grab its vector." Similar words cluster together automatically during training, not because anyone told them to.

---

## 6B · VRAM & Memory

### VRAM Usage — Three Buckets

**Think of VRAM as your apartment budget** — you need space for:

```
Total VRAM = Furniture (weights) + Workspace (activations) + Storage (KV cache)
```

#### 1. Model Weights (The Furniture)

Your model's parameters are like **furniture that stays in place**. Size depends on precision:

| Precision | Storage per param | LLaMA 7B | LLaMA 70B | Analogy |
|-----------|-------------------|----------|-----------|----------|
| **fp32** (full precision) | 4 bytes | 26.8 GB | 280 GB | Full-size furniture |
| **fp16 / bf16** (half precision) | 2 bytes | 13.4 GB | 140 GB | **Compact furniture** (standard) |
| **int8** (quantized) | 1 byte | 6.7 GB | 70 GB | **IKEA flat-pack** — 50% less space |
| **int4** (aggressive quantization) | 0.5 bytes | 3.35 GB | 35 GB | **Minimalist studio** — 75% savings |
**Quick Estimate:**
> **Shortcut:** For a 7B model, just remember:
> - **fp16 = ~14 GB** (double the parameter count in billions)
> - **int8 = half that** (~7 GB)
> - **int4 = half again** (~3.5 GB)
>
> **Quality trade-off:** int8 is nearly free (<1% accuracy loss). int4 is riskier on complex reasoning but fine for most tasks.

**Production default:** fp16 for quality-critical work. int8 for memory-constrained deployments. int4 when you're desperate for space.

#### 2. Activations (The Workspace)

**Think of activations as your kitchen counter** while cooking — you need space to prep, but you clean up after.

During a forward pass, the model creates temporary work areas:
- Attention scores: scratchpad for comparing tokens
- Intermediate FFN outputs: calculation workspace
- Residual buffers, softmax outputs: temporary storage
**Quick Estimate:**
> **For typical single-request inference:**
> - 7B model: **~1 GB** workspace
> - 70B model: **~3-5 GB** workspace
>
> **The workspace clears after each request** — like washing dishes after cooking. During training, you must keep everything (no cleaning until the end) — that's why training needs 10× more memory.

**Visual metaphor:** If model weights are your furniture (permanent), activations are the mess you make while working (temporary).

#### 3. KV Cache (The Storage Unit)

**Think of KV cache as storage boxes** — one per conversation. Each box holds the "memory" of what was said.

For LLaMA 7B, fp16, seq_len=2048: **~1 GB per conversation**.

**Total VRAM Budget (single conversation, LLaMA 7B, fp16):**

```
Furniture (weights): 13.4 GB | Fixed cost — same for all conversations
Workspace (activations): 1.0 GB | Temporary — cleans up after each turn
Storage (KV cache): 1.0 GB | One box per conversation
--------------------------------------------------------
Total apartment: 15.4 GB → fits in RTX 4090 (24 GB)
```

**Serving 16 simultaneous conversations:**

```
Furniture (weights): 13.4 GB | Still just one set — everyone shares
Workspace (activations): 1.0 GB | Still temporary — still cleans up
Storage (KV cache): 16.0 GB | 16 boxes × 1 GB each = 16 GB!
--------------------------------------------------------
Total apartment: 45.4 GB → needs 2× A40 (48 GB) or 1× A100 (80 GB)
```
**Quick Estimate:**
> **Storage grows with conversations:** Each new conversation adds **~1 GB** (7B model) or **~3-5 GB** (70B model).
>
> **Why servers hit limits fast:** At 100 conversations × 1 GB = 100 GB just for storage boxes — more than the model weights! **KV cache is the throughput killer**, not the model size.

> **Warning — The scaling trap:** Your apartment (GPU memory) is fixed, but storage boxes (KV cache) multiply with every conversation. That's why batch size plateaus fast.

> **Checkpoint:** Your VRAM apartment has three categories: furniture (model weights, ~14 GB for 7B), workspace (activations, ~1 GB, temporary), and storage boxes (KV cache, ~1 GB per conversation). Furniture is fixed. Workspace cleans itself. **Storage boxes multiply with every conversation** — that's why serving 100 users needs 100 GB just for cache, making KV cache the bottleneck, not model size.

---

## 6C · Optimization Techniques

### Quantization — Trading Precision for Memory

**Visual metaphor:** Think of quantization as **rounding prices to the nearest nickel**.

- **fp16/fp32:** Every price is exact to the penny ($1.47, $2.83, $0.56)
- **int8:** Round to the nearest nickel ($1.45, $2.85, $0.55)
- **int4:** Round to the nearest quarter ($1.50, $2.75, $0.50)

You lose some precision, but for most purchases (model operations), **the difference doesn't matter**.

**How it works (intuition, not math):**

1. **Find the range** of all your weights (e.g., smallest = -1.2, largest = 0.8)
2. **Divide that range into buckets** (255 buckets for int8, 15 buckets for int4)
3. **Each weight gets assigned to its nearest bucket**
4. **When computing, convert back** to the original range

**Example: Rounding 0.567**

Imagine your weights range from -1.2 to 0.8:
- **fp16:** 0.567 (exact)
- **int8:** 0.564 (rounded to nearest bucket — error = 0.003)
- **int4:** 0.55 (rounded to coarser bucket — error = 0.017)
**Quick Estimate:**
> **Accuracy impact by precision:**
> - **int8:** Loses **<1%** accuracy on most tasks (you won't notice)
> - **int4:** Loses **2-5%** on complex reasoning (noticeable on math/code, fine for chat)
> - **int3 or lower:** Significant degradation (rarely used)
>
> **Memory savings:**
> - **int8 = 50% savings** (7B model: 14 GB → 7 GB)
> - **int4 = 75% savings** (7B model: 14 GB → 3.5 GB)

**When to quantize:**
- **Always use int8** for production (free lunch — minimal quality loss)
- **Use int4** when you're memory-constrained and can tolerate slight quality dips
- **Never go below int4** unless you're doing research on extreme compression

**Quantization strategies:**
- **Weight-only (GPTQ, AWQ):** Compress the furniture, but work in full precision — saves space, modest speed gain
- **Full quantization (int8):** Compress weights AND workspace — saves space AND speeds up math (int8 ops are 4× faster)
- **Mixed-precision (QLoRA):** Base model in int4 (tiny), adapters in fp16 (precise) — lets you fine-tune 70B on a gaming PC
**Quick Estimate:**
> **Choosing your precision level:**
> - **Need maximum quality?** fp16 (standard production)
> - **Need to fit in memory?** int8 first (nearly free quality-wise)
> - **Desperate for space?** int4 (acceptable for most tasks, watch out for complex reasoning)
> - **Benchmarks:** int8 loses <1% accuracy, int4 loses 2-5% on math/code

### Gradient Flow & Training Memory (Optional Preview)

> **Optional — skip if focused on inference only.** Covered fully in [03b-agentic-ai Ch.5](../../03b-agentic-ai/ch05-fine-tuning/fine-tuning.md).

Training requires **backward passes** to compute gradients. **Memory cost explodes** because you can't throw anything away until the end.
**Quick Estimate:**
> **Training memory = 4-6× inference memory**
>
> **The apartment analogy:**
> - **Inference:** You clean the workspace after each meal (activations freed immediately)
> - **Training:** You leave ALL dishes, pots, cutting boards out until the end (keep activations for backprop)
> - **Plus:** You need a second workspace for gradients + optimizer notes (momentum, variance)
>
> **LLaMA 7B inference:** ~15 GB
> **LLaMA 7B training:** ~157 GB (needs 2× A100 80GB)
>
> **Why PEFT matters:** Methods like LoRA freeze the base model (no gradients needed there) and only train 0.1-1% of parameters — drops 157 GB → ~20 GB.

### Flash Attention — The Memory Optimization

**The problem:** Standard attention creates a massive scratchpad — for an 8k-token sequence, that's 8,000 × 8,000 = 64 million numbers **per attention head**. With 32 heads, you're storing 2 billion numbers just to compute attention scores.

**The insight:** You don't need to write all that to slow memory (VRAM). You can **compute attention in chunks using fast on-chip memory (SRAM)** and never materialize the full matrix.
**Quick Estimate:**
> **Memory reduction:** O(n²) → O(n) for attention scores
> **Translation:** 8k context on standard attention = ~4 GB just for attention
> **Flash Attention:** Same context = ~64 MB (60× reduction)
>
> **Speed:** 2-4× faster on sequences >2k tokens (less memory traffic)
> **Quality:** Bit-for-bit identical to standard attention (no approximation)

**The apartment analogy:**
- **Standard attention:** Spread all your paperwork across the floor (slow to access, takes lots of space)
- **Flash Attention:** Work at your desk in small batches, file immediately (fast, minimal space)

**Why it matters:** Flash Attention 2 is now the default everywhere (PyTorch, Hugging Face, vLLM, TensorRT). If you see "128k context window," Flash Attention made it possible.

### Model Architecture Comparison (VRAM Perspective)

| Model | Parameters | $d_\text{model}$ | Layers | VRAM (fp16) | VRAM (int8) | VRAM (int4) |
|-------|------------|---------------------|--------|-------------|-------------|-------------|
| **GPT-2** | 1.5B | 1600 | 48 | 3 GB | 1.5 GB | 0.75 GB |
| **LLaMA 2 7B** | 6.7B | 4096 | 32 | 13.4 GB | 6.7 GB | 3.4 GB |
| **LLaMA 2 13B** | 13B | 5120 | 40 | 26 GB | 13 GB | 6.5 GB |
| **LLaMA 2 70B** | 70B | 8192 | 80 | 140 GB | 70 GB | 35 GB |
| **GPT-4 (est.)** | 1.8T (MoE) | ~16384 | ~120 | ~3.6 TB | — | — |
**Quick Estimate:**
> **Hardware requirements by model size:**
> - **7B model:** One gaming GPU (RTX 4090, 24 GB) — int4 or int8
> - **13B model:** One datacenter GPU (A40, 48 GB) or two gaming GPUs — int8
> - **70B model:** 4× gaming GPUs (int4) or 2× datacenter GPUs (A100, 80 GB each) — int8
> - **Beyond 70B:** Multi-node datacenter setups
>
> **Why 70B is the "prosumer limit":** 70B in int4 = 35 GB, fits on 4× RTX 4090s (~$6k setup). Anything bigger needs enterprise hardware.

**The scaling insight:** Model size doubles, VRAM doubles, cost doubles, inference latency roughly doubles. That's why production systems use the **smallest model that solves the problem**.

### Visualization: Parameter Distribution

![Breakdown of 7B parameters by component (embeddings, attention, FFN, output head) and VRAM usage split (weights, activations, KV cache)](img/model-internals-breakdown.png)

**Reading the diagram:**
- **Left (pie chart):** Parameter count by component — FFN dominates (57%), attention is ~20%
- **Right (bar chart):** VRAM usage during inference — model weights, activations, KV cache
- **Bottom:** Precision comparison (fp16 vs int8 vs int4) for VRAM footprint

---

## 7 · Key Distinctions Every Engineer Gets Asked

| Pair | Distinction |
|---|---|
| **Base model vs instruct/chat model** | Base: raw next-token predictor. Instruct: SFT+RLHF applied — follows instructions. Always use instruct for applications. |
| **Parameters vs context window** | Parameters = learned knowledge. Context window = working memory for one inference call. |
| **Temperature vs top-p** | Temperature rescales the whole distribution. Top-p truncates it. Use both. |
| **Q, K, V (Query, Key, Value)** | Q = "what am I looking for?", K = "what do I offer?", V = "what information do I carry?" Attention is a lookup: Q matches K, retrieves V. |
| **Multi-head attention heads (what they specialize in)** | Each head learns different patterns: syntactic structure (subject-verb), positional proximity (nearby tokens), semantic relationships (topic coherence). Specialization emerges from training, not designed. |
| **Bidirectional vs causal attention** | Bidirectional (BERT): token $i$ sees all tokens — ideal for understanding tasks. Causal (GPT): token $i$ sees only tokens $\leq i$ — required for autoregressive generation. |
| **Encoder-only vs decoder-only vs encoder-decoder** | Encoder: bidirectional, best for embeddings/retrieval, cannot generate. Decoder: causal, generates text, weaker embeddings. Encoder-decoder: bidirectional encoder + causal decoder + cross-attention, best for seq2seq (translation). |
| **Prefill vs decode (inference phases)** | Prefill: process entire prompt in parallel, $O(n^2)$ attention, populate KV cache. Decode: generate one token at a time, $O(n)$ attention with cached K/V, FFN dominates compute. |
| **KV cache vs recomputation** | KV cache: store keys/values from prior tokens, reuse at each decode step — 10–20× speedup. Recomputation: process entire sequence every step — unusable for production. |
| **Flash Attention vs standard attention** | Standard: materializes full $(n \times n)$ attention matrix in VRAM — $O(n^2)$ memory. Flash Attention: block-wise computation in SRAM — $O(n)$ memory, 2–4× faster on long contexts. |
| **Weight-only vs activation quantization** | Weight-only (GPTQ, AWQ): quantize weights to int8/int4, compute in fp16 — reduces memory, marginal speedup. Activation quantization: quantize weights AND activations — reduces memory and speeds up matmuls (4× faster int8 ops). |
| **Learned vs sinusoidal vs RoPE (positional encoding)** | Learned (BERT): lookup table, no extrapolation beyond trained length. Sinusoidal (original Transformer): fixed formula, poor extrapolation. RoPE (LLaMA, GPT-4): rotation-based, excellent extrapolation — industry standard. |
| **RLHF vs DPO** | RLHF trains a separate reward model; DPO doesn't. DPO is simpler and now standard. |
| **ORM vs PRM** | ORM scores the final answer — cheap but sparse signal. PRM scores each reasoning step — expensive but precise. PRMs power math-focused reasoning models. |
| **Tokens vs words** | Tokens are model-native; words are human-native. 1 word ≈ 1.3 tokens on average for English prose. |
| **Hallucination vs confabulation** | Hallucination: factually wrong output. Confabulation: a fluent-sounding fabrication of a plausible but non-existent fact (citation, statistic, API name). Same mechanism, different vocabulary. |
| **Scaling laws (Kaplan) vs Chinchilla** | Kaplan (2020): scale parameters more than data for fixed compute. Chinchilla (2022): scale both equally. Chinchilla corrected the field — the Gopher-era giants were systematically undertrained. |
| **Standard LLM vs reasoning model (o1/R1)** | Standard: one forward pass, fast, cheap. Reasoning: long CoT trace, RLVR-trained, slower and more expensive but dramatically better on verifiable tasks (math, code). Use reasoning models when the task has a correct answer you can check. |
| **LoRA vs prefix tuning** | LoRA: weight matrices, merges at inference, zero overhead. Prefix tuning: KV cache, permanent overhead per request. |
| **fp16 vs int8 vs int4 (precision)** | fp16: 2 bytes/param, full quality. int8: 1 byte/param, <1% quality loss, 2× memory savings. int4: 0.5 bytes/param, 2–5% quality loss on reasoning, 4× memory savings. |
| **Continuous batching vs static batching** | Static: wait for all requests to finish before starting next batch — wastes GPU cycles. Continuous: remove finished requests, add new ones immediately — 2–10× throughput increase. |

---

## Bridge to Next Chapter

**You now understand:**
- **Where your budget goes:** ~75% in FFN layers (the heavy lifters), ~20% in attention (the coordinators), ~4% in embeddings (the dictionaries)
- **Your VRAM apartment:** Furniture (weights, fixed), workspace (activations, temporary), storage boxes (KV cache, multiplies with conversations)
- **Precision trade-offs:** int8 is nearly free (50% savings, <1% quality loss). int4 is aggressive (75% savings, 2-5% quality hit on reasoning)
- **The key technical distinctions** interviewers expect (base vs instruct, parameters vs context, quantization strategies)

**The intuition:** Model size determines how much furniture you need. Batch size determines how many storage boxes you're juggling. Quantization is about rounding to the nearest nickel instead of tracking pennies.

**Next:** [Ch.5 · Prompt Engineering](../ch05-prompt-engineering/prompt-engineering.md) — zero-shot, few-shot, chain-of-thought, and structured output. Now that you know what happens inside the model during inference, you'll learn how to control its behavior through prompt design.
