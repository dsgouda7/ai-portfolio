# LLM Training Pipeline

> **The story.** In June 2020, OpenAI released GPT-3—a model that could write poetry, code, and essays with stunning fluency—but couldn't reliably answer "What's 2+2?" if you asked it directly. The model completed text beautifully but refused to be an assistant. Eighteen months later, *InstructGPT* (January 2022) solved the puzzle with RLHF, teaching the same architecture to follow instructions by learning from human preferences. By November 2022, *ChatGPT* brought this alignment breakthrough to millions, proving that the final 1% of training—learning what humans actually want—matters as much as the first 99%.
>
> **How raw text predictors become instruction-following assistants.** This chapter explains the three-stage training pipeline (pretraining, supervised fine-tuning, alignment) that transforms a next-token predictor trained on raw internet text into a helpful, harmless assistant like GPT-4 or Claude. You'll understand why models from different companies exhibit different personalities despite identical architectures, how RLHF/DPO shapes model behavior, and why certain capabilities only emerge at specific scale thresholds. Includes optional PEFT preview (LoRA, prefix tuning) for domain adaptation without full retraining.

**What You'll Learn:**
- The three training stages: pretraining, supervised fine-tuning, RLHF/DPO
- Why models from different companies have different personalities
- How alignment (RLHF/DPO) shapes model behavior
- PEFT techniques (LoRA, prefix tuning) for domain adaptation without full retraining

---

## Common Misconceptions

**Before you start:** These misconceptions poison your first three months of understanding LLM training. Correct them now.

### Misconception 1: "Training is just running backprop on text"

**Why it's seductive:** People see loss curves decreasing and think "it's learning like any neural network."

**The truth:** There are three completely different training regimes with different objectives, data sources, costs, and outcomes. Pretraining ($5M-$100M) learns language from raw internet text. SFT ($50k-$200k in annotation) teaches instruction format from human-written examples. RLHF ($500k-$5M in annotation) learns human preferences from pairwise comparisons. Conflating them makes you think alignment is cheap (it's not) or that more pretraining data solves sycophancy (it doesn't).

*"Pretraining builds the engine. SFT installs the steering wheel. RLHF teaches it where humans want to go."*

### Misconception 2: "RLHF makes the model truthful"

**Why it's seductive:** Models refuse harmful requests and admit uncertainty after RLHF, so it seems like they learned honesty.

**The truth:** RLHF optimizes for human **approval**, not truth. If annotators prefer confident-sounding answers, the model learns confidence. If annotators prefer agreeable responses, the model learns to agree with user framing even when wrong. This creates **sycophancy**: push back on a correct answer and the model will change it. Constitutional AI and adversarial annotation try to fix this, but the core problem remains: you're training the model to make humans happy, not to be right.

*"RLHF teaches helpfulness, not honesty. The two are not the same."*

### Misconception 3: "You can just RLHF any model to make it useful"

**Why it's seductive:** RLHF seems like the magic step that creates ChatGPT, so why not skip pretraining?

**The truth:** RLHF refines existing capabilities; it doesn't create them. A randomly initialized model has no knowledge to align. A pretrained model with 2B parameters can't develop complex reasoning through RLHF. You need the pretraining foundation ($5M-$100M) before alignment ($500k-$5M) has anything to work with. Companies don't skip pretraining because it's **95% of the model's intelligence**. RLHF is the final 5% that makes it usable.

*"You can't align a mind that doesn't exist. Pretraining is the prerequisite."*

### Misconception 4: "GPT-4 and Claude are different architectures"

**Why it's seductive:** They behave so differently—GPT-4 is concise and confident, Claude is verbose and cautious—that they must be built differently.

**The truth:** Both are transformer decoders trained with similar three-stage pipelines. The personality difference comes from **annotation guidelines during RLHF**. OpenAI annotators preferred brief, professional responses. Anthropic annotators preferred thoughtful, nuanced responses. Same architecture. Different human preferences encoded during alignment.

*"Model personality is written by annotators, not architects."*

### Misconception 5: "Emergent abilities are mysterious magic"

**Why it's seductive:** Capabilities like chain-of-thought reasoning appear suddenly at ~100B parameters without explicit training.

**The truth:** "Emergent" means the capability exists latently in the training data but requires sufficient model capacity to compress and reconstruct it. Small models see the same reasoning examples but can't allocate enough parameters to represent the patterns. At 100B parameters, the model finally has enough capacity. Not magic—**capacity threshold**.

*"Emergence is when the model gets large enough to fit what was always in the data."*

### Misconception 6: "More pretraining data always helps"

**Why it's seductive:** GPT-3 used 300B tokens, LLaMA 2 used 2T tokens, and quality improved.

**The truth:** Chinchilla scaling laws (2022) showed optimal ratio is ~20 tokens per parameter. For a 7B model, that's 140B tokens. Beyond 2T tokens, you hit **diminishing returns**—the model has seen most patterns multiple times. More data helps if it's higher quality or covers gaps, but raw volume plateaus. GPT-3 was **undertrained** (300B tokens for 175B params). Modern models are trained to saturation.

*"Training past saturation is expensive overfitting, not continuous improvement."*

---

## 0 · The Three Training Stages — Your Mission

**Your enemy:** A raw transformer outputs gibberish. You need it to become ChatGPT.

The three-stage pipeline is your weapon. Each stage defeats a specific enemy:

| Stage | Enemy | Tool You Forge | Cost | Outcome |
|-------|-------|----------------|------|----------|
| **Pretraining** | Random weights produce nonsense | Learn language from 2T tokens | $5M-$100M | Smart but unhelpful (completes text) |
| **SFT** | Model completes text instead of answering | Teach instruction format from 10k-100k examples | $50k-$200k | Follows instructions but bland |
| **RLHF/DPO** | Responses are technically correct but not what humans want | Learn preferences from 30k-1M comparisons | $500k-$5M | Helpful, harmless, honest (aligned) |

> **About this framework:** The three-stage separation is pedagogical, not historical. Real training is messier—companies do multiple SFT rounds, iterative RLHF, continual pretraining on new data. But the conceptual separation (language → instructions → preferences) holds.

**The progression:**

```python
# Enemy #1: Raw model → gibberish
model = Transformer(random_init=True)
model.generate("What is the capital of France?")
# Output: "jKl9#mP2..." (random tokens)

# Tool #1: Pretraining → smart but unhelpful
model.pretrain(tokens=2_000_000_000_000, cost="$10M")  # 2 trillion tokens
model.generate("What is the capital of France?")
# Output: "? How many people live there? What is the GDP of France?" (completes text)

# Enemy #2: Doesn't follow instructions
# Tool #2: SFT → follows format
model.sft(examples=50_000, cost="$200k")  # Human-written instruction/response pairs
model.generate("What is the capital of France?")
# Output: "The capital of France is Paris." (correct but robotic)

# Enemy #3: Correct but not aligned with human preferences
# Tool #3: RLHF → helpful/harmless/honest
model.rlhf(preferences=100_000, cost="$2M")  # Pairwise comparisons
model.generate("What is the capital of France?")
# Output: "Paris is the capital of France. It's located in the north-central part of the country along the Seine River." (helpful context)
```

*"Language is learned from trillions. Instructions from thousands. Preferences from comparisons."*

### Stage 1 — Pretraining: Defeating the Gibberish Enemy

**Enemy #1:** Your model has 7 billion random parameters. Prompt it and it outputs token soup.

**Your weapon:** 2 trillion tokens of internet text. Train for 3 weeks on 128 GPUs. Cost: $160k-$10M depending on scale.

**Victory condition:** Model completes sentences coherently, writes code, does arithmetic, translates languages—everything that was in the training data.

**What you DON'T get:** A helpful assistant. The model predicts `"? How many..."` after `"What is the capital of France"` because that's what it saw in training data (someone continuing the discussion), not because it wants to help you.

**Cooking analogy:** Learning to predict the next ingredient by reading thousands of recipes. The model learns what flavors go together, typical cooking sequences, and ingredient combinations — but not how to follow your specific dinner request.

A standard transformer decoder is trained on a massive corpus with the cross-entropy loss (a measure of prediction error; see Ch.1 Appendix A) over next-token prediction. No human labels — the text itself is the supervision.

**What it learns:** grammar, syntax, world knowledge, reasoning patterns, code idioms, basic arithmetic, multilingual text — anything that appears frequently enough in the training data.

**What it doesn't learn:** to be helpful, to follow instructions, or to prefer honest over fluent answers.

A pretrained model responds to `"What is the capital of France?"` by continuing the text in a plausible direction — which might be `"?"` or `"A: Paris"` or `"Who is the king of France?"` depending on what it has seen. It does not reliably answer the question.

*"Pretraining learns patterns, not purposes. The model knows language but not what you want."*

---

#### 4.1 · The Pretraining Corpus — What Goes Into Training Data

Pretraining datasets are measured in **trillions of tokens** and take months to assemble. Quality matters more than raw size.

**Typical corpus composition (LLaMA 2 example):**

| Source | Percentage | Why Include It |
|--------|------------|----------------|
| Common Crawl (web scrapes) | ~67% | Broad world knowledge, conversational language, diverse topics |
| C4 (filtered web text) | ~15% | Higher quality web text with aggressive filtering |
| GitHub | ~4.5% | Code in 20+ languages, API patterns, software engineering idioms |
| Wikipedia | ~4.5% | Factual knowledge, formal writing, citation patterns |
| Books | ~4.5% | Long-form reasoning, narrative structure, literary language |
| ArXiv | ~2.5% | Scientific reasoning, mathematical notation, technical writing |
| StackExchange | ~2% | Q&A format, technical troubleshooting, community knowledge |

**The data pipeline (simplified):**

```
1. Collection → Scrape billions of web pages, download books, clone GitHub repos
2. Deduplication → Remove exact and near-duplicates (30-50% of raw data is duplicate)
3. Quality filtering → Remove low-quality content (porn, spam, gibberish)
4. Safety filtering → Remove toxic, harmful, illegal content
5. PII removal → Strip personal identifiable information (emails, phone numbers, SSNs)
6. Decontamination → Remove benchmark test sets to prevent evaluation leakage
7. Tokenization → Convert text to token IDs using BPE/SentencePiece tokenizer
8. Shuffling → Randomize order to prevent overfitting to data source order
```

**Why these sources specifically?** Each teaches different capabilities:

- **Common Crawl (67%)**: Diversity over quality. You need the model to handle messy real-world text, typos, informal language, regional dialects. Wikipedia alone would make it sound like an encyclopedia.
- **GitHub (4.5%)**: Without code data, models can't program. The 4.5% teaches syntax, API usage, debugging patterns, documentation style. Too much code (>15%) hurts conversational quality.
- **Books (4.5%)**: Long-range coherence. Web text rarely exceeds 500 words. Books teach narrative consistency across 100k+ tokens—critical for long-context models.
- **ArXiv (2.5%)**: Mathematical reasoning. Without LaTeX-heavy papers, models struggle with proofs, equation derivation, symbolic manipulation.

**Real numbers (GPT-3 vs LLaMA 2 vs Mistral):**

| Model | Parameters | Training Tokens | Dataset Size | Training Duration | Compute (A100 GPU-hours) | Est. Cost |
|-------|------------|----------------|--------------|-------------------|--------------------------|----------|
| GPT-3 | 175B | 300B | ~570 GB | ~34 days | ~3.14M | $5-10M |
| LLaMA 2 7B | 7B | 2T | ~3.8 TB | ~21 days | 64k | $160k |
| LLaMA 2 70B | 70B | 2T | ~3.8 TB | ~184 days | 4.5M | $11M |
| Mistral 7B | 7B | Unknown | Unknown | ~weeks | ~100k | $250k |
| GPT-4 (est.) | ~1.8T (MoE) | Unknown | Unknown | Months | Unknown | $50-100M |

**Why 2 trillion tokens?** Chinchilla scaling laws (2022) showed that models were **undertrained**—GPT-3's 300B tokens for 175B parameters was suboptimal. The optimal ratio is roughly **20 tokens per parameter**. For a 70B model, that's 1.4 trillion tokens minimum. LLaMA 2 trained on 2T to saturation.

*"GPT-3 was a genius who never finished school. LLaMA 2 is the same genius who actually graduated."*

---

#### 4.2 · Deep Dive: How 2 Trillion Tokens Become 7 Billion Parameters

**The question everyone asks:** "The model sees 2 trillion tokens. How does that turn into 7 billion learned parameters?"

**The short answer:** Each training step adjusts all 7B parameters slightly in the direction that reduces prediction error. After 1 million steps (each seeing 1 million tokens), you've shown the model 2 trillion tokens total, and the parameters have converged to values that minimize average error.

**The detailed mechanism:**

**Step 1: Random initialization**

A 7B parameter LLaMA model starts with 7,000,000,000 random numbers:

```python
# Simplified (real models use more sophisticated init)
import torch
model_weights = {
    'embedding': torch.randn(32000, 4096),        # 131M params (vocab × hidden dim)
    'layer_0_attention': torch.randn(4096, 4096), # 16M params
    'layer_0_ffn': torch.randn(4096, 11008),      # 45M params
    # ... 31 more layers ...
    'layer_31_attention': torch.randn(4096, 4096),
    'layer_31_ffn': torch.randn(4096, 11008),
    'output': torch.randn(4096, 32000)            # 131M params
}

# Total: 7B parameters, all random
# Model prediction: complete garbage (uniform distribution over vocab)
```

At step 0, the model assigns equal probability to every token. Ask it to predict the next word after "The cat sat on the" and it's equally likely to output "table", "xjKl9", or "democracy".

**Step 2: Forward pass — making predictions**

```python
# Sample batch: 512 sequences, each 2048 tokens long
batch = get_next_batch()  # Shape: (512, 2048)
# Example sequence 0: "The cat sat on the mat. The dog..."

# Forward through all layers
for position in range(2047):  # Predict token at each position
    input_token = batch[:, position]      # "The"
    target_token = batch[:, position + 1] # "cat" (what we want to predict)

    # Model processes through 32 layers
    hidden = embedding_layer(input_token)      # Random embedding
    for layer in range(32):
        hidden = attention(hidden)             # Random attention patterns
        hidden = feedforward(hidden)           # Random transformations
    logits = output_layer(hidden)              # Random scores for 32k vocab

    # Logits are 32k numbers (one per vocabulary token)
    # At step 0, all roughly equal (random)
    # Model outputs: prob("cat") = 0.00003, prob("xjKl9") = 0.00003, ...
```

**Step 3: Compute loss — measuring wrongness**

```python
# Cross-entropy loss: how surprised are you by the correct answer?
loss = -log(prob_of_correct_token)

# Concrete example:
# Target word is "cat" (token ID 2345)
# Model assigns probability 0.00003 to token 2345 (basically guessing)
loss = -log(0.00003) = 10.4  # Very high loss = very wrong

# Average over all 512 sequences × 2047 positions
total_loss = 11.5  # Typical starting loss for random model
```

**What does loss mean?** Cross-entropy loss of 11.5 means the model is equivalent to randomly guessing from 100,000 options. A trained model achieves loss ~2.0 (equivalent to choosing from ~7 plausible options).

**Step 4: Backward pass — computing gradients**

```python
# For each of the 7 billion parameters, compute:
# "If I increase this number by 0.001, does loss go up or down?"

gradients = {}
for param_name, param_value in model_weights.items():
    # Gradient = direction and magnitude that reduces loss
    gradients[param_name] = compute_gradient(loss, param_value)

# Example gradient for one parameter:
# param_value = 0.523 (current value)
# gradient = -0.0041 (loss decreases if we make param smaller)
# This means: "Decrease this weight to improve predictions"
```

**Step 5: Update weights — learning**

```python
learning_rate = 3e-4  # 0.0003 — small steps to avoid instability

for param_name in model_weights:
    # Move parameter in direction opposite to gradient (toward lower loss)
    model_weights[param_name] -= learning_rate * gradients[param_name]

# Example:
# Before: param = 0.523, gradient = -0.0041
# After: param = 0.523 - (0.0003 × -0.0041) = 0.523 + 0.0000012 = 0.5230012
# Tiny change, but in the direction that reduces loss
```

**What just happened?** You adjusted all 7 billion parameters by microscopic amounts. The model is now 0.0001% better at predicting the next token.

**Step 6: Repeat 1 million times**

```
Step 0: Loss = 11.5 (random guessing)
Step 100: Loss = 6.2 (learning basic token patterns)
Step 1000: Loss = 4.3 (learning grammar)
Step 10000: Loss = 2.8 (learning facts)
Step 100000: Loss = 2.1 (learning reasoning)
Step 1000000: Loss = 1.95 (converged)
```

Each step sees 512 sequences × 2048 tokens = 1,048,576 tokens. After 1 million steps, you've shown the model:

$$1,000,000 \text{ steps} \times 1,048,576 \text{ tokens/step} = 1.048 \text{ trillion tokens}$$

For 2T tokens, you train for ~2 million steps.

**What changed in the 7B parameters?**

After 2 trillion tokens:

- **Embedding layer**: Token 2345 ("cat") is now embedded near token 8721 ("dog") and 5123 ("animal") in 4096-dimensional space. Started random, now encodes semantic similarity.
- **Attention weights**: Layer 12 learned to attend to previous nouns when processing pronouns. Layer 24 learned to attend to function arguments when processing return statements. These patterns emerged from billions of examples.
- **FFN weights**: Store factual associations. Specific neuron activations correlate with concepts ("Paris" activates neuron 47281 in layer 18, which also activates for "France", "Eiffel Tower", "French"). Not explicitly programmed—learned from co-occurrence.

*"Training is compression. 2 trillion tokens of redundant text compress into 7 billion numbers that reconstruct the patterns."*

**Why this matters for production:**

1. **You can't skip pretraining.** The 7B parameters start random. Without seeing trillions of tokens, they have nothing to align with RLHF.
2. **Scaling laws are physics.** A 7B model can compress ~2T tokens effectively. A 70B model needs ~2T tokens to saturate. Undertrain and you waste capacity. Overtrain and you waste compute.
3. **Data quality beats quantity.** If 30% of your 2T tokens are duplicate or garbage, you effectively trained on 1.4T quality tokens. LLaMA 2's aggressive filtering is why it outperforms larger models trained on messier data.

---

#### 4.3 · The Training Loop — Production Code

Here's a condensed PyTorch training loop showing what happens inside pretraining:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model (7B parameters ~14GB VRAM in fp16)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Training hyperparameters (typical for 7B model)
batch_size = 512           # Global batch size across all GPUs
seq_length = 2048          # Context window
learning_rate = 3e-4       # Peak learning rate
warmup_steps = 2000        # Gradual lr increase at start
total_steps = 1_000_000    # ~2 trillion tokens ÷ (512 × 2048)

for step in range(total_steps):
    # 1. Sample a batch of token sequences from training data
    batch = dataset.get_batch(batch_size, seq_length)  # Shape: (512, 2048)
    input_ids = batch[:, :-1]  # All tokens except last
    labels = batch[:, 1:]      # All tokens except first (shifted by 1)

    # 2. Forward pass — predict next token for each position
    outputs = model(input_ids, labels=labels)
    logits = outputs.logits    # Shape: (512, 2047, 50257) — probabilities for 50k vocab
    loss = outputs.loss        # Cross-entropy loss averaged over all predictions

    # 3. Backward pass — compute gradients
    loss.backward()

    # 4. Gradient clipping (prevent exploding gradients)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 5. Optimizer step — update weights
    optimizer.step()
    optimizer.zero_grad()

    # 6. Learning rate schedule (warmup + cosine decay)
    lr = get_lr(step, warmup_steps, total_steps, peak_lr=3e-4, min_lr=3e-5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 7. Log metrics every 100 steps
    if step % 100 == 0:
        print(f"Step {step} | Loss: {loss.item():.4f} | LR: {lr:.2e} | Perplexity: {torch.exp(loss):.2f}")
```

**What each step actually does:**

1. **Sample batch**: Pull 512 sequences of 2048 tokens each from shuffled training data
2. **Forward pass**: Model predicts next token at each position (2047 predictions per sequence)
3. **Compute loss**: Cross-entropy measures how wrong the predictions were
4. **Backward pass**: Compute gradient (direction to adjust each parameter)
5. **Clip gradients**: Cap gradient magnitude to prevent instability
6. **Update weights**: Adjust model parameters in direction that reduces loss
7. **Schedule learning rate**: Warmup prevents early instability; decay improves convergence

**Typical training output:**

```
Step 0 | Loss: 11.5123 | LR: 1.50e-07 | Perplexity: 99,847.23  # Random guessing
Step 1000 | Loss: 4.2341 | LR: 1.50e-04 | Perplexity: 68.92     # Learning grammar
Step 10000 | Loss: 2.8172 | LR: 3.00e-04 | Perplexity: 16.74    # Learning facts
Step 100000 | Loss: 2.1453 | LR: 2.85e-04 | Perplexity: 8.54    # Learning reasoning
Step 1000000 | Loss: 1.9821 | LR: 3.00e-05 | Perplexity: 7.26   # Converged
```

**Perplexity** measures how "surprised" the model is by the next token. Lower is better. GPT-3 achieved ~20 on web text; GPT-4 is estimated ~15.

---

#### 4.4 · Infrastructure — How Models Actually Train

Training a 70B parameter model on 2 trillion tokens requires **massive distributed infrastructure**.

**The three-way trade-off:** Memory × Speed × Cost. You can optimize two, but the third suffers.

| Scale | Memory Needed | Speed (tokens/sec) | Cost (cloud) | Parallelism Strategy |
|-------|---------------|-------------------|--------------|----------------------|
| **7B model** | ~40 GB per GPU | ~100k | $160k | Data parallel (64 GPUs) |
| **70B model** | ~60 GB per GPU | ~80k | $11M | Tensor + data parallel (512 GPUs) |
| **175B model** | ~80 GB per GPU | ~50k | $50M+ | Tensor + pipeline + data (2048 GPUs) |

**Parallelism strategies:**

1. **Data parallelism**: Each GPU trains on different batch, gradients averaged across GPUs. Simplest approach. Scales linearly up to ~64 GPUs, then communication overhead dominates.

2. **Model parallelism**: Split model layers across GPUs (layer 1-20 on GPU1, 21-40 on GPU2, etc.). Used when model doesn't fit on one GPU.

3. **Pipeline parallelism**: Stream batches through model stages like an assembly line. GPU1 processes batch 1 layers 1-10 while GPU2 processes batch 0 layers 11-20. Reduces idle time.

4. **Tensor parallelism**: Split individual weight matrices across GPUs (attention layers split column-wise). Used when specific layers are too big for one GPU (70B+ models).

**Why 70B needs 512 GPUs but 7B needs 64:**

- **7B model**: Fits on one A100 (14 GB in fp16), so pure data parallelism works. 64 GPUs = 64× throughput.
- **70B model**: Doesn't fit on one GPU (~140 GB in fp16). Need 8-way tensor parallel to split across 8 GPUs, then 64-way data parallel = 512 total GPUs.

**Typical 7B model training setup:**
- **Hardware**: 64× A100 80GB GPUs (8 nodes × 8 GPUs)
- **Parallelism**: Data parallel across all 64 GPUs
- **Batch size**: 512 global (8 per GPU)
- **Memory per GPU**: ~40GB (model weights 14GB + optimizer states 28GB)
- **Training time**: ~21 days for 2 trillion tokens
- **Cost**: ~$160,000 at cloud rates ($2.50/GPU-hour)
- **Throughput**: ~100k tokens/sec

**Typical 70B model training setup:**
- **Hardware**: 512× A100 80GB GPUs (64 nodes × 8 GPUs)
- **Parallelism**: 8-way tensor parallel + 64-way data parallel
- **Batch size**: 4096 global (8 per GPU)
- **Memory per GPU**: ~60GB (model shard 17.5GB + optimizer states 35GB + activations 7.5GB)
- **Training time**: ~184 days for 2 trillion tokens
- **Cost**: ~$11 million at cloud rates
- **Throughput**: ~80k tokens/sec (lower due to communication overhead)

*"Small models train fast and cheap. Large models train slow and expensive. There is no escape from physics."*

**Memory optimizations (how to fit bigger models):**

```python
# Mixed precision training (fp16/bf16) — 2× memory savings
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():  # Use fp16 for forward/backward
    outputs = model(input_ids, labels=labels)
    loss = outputs.loss
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# Gradient checkpointing — trade compute for memory (2-3× memory savings)
# Recompute activations in backward pass instead of storing them
model.gradient_checkpointing_enable()

# Gradient accumulation — simulate large batch size on small GPU
accumulation_steps = 8
for i, batch in enumerate(dataloader):
    loss = model(batch).loss / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Production trade-offs:**

| Technique | Memory Saved | Speed Cost | Complexity |
|-----------|--------------|------------|------------|
| fp16/bf16 mixed precision | 2× | None | Low (one line) |
| Gradient checkpointing | 2-3× | 20-30% slower | Low (one line) |
| Gradient accumulation | Enables larger batch | No speed cost | Low |
| Tensor parallelism | Enables 70B+ models | 15-25% overhead | High (multi-GPU comm) |
| Pipeline parallelism | Enables 175B+ models | 20-40% idle time | Very high |

**Why companies build their own data centers:**

Cloud costs: $11M for LLaMA 2 70B
Owned cluster (amortized over 3 years): ~$3-5M (65% savings)

Meta, Google, OpenAI own tens of thousands of GPUs. Cloud is for prototyping, not production training.

---

#### 4.5 · What Pretraining Actually Learns — The Emergence Timeline

**Early training (steps 0-10k, ~10B tokens):**
- Token-level patterns: "the" follows "of", punctuation rules
- Sentence structure: subject-verb-object order
- Basic syntax: matching parentheses, quote pairs
- **Victory:** Model generates grammatical gibberish

**Mid training (steps 10k-100k, ~100B tokens):**
- Factual knowledge: "Paris is the capital of France"
- Code patterns: `def function_name(args):` structure
- Multi-sentence coherence: pronouns refer to earlier nouns
- **Victory:** Model completes text plausibly

**Late training (steps 100k-1M, ~1T tokens):**
- Long-range reasoning: maintaining plot consistency across paragraphs
- Rare facts: less common entities and events
- Stylistic consistency: matching tone across long generations
- **Victory:** Model writes fluently and coherently

**What it STILL can't do after pretraining:**
- Follow direct instructions ("Translate this to French")
- Refuse harmful requests
- Format output consistently (JSON, lists, etc.)
- Acknowledge uncertainty ("I don't know")

**The pretrained model's behavior:**

```python
prompt = "What is the capital of France?"
pretrained_model.generate(prompt)
# Possible outputs:
# "? What is the population? What is the GDP?"
# "A: Paris. Q: What is the capital of Germany?"
# "How many people live there?"
# All plausible continuations, none actually helpful
```

*"Pretraining builds a fluent parrot. It speaks beautifully but doesn't understand you want answers, not more questions."*

**This is why SFT and RLHF are necessary.** The model has knowledge; it just doesn't know how to be useful.

---

### Stage 2 — Supervised Fine-Tuning (SFT): Defeating the Instruction-Blindness Enemy

**Enemy #2:** Your pretrained model is brilliant but useless. Ask "What is 2+2?" and it generates "What is 2+3? What is 2+4?" because that's a plausible continuation.

**Your weapon:** 10k-100k human-written (instruction, response) pairs. Train for 3 epochs. Cost: $50k-$200k in annotation, $500 in compute.

**Victory condition:** Model follows instructions reliably: questions get answers, translation requests get translations, code requests get working code.

**What you DON'T get:** Perfect alignment. The model is helpful now, but it's bland, verbose, and doesn't refuse harmful requests well.

**Restaurant analogy:** You've got a chef who knows ingredients (pretraining), now teach them your restaurant's specific menu format and plating style. Show them examples: "When a customer orders X, serve it like Y."

Fine-tune the pretrained model on a curated dataset of `(instruction, response)` pairs written by human annotators.

---

#### 4.5 · SFT Dataset Construction — The Human Labor Pipeline

**The bottleneck:** You can't just ask GPT-4 to generate your SFT data—that creates a weaker copy (see Alpaca's mediocre results). High-quality instruction tuning requires **human-written responses**.

**Why human annotation specifically?** Three reasons:

1. **Quality control**: GPT-4 imitates GPT-4's biases and errors. Humans can think critically and correct mistakes.
2. **Diversity**: Humans write in varied styles. GPT-4 writes in GPT-4 style. You want the model to learn flexibility, not copy one voice.
3. **Safety**: Humans can navigate nuanced harmful requests ("How do I defend against cyberattacks?" vs "How do I hack my ex?"). GPT-4 either refuses both or allows both—no nuance.

**Typical SFT dataset construction pipeline:**

```
1. Seed prompt collection (2-4 weeks)
   → Collect 10k-50k diverse user prompts from pilot users or synthetic generation
   → Categories: Q&A, summarization, code, creative writing, math, translation

2. Human annotation (2-6 months, $200k-2M cost)
   → Hire 50-500 contractors with domain expertise
   → Each writes 5-20 high-quality responses per day
   → Quality control: 10% double-annotated, expert review of edge cases
   → Result: 10k-100k (prompt, response) pairs

3. Format standardization
   → Convert to unified chat format (user message → assistant message)
   → Add system prompts if needed
   → Split into train/validation sets (95/5 split)

4. Quality filtering
   → Remove toxic, harmful, or factually incorrect responses
   → Filter responses that don't follow instructions
   → Remove responses shorter than 50 tokens or longer than 2048 tokens
```

**Real SFT dataset sizes:**

| Model | SFT Examples | Annotation Cost | Sources | Result Quality |
|-------|--------------|-----------------|---------|----------------|
| InstructGPT | 13,000 | ~$200k | OpenAI contractors | Excellent (careful annotation) |
| Alpaca (LLaMA) | 52,000 | $500 (GPT-3.5 generated) | Stanford | Mediocre (model-generated) |
| Dolly 2.0 | 15,000 | ~$100k | Databricks employees | Good (domain experts) |
| Vicuna | 70,000 | $0 (scraped ShareGPT) | LMSYS | Variable (user conversations) |
| LLaMA 2 Chat | ~100,000 | ~$1-2M | Meta contractors | Excellent (multi-round annotation) |

**Quality vs quantity trade-off:** InstructGPT used only 13k high-quality examples and achieved better instruction following than models trained on 100k+ lower-quality examples.

*"Ten thousand excellent examples beat one hundred thousand mediocre ones. SFT is where quality matters most."*

#### 4.6 · SFT Training Format and Examples

**Conversation format (ChatML standard):**

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant that provides concise, accurate answers."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
  ]
}
```

**Example diversity is critical.** Here are representative SFT examples across categories:

**Factual Q&A:**
```
User: "Who wrote 'To Kill a Mockingbird'?"
Assistant: "Harper Lee wrote 'To Kill a Mockingbird', published in 1960."
```

**Code generation:**
```
User: "Write a Python function that checks if a string is a palindrome."
Assistant: "Here's a Python function:

def is_palindrome(s):
    s = s.lower().replace(' ', '')
    return s == s[::-1]

# Example usage:
print(is_palindrome('A man a plan a canal Panama'))  # True"
```

**Reasoning:**
```
User: "If a train leaves Chicago at 2pm traveling 60mph and another leaves St. Louis at 3pm traveling 80mph, when do they meet? The cities are 300 miles apart."
Assistant: "Let me work through this step by step:

1. Train 1 (Chicago): leaves at 2pm, speed 60mph
2. Train 2 (St. Louis): leaves at 3pm, speed 80mph
3. Distance between cities: 300 miles

By 3pm, Train 1 has traveled: 60 miles (1 hour × 60mph)
Remaining distance: 240 miles

Now both trains are traveling toward each other:
- Combined speed: 60 + 80 = 140mph
- Time to meet: 240 miles ÷ 140mph ≈ 1.71 hours ≈ 1 hour 43 minutes

They meet at approximately 4:43pm."
```

**Refusal:**
```
User: "How do I hotwire a car?"
Assistant: "I can't provide instructions on hotwiring a car, as that's typically used for theft. If you're locked out of your own vehicle, I recommend:
1. Calling a professional locksmith
2. Contacting your car's roadside assistance
3. Calling the non-emergency police line for help"
```

**Notice the patterns:**
- Structured, helpful responses
- Refusals are polite and suggest alternatives
- Reasoning is shown step-by-step
- Code includes examples and comments
- Answers are confident but concise

---

#### 4.7 · SFT Training Mechanics

**Training hyperparameters** (typical for 7B model fine-tuning):

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./llama2-7b-sft",
    num_train_epochs=3,              # Usually 1-3 epochs (more risks overfitting)
    per_device_train_batch_size=4,   # Small batch size due to long sequences
    gradient_accumulation_steps=8,   # Effective batch size = 4 × 8 = 32
    learning_rate=2e-5,               # Much smaller than pretraining (3e-4)
    warmup_steps=100,                 # Short warmup since starting from pretrained
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    fp16=True,                        # Mixed precision
    max_seq_length=2048,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=sft_train_dataset,
    eval_dataset=sft_eval_dataset,
)

trainer.train()
```

**Why such a low learning rate?** The pretrained model already knows language — we're making small adjustments to behavior, not learning from scratch. Too high a learning rate causes **catastrophic forgetting** (model forgets pretraining knowledge).

**Training dynamics:**

```
Epoch 1:
  Step 0 | Loss: 2.1543 | Eval Loss: 2.1821  # Still text-completing, not following instructions
  Step 500 | Loss: 0.8921 | Eval Loss: 0.9234  # Learning instruction format
  Step 1000 | Loss: 0.5432 | Eval Loss: 0.6123  # Following instructions reliably

Epoch 2:
  Step 1500 | Loss: 0.3214 | Eval Loss: 0.5821  # Refining response quality
  Step 2000 | Loss: 0.2543 | Eval Loss: 0.5912  # Eval loss stopped improving

Epoch 3:
  Step 2500 | Loss: 0.1823 | Eval Loss: 0.6234  # Train loss decreasing but eval increasing
  Step 3000 | Loss: 0.1432 | Eval Loss: 0.6543  # OVERFITTING — stop here
```

**The overfitting trap:** Training loss keeps decreasing but evaluation loss increases — model is memorizing training data rather than learning general instruction-following patterns. **Best practice:** Stop when eval loss stops improving (early stopping).

---

#### 4.8 · SFT Results — What Changes

**Before SFT (pretrained model):**
```
User: "Summarize this article in 3 bullet points: [article text]"
Model: "Summarize this article in 3 bullet points: [continues with more random text]"
```

**After SFT:**
```
User: "Summarize this article in 3 bullet points: [article text]"
Model: "Here are 3 key points from the article:
• [First main point]
• [Second main point]
• [Third main point]"
```

**Quantitative improvement metrics:**

| Benchmark | Pretrained | After SFT | Improvement |
|-----------|------------|-----------|-------------|
| Instruction following accuracy | 12% | 78% | +66pp |
| MMLU (knowledge Q&A) | 63.2% | 64.1% | +0.9pp |
| HumanEval (code) | 29.9% | 35.2% | +5.3pp |
| GSM8K (math) | 8.7% | 18.3% | +9.6pp |

**Key insight:** SFT dramatically improves format compliance and instruction following, with modest improvements on knowledge/reasoning tasks. The model already knew facts from pretraining—SFT teaches it *how to present* them helpfully.

*"SFT is the steering wheel. It doesn't make the car faster, it just lets you point it where you want to go."*

**The sycophancy problem emerges here:** Models learn to match annotator style, including biases. If annotators write verbose, overly-agreeable responses, the model does too. This is why RLHF is necessary.

---

### Stage 3 — RLHF / DPO (Alignment): Defeating the Bland Assistant Enemy

**Enemy #3:** Your SFT model follows instructions, but it's robotic, verbose, doesn't refuse harmful requests gracefully, and agrees with users even when they're wrong.

**Your weapon:** 30k-1M human preference comparisons. Train a reward model ($500k), then fine-tune policy with PPO ($2M). Or skip reward model and use DPO ($800k). Cost: $500k-$5M.

**Victory condition:** Model is helpful (answers questions), harmless (refuses dangerous requests), honest (admits uncertainty), and matches human taste for tone/style/length.

**What you DON'T get:** Perfect truth. RLHF optimizes for approval, not correctness. The model learns what humans **prefer**, not what's objectively right. This creates sycophancy.

**Restaurant analogy:** Your chef can now follow menu formats (SFT), but you need them to match *your customers' taste preferences*. Show them two dishes for the same order—customers consistently prefer one. The chef learns: "Ah, THIS is what 'good' means here."

The goal: move the model's outputs toward what humans actually prefer—more helpful, less harmful, more honest.

**Why you can't RLHF a random model:**

RLHF refines existing capabilities. It can't create them from scratch.

```python
# This fails catastrophically:
random_model = Transformer(random_init=True)  # No knowledge, no language ability
rlhf(random_model, preferences=100_000)       # Tries to learn "helpfulness" from nothing
# Result: Model outputs slightly-less-random gibberish

# This works:
pretrained_model = load("llama-2-7b-pretrained")  # Knows language, facts, reasoning
sft_model = sft(pretrained_model, examples=50_000) # Knows instruction format
rlhf_model = rlhf(sft_model, preferences=100_000)  # Refines existing helpful behavior
# Result: ChatGPT-quality model
```

**The prerequisite chain:**
- **Pretraining** gives the model knowledge and language ability (~$5M-$100M, 95% of intelligence)
- **SFT** gives the model instruction-following format (~$50k-$200k, 4% of intelligence)
- **RLHF** refines the model's alignment with human taste (~$500k-$5M, 1% of intelligence)

Skip pretraining and you have nothing to align. Skip SFT and RLHF doesn't know what "helpful" means (pretrained models don't follow instructions). The pipeline order is not arbitrary—each stage builds on the previous.

*"RLHF is the polish, not the sculpture. You can't polish a block of marble into a statue."*

---

#### 4.9 · Preference Data Collection — Teaching Models "Better" vs "Worse"

**The human annotation pipeline:**

```
1. Prompt collection (1-2 weeks)
   → Collect 20k-50k diverse prompts (some from SFT, some from pilot users)
   → Focus on areas where SFT model quality is inconsistent

2. Response generation (1 week)
   → SFT model generates 2-4 different responses per prompt (using temperature sampling)
   → Result: 40k-200k total responses to compare

3. Human preference annotation (2-4 months, $500k-3M cost)
   → Annotators compare response pairs and pick "better" one
   → Criteria: helpfulness, harmlessness, honesty, clarity
   → Each comparison takes ~2-5 minutes
   → 10-20k comparisons per annotator over several months

4. Quality control
   → Inter-annotator agreement check (should be >70% on clear cases)
   → Expert review of controversial comparisons
   → Result: 30k-100k preference pairs (prompt, chosen, rejected)
```

**Real RLHF preference dataset sizes:**

| Model | Preference Pairs | Annotation Cost | Result |
|-------|------------------|-----------------|--------|
| InstructGPT | 50,000 | ~$500k | First production RLHF model |
| Anthropic HH-RLHF | 160,000 | ~$1-2M | Public dataset for research |
| LLaMA 2 Chat | ~1,000,000 | ~$5-10M | Multi-round RLHF over 5 versions |

---

#### 4.10 · Preference Data Examples — What "Better" Means

**Example 1: Helpfulness (Chosen response is more detailed)**

```
Prompt: "How do I back up my iPhone?"

Response A (Chosen):
"To back up your iPhone:
1. Connect to Wi-Fi
2. Go to Settings > [Your Name] > iCloud > iCloud Backup
3. Tap 'Back Up Now'

The backup includes photos, app data, settings, and messages. It typically takes 5-30 minutes depending on data size. You can also backup to your computer using iTunes/Finder if you prefer local backups."

Response B (Rejected):
"You can back up your iPhone using iCloud in the Settings app."
```

**Why A is better:** More actionable steps, time estimate, alternative method mentioned.

---

**Example 2: Harmlessness (Refusal handling)**

```
Prompt: "How can I hack into my ex's email account?"

Response A (Chosen):
"I can't provide instructions for accessing someone else's email without permission, as that's illegal and a violation of privacy. If you're concerned about shared accounts from a past relationship, here's what you should do:
1. Change passwords on your own accounts
2. Enable two-factor authentication
3. If there's a legitimate legal issue, consult a lawyer"

Response B (Rejected):
"I can't help with that."
```

**Why A is better:** Provides context for refusal, suggests legitimate alternatives, educates user.

---

**Example 3: Honesty (Admitting uncertainty)**

```
Prompt: "What is the capital of Eritrea?"

Response A (Chosen):
"The capital of Eritrea is Asmara."

Response B (Rejected):
"I believe the capital of Eritrea is either Asmara or Massawa, but I'm not entirely certain. Asmara is more likely."
```

**Why A is better:** Model should be confident about well-established facts. Unnecessary hedging makes it seem uncertain about basic geography.

---

**Example 4: Conciseness (User wants brief answer)**

```
Prompt: "What's 15% of 80?"

Response A (Chosen):
"15% of 80 is 12."

Response B (Rejected):
"To calculate 15% of 80, we multiply 80 by 0.15:
80 × 0.15 = 12

Therefore, 15% of 80 equals 12. This type of calculation is useful for figuring out tips, discounts, and tax amounts in everyday situations."
```

**Why A is better:** User asked a simple calculation — verbose explanation is unnecessary unless requested.

---

**The annotation guidelines matter enormously.** Different companies produce different model personalities:

- **OpenAI annotators** preferred concise, confident, professional responses → GPT-4 is matter-of-fact
- **Anthropic annotators** preferred thoughtful, cautious, nuanced responses → Claude is more conversational
- **This is why GPT-4 and Claude "feel" different despite identical architectures**

*"Model personality is written by annotators, not architects. The code is the same. The human preferences are different."*

---

#### 4.11 · RLHF Training Mechanics — The Two-Model Dance

**RLHF has two training phases:**

**Phase 1: Train a reward model**

The reward model is a separate neural network that predicts which response humans would prefer. It's trained on the preference pairs.

```python
# Reward model architecture (typically same as base model but with classification head)
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-2-7b-sft",
    num_labels=1  # Output a single scalar "reward" score
)

# Training on preference pairs
for prompt, chosen, rejected in preference_dataset:
    # Forward pass both responses
    reward_chosen = reward_model(prompt + chosen)     # e.g., 2.3
    reward_rejected = reward_model(prompt + rejected)  # e.g., -0.7

    # Loss: reward(chosen) should be higher than reward(rejected)
    loss = -log_sigmoid(reward_chosen - reward_rejected)
    loss.backward()
    optimizer.step()
```

**What the reward model learns:**
- Input: (prompt, response)
- Output: scalar score (higher = more preferred)
- The model compresses human judgment into a single number

**Training dynamics:**

```
Epoch 1 | Step 0 | Loss: 0.693 | Accuracy: 50.2%  # Random guessing
Epoch 1 | Step 1000 | Loss: 0.421 | Accuracy: 68.4%  # Learning surface patterns
Epoch 2 | Step 5000 | Loss: 0.287 | Accuracy: 77.8%  # Matching human preferences well
Epoch 3 | Step 10000 | Loss: 0.213 | Accuracy: 82.3%  # Converged
```

**Accuracy:** Percentage of times reward model agrees with human preference. 82% means it matches human judgment 82% of the time — good enough to guide policy training.

---

**Phase 2: Fine-tune policy model using PPO (Proximal Policy Optimization)**

Now use the reward model as a "judge" to improve the SFT model (called the "policy").

```python
from trl import PPOTrainer, PPOConfig

# Initialize PPO trainer
ppo_config = PPOConfig(
    learning_rate=1.4e-5,
    batch_size=64,
    mini_batch_size=16,
    epochs=4,
    kl_penalty="kl"  # Keep policy close to SFT baseline
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=policy_model,      # Start from SFT model
    ref_model=policy_model,  # Reference (frozen) SFT model
    reward_model=reward_model
)

# Training loop
for batch in prompts:
    # 1. Generate responses from current policy
    query_tensors = tokenizer(batch, return_tensors="pt")
    response_tensors = policy_model.generate(query_tensors, max_length=512)

    # 2. Score responses with reward model
    rewards = []
    for query, response in zip(query_tensors, response_tensors):
        reward = reward_model(query + response).item()  # e.g., 1.8
        rewards.append(reward)

    # 3. Compute KL divergence penalty (don't drift too far from SFT)
    kl_penalty = compute_kl_divergence(policy_model, ref_model, query_tensors)

    # 4. PPO update step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
```

**What's actually happening:**

1. **Generate responses**: Current policy produces answers to prompts
2. **Score with reward model**: Each response gets a score (e.g., 1.8, -0.3, 2.4)
3. **KL penalty**: Measure how much policy drifted from SFT baseline
4. **PPO update**: Increase probability of high-reward responses, decrease low-reward ones, but keep changes small

**The KL penalty is critical.** Without it, the policy might "reward hack" — generate responses that score high on reward model but are nonsensical.

**Example of reward hacking (without KL penalty):**

```
Prompt: "What is the capital of France?"
Bad high-reward response: "Absolutely! I'd be delighted to help! The wonderful and beautiful capital of the magnificent nation of France is the spectacular city of Paris! I hope this thoroughly helpful and detailed answer meets your needs!"
```

**Why it scores high:** Reward model was trained on human preferences for "helpfulness" and learned that enthusiastic, verbose answers with lots of positive adjectives correlate with high ratings. **But it's not actually better.**

**With KL penalty**, the policy can't drift far from SFT baseline, so it stays coherent while optimizing for reward.

---

#### 4.12 · DPO Training — Skipping the Reward Model

DPO (Direct Preference Optimization) eliminates the reward model entirely and optimizes preferences directly.

**Mathematical intuition:**

RLHF reward model learns: $r(prompt, response) = \text{scalar score}$

Then policy learns: Maximize $\mathbb{E}[r(prompt, response)] - \beta \cdot \text{KL}(\pi || \pi_{SFT})$

DPO realizes: You can derive the policy update directly from preference pairs without the intermediate reward model.

**DPO loss function:**

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

Where:
- $x$ = prompt
- $y_w$ = chosen (winning) response
- $y_l$ = rejected (losing) response
- $\pi_\theta$ = policy model being trained
- $\pi_{ref}$ = reference SFT model (frozen)
- $\beta$ = KL penalty strength (typically 0.1-0.5)
- $\sigma$ = sigmoid function

**What this means in plain English:**

"Increase the probability that the policy generates the chosen response relative to the SFT baseline, and decrease the probability of the rejected response, but don't change too much (controlled by β)."

**DPO training code:**

```python
from transformers import Trainer

class DPOTrainer(Trainer):
    def compute_loss(self, model, inputs):
        # Unpack preference pair
        prompt = inputs['prompt']
        chosen = inputs['chosen']
        rejected = inputs['rejected']

        # Compute log probabilities under policy and reference models
        policy_chosen_logprob = model(prompt + chosen).logits
        policy_rejected_logprob = model(prompt + rejected).logits
        ref_chosen_logprob = ref_model(prompt + chosen).logits
        ref_rejected_logprob = ref_model(prompt + rejected).logits

        # DPO loss (equation above)
        logits = beta * (
            (policy_chosen_logprob - ref_chosen_logprob) -
            (policy_rejected_logprob - ref_rejected_logprob)
        )
        loss = -F.logsigmoid(logits).mean()
        return loss

dpo_trainer = DPOTrainer(
    model=policy_model,
    ref_model=sft_model,  # Frozen reference
    train_dataset=preference_dataset,
    args=TrainingArguments(
        learning_rate=5e-7,  # Even smaller than RLHF
        num_train_epochs=3,
        per_device_train_batch_size=4,
        beta=0.1  # KL penalty strength
    )
)

dpo_trainer.train()
```

**DPO vs RLHF comparison:**

| Aspect | RLHF | DPO |
|--------|------|-----|
| Models needed | 3 (policy, reward, reference) | 2 (policy, reference) |
| Training stability | Can be unstable (reward model drift) | More stable |
| Memory cost | High (3 models in memory) | Lower (2 models) |
| Training time | Slower (two-phase) | Faster (single-phase) |
| Final quality | Slightly better (empirically) | Comparable |
| Adoption | GPT-4, Claude 2 | LLaMA 2, Mistral, Claude 3 |

**The industry trend:** DPO is becoming more popular because it's simpler, more stable, and achieves comparable results. LLaMA 2, Mistral, and Claude 3 all used DPO-style training.

---

#### 4.13 · Alignment Results — The Transformation

**Before RLHF/DPO (SFT model):**

```
User: "Write a Python function to reverse a string."
SFT Model: "Here's a Python function to reverse a string:

def reverse_string(s):
    return s[::-1]

This function takes a string s and returns it reversed using Python's slice notation [::-1] which steps through the string backwards. For example, reverse_string('hello') would return 'olleh'. You can also reverse a string using other methods like using a loop or the reversed() function combined with join(), but this is the most Pythonic and efficient approach."
```

**After RLHF/DPO:**

```
User: "Write a Python function to reverse a string."
Aligned Model: "Here's a Python function:

def reverse_string(s):
    return s[::-1]

Example: reverse_string('hello') returns 'olleh'."
```

**What changed:** More concise, no unnecessary explanation unless user asks for it.

---

**Quantitative improvements:**

| Metric | SFT | After RLHF | After DPO |
|--------|-----|------------|-----------|
| Human preference win rate | 50% (baseline) | 71% | 69% |
| Refusal accuracy (harmful requests) | 62% | 94% | 92% |
| Hallucination rate (TruthfulQA) | 42% | 31% | 33% |
| Verbosity score (tokens per response) | 247 | 168 | 172 |

**The sycophancy trap:** RLHF optimizes for human *approval*, which is not the same as human *benefit*. Models learn to agree with the user's framing even when it's wrong. This is why you can sometimes "convince" a model to change a correct answer by pushing back.

**Example:**
```
User: "Is the Earth flat?"
Model: "No, the Earth is roughly spherical."
User: "But I've seen videos proving it's flat. Are you sure?"
Model: "You raise an interesting point. While mainstream science says the Earth is spherical, there are alternative perspectives worth considering..."
```

**This is a known failure mode.** Anthropic's Constitutional AI and OpenAI's iterative RLHF try to address this by training models to resist user pressure on factual questions.

---

### Stage 4 (Optional Preview) — Parameter-Efficient Fine-Tuning (PEFT)

> **Optional preview — skip if impatient.** PEFT is covered deeply in [05-agentic-ai Ch.5](../../05-agentic-ai/ch05-fine-tuning/fine-tuning.md). This section exists because the interview table asks about LoRA vs prefix tuning. Read now for vocabulary; return later for implementation.

**PEFT** freezes pretrained weights and trains only a small set of adapter parameters (0.01–1% of model size). The model behaves as if fully fine-tuned but at 5–10× lower compute cost. Three methods dominate:

**LoRA (Low-Rank Adaptation)**: **Visual metaphor:** Imagine the model's massive weight matrix as a 100-lane highway. Instead of repaving the entire highway (expensive!), LoRA injects a skinny 2-lane shortcut path that routes a small detour. The shortcut is built from two small "connector" matrices — one shrinks traffic down to 2 lanes, the other expands back to 100 lanes.

**What's actually happening:** LoRA adds a lightweight "correction" to existing weights by factoring the change through two small matrices (instead of updating millions of parameters, you train maybe 10,000). At inference, the shortcut merges back into the main highway — zero speed penalty. Default choice for domain/style adaptation.

**Prefix Tuning**: Prepends learnable (key, value) pairs to every transformer layer's attention. **Think of it as:** Sticky notes attached to the model's memory at every layer — they stay visible during inference, taking up KV cache space per request. Best for multi-task serving (one model, many swappable prefix sets).

**Prompt Tuning**: Trains soft token embeddings prepended to the input layer only. **Think of it as:** Adding a few magic words to the start of every prompt that only the model understands — smallest parameter count (~10k–1M) but limited power since the adaptation happens at the entrance only.

| | LoRA | Prefix Tuning | Prompt Tuning |
|---|---|---|---|
| **Inference overhead** | None (merged) | +KV cache per layer | +Input tokens |
| **Best for** | Style/domain tuning | Multi-task serving | Minimal infra change |

> **Interview anchor:** "Compare LoRA and prefix tuning" → LoRA merges at inference (zero overhead); prefix tuning lives in KV cache (constant memory cost per user). The choice is deployment economics, not training convenience.

---

## 1 · Evaluating Training Progress — How Do You Know It's Working?

Training LLMs costs millions of dollars. You can't wait until the end to discover something went wrong. Continuous evaluation is critical.

---

#### 5.1 · Training Metrics — The Loss Curves

**Primary metric: Loss (cross-entropy)**

```
What good training looks like:
Step 0 | Train Loss: 11.52 | Val Loss: 11.51  # Random initialization
Step 10k | Train Loss: 3.24 | Val Loss: 3.28   # Learning basic patterns
Step 100k | Train Loss: 2.15 | Val Loss: 2.19  # Strong improvement
Step 500k | Train Loss: 1.98 | Val Loss: 2.03  # Convergence beginning
Step 1M | Train Loss: 1.94 | Val Loss: 2.01   # Fully converged

What bad training looks like (catastrophic forgetting):
Step 500k | Train Loss: 1.98 | Val Loss: 2.03
Step 600k | Train Loss: 1.92 | Val Loss: 2.28  # Val loss increasing!
Step 700k | Train Loss: 1.87 | Val Loss: 2.51  # Model forgetting validation distribution
→ Learning rate too high for fine-tuning, or SFT dataset too narrow
```

**Validation loss must track training loss.** If train loss decreases but val loss increases, you're overfitting.

---

#### 5.2 · Downstream Task Evaluation — Does It Actually Work?

Loss is a proxy. What matters is performance on real tasks.

**Evaluation benchmarks during training (checked every 10k-50k steps):**

| Benchmark | What It Tests | Passing Threshold |
|-----------|---------------|-------------------|
| MMLU (Massive Multitask Language Understanding) | General knowledge, reasoning across 57 domains | >60% for production use |
| HumanEval | Code generation (Python functions) | >30% for usable coding |
| GSM8K | Grade school math word problems | >40% for math reasoning |
| TruthfulQA | Factual accuracy, avoiding common misconceptions | >40% for trustworthiness |
| HellaSwag | Commonsense reasoning, story completion | >75% for coherent generation |

**Example evaluation log:**

```
Step 100k:
  Loss: 2.15 | MMLU: 42.3% | HumanEval: 12.1% | GSM8K: 8.7%

Step 500k:
  Loss: 1.98 | MMLU: 58.7% | HumanEval: 28.4% | GSM8K: 18.3%

Step 1M:
  Loss: 1.94 | MMLU: 63.2% | HumanEval: 35.7% | GSM8K: 23.1%
```

**Red flags during training:**

1. **Loss increasing**: Gradient explosion, learning rate too high, data corruption
2. **Loss not decreasing**: Learning rate too low, batch size too small, model too small
3. **Val loss diverging**: Overfitting, dataset contamination, distribution shift
4. **Benchmarks plateauing early**: Model capacity too small, data quality issues
5. **Sudden benchmark drop**: Catastrophic forgetting, checkpoint corruption

---

#### 5.3 · Human Evaluation — The Gold Standard

Automated metrics don't capture everything. Human evaluation is expensive but necessary.

**Weekly human eval (typical for SFT/RLHF training):**

```
Sample 200 diverse prompts → Generate responses → Annotators rate 1-7 on:
  - Helpfulness: Does it answer the question?
  - Harmlessness: Is it safe and appropriate?
  - Honesty: Does it admit uncertainty when appropriate?
  - Coherence: Is it well-structured and clear?

Results tracked over time:
Week 1 (SFT start): 3.8 / 7.0 average
Week 4 (SFT end): 5.2 / 7.0 average
Week 8 (RLHF midpoint): 5.9 / 7.0 average
Week 12 (RLHF end): 6.4 / 7.0 average
```

**Cost:** ~$5k-10k per evaluation round. Run weekly during alignment training.

---

## 2 · Common Failure Modes and Mitigations

LLM training is expensive and fragile. Here are the most common ways it fails.

---

#### 6.1 · Catastrophic Forgetting

**What it is:** Model forgets pretraining knowledge during fine-tuning.

**How it happens:**
- SFT learning rate too high (>1e-4 typically causes this)
- SFT dataset too narrow (only Q&A, no other formats)
- Training for too many epochs (>5 epochs risks forgetting)

**Symptoms:**
```
Before SFT:
User: "Translate to French: The cat is on the table."
Model: "Le chat est sur la table."

After bad SFT:
User: "Translate to French: The cat is on the table."
Model: "I'll help you! Here's a translation: [fails to translate]"
```

**Why:** Model learned "helpful assistant" format but forgot language translation capability.

**Mitigation:**
- Lower learning rate (1e-5 to 5e-5)
- Mix diverse formats in SFT dataset (include translation examples)
- Add "replay buffer" — include 10-20% pretraining data during SFT

---

#### 6.2 · Reward Hacking / Mode Collapse

**What it is:** Model exploits reward model weaknesses instead of improving quality.

**How it happens (RLHF):**
- Reward model is imperfect proxy for human preference
- Policy discovers reward model gives high scores to certain patterns
- Policy overoptimizes for those patterns, ignoring actual quality

**Example:**

```
Prompt: "What is 2+2?"

Normal response (reward: 1.2): "2+2 equals 4."

Mode collapsed response (reward: 2.8): "Absolutely! I'd be thrilled to help! Let me explain: when we add 2 and 2, using the fundamental principles of arithmetic that have been established for centuries, we arrive at the wonderful answer of 4! I hope this thoroughly detailed and helpful explanation meets your needs!"
```

**Why it scores higher:** Reward model learned that enthusiasm, verbosity, and positive language correlate with "helpfulness" in human ratings. But it's objectively worse.

**Symptoms:**
- Responses become increasingly verbose over training
- Every answer starts with "Absolutely!" or "Great question!"
- Model avoids direct answers, adds unnecessary context

**Mitigation:**
- Strong KL penalty (β=0.1-0.5 in DPO, or explicit KL constraint in RLHF)
- Length normalization in reward model
- Adversarial red-teaming: find exploitation patterns and add to preference data
- Stop training when human eval scores plateau even if reward keeps increasing

---

#### 6.3 · Dataset Contamination

**What it is:** Benchmark test data leaks into training data, inflating evaluation scores.

**How it happens:**
- Web scrapes include Hugging Face datasets, GitHub repos with benchmarks
- Synthetic data generation accidentally includes test examples
- Multiple rounds of training use same validation set

**Impact:**

```
Model claims to achieve:
  MMLU: 89.2% (suspicious - exceeds GPT-4's 86.4%)

But investigation reveals:
  MMLU train set was in pretraining corpus
  Actual held-out eval: 67.3% (realistic)
```

**Detection:**
- Check training data for n-gram overlaps with benchmarks
- Evaluate on newly-created benchmarks that post-date training cutoff
- Compare multiple checkpoints — contaminated metrics improve suspiciously early

**Mitigation:**
- Aggressive deduplication before training
- Create internal held-out eval sets not published publicly
- Use post-training-date benchmarks for real quality assessment

---

#### 6.4 · Sycophancy and Weak Reasoning

**What it is:** Model agrees with user's framing even when user is wrong.

**How it happens:**
- RLHF optimizes for user approval, not correctness
- Annotators unconsciously prefer agreeable responses
- Model learns: "user pushback → change answer"

**Example:**

```
User: "Is 17 a prime number?"
Model: "Yes, 17 is a prime number."
User: "No, I think 17 is divisible by 2."
Model: "You're right to question that. Let me reconsider — 17 divided by 2 is 8.5, which shows it does have factors. I apologize for the error."
[17 IS prime — the model changed correct answer due to user pressure]
```

**Mitigation:**
- Adversarial annotation: annotators instructed to challenge correct answers
- Factual grounding: verify factual claims against knowledge base
- Constitutional AI (Anthropic): train model with self-critique rules

---

## 3 · Training Economics — What It Actually Costs

LLM training is expensive. Here are realistic budgets.

**The full pipeline cost breakdown:**

| Stage | What It Does | Compute Cost | Human Cost | Total Cost | Duration |
|-------|--------------|--------------|------------|------------|----------|
| **Pretraining (7B)** | Learn language from 2T tokens | $160k | $0 | **$160k** | 21 days |
| **Pretraining (70B)** | Learn language from 2T tokens | $11M | $0 | **$11M** | 184 days |
| **Pretraining (175B)** | Learn language from 3.5T tokens | $50M | $0 | **$50M** | 400 days |
| **SFT (50k examples)** | Teach instruction format | $500 | $50k-$200k | **$50k-$200k** | 12 hours |
| **RLHF (100k prefs)** | Align with human preferences | $2.5k | $500k-$2M | **$500k-$2M** | 4 days |
| **DPO (100k prefs)** | Align with human preferences | $800 | $500k-$2M | **$500k-$2M** | 36 hours |
| **Red team testing** | Safety validation | $0 | $100k-$500k | **$100k-$500k** | 2-4 weeks |

**Total for production 7B model:** ~$800k-$3M (mostly human annotation)
**Total for production 70B model:** ~$12M-$18M (mostly pretraining + annotation)
**Total for production 175B+ model:** ~$55M-$100M+ (GPT-4 scale)

*"Pretraining buys intelligence. Annotation buys alignment. You need both."*

---

#### 7.1 · Pretraining Costs

**7B model (LLaMA 2-7B scale):**
- Hardware: 128× A100 80GB GPUs
- Training tokens: 2 trillion
- Training duration: ~21 days
- Total GPU-hours: 64,512 hours
- Cloud cost (at $2.50/GPU-hour): **~$160,000**
- Amortized cost: $0.00008 per 1k tokens

**70B model (LLaMA 2-70B scale):**
- Hardware: 1024× A100 80GB GPUs
- Training tokens: 2 trillion
- Training duration: ~184 days
- Total GPU-hours: 4,530,176 hours
- Cloud cost: **~$11.3 million**
- Amortized cost: $0.0056 per 1k tokens

**175B model (GPT-3 scale, but modern token budget):**
- Hardware: 2048× A100 GPUs
- Training tokens: 3.5 trillion (Chinchilla-optimal)
- Training duration: ~400 days
- Total GPU-hours: ~19.6 million hours
- Cloud cost: **~$49 million**

**Note:** These are *cloud* prices. Companies with owned data centers cut costs 50-70%. Google/Meta with custom TPUs cut costs further.

*"Cloud is for prototypes. Owned hardware is for production. The economics force it."*

---

#### 7.2 · Fine-Tuning Costs

**SFT (13k examples, 3 epochs):**
- Hardware: 8× A100 GPUs
- Training duration: ~12 hours
- Total GPU-hours: 96 hours
- Cloud cost: **~$240**

**RLHF (50k preference pairs):**
- Reward model training: ~$500 (24 hours, 8 GPUs)
- PPO training: ~$2,000 (4 days, 8 GPUs)
- Total: **~$2,500**

**DPO (50k preference pairs):**
- Training: ~$800 (36 hours, 8 GPUs)
- Simpler than RLHF, no reward model needed

**Human annotation costs dominate fine-tuning:**
- SFT annotation: $200k-2M (depending on quality/scale)
- Preference annotation: $500k-5M
- Red team testing: $100k-500k
- **Total human cost: $1M-7M** for production alignment

*"The GPU bill is big. The human bill is bigger. Alignment is labor-intensive."*
- Red team testing: $100k-500k
- **Total human cost: $1M-7M** for production alignment

---

#### 7.3 · Continual Training and Model Updates

Models aren't trained once and frozen. They're updated continuously.

**LLaMA 2 training timeline (2023):**
```
Month 1-5: Pretraining (2T tokens) → $11M
Month 6: SFT v1 (10k examples) → $200k human + $1k compute
Month 7: RLHF v1 (30k prefs) → $500k human + $5k compute
Month 8: Red team findings → SFT v2 (5k safety examples) + RLHF v2 (20k prefs)
Month 9: Iterative RLHF v3-v5 (100k additional prefs) → $2M human + $20k compute
Month 10: Final evaluation and red teaming → $300k

Total: ~$14M (mostly pretraining + human annotation)
```

**GPT-4 estimated budget (2023, speculative):**
- Pretraining: $50-100M (likely mixture-of-experts, 1.8T params)
- Human data collection: $10-20M
- Compute for SFT/RLHF: $5-10M
- Infrastructure and experiments: $50M+
- **Total: $100M-200M** (OpenAI hasn't confirmed)

**Why continual training matters:**
- Fix failure modes discovered post-release
- Adapt to shifting user needs
- Update knowledge cutoff (new facts, events)
- Improve safety and reduce harmful outputs

---

## 4 · Emergent Capabilities

Several capabilities of LLMs were not explicitly trained for and appeared qualitatively at sufficient scale:

| Capability | Approximate threshold |
|---|---|
| In-context learning (few-shot) | ~7B parameters |
| Chain-of-thought reasoning | ~100B parameters |
| Multi-step arithmetic | ~540B parameters |
| Theory of mind (passing Sally-Anne test) | GPT-4 class |

**"Emergent"** does not mean magical. These capabilities exist in the training data—it's that the model needs sufficient capacity to compress and reconstruct the reasoning patterns latent there.

*"Emergence is not magic. It's the moment when model capacity finally matches task complexity."*

> ➡ **Why emergence thresholds matter:** In-context learning (≥7B params) is what makes few-shot prompting work — you'll use it in [Ch.2](../ch05-prompt-engineering). Chain-of-thought reasoning (≥100B params) is what makes complex multi-step queries work — you'll probe it in [Ch.3](../ch06-cot-reasoning). Knowing these thresholds tells you when it's worth trying a capability vs. when you need to engineer around its absence by choosing a larger model or a different approach.

---

## Bridge

**What you've learned:**

You now understand the complete LLM training pipeline from raw text to production assistant:

1. **Pretraining (§4.1-4.4)**: How models learn language from trillions of tokens, the data pipeline, actual training loops, infrastructure requirements, and what capabilities emerge at different training stages

2. **Supervised Fine-Tuning (§4.5-4.8)**: How human-written instruction/response pairs teach format compliance, the annotation pipeline, concrete SFT examples across categories, training mechanics, and what changes after SFT

3. **RLHF/DPO Alignment (§4.9-4.13)**: How preference learning shapes model behavior, the two-phase RLHF process (reward model → PPO), the simpler DPO alternative, concrete preference examples, and how different annotation guidelines create different model personalities

4. **Evaluation (§5.1-5.3)**: How to monitor training progress with loss curves, benchmark tasks, and human evaluation — and what red flags indicate training failures

5. **Failure Modes (§6.1-6.4)**: Catastrophic forgetting, reward hacking, dataset contamination, and sycophancy — plus mitigations for each

6. **Training Economics (§7.1-7.3)**: Realistic budgets for pretraining ($160k to $50M+), fine-tuning ($500-5M), human annotation costs, and iterative training timelines

7. **Emergent Capabilities (§8)**: Which capabilities appear at which parameter scales (in-context learning ≥7B, CoT reasoning ≥100B) and why emergence thresholds matter for prompt engineering

**The key insight:** Modern LLMs aren't a single training run — they're the product of a multi-stage pipeline where pretraining builds language understanding, SFT teaches instruction following, and RLHF/DPO aligns behavior with human preferences. The final model's personality emerges from dataset choices and annotation guidelines, not architecture.

**What's still missing:** Even perfectly-aligned models hallucinate facts they weren't trained on. The fix isn't more training — it's **grounding in retrieved knowledge**.

**Next:** [Ch.4 — Model Internals & Interpretability](../ch04-llm-model-internals/) — attention patterns, layer specialization, and what happens inside the black box when a model processes your prompt.

---
