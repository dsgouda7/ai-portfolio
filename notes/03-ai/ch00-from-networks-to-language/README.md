# Ch.0 — Prerequisites: From Neural Networks to Transformers

> **Reading order:** Read this chapter BEFORE Ch.1 if you need a refresher on RNNs, attention, embeddings, or skip connections. If you've recently completed notes/01 (ML Track) Ch.6, Ch.9, Ch.10 and notes/02 (Advanced Deep Learning) Ch.1, you can skip directly to Ch.1.
>
> **What this chapter does:** Condenses the specific concepts from notes/01-02 that Ch.1 assumes you know. Every section includes worked examples with actual numbers—no hand-waving.

> **Where you are:** You've learned machine learning fundamentals (notes/01) and advanced architectures (notes/02). This chapter bridges those tracks to the LLM content in notes/03. It extracts only the concepts Ch.1 references without explaining, teaching them with minimal examples.

> **Notation:** $h_t$ = hidden state at time $t$; $x_t$ = input at time $t$; $W$ = weight matrix; $Q, K, V$ = query, key, value matrices; $\alpha$ = attention weights; $d_k$ = key/query dimension.

---

## 0 · Why This Chapter Exists

**The gap:** Ch.1 (Transformer Architecture) opens with statements like:

- "RNNs process sequences one token at a time" (assumes you know what hidden states are)
- "Vanishing gradients beyond 100-200 tokens" (assumes you've seen exponential gradient decay)
- "Attention is a soft dictionary lookup with Q/K/V" (assumes you understand dot-product similarity)
- "Skip connections prevent gradient vanishing" (assumes you know the "+1" gradient term)
- "Embeddings as vector representations" (assumes you know why we need vectors for text)

If those statements make complete sense, **skip this chapter**. If any are fuzzy, **this chapter builds the intuition** Ch.1 assumes.

**What you'll learn:**
- §1: Why text becomes vectors (embeddings), what dimensions mean
- §2: How RNNs process sequences, why gradients vanish (with 0.8^T calculation)
- §3: Attention mechanism with a 3-token numeric walkthrough
- §4: Skip connections and the "+1" gradient highway
- §5: Encoder-decoder architectural pattern
- §6: What "training" means (loss, backprop, parameters)

**Time investment:** 20-30 minutes if skimming worked examples, 45-60 minutes if working through calculations.

---

## 1 · Embeddings: Why Text Becomes Vectors

### The Problem

Transformers operate on **vector representations** of tokens, not raw text. Ch.1 jumps straight into formulas like $Q = X W_Q$ where $X$ is an embedding matrix. Why vectors?

**Answer:** Math operations (dot products, matrix multiplication, gradient descent) require numbers. "cat" is a string—we need a numeric representation that preserves meaning.

### What is an Embedding?

An **embedding** maps each token to a point in high-dimensional space where **distance = similarity**.

**Example:** 3-dimensional embedding space (real embeddings are 512-4096 dims):
```
"cat"   → [0.8, 0.9, 0.1]
"dog"   → [0.7, 0.8, 0.2]  # Close to "cat" (both animals)
"car"   → [0.1, 0.2, 0.9]  # Far from "cat" (different concept)
```

**Dot product measures similarity:**
```
cat · dog = 0.8×0.7 + 0.9×0.8 + 0.1×0.2 = 0.56 + 0.72 + 0.02 = 1.30 (high)
cat · car = 0.8×0.1 + 0.9×0.2 + 0.1×0.9 = 0.08 + 0.18 + 0.09 = 0.35 (low)
```

Higher dot product → more similar meaning.

### Where Embeddings Come From

In transformers:
1. **Token ID:** "cat" → token 4517 (from vocabulary)
2. **Lookup:** Row 4517 of embedding matrix $E$ (shape: [vocab_size, d_model])
3. **Vector:** Returns $d_{model}$-dimensional vector (e.g., 768-d for BERT)

**These embeddings are learned during training**—the model adjusts them so semantically similar words end up nearby in vector space.

### Why This Matters for Ch.1

- **Attention operates on embeddings:** Q/K/V matrices project these vectors
- **"Sequence of vectors" = embeddings:** When Ch.1 says "(n, d_model)", it means n token embeddings
- **Dimensionality reasoning:** Why d_model=768? Enough dimensions to represent rich semantics without wasting memory

> **Checkpoint:** Can you explain why we can't just use one-hot encoding (binary vectors with single 1)? 
>
> *Answer: One-hot vectors have no semantic structure. "cat" = [0,0,1,0,0] and "dog" = [0,1,0,0,0] have dot product 0 (orthogonal) despite being semantically similar. Embeddings learn similarity relationships.*

---

## 2 · Sequential Models: Why RNNs Failed at Scale

### 2.1 · How RNNs Work

A **Recurrent Neural Network** processes sequences one token at a time, maintaining a **hidden state** that threads information forward:

$$
h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)
$$

- $h_t$: Hidden state at step $t$ (e.g., 128-d vector)
- $x_t$: Current input embedding
- $h_{t-1}$: Previous hidden state (memory from earlier tokens)

**Concrete example (2-D hidden state for simplicity):**

```python
# Sentence: "The cat sat"
# Embeddings: x₁=[1.0, 0.5], x₂=[0.8, 0.9], x₃=[0.3, 1.2]

W_hh = [[0.5, 0.2],   W_xh = [[0.3, 0.1],
        [0.1, 0.4]]           [0.2, 0.5]]

# Step 1: Process "The"
h₁ = tanh(W_hh @ h₀ + W_xh @ x₁)
   = tanh([[0,0]] + [[0.3×1.0 + 0.1×0.5], [0.2×1.0 + 0.5×0.5]])
   = tanh([0.35, 0.45]) = [0.336, 0.422]

# Step 2: Process "cat" (uses h₁)
h₂ = tanh(W_hh @ h₁ + W_xh @ x₂)
   = tanh([[0.5×0.336 + 0.2×0.422], [0.1×0.336 + 0.4×0.422]] + [[0.3×0.8 + 0.1×0.9], [0.2×0.8 + 0.5×0.9]])
   = tanh([0.252, 0.202] + [0.33, 0.61]) = tanh([0.582, 0.812]) = [0.524, 0.671]

# Step 3: Process "sat" (uses h₂)
h₃ = tanh(W_hh @ h₂ + W_xh @ x₃)
   = [final hidden state encoding entire sentence]
```

**Key insight:** Each step depends on the previous one—**cannot parallelize**.

### 2.2 · The Vanishing Gradient Problem

**Why deep RNNs fail:** Gradients decay exponentially when backpropagating through time.

**From notes/01 Ch.6, explicit calculation:**

Suppose each time step multiplies the gradient by 0.8 (typical for tanh activation). After $T$ steps:

| Steps (T) | Gradient Magnitude | % of Original |
|-----------|-------------------|---------------|
| 1         | 0.8               | 80%           |
| 5         | 0.8^5 = 0.33      | 33%           |
| 10        | 0.8^10 = 0.11     | 11%           |
| 20        | 0.8^20 = 0.012    | 1.2%          |
| 50        | 0.8^50 = 0.000014 | 0.0014%       |

**After 50 tokens, gradients are essentially zero**—early tokens never learn.

**LSTMs help but don't solve:**
- Gates control information flow (forget gate, input gate)
- Reduces vanishing but still serial (token $t$ waits for $t-1$)
- Training 1.5B-parameter LSTM on 40GB text: **months** even on 256 GPUs

### 2.3 · Why This Motivated Transformers

Two fatal flaws:
1. **No parallelization:** GPU must process tokens sequentially
2. **Gradient vanishing:** Information from token 1 doesn't reach token 100

**Transformer solution preview:**
- All tokens process simultaneously (parallelization)
- Direct attention paths (no gradient decay over distance)

> **Checkpoint:** Calculate gradient magnitude after 100 time steps if each step multiplies by 0.9.
>
> *Answer: 0.9^100 ≈ 0.000027 (0.0027% of original) — effectively vanished.*

---

## 3 · Attention: The Core Mechanism

### 3.1 · The Big Idea

**From notes/01 Ch.9:** Attention is a **soft dictionary lookup**.

- **Query (Q):** "What am I looking for?"
- **Key (K):** "What do I offer?"
- **Value (V):** "What information do I carry?"

**Process:**
1. Compute similarity: Q · K (dot product)
2. Normalize: softmax → probability distribution
3. Retrieve: weighted sum of V

### 3.2 · Worked Example: 3-Token Attention

**Sentence:** "The river bank"

**Embeddings (2-D for simplicity):**
```
x₁ = "The"   = [1.0, 0.2]
x₂ = "river" = [0.8, 0.9]
x₃ = "bank"  = [0.5, 1.1]
```

**Step 1: Create Q, K, V**

For simplicity, use identity projection (Q=K=V=X):
```
Q = K = V = [[1.0, 0.2],
             [0.8, 0.9],
             [0.5, 1.1]]
```

**Step 2: Compute attention scores (Q·K^T)**

```
Scores = Q · K^T

Row 1 (query="The"):
  score₁₁ = [1.0, 0.2] · [1.0, 0.2] = 1.0×1.0 + 0.2×0.2 = 1.04
  score₁₂ = [1.0, 0.2] · [0.8, 0.9] = 1.0×0.8 + 0.2×0.9 = 0.98
  score₁₃ = [1.0, 0.2] · [0.5, 1.1] = 1.0×0.5 + 0.2×1.1 = 0.72

Row 2 (query="river"):
  score₂₁ = [0.8, 0.9] · [1.0, 0.2] = 0.8×1.0 + 0.9×0.2 = 0.98
  score₂₂ = [0.8, 0.9] · [0.8, 0.9] = 0.8×0.8 + 0.9×0.9 = 1.45
  score₂₃ = [0.8, 0.9] · [0.5, 1.1] = 0.8×0.5 + 0.9×1.1 = 1.39

Row 3 (query="bank"):
  score₃₁ = [0.5, 1.1] · [1.0, 0.2] = 0.5×1.0 + 1.1×0.2 = 0.72
  score₃₂ = [0.5, 1.1] · [0.8, 0.9] = 0.5×0.8 + 1.1×0.9 = 1.39
  score₃₃ = [0.5, 1.1] · [0.5, 1.1] = 0.5×0.5 + 1.1×1.1 = 1.46

Score matrix:
      The    river  bank
The  [1.04   0.98   0.72]
river[0.98   1.45   1.39]
bank [0.72   1.39   1.46]
```

**Step 3: Apply softmax (row-wise)**

```
Row 1 (The):
  exp(1.04) = 2.83, exp(0.98) = 2.66, exp(0.72) = 2.05
  sum = 7.54
  α₁ = [2.83/7.54, 2.66/7.54, 2.05/7.54] = [0.375, 0.353, 0.272]

Row 2 (river):
  exp(1.45) = 4.26, exp(0.98) = 2.66, exp(1.39) = 4.01
  sum = 10.93
  α₂ = [0.244, 0.390, 0.367]

Row 3 (bank):
  exp(0.72) = 2.05, exp(1.39) = 4.01, exp(1.46) = 4.31
  sum = 10.37
  α₃ = [0.198, 0.387, 0.416]
```

**Step 4: Weighted sum of values**

```
Output for "bank" (row 3):
  out₃ = 0.198×[1.0, 0.2] + 0.387×[0.8, 0.9] + 0.416×[0.5, 1.1]
       = [0.198, 0.040] + [0.310, 0.348] + [0.208, 0.458]
       = [0.716, 0.846]
```

**Interpretation:** "bank" attended most strongly to itself (41.6%) and "river" (38.7%), incorporating contextual information that disambiguates it as a geographic feature.

### 3.3 · Why Scaled Dot-Product?

**The formula in Ch.1:**
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

**Why divide by $\sqrt{d_k}$?**

Without scaling, dot products grow with dimension. For $d_k=64$ (typical):
- Unscaled scores might be [82, 15, -38] → softmax → [0.9998, 0.0002, 0]
- Softmax saturates (all weight on one token)
- Gradients vanish

**With scaling:**
- Divide by $\sqrt{64}=8$ → [10.25, 1.875, -4.75]
- Softmax → [0.87, 0.09, 0.01] (balanced distribution)
- Gradients flow cleanly

> **Checkpoint:** In the 3-token example, if we used 64-D embeddings instead of 2-D, how would scores change?
>
> *Answer: Dot products would be ~8× larger (sum of 64 products vs 2). Without √d_k scaling, softmax would collapse to near-one-hot (gradient issues).*

---

## 4 · Skip Connections: The Gradient Highway

### 4.1 · The ResNet Insight

**From notes/02 Ch.1:** Residual connections solve vanishing gradients by adding an identity path.

**Standard network:**
$$
y = F(x)
$$

**Residual network:**
$$
y = F(x) + x
$$

Where $F(x)$ is a non-linear transformation (e.g., two Conv layers in ResNets, attention block in transformers).

### 4.2 · Why This Works: The "+1" Gradient Term

**Backward pass through residual connection:**

$$
\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial x}
$$

Since $y = F(x) + x$:

$$
\frac{\partial y}{\partial x} = \frac{\partial F(x)}{\partial x} + \frac{\partial x}{\partial x} = \frac{\partial F(x)}{\partial x} + 1
$$

Therefore:

$$
\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \left( \frac{\partial F(x)}{\partial x} + 1 \right)
$$

**The "+1" term is the gradient highway:**
- Even if $\frac{\partial F(x)}{\partial x} \to 0$ (transformation branch saturates), gradient still flows via the "+1"
- Through 50 residual blocks: gradient = $\frac{\partial \mathcal{L}}{\partial y_{50}} \times (1 + \epsilon)^{50}$ where $\epsilon$ is small
- Compare to plain network: $\frac{\partial \mathcal{L}}{\partial y_{50}} \times 0.9^{50} \approx 0$

### 4.3 · Transformers Use Skip Connections Everywhere

**Every transformer block has TWO residual connections:**

```
x_1 = x + MultiHeadAttention(x)  # Skip around attention
x_2 = x_1 + FeedForward(x_1)     # Skip around FFN
```

This is why **96-layer GPT-3** can train—gradients flow directly through all layers via the addition operations.

**Numerical comparison (from notes/02 Ch.1):**

| Architecture | Layers | Gradient at Layer 1 |
|--------------|--------|---------------------|
| Plain network| 40     | 2% of final layer   |
| ResNet-40    | 40     | 75% of final layer  |
| Transformer-96| 96    | 60-70% of final layer|

> **Checkpoint:** Why can't we just make learning rates smaller to compensate for vanishing gradients?
>
> *Answer: Vanishing happens in forward/backward pass, not learning rate. If gradient is 0.02 at layer 1, multiplying by any learning rate still gives tiny updates. Skip connections fix the gradient flow itself.*

---

## 5 · Encoder-Decoder Architecture

### 5.1 · The Pattern

**From notes/02 Ch.5 (U-Net) and notes/03 Ch.1:**

**Encoder:** Processes input sequence bidirectionally
- Reads full sentence
- Builds contextualized representations
- Examples: BERT, vision ResNets

**Decoder:** Generates output sequence autoregressively
- Produces one token at a time
- Can only see previous outputs (causal masking)
- Examples: GPT, language model heads

**Encoder-Decoder:** Combines both
- Encoder processes source (e.g., English sentence)
- Decoder generates target (e.g., French translation)
- Cross-attention: decoder attends to encoder outputs
- Examples: T5, original Transformer (2017)

### 5.2 · Information Flow

```
┌─────────────────────────────────────┐
│ Encoder (bidirectional attention)  │
│ Token 1 ↔ Token 2 ↔ Token 3       │
│   ↕         ↕         ↕             │
│ All tokens see all tokens          │
└───────────────┬─────────────────────┘
                │ encoder outputs
                ↓
┌─────────────────────────────────────┐
│ Decoder (causal attention)          │
│ Token 1 → Token 2 → Token 3        │
│   (can only see previous tokens)   │
│                                     │
│ Cross-attention: attends to ↑      │
└─────────────────────────────────────┘
```

**Why different attention masks?**
- **Encoder:** Understanding tasks benefit from full context (classification, retrieval)
- **Decoder:** Generation requires causal masking (can't peek at future)

> **Checkpoint:** Why can't decoders use bidirectional attention?
>
> *Answer: During generation, future tokens don't exist yet. At step 5, we're predicting token 6—we can't "attend to" tokens 7-10 because they haven't been generated. Bidirectional attention would be peeking at the answer.*

---

## 6 · Training Foundations

### 6.1 · What are Parameters?

**Parameters = the numbers the model learns.**

In a dense layer: $y = Wx + b$
- $W$: weight matrix (e.g., 768×3072 = 2.4M parameters)
- $b$: bias vector (e.g., 3072 parameters)

**Total for GPT-3:** 175 billion parameters across all layers.

### 6.2 · Loss Functions

**Loss measures how wrong predictions are.**

**For language modeling (predicting next token):**

Cross-entropy loss (from notes/01 Ch.7, MLE framework):

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log P(\text{correct token}_i | \text{context}_i)
$$

**Example:** Predicting "cat" after "The"
- Model outputs probabilities: P(cat)=0.3, P(dog)=0.4, P(car)=0.1, ...
- True next word is "cat"
- Loss = $-\log(0.3) = 1.20$

**If model improves:** P(cat)=0.8
- Loss = $-\log(0.8) = 0.22$ (lower = better)

### 6.3 · Backpropagation (Conceptual)

**How parameters update:**

1. Forward pass: Compute predictions
2. Calculate loss: How wrong were we?
3. Backward pass: Compute gradients $\frac{\partial \mathcal{L}}{\partial W}$ for every parameter
4. Update: $W_{\text{new}} = W_{\text{old}} - \eta \frac{\partial \mathcal{L}}{\partial W}$ (where $\eta$ is learning rate)

**Chain rule propagates errors backward:**
- Layer 96 gradient → Layer 95 → ... → Layer 1
- Skip connections ensure gradients don't vanish (§4)

### 6.4 · Optimizers

**Adam (standard for transformers):**
- Adaptive learning rates per parameter
- Fast convergence
- Used in BERT, GPT, T5, nearly all modern LLMs

**You don't need the formula—just know:** Adam adjusts each parameter's learning rate based on gradient history (recent large gradients → smaller steps; recent small gradients → larger steps).

> **Checkpoint:** If a model has 7B parameters and trains on 1T tokens, roughly how many gradient updates happen?
>
> *Answer: Depends on batch size. With batch_size=1M tokens, that's 1T/1M = 1M updates. Each update adjusts all 7B parameters using their computed gradients.*

---

## 7 · Bridge to Ch.1

You now understand the foundations Ch.1 assumes:

**✓ Embeddings:** Why text → vectors, what dimensions mean  
**✓ RNNs:** Sequential processing, hidden states, vanishing gradients  
**✓ Attention:** Q/K/V, dot-product similarity, softmax, weighted sums  
**✓ Skip connections:** "+1" gradient term, residual learning  
**✓ Encoder-decoder:** Bidirectional vs causal, cross-attention  
**✓ Training:** Parameters, loss, backprop, optimizers  

**Ch.1 builds on this:**
- **§0:** Historical context (RNN → Transformer evolution)
- **§2:** Tokenization (BPE algorithm)
- **§2A:** Multi-head attention (parallel Q/K/V projections)
- **§2A:** Positional encoding (injecting sequence order)
- **§2B:** Three architectures (encoder-only, decoder-only, encoder-decoder)
- **§3:** Interview questions and production knowledge

**What changes:** Ch.1 has production details, code examples, and architectural comparisons. This chapter gave you the mechanical intuition—Ch.1 shows you how it scales to billion-parameter models.

---

## Summary

This chapter condensed **6 key concepts** from notes/01-02 that enable understanding transformers:

1. **Embeddings** (§1): Text as vectors, similarity via dot products
2. **RNNs & vanishing gradients** (§2): Why sequential models fail (0.8^T decay)
3. **Attention mechanism** (§3): Q/K/V with 3-token worked example
4. **Skip connections** (§4): "+1" gradient highway from ResNets
5. **Encoder-decoder pattern** (§5): Bidirectional vs causal attention
6. **Training basics** (§6): Parameters, cross-entropy loss, Adam optimizer

**Time to Ch.1:** You're ready. Every concept Ch.1 uses is now in your mental model.
