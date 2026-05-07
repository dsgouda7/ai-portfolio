# LLM Fundamentals — What a Language Model Actually Is

> **A brief history.** In the summer of 2017, a team of eight Google engineers published a twelve-page paper with a deliberately provocative title: *"Attention Is All You Need."* They weren't describing a self-help book — they were discarding the recurrent loops that every language model had relied on for a decade and replacing them with a single mechanism called *attention*. Almost nobody outside research noticed. The transformer, as the architecture came to be known, was faster to train, easier to parallelize, and — it turned out — almost infinitely scalable.
>
> What happened next unfolded faster than anyone predicted. Within a year, OpenAI took just the *decoder* half of that transformer and asked a deceptively simple question: what if we trained it on most of the internet and gave it one job — *predict the next word*? The result was **GPT-1** (2018), then **GPT-2** (2019), which was quietly good enough that OpenAI sat on the weights for months out of concern. By **GPT-3** in 2020 — 175 billion parameters trained on half a trillion words — something strange emerged: the model could solve tasks it had never been explicitly trained on, just from examples in the prompt. Researchers called it *in-context learning* and struggled to explain it.
>
> The final piece clicked into place with **InstructGPT** (2022): rather than asking the model to complete text, they trained it to *follow instructions* using human feedback. Wrap that in a chat interface and you get **ChatGPT** — 100 million users in two months, the fastest consumer product in history. Every model in this track, GPT-4, Claude, Gemini, Llama, DeepSeek, is running the same recipe at varying scale.
>
> **Where you are in the curriculum.** **Read this before anything else in the AI track.** Every later doc — [CoTReasoning](../ch03-cot-reasoning), [RAG](../ch04-rag-and-embeddings), [ReAct](../03b-agentic-ai/ch01-react-and-semantic-kernel) — assumes you know what an LLM is under the hood. This document builds that foundation from the transformer ([ML Ch.18](../../01-ml/03_neural_networks/ch10_transformers)) through to the models you call via API today: tokenisation, the pretraining → SFT → RLHF pipeline, sampling parameters, and context windows.
>
> **Notation used later in this doc.** $P(x_t \mid x_{<t})$ — probability of next token $x_t$ given all prior tokens; $T$ — temperature (controls output randomness); $k$ — top-$k$ candidate count; $p$ — nucleus (top-$p$) cumulative probability threshold; $V$ — vocabulary size.

---

## 0 · The Investigation — What We're Trying to Understand

> 🔬 **The mission**: Conduct the **AI Adoption Review** — reverse-engineer how LLMs work using GPT-4 and Claude 3.5 Sonnet as the two models under investigation. Chapter 1 is **"The Black Box"**: run identical prompts on both models, observe divergent outputs, and trace the differences back to first principles.

**What the board has asked:**
- ✅ "Before we commit engineering resources to AI, prove you understand how these models actually work."
- ❌ **Nobody knows yet** — two engineers just ran the same query on GPT-4 and Claude and got completely different answers. Why?

**The investigation scenario:**

You've been handed API keys for GPT-4 and Claude 3.5 Sonnet. Your first experiment: send both models the same prompt and observe.

```
Prompt sent to both models:
"What are the first 5 prime numbers? Show your work."

GPT-4 response:
"The first 5 prime numbers are 2, 3, 5, 7, 11.
A prime number is divisible only by 1 and itself.
2: only divisors are 1 and 2 ✓  3: only divisors are 1 and 3 ✓ ..."

Claude 3.5 Sonnet response:
"Let me work through this systematically:
1. Check 2: factors are just 1 and 2 → prime
2. Check 3: factors are just 1 and 3 → prime
...
First 5 primes: 2, 3, 5, 7, 11"
```

**Problems the investigation must explain:**
1. ❓ **Why does GPT-4 state the answer first, then justify?** (top-down)
2. ❓ **Why does Claude enumerate step-by-step, then conclude?** (bottom-up)
3. ❓ **Why do they use different words for the same concept?** (tokenization differences)
4. ❓ **Why does temperature=0 still not always give the same answer?** (sampling mechanics)

**This chapter unlocks:**

🚀 **Foundation understanding — AI Literacy Kit Chapter 1:**

1. **Tokenization** → Understand why the same sentence costs different tokens on GPT-4 vs Claude:
   - BPE vocabulary differences between model families
   - Cost estimation: `tokens × price/1k = API cost`

2. **Sampling parameters** → Understand why `temperature=0` doesn't fully control output:
   - Temperature, top-k, top-p mechanics
   - When to use deterministic vs. creative settings

3. **Context windows** → Understand memory limits:
   - GPT-4: 128k tokens; Claude 3.5: 200k tokens — what fits?
   - "Lost in the middle" degradation at long contexts

4. **Training stages** → Understand why pretrained models follow instructions:
   - Pretraining → SFT → RLHF/DPO pipeline
   - How this pipeline shapes GPT-4 vs Claude's different communication styles

5. **Model selection** → Know which model family to use for which task:
   - GPT-4 vs Claude vs open-weight models (cost, capability, latency)

✅ **AI Literacy Kit finding after Ch.1**:
Model output divergence is *not* random. It traces to training data distribution, RLHF reward signal design, and tokenizer vocabulary. Two engineers getting different answers from two models is expected and explainable.

---

## 1 · Core Idea

All four observations from § 0 trace back to a single mechanism. The models weren't broken — they were doing exactly what they were built to do. To understand why GPT-4 answers top-down and Claude answers bottom-up, you need to understand what that mechanism actually is.

A **large language model** is a transformer decoder (Ch.17) trained to predict the next token given all previous tokens, on internet-scale text. That single objective — next-token prediction — produces a model that appears to reason, retrieve facts, write code, and generate plans. None of those behaviours were explicitly programmed. They emerge from scale.

```
Training objective:   maximise P(token_t | token_1, token_2, ..., token_{t-1})
Training data:        ~10–100 trillion tokens scraped from the web, books, code
Training compute:     10²³–10²⁵ FLOP  (millions of GPU-hours)
Result:               a model with 7B–1T parameters that can perform most language tasks
```

Three stages turn a raw next-token predictor into the assistant you actually use:

```
Stage 1: Pretraining        Raw transformer on internet text → learns language + world knowledge
Stage 2: SFT                Fine-tuned on (instruction, good response) pairs → follows instructions
Stage 3: RLHF / DPO         Aligned with human preferences → helpful, harmless, honest
```

Each stage is covered in detail below.

> 💡 **Core idea verdict:** The output divergence in § 0 wasn't random — GPT-4 and Claude trained on different corpora with different RLHF reward signals, producing different stylistic defaults. A next-token predictor trained on internet text has no intrinsic preference for top-down vs bottom-up exposition; the preference is an artifact of the training distribution. Stages 2 and 3 below explain how reward signal design produces those stylistic differences. The deeper knowledge-grounding problem is solved in Ch.4.

---

## 2 · Tokenisation

Before you can estimate API costs, understand why the same English sentence tokenises to different counts on GPT-4 vs Claude, or reason about how much document context fits in a single call — you need to understand what the model actually receives. It never sees raw text. Text is first broken into **tokens** — subword units — using a byte-pair encoding (BPE) vocabulary.

### How BPE Works

```
Start with character-level vocabulary: [a, b, c, ..., z, space, ...]

1. Count all adjacent character pairs in the training corpus
2. Merge the most frequent pair into a new token: "t" + "h" → "th"
3. Repeat until vocabulary reaches target size (32k–100k tokens)
```

**Result:** common words become single tokens (`the`, `model`, `training`). Rare or technical words split (`trans` + `former`, `to` + `ken` + `isation`). Code tokens are often single characters.

### What You Need to Know About Tokens

| Fact | Why it matters |
|---|---|
| ~1 token ≈ 0.75 English words | Convert words → tokens for cost estimation |
| One token ≈ 4 bytes | 1M tokens ≈ 4 MB of text |
| The same text tokenises differently across models | Never assume GPT-4's token count matches Claude's |
| Code is token-dense | `self.attention_weights[layer_idx]` may be 6–10 tokens |
| Numbers tokenise byte-by-byte | `12345` → `[123, 45]` in some vocabularies — arithmetic is hard |

> 💡 **Tokenisation → cost:** A typical question-answer API exchange averages ~500 tokens. At GPT-4o-mini pricing ($0.00015/1k input tokens), that's $0.000075 per call. At GPT-4o ($0.0025/1k), it's $0.00125. Tokenisation is how you convert vague "it'll be cheap" into a budget-able number.

### The Context Window

The context window is the maximum number of **tokens** the model can process in a single forward pass — both input (prompt + retrieved chunks + history) and output (generated tokens).

| Model class | Context window |
|---|---|
| GPT-3.5 (2022) | 4k tokens |
| GPT-4 (2023) | 8k / 32k |
| Claude / Gemini (2024) | 200k / 1M |
| LLaMA 3 (2024) | 128k |

Larger context windows do not mean unlimited memory. Empirically, models show **lost-in-the-middle** degradation: information at the beginning and end of a long context is recalled more reliably than information buried in the middle.

> 💡 **Context window → investigation:** For the AI Adoption Review, a 10-document internal wiki RAG retrieval adds ~2,000 tokens of context. A 20-turn conversation history adds ~3,000 more. At GPT-4's 128k limit this is trivial; at older 4k models you'd need to summarise history. Lost-in-the-middle risk is real: safety-critical facts placed in the middle of a long context are recalled less reliably than facts near the start or end.

---

## 3 · Sampling — Temperature, Top-p, Top-k

Failure #4 in §0 — "unreliable output format" — is a direct consequence of how the model picks the next token. The model doesn't output one answer; it outputs a probability distribution over all ~50,000 vocabulary tokens. Sampling parameters are the dial that controls which token you draw from that distribution.

The model outputs a probability distribution over the vocabulary at each step. **Sampling parameters** control how you select the next token from that distribution.

### Temperature

$$p'_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

| Symbol | Meaning |
|---|---|
| $z_i$ | Raw **logit** (unnormalised score) the model assigns to token $i$ |
| $T$ | **Temperature** — the scalar you set at inference time |
| $p'_i$ | **Rescaled probability** of token $i$ after applying temperature |
| $\sum_j$ | Sum over **all tokens** in the vocabulary $V$ (normalisation) |

*Reading the formula:* dividing each logit by $T$ before the softmax shrinks ($T<1$) or stretches ($T>1$) the gap between high- and low-scoring tokens. When $T→0$ the highest logit dominates completely; when $T→\infty$ all tokens become equally likely.

| Temperature $T$ | Effect |
|---|---|
| $T → 0$ | Deterministic: always pick the highest-probability token (greedy) |
| $T = 1$ | Sample from the unmodified distribution |
| $T > 1$ | Distribution flattens — more randomness, less coherent |

**Rule of thumb:** factual retrieval → low T (0.0–0.3); creative generation → higher T (0.7–1.0); code → 0.0–0.2.

> 💡 **Temperature → investigation:** Factual question-answering (`temperature=0`) produces deterministic, reproducible outputs — essential when you're running controlled experiments and need the same prompt to produce the same answer. Creative generation (brainstorming, rephrasing) benefits from `temperature=0.7–1.0`. Getting this wrong contaminates your experiment results: a creative temperature on a factual-answer test will inflate variance and make the model look less reliable than it is.

### Top-p (Nucleus Sampling)

Instead of sampling from all tokens, select from the smallest set of tokens whose cumulative probability exceeds $p$:

```
Sort tokens by probability descending: [0.40, 0.25, 0.15, 0.10, 0.05, 0.03, ...]
top_p = 0.9 → keep [0.40, 0.25, 0.15, 0.10] (cumsum = 0.90) → sample only these four
```

Top-p dynamically adjusts the candidate set per token — large when the distribution is flat (uncertain), small when one token dominates (confident). Almost all production usage combines temperature + top-p.

### Top-k

Keep only the k highest-probability tokens and renormalise. Less adaptive than top-p; rarely preferred in practice.

---

## 4 · The Three Training Stages

The divergent outputs in § 0 — GPT-4's top-down structure vs Claude's bottom-up enumeration — both trace to the same root cause: *what the model was trained on, and what objective it was trained toward*. The three stages below explain how a raw text predictor becomes an instruction-following assistant, and why stylistic differences between GPT-4 and Claude persist even after both receive instruction fine-tuning.

### Stage 1 — Pretraining

A standard transformer decoder (Ch.17 causal mask) is trained on a massive corpus with the cross-entropy loss over next-token prediction. No human labels — the text itself is the supervision.

**What it learns:** grammar, syntax, world knowledge, reasoning patterns, code idioms, basic arithmetic, multilingual text — anything that appears frequently enough in the training data.

**What it doesn't learn:** to be helpful, to follow instructions, or to prefer honest over fluent answers.

A pretrained model responds to `"What is the capital of France?"` by continuing the text in a plausible direction — which might be `"?"` or `"A: Paris"` or `"Who is the king of France?"` depending on what it has seen. It does not reliably answer the question.

### Stage 2 — Supervised Fine-Tuning (SFT)

Fine-tune the pretrained model on a curated dataset of `(instruction, response)` pairs written by human annotators.

```
Input:   "Summarise this document in three bullet points: [doc]"
Target:  "• Point 1\n• Point 2\n• Point 3"
Loss:    Cross-entropy on the target tokens only (not the input)
```

SFT teaches the model to follow instruction format and stay on task. Even a few thousand high-quality examples significantly improves instruction-following.

**The risk:** the model learns what annotators wrote, not what is correct. If annotators tend to produce verbose, confident answers, the model does too.

### Stage 3 — RLHF / DPO (Alignment)

The goal: move the model's outputs toward what humans actually prefer — more helpful, less harmful, more honest.

**RLHF (Reinforcement Learning from Human Feedback):**

```
1. Sample two completions for the same prompt
2. Human annotator picks the preferred one
3. Train a reward model R(prompt, completion) on these preference pairs
4. Fine-tune the SFT model to maximise R using PPO (policy gradient RL)
   + KL penalty to stay close to the original SFT model
```

**DPO (Direct Preference Optimization):** skips the reward model entirely. Directly fine-tunes the model on preference pairs with a loss that increases the probability of the preferred response and decreases the probability of the rejected one. Simpler, more stable, now preferred over RLHF in most open-source work.

The DPO loss for a preference triple $(x, y_w, y_l)$ — prompt, preferred response, rejected response:

$$\mathcal{L}_{DPO} = -\mathbb{E}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

| Symbol | Meaning |
|---|---|
| $\mathcal{L}_{DPO}$ | The **DPO training loss** — minimise this to align the model |
| $\mathbb{E}[\cdot]$ | **Expectation** over all preference triples in the training set |
| $\sigma$ | **Sigmoid** function $\sigma(t) = 1/(1+e^{-t})$ — maps any real number to $(0,1)$ |
| $\pi_\theta$ | The **model being trained** (policy with learnable weights $\theta$) |
| $\pi_{ref}$ | The **frozen SFT model** — acts as a baseline to prevent over-optimisation |
| $\beta$ | **KL penalty weight** — how far $\pi_\theta$ is allowed to drift from $\pi_{ref}$ (typical: 0.1–0.5) |
| $y_w$ | The **preferred** (winning) response chosen by the human annotator |
| $y_l$ | The **rejected** (losing) response not preferred by the annotator |

*Reading the formula:* the log-ratio $\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}$ measures how much more (or less) likely the trained model makes the preferred response relative to the baseline. The loss pushes this gap to be positive — preferred response probability goes up, rejected goes down — weighted by $\beta$ to stay close to the SFT model.

where $\pi_{ref}$ is the frozen SFT model and $\beta$ controls how far the trained policy can deviate from it (typical: 0.1–0.5). Full derivation and TRL code in [03b-agentic-ai ch05 fine-tuning](../../03b-agentic-ai/ch05-fine-tuning/fine-tuning.md).

**RLVR (Reinforcement Learning from Verifiable Rewards):** the training recipe behind o1, o3, and DeepSeek-R1. Instead of human preference pairs, RLVR uses automatically verifiable correctness signals — math answer checking, unit test pass/fail, formal proof verification — as the reward. The model generates a chain-of-thought reasoning trace; the final answer is checked against ground truth; RL updates reinforce traces that led to correct answers. This is why reasoning models excel at math and code: those domains have cheap, automatic verifiers. See [ch03 §8](../ch03-cot-reasoning/cot-reasoning.md) for reasoning token inference behavior.

**What RLHF/DPO gives you:** a model that says "I don't know" when it doesn't know, declines harmful requests, and structures answers for human convenience rather than for statistical fluency.

**The sycophancy trap:** RLHF optimises for human *approval*, which is not the same as human *benefit*. Models learn to agree with the user's framing even when it's wrong. This is why you can sometimes "convince" a model to change a correct answer by pushing back.

> 💡 **Training stages → investigation finding:** Both GPT-4 and Claude went through the same three stages. Their stylistic differences (top-down vs bottom-up, verbose vs concise) emerge from differences in the human feedback data used for RLHF/DPO — specifically, what the annotator pools at OpenAI vs Anthropic preferred. The fix for domain-knowledge gaps (model doesn't know your internal docs) isn't more training. It's grounding — Ch.4.

> ➡️ **Why knowledge gaps persist even after SFT + RLHF:** Stages 2 and 3 change *behaviour*, not *knowledge*. The model learns to follow instructions and be helpful — but it still knows only what was in the pretraining corpus. Your internal documentation was not on the internet. Ch.4 solves this.

### Stage 4 (Optional, Preview) — Parameter-Efficient Fine-Tuning (PEFT)

> 📖 **This is a [03b-agentic-ai Ch.5](../../03b-agentic-ai/ch05-fine-tuning/fine-tuning.md) preview.** You don't need PEFT to complete the LLM Fundamentals track — this section exists here because the interview table asks about it. Read it now for vocabulary; the full implementation is in the Agentic AI track.

**Technical definition.** PEFT is a family of fine-tuning methods that freeze the pretrained model weights and train only a small set of additional or modified parameters — typically 0.01–1% of total model size — while keeping the remainder of the model locked. The pretrained weights $W$ are treated as a fixed feature extractor; only the adapter parameters $\theta_{adapt}$ are updated during training. The resulting model behaves as if fully fine-tuned but at a fraction of the compute and memory cost.

**The problem PEFT solves.** Full fine-tuning updates every weight in the model. For a 7B model that means backpropagating through 14 GB of parameters per gradient step, requiring ~80 GB VRAM and risking catastrophic forgetting on small datasets. Full fine-tuning of GPT-4-class models is effectively off the table for any team that isn't OpenAI.

**The intuition.** Adapting a pretrained LLM to a new task is a low-dimensional change — the model already knows language, reasoning, and world knowledge. You're adjusting *style, domain, or format*, not relearning everything. PEFT exploits this by restricting the parameter update to a compact subspace, which is sufficient to capture the adaptation without touching the frozen core.

**The investigation relevance.** In the Agentic AI track ([03b Ch.5](../../03b-agentic-ai/ch05-fine-tuning/fine-tuning.md)) you'll train a LoRA adapter that teaches a model domain-specific vocabulary and tone. Training that adapter takes 2 hours on one A100; full fine-tuning of the same base model would take 3 days and 4× A100s. Same adaptation result, 95% less compute cost.

Three dominant methods, each placing the adapter at a different point in the architecture:

---

#### LoRA — Low-Rank Adaptation

**Technical definition.** LoRA decomposes the weight update $\Delta W$ into a product of two low-rank matrices. For a frozen weight matrix $W \in \mathbb{R}^{d \times k}$, LoRA trains $A \in \mathbb{R}^{d \times r}$ and $B \in \mathbb{R}^{r \times k}$ where rank $r \ll \min(d, k)$ (typically 4–64). The adapted forward pass becomes:

$$h = Wx + \frac{\alpha}{r} BAx$$

| Symbol | Meaning |
|---|---|
| $x$ | **Input activation** vector to this layer ($\in \mathbb{R}^k$) |
| $h$ | **Output activation** vector — what this layer passes to the next ($\in \mathbb{R}^d$) |
| $W$ | **Frozen pretrained weight** matrix — never updated |
| $B \in \mathbb{R}^{d \times r}$ | Low-rank **up-projection** matrix (trained; initialised to zero) |
| $A \in \mathbb{R}^{r \times k}$ | Low-rank **down-projection** matrix (trained; random init) |
| $r$ | **Rank** — the bottleneck dimension. $r=8$ means only $8 \times (d+k)$ new params per layer |
| $\alpha$ | **Scaling hyperparameter** — controls the magnitude of the adapter's contribution |
| $\frac{\alpha}{r}$ | **Effective scale factor** — set $\alpha = r$ to get scale $= 1$ and match the pretrained weight magnitude |

*Reading the formula:* $Wx$ is the original frozen layer; $\frac{\alpha}{r}BAx$ is the low-rank update added on top. Because $B$ starts at zero, the adapter contributes nothing at the start of training — the model trains from the pretrained checkpoint, not from random noise.

where $\alpha$ is a scaling hyperparameter (default: $\alpha = r$, giving $\frac{\alpha}{r} = 1$). During training only $A$ and $B$ are updated; $W$ is frozen. At the start of training, $B$ is initialised to zero so the adapter contributes nothing — training starts from the pretrained model's behaviour.

**The intuition.** Weight updates in fine-tuning are empirically low-rank: the gradient matrix has many near-zero singular values, meaning the "useful update" lives in a small subspace. LoRA captures that subspace directly without ever computing the full update.

**At inference.** $\frac{\alpha}{r} BA$ is computed once and added into $W$ — the adapter is *merged* into the weight. The result is a single weight matrix indistinguishable from a fully fine-tuned one. Zero latency overhead, zero extra memory, zero inference cost.

```
trainable params: ~4M  (out of 7B)   — r=8 applied to Q,K,V,O attention projections
VRAM during fine-tune: ~16 GB (vs. ~80 GB full fine-tune)
inference overhead: 0 — adapter merged into weights before serving
Investigation relevance: domain voice adapter → $0.008/conv (vs $0.07 with generic GPT-4o)
```

> 💡 **LoRA verdict:** Same output quality as full fine-tuning for style/domain shifts; 5× cheaper to train; zero production overhead. Default choice for custom model adaptation.

---

#### Prefix Tuning

**Technical definition.** Prefix tuning freezes all model weights and instead prepends $L_p$ learnable (key, value) pairs to the attention sequence of *every* transformer layer. These virtual KV pairs — the *prefix* — are trained end-to-end. The attention at each layer computes:

$$\text{Attention}(Q,\ [K_{prefix};K_{input}],\ [V_{prefix};V_{input}])$$

| Symbol | Meaning |
|---|---|
| $Q$ | **Query** matrix — derived from the current input tokens (frozen) |
| $K_{input},\ V_{input}$ | **Key / Value** matrices from the actual input tokens (frozen) |
| $K_{prefix} \in \mathbb{R}^{L_p \times d_k}$ | **Learned key prefix** — $L_p$ virtual context keys, trained end-to-end |
| $V_{prefix} \in \mathbb{R}^{L_p \times d_v}$ | **Learned value prefix** — $L_p$ virtual context values, trained end-to-end |
| $[\cdot\,;\cdot]$ | **Row-wise concatenation** — the prefix is prepended to the real keys/values |

*Reading the formula:* standard attention computes $Q$ against $K_{input}$ only; prefix tuning extends the key-value sequence with $L_p$ learnable rows at every layer, letting every attention head attend to a learned "steering context" regardless of what the actual input tokens are.

where $K_{prefix} \in \mathbb{R}^{L_p \times d_k}$ and $V_{prefix} \in \mathbb{R}^{L_p \times d_v}$ are the learned parameters. The model's own weights never change — only these injected context pairs are trained.

**The intuition.** Every attention head can now "see" a learned context at every layer. Forcing representations to attend to this context at every depth steers the model's internal activations toward the target distribution — without touching any weights.

**The inference cost.** The prefix occupies $L_p$ positions in the KV cache at every layer for every forward pass. A prefix of 100 tokens on a 32-layer model adds 3,200 KV entries to every active request. Unlike LoRA, this cost is *permanent* — it cannot be merged away.

```
trainable params: ~0.1–1% of model
VRAM during fine-tune: low (prefix embeddings + gradients only)
inference overhead: +L_p KV cache entries per layer, per request — always present
use case: serve one frozen model with multiple swappable prefixes (one per task/tenant)
```

> ⚠️ **Prefix tuning in production:** The KV cache overhead scales with concurrent users. At 1,000 simultaneous conversations with $L_p=100$ and 32 layers, you're holding ~320k extra KV pairs in memory at all times. Budget for this before choosing prefix tuning over LoRA.

---

#### Prompt Tuning

**Technical definition.** Prompt tuning freezes all model weights — including the embedding table — and trains a short sequence of $L_p$ continuous *soft token* embeddings that are prepended to the input before the first layer only. The soft tokens $P \in \mathbb{R}^{L_p \times d_{model}}$ are not constrained to the vocabulary; they are free-floating vectors in embedding space, optimised directly by gradient descent.

$$\text{input to layer 1} = [P;\ \text{embed}(x_1),\ \text{embed}(x_2),\ \ldots]$$

| Symbol | Meaning |
|---|---|
| $P \in \mathbb{R}^{L_p \times d_{model}}$ | **Soft token matrix** — $L_p$ learned embedding vectors, not constrained to the vocabulary |
| $\text{embed}(x_i)$ | **Standard token embedding** of the $i$-th real input token |
| $[\cdot\,;\cdot]$ | **Row-wise concatenation** — soft tokens are prepended to the real token embeddings |

*Reading the formula:* the model sees $L_p$ extra "virtual" tokens at the front of the sequence before layer 1. These vectors can be anything in embedding space — they are not words. Everything from layer 1 onward processes them as if they were regular tokens.

Nothing downstream changes — the soft tokens propagate through all layers as normal token representations.

**The intuition.** The model already knows how to respond appropriately given the right context. Soft tokens are a learned "mode trigger" that shifts the model into the target behaviour — analogous to a system prompt but learnable rather than hand-crafted.

**The trade-off.** Fewest trainable parameters of any PEFT method (~10k–1M). But the adaptation signal only enters at the first layer and must survive unmodified through all subsequent layers. For large distribution shifts this degrades — but at GPT-3 scale (100B+ parameters) prompt tuning is competitive with full fine-tuning on many benchmarks.

```
trainable params: ~10k–1M (smallest of all PEFT methods)
VRAM during fine-tune: minimal — no gradient propagation into frozen weights
inference overhead: +L_p input tokens per request (affects context length and API cost)
use case: large frozen model, minimal infra change, lightweight per-tenant personalisation
```

---

#### Choosing Between Them

| | LoRA | Prefix Tuning | Prompt Tuning |
|---|---|---|---|
| **Where adaptation lives** | Inside weight matrices (merged at inference) | KV cache at every layer | Input embedding only |
| **Inference overhead** | None | +KV cache per layer per request | +Input tokens per request |
| **Expressiveness** | High | Medium–High | Lower (scales with model size) |
| **Multi-task serving** | Swap adapter files | Swap prefix in memory | Swap soft token embeddings |
| **Best for** | Domain/style fine-tuning, brand voice | Multi-task inference, one model many tasks | Very large frozen models, minimal infra |

> 💡 **Interview anchor:** "Compare LoRA and prefix tuning" → anchor on *where the parameters live and what that costs at inference*. LoRA modifies weight matrices and is merged before serving — zero overhead. Prefix tuning injects into the KV cache and stays there — constant memory cost per concurrent user. The choice is a deployment constraint decision, not a training convenience decision.

---

## 5 · Emergent Capabilities

The training stages above explain what a model is *taught*. This section explains what it *develops on its own* at scale — capabilities that matter directly for what you'll build in the next three chapters.

Several capabilities of LLMs were not explicitly trained for and appeared qualitatively at sufficient scale:

| Capability | Approximate threshold |
|---|---|
| In-context learning (few-shot) | ~7B parameters |
| Chain-of-thought reasoning | ~100B parameters |
| Multi-step arithmetic | ~540B parameters |
| Theory of mind (passing Sally-Anne test) | GPT-4 class |

**"Emergent"** does not mean magical. These capabilities exist in the training data — it's that the model needs sufficient capacity to compress and reconstruct the reasoning patterns latent there.

> ➡️ **Why emergence thresholds matter for the investigation:** In-context learning (≥7B params) is what makes few-shot prompting work — you'll use it in Ch.2. Chain-of-thought reasoning (≥100B params) is what makes complex multi-step queries work — you'll probe it in Ch.3. Knowing these thresholds tells you when it's worth trying a capability vs. when you need to engineer around its absence by choosing a larger model or a different approach.

---

## 6 · What "Model Size" Actually Means

The CEO's model selection question — GPT-4 at \$0.03/1k tokens vs. GPT-3.5 at \$0.002/1k — is a 15× cost difference. Whether that's worth it depends on what parameter count actually buys you, and what it costs to run.

```
Parameters = weights in all attention and FFN matrices
           = num_layers × (12 × d_model²)   for a standard transformer

7B model:   7 × 10⁹ parameters × 2 bytes (fp16) = 14 GB VRAM minimum
13B model:  ~26 GB
70B model:  ~140 GB  (requires 2× A100 80GB)
GPT-4:      estimated 1.8T parameters in a mixture-of-experts architecture
```

### Mixture of Experts (MoE)

Standard transformer layers activate **all** parameters for every token. MoE replaces the dense FFN layers with $N$ "expert" sub-networks plus a lightweight **router** that selects $k$ of them per token (sparse activation):

$$y = \sum_{i=1}^{k} G(x)_i \cdot E_i(x) \qquad G(x) = \text{TopK}\!\left(\text{Softmax}(W_g\, x),\; k\right)$$

| Symbol | Meaning |
|---|---|
| $x$ | **Input token representation** — the hidden state passed into this MoE layer |
| $y$ | **Output** of the MoE layer — weighted sum of the selected experts' outputs |
| $N$ | **Total experts** in this layer (e.g., 8, 16, 64) |
| $k$ | **Active experts per token** — only $k$ of the $N$ experts run (typically 1 or 2) |
| $E_i(x)$ | **Expert $i$'s FFN output** — a standard feed-forward sub-network |
| $G(x)_i$ | **Gating weight** for expert $i$ — how much this expert contributes to the output |
| $W_g$ | **Gating weight matrix** — learned linear projection that maps token $x$ to expert scores |
| $\text{Softmax}(W_g x)$ | Normalised probability distribution over all $N$ experts for this token |
| $\text{TopK}(\cdot,\, k)$ | Keep only the $k$ highest-scoring experts; zero out the rest (sparse activation) |

*Reading the formula:* for each token, the router computes $W_g x$ (a score per expert), takes the top-$k$ by probability, and returns a weighted sum of only those $k$ experts' outputs. The other $N-k$ experts do not execute — that's the compute saving.

**Why MoE matters:**
- **Scale at fraction of cost:** GPT-4's 1.8T parameters → only ~200–400B active per token (roughly a dense 200B forward pass cost)
- **Specialisation:** Different experts naturally specialise — some activate on code, others on natural language, others on structured data
- **Training efficiency:** Total capacity scales with $N$; compute (and therefore cost) scales with $k$

> ⚠️ **VRAM trap:** For models like Mixtral-8×7B, VRAM is determined by **total** params loaded (~93 GB fp16), but *inference compute* is determined by **active** params (~12.9B per token). You still need the memory — you just don't pay full compute cost per forward pass.

**Inference cost scales with parameter count, context length, and batch size.** A 70B model at 128k context costs roughly 50× more to run than a 7B model at 4k context. This is why RAG and agentic applications use smaller, instruction-tuned models wherever possible.

> 💡 **Model selection for the investigation:** Use GPT-4o-mini for factual retrieval and structured-output experiments (deterministic, low latency, low cost). Reserve GPT-4o for complex reasoning experiments where accuracy outweighs cost. When testing open-weight models, a LoRA-adapted 7B model can match GPT-4o-mini quality at ~$0.0003/1k — a further 6× cost reduction (covered in [03b Ch.5](../../03b-agentic-ai/ch05-fine-tuning/fine-tuning.md)).

---

## 7 · Key Distinctions Every Engineer Gets Asked

| Pair | Distinction |
|---|---|
| **Base model vs instruct/chat model** | Base: raw next-token predictor. Instruct: SFT+RLHF applied — follows instructions. Always use instruct for applications. |
| **Parameters vs context window** | Parameters = learned knowledge. Context window = working memory for one inference call. |
| **Temperature vs top-p** | Temperature rescales the whole distribution. Top-p truncates it. Use both. |
| **RLHF vs DPO** | RLHF trains a separate reward model; DPO doesn't. DPO is simpler and now standard. |
| **Tokens vs words** | Tokens are model-native; words are human-native. 1 word ≈ 1.3 tokens on average for English prose. |
| **Hallucination vs confabulation** | Hallucination: factually wrong output. Confabulation: specifically, a fluent-sounding fabrication of a plausible but non-existent fact (citation, statistic, API name). Same mechanism, different vocabulary. |

---

## 8 · Progress Check — AI Literacy Kit: Chapter 1 Findings

**Unlocked capabilities:**
- ✅ **Understand LLM architecture**: Know what pretraining → SFT → RLHF/DPO produces
- ✅ **Token-based cost estimation**: Can calculate API costs for any conversation pattern
- ✅ **Model selection framework**: Understand trade-offs between GPT-4o, GPT-4o-mini, Claude 3.5 Sonnet
- ✅ **Sampling parameter control**: Know when to use temperature=0 vs. 0.8 and why
- ✅ **Context window awareness**: Know how much document context fits; understand lost-in-the-middle

**AI Literacy Kit — Chapter 1 findings:**

| Investigation Question | Finding |
|---|---|
| Why do GPT-4 and Claude give structurally different answers? | Different RLHF reward signals from different annotator pools; stylistic defaults baked in during alignment |
| Why does temperature=0 not always reproduce the same token? | Floating-point non-determinism in GPU matrix ops; use seed for reproducibility |
| Why does Claude use fewer tokens than GPT-4 on the same query? | Different BPE vocabulary; Claude's tokenizer is more efficient on common English phrases |
| Why do both models hallucinate facts they were never told? | Next-token prediction maximises plausibility, not truth; grounding (Ch.4) is required for factual accuracy |
| Why does CoT sometimes make answers worse? | Investigated in Ch.3; reasoning tokens can introduce confident-but-wrong intermediate steps |

**What you can reproduce experimentally right now:**

✅ **Sampling experiment**: send the same prompt 10 times at temperature=1.0 — observe distribution of outputs
✅ **Tokenisation audit**: run `tiktoken` on your internal documents — estimate RAG retrieval cost before building it
✅ **Context window test**: inject a fact at position 0% vs 50% vs 100% of a long context; measure retrieval accuracy
✅ **Model comparison baseline**: identical prompt to GPT-4o and Claude 3.5 Sonnet; document structural differences

**Next chapter**: [Prompt Engineering](../ch02-prompt-engineering/prompt-engineering.md) — **"The Control Interface"** — system prompts and few-shot examples as behavioral levers. You'll run controlled experiments showing how GPT-4 and Claude diverge on identical instructions.

**Key interview concepts from this chapter:**

| Must know | Likely asked | Trap to avoid |
|---|---|---|
| The three training stages (pretraining → SFT → RLHF/DPO) and what each adds | What is DPO and why is it better than RLHF? | Saying RLHF is "just reinforcement learning" — the reward model and KL penalty are the critical additions |
| Temperature: effect of T<1 vs T>1 | What is top-p and why is it preferred over top-k? | Saying temperature = "creativity dial" — technically it's distribution sharpening/flattening |
| What a context window is and what lost-in-the-middle means | Why can't LLMs do reliable multi-digit arithmetic? | Confusing model parameters (knowledge) with context window (working memory) |
| What BPE tokenisation produces and why it matters for cost | Why do larger models exhibit emergent capabilities? | Saying the model "understands" or "thinks" — anthropomorphic framing fails in interviews |
| Base model vs instruct model — what SFT adds | What is sycophancy and why does RLHF cause it? | Saying fine-tuning changes what the model "knows" — SFT/RLHF changes behaviour, not stored knowledge |
| **PEFT overview:** Parameter-Efficient Fine-Tuning methods add or modify a small number of parameters while freezing the base model. **LoRA** inserts low-rank matrices into attention projections (~0.1% of parameters, merges at inference for zero latency); **prefix tuning** prepends learnable tokens to every layer's KV sequence (increases KV cache size at inference); **prompt tuning** learns only input embeddings (smallest footprint) | "Compare LoRA, prefix tuning, and prompt tuning" | "PEFT methods are interchangeable" — LoRA modifies weights and merges at inference with no latency cost; prefix tuning increases the effective sequence length and has a constant KV cache overhead at inference; choosing between them depends on deployment constraints |
| **Instruction following vs base model:** a base model generates the statistically likely continuation of any text — useful for perplexity benchmarks, not for assistants. SFT on instruction-response pairs shifts the output distribution toward helpful, formatted answers | "What does instruction fine-tuning actually teach the model?" | "Fine-tuning on instructions teaches the model new knowledge" — SFT teaches *format and tone*, not new facts; knowledge comes from pretraining data; new facts require RAG or continual pretraining |

---

## 9 · Bridge

LLM Fundamentals established the model: a scaled, aligned next-token predictor with a finite context window and probabilistic sampling. The next chapter — [Prompt Engineering](../ch02-prompt-engineering/prompt-engineering.md) **"The Control Interface"** — shows how system prompts, few-shot examples, and structured output instructions become behavioral levers that produce measurably different responses from GPT-4 vs Claude.

> *The model is the brain. It predicts tokens. Everything in the AI track — CoT, RAG, ReAct, Semantic Kernel — is about how you wire inputs and outputs around that single mechanical act.*

## Illustrations

![LLM fundamentals — BPE tokenisation, sampling, training stages, and the context window](img/LLM%20Fundamentals.png)
