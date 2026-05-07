# LLM Fundamentals — What a Language Model Actually Is

> **A brief history.** In the summer of 2017, a team of eight Google engineers published a twelve-page paper with a deliberately provocative title: *"Attention Is All You Need."* They weren't describing a self-help book — they were discarding the recurrent loops that every language model had relied on for a decade and replacing them with a single mechanism called *attention*. Almost nobody outside research noticed. The transformer, as the architecture came to be known, was faster to train, easier to parallelize, and — it turned out — almost infinitely scalable.
>
> What happened next unfolded faster than anyone predicted. Within a year, OpenAI took just the *decoder* half of that transformer and asked a deceptively simple question: what if we trained it on most of the internet and gave it one job — *predict the next word*? The result was **GPT-1** (2018), then **GPT-2** (2019), which was quietly good enough that OpenAI sat on the weights for months out of concern. By **GPT-3** in 2020 — 175 billion parameters trained on half a trillion words — something strange emerged: the model could solve tasks it had never been explicitly trained on, just from examples in the prompt. Researchers called it *in-context learning* and struggled to explain it.
>
> The final piece clicked into place with **InstructGPT** (2022): rather than asking the model to complete text, they trained it to *follow instructions* using human feedback. Wrap that in a chat interface and you get **ChatGPT** — 100 million users in two months, the fastest consumer product in history. Every model in this track, GPT-4, Claude, Gemini, Llama, DeepSeek, is running the same recipe at varying scale.
>
> **Where you are in the curriculum.** **Read this before anything else in the AI track.** Every later doc — [CoTReasoning](../ch03_cot_reasoning), [RAG](../ch04_rag_and_embeddings), [ReAct](../ch06_react_and_semantic_kernel), every agent framework — assumes you know what an LLM is under the hood. This document builds that foundation from the transformer ([ML Ch.18](../../01-ml/03_neural_networks/ch10_transformers)) through to the models you call via API today: tokenisation, the pretraining → SFT → RLHF pipeline, sampling parameters, and context windows.
>
> **Notation used later in this doc.** $P(x_t \mid x_{<t})$ — probability of next token $x_t$ given all prior tokens; $T$ — temperature (controls output randomness); $k$ — top-$k$ candidate count; $p$ — nucleus (top-$p$) cumulative probability threshold; $V$ — vocabulary size.

---

## 0 · The Challenge — Where We Are

> 🎯 **The mission**: Launch **Mamma Rosa's PizzaBot** — satisfying 6 constraints:
> 1. **BUSINESS VALUE**: >25% conversion + +$2.50 AOV + 70% labor savings — 2. **ACCURACY**: <5% error — 3. **LATENCY**: <3s p95 — 4. **COST**: <$0.08/conv — 5. **SAFETY**: Zero attacks — 6. **RELIABILITY**: >99% uptime

**What we know so far:**
- ❌ **This is Chapter 1** — the foundation. We're starting from scratch.
- ❌ **All constraints unmet** — raw GPT-3.5 delivers 8% conversion (phone baseline: 22%)
- 📊 **Phone baseline metrics**: 22% conversion, $38.50 AOV, $157,680/year labor cost

**The business context:**

You're the Lead AI Engineer at Mamma Rosa's Pizza. The CEO is skeptical about AI: "We have phone staff who take orders perfectly. Why should I invest $300k in a chatbot?"

**Traditional phone order baseline:**
- 22% order conversion rate (of customers who call)
- $38.50 average order value
- $157,680/year in phone staff labor costs (3 staff × $18/hr × 8hr × 365 days)
- ~45 simultaneous call capacity during peak hours

**Your mission:** Prove AI can beat human performance on business metrics while maintaining accuracy, speed, and safety.

**What's blocking us:**

🚨 **We need to understand what an LLM even is before we can build anything**

The team downloaded GPT-3.5 and sent it this test query:

```
User: "What's your cheapest gluten-free pizza under 600 calories?"

GPT-3.5 (raw, no prompt engineering):
"I apologize, but I don't have access to real-time menu data or calorie information
for specific restaurants. However, many pizza chains offer gluten-free options.
You might want to try Margherita pizza which is typically around 500-600 calories.
Would you like me to help you find..."
```

**Problems:**
1. ❌ **Doesn't know Mamma Rosa's menu** (trained on internet text, not our private data)
2. ❌ **Hallucinates calorie counts** (makes up numbers that sound plausible)
3. ❌ **Doesn't attempt to look up real data** (no tool use, no grounding)
4. ❌ **Unreliable output format** (conversational, not structured for order processing)

**Business impact:**
- **Conversion rate with raw LLM: 8%** (14 points below phone baseline of 22%!)
- **Error rate: ~40%** (hallucinates menu items, prices, calorie counts)
- **Customer trust: destroyed** (one wrong allergen claim = lawsuit risk)

CEO's reaction: "This is embarrassing. My phone staff would never give wrong information. Pull the plug unless you can fix this."

**What this chapter unlocks:**

🚀 **Foundation knowledge — no constraint achievements yet:**

1. **Tokenization** → Estimate API costs:
   - Understand how "gluten-free" becomes tokens (3 tokens in GPT-3.5)
   - Calculate: 500 tokens/conv × $0.002/1k = $0.001 LLM cost baseline

2. **Sampling parameters** → Control output behavior:
   - Temperature=0 for deterministic order confirmations
   - Temperature=0.7-1.0 for creative menu recommendations
   - Top-p nucleus sampling for production-quality text

3. **Context windows** → Understand memory limits:
   - Know how much menu data fits (4k tokens = ~3,000 words)
   - Recognize "lost-in-the-middle" risk for long conversations

4. **Training stages** → Understand why base models fail:
   - Pretraining → SFT → RLHF pipeline
   - Base model vs. instruct model behavior differences

5. **Model selection** → Choose appropriate tiers:
   - GPT-4 ($0.03/1k) vs GPT-3.5 ($0.002/1k) vs Claude ($0.015/1k)
   - Cost/capability trade-offs for different tasks

⚡ **Constraint status after Ch.1**:
- ❌ **All 6 constraints BLOCKED** — This is **foundation knowledge only**
- 🔧 **Next steps required**: Prompt engineering (Ch.2) → reasoning (Ch.3) → RAG (Ch.4) before system becomes usable

**Business impact of this chapter:**
- ✅ **Enables cost estimation**: CEO now has realistic budget projections
- ✅ **Informs model selection**: Know when to use GPT-4 vs. GPT-3.5
- ✅ **Sets realistic expectations**: Team understands why raw LLMs need engineering
- ❌ **But no revenue yet**: 8% conversion doesn't justify $300k investment

---

## 1 · Core Idea

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

---

## 2 · Tokenisation

The model never sees raw text. Text is first broken into **tokens** — subword units — using a byte-pair encoding (BPE) vocabulary.

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

### The Context Window

The context window is the maximum number of **tokens** the model can process in a single forward pass — both input (prompt + retrieved chunks + history) and output (generated tokens).

| Model class | Context window |
|---|---|
| GPT-3.5 (2022) | 4k tokens |
| GPT-4 (2023) | 8k / 32k |
| Claude / Gemini (2024) | 200k / 1M |
| LLaMA 3 (2024) | 128k |

Larger context windows do not mean unlimited memory. Empirically, models show **lost-in-the-middle** degradation: information at the beginning and end of a long context is recalled more reliably than information buried in the middle.

---

## 3 · Sampling — Temperature, Top-p, Top-k

The model outputs a probability distribution over the vocabulary at each step. **Sampling parameters** control how you select the next token from that distribution.

### Temperature

$$p'_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

| Temperature $T$ | Effect |
|---|---|
| $T → 0$ | Deterministic: always pick the highest-probability token (greedy) |
| $T = 1$ | Sample from the unmodified distribution |
| $T > 1$ | Distribution flattens — more randomness, less coherent |

**Rule of thumb:** factual retrieval → low T (0.0–0.3); creative generation → higher T (0.7–1.0); code → 0.0–0.2.

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

where $\pi_{ref}$ is the frozen SFT model and $\beta$ controls how far the trained policy can deviate from it (typical: 0.1–0.5). Full derivation, PizzaBot training data, and TRL code in [ch10 §5.5](../ch10-fine-tuning/fine-tuning.md).

**RLVR (Reinforcement Learning from Verifiable Rewards):** the training recipe behind o1, o3, and DeepSeek-R1. Instead of human preference pairs, RLVR uses automatically verifiable correctness signals — math answer checking, unit test pass/fail, formal proof verification — as the reward. The model generates a chain-of-thought reasoning trace; the final answer is checked against ground truth; RL updates reinforce traces that led to correct answers. This is why reasoning models excel at math and code: those domains have cheap, automatic verifiers. See [ch03 §8](../ch03-cot-reasoning/cot-reasoning.md) for reasoning token inference behavior.

**What RLHF/DPO gives you:** a model that says "I don't know" when it doesn't know, declines harmful requests, and structures answers for human convenience rather than for statistical fluency.

**The sycophancy trap:** RLHF optimises for human *approval*, which is not the same as human *benefit*. Models learn to agree with the user's framing even when it's wrong. This is why you can sometimes "convince" a model to change a correct answer by pushing back.

### Stage 4 (Optional) — Parameter-Efficient Fine-Tuning (PEFT)

**Technical definition.** PEFT is a family of fine-tuning methods that freeze the pretrained model weights and train only a small set of additional or modified parameters — typically 0.01–1% of total model size — while keeping the remainder of the model locked. The pretrained weights $W$ are treated as a fixed feature extractor; only the adapter parameters $\theta_{adapt}$ are updated during training. The resulting model behaves as if fully fine-tuned but at a fraction of the compute and memory cost.

**The problem PEFT solves.** Full fine-tuning updates every weight in the model. For a 7B model that means backpropagating through 14 GB of parameters per gradient step, requiring ~80 GB VRAM and risking catastrophic forgetting on small datasets. Full fine-tuning of GPT-4-class models is effectively off the table for any team that isn't OpenAI.

**The intuition.** Adapting a pretrained LLM to a new task is a low-dimensional change — the model already knows language, reasoning, and world knowledge. You're adjusting *style, domain, or format*, not relearning everything. PEFT exploits this by restricting the parameter update to a compact subspace, which is sufficient to capture the adaptation without touching the frozen core.

**The PizzaBot connection.** In Ch.8 you'll train a LoRA adapter that teaches Mamma Rosa's brand voice — "Try our garlic bread, it pairs perfectly with that!" instead of the generic GPT tone. Training that adapter takes 2 hours on one A100; full fine-tuning of the same base model would take 3 days and 4× A100s. Same business result, 95% less compute cost.

Three dominant methods, each placing the adapter at a different point in the architecture:

---

#### LoRA — Low-Rank Adaptation

**Technical definition.** LoRA decomposes the weight update $\Delta W$ into a product of two low-rank matrices. For a frozen weight matrix $W \in \mathbb{R}^{d \times k}$, LoRA trains $A \in \mathbb{R}^{d \times r}$ and $B \in \mathbb{R}^{r \times k}$ where rank $r \ll \min(d, k)$ (typically 4–64). The adapted forward pass becomes:

$$h = Wx + \frac{\alpha}{r} BAx$$

where $\alpha$ is a scaling hyperparameter (default: $\alpha = r$, giving $\frac{\alpha}{r} = 1$). During training only $A$ and $B$ are updated; $W$ is frozen. At the start of training, $B$ is initialised to zero so the adapter contributes nothing — training starts from the pretrained model's behaviour.

**The intuition.** Weight updates in fine-tuning are empirically low-rank: the gradient matrix has many near-zero singular values, meaning the "useful update" lives in a small subspace. LoRA captures that subspace directly without ever computing the full update.

**At inference.** $\frac{\alpha}{r} BA$ is computed once and added into $W$ — the adapter is *merged* into the weight. The result is a single weight matrix indistinguishable from a fully fine-tuned one. Zero latency overhead, zero extra memory, zero inference cost.

```
trainable params: ~4M  (out of 7B)   — r=8 applied to Q,K,V,O attention projections
VRAM during fine-tune: ~16 GB (vs. ~80 GB full fine-tune)
inference overhead: 0 — adapter merged into weights before serving
PizzaBot use: Ch.8 brand-voice adapter → $0.008/conv (vs $0.07 with generic GPT-3.5)
```

> 💡 **LoRA verdict:** Same output quality as full fine-tuning for style/domain shifts; 5× cheaper to train; zero production overhead. Default choice for custom model adaptation.

---

#### Prefix Tuning

**Technical definition.** Prefix tuning freezes all model weights and instead prepends $L_p$ learnable (key, value) pairs to the attention sequence of *every* transformer layer. These virtual KV pairs — the *prefix* — are trained end-to-end. The attention at each layer computes:

$$\text{Attention}(Q,\ [K_{prefix};K_{input}],\ [V_{prefix};V_{input}])$$

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

Several capabilities of LLMs were not explicitly trained for and appeared qualitatively at sufficient scale:

| Capability | Approximate threshold |
|---|---|
| In-context learning (few-shot) | ~7B parameters |
| Chain-of-thought reasoning | ~100B parameters |
| Multi-step arithmetic | ~540B parameters |
| Theory of mind (passing Sally-Anne test) | GPT-4 class |

**"Emergent"** does not mean magical. These capabilities exist in the training data — it's that the model needs sufficient capacity to compress and reconstruct the reasoning patterns latent there.

---

## 6 · What "Model Size" Actually Means

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

| Term | Meaning |
|---|---|
| $N$ | Total experts (e.g., 8, 16, 64) |
| $k$ | Active experts per token (typically 1 or 2) |
| $G(x)$ | Router — gating weights over experts for input $x$ |
| $E_i(x)$ | Expert $i$'s FFN output |

**Why MoE matters:**
- **Scale at fraction of cost:** GPT-4's 1.8T parameters → only ~200–400B active per token (roughly a dense 200B forward pass cost)
- **Specialisation:** Different experts naturally specialise — some activate on code, others on natural language, others on structured data
- **Training efficiency:** Total capacity scales with $N$; compute (and therefore cost) scales with $k$

> ⚠️ **VRAM trap:** For models like Mixtral-8×7B, VRAM is determined by **total** params loaded (~93 GB fp16), but *inference compute* is determined by **active** params (~12.9B per token). You still need the memory — you just don't pay full compute cost per forward pass.

**Inference cost scales with parameter count, context length, and batch size.** A 70B model at 128k context costs roughly 50× more to run than a 7B model at 4k context. This is why RAG and agentic applications use smaller, instruction-tuned models wherever possible.

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

## 8 · PizzaBot Connection

> See [AIPrimer.md](../ai-primer.md) for the full system definition.

| Concept | Where it shows up in PizzaBot |
|---|---|
| **Temperature** | `temperature=0` for the JSON order confirmation (no hallucinated prices). `temperature=0.8` for the "surprise me with a pizza" recommendation path. |
| **BPE tokenisation** | `"pepperoni"` → 3 tokens; `"gluten-free"` → 3 tokens. Matters when estimating the cost of embedding the menu corpus and when counting context window usage per turn. |
| **Context window** | A long SMS session (20+ back-and-forth turns) plus 5 retrieved chunks plus the system prompt approaches 8k tokens. This is why the agent must summarise older turns rather than pass the full history. |
| **Lost-in-the-middle** | If the allergen chunk is injected in the middle of a long context, the model may miss the gluten flag. Keep critical safety facts (allergens) near the system prompt or in a separate grounding block. |
| **RLHF sycophancy** | User: "I was told the Margherita is £8 today." Model (RLHF-aligned) may agree even if the RAG corpus says £13.99. Mitigation: grounding constraint + fact-check against retrieved price. |

---

## 9 · Progress Check — What We Can Solve Now

**Unlocked capabilities:**
- ✅ **Understand LLM architecture**: Know what pretraining → SFT → RLHF produces
- ✅ **Token-based cost estimation**: Can calculate API costs for PizzaBot conversations
- ✅ **Model selection framework**: Understand trade-offs between GPT-4, GPT-3.5, Claude tiers
- ✅ **Sampling parameter control**: Know when to use temperature=0 vs. 0.8
- ✅ **Context window awareness**: Understand how much menu data fits in one call

**Progress toward constraints:**

| Constraint | Status | Current State |
|------------|--------|---------------|
| #1 BUSINESS VALUE | ❌ **BLOCKED** | 8% conversion with raw GPT-3.5 (target >25%, phone baseline 22%) — System unusable |
| #2 ACCURACY | ❌ **BLOCKED** | ~40% error rate (hallucinates menu items, prices, calories) — Need grounding (Ch.4 RAG) |
| #3 LATENCY | ❌ **BLOCKED** | Simple queries take 2-5s but system can't complete orders yet |
| #4 COST | ⚡ **FOUNDATION** | Can estimate: 500 tokens/conv × $0.002/1k = $0.001 LLM cost (before RAG/tools add overhead) |
| #5 SAFETY | ❌ **BLOCKED** | No prompt injection defense, no guardrails — completely vulnerable |
| #6 RELIABILITY | ❌ **BLOCKED** | No error handling, no tool fallback mechanisms |

**What we can solve:**

✅ **Basic conversations (but unreliable)**:
- **User**: "What sizes do you have?"
- **GPT-3.5**: "We offer small, medium, and large pizzas."
- **Result**: ❌ Wrong! (Mamma Rosa's has personal, medium, large, extra-large)
  - ❌ Hallucinated answer not grounded in real menu
  - **Business impact**: Customer orders wrong size → confusion, refund request

✅ **Cost estimation framework**:
- **Calculation**: 3 turns × 150 tokens/turn = 450 tokens
- **LLM cost**: 450 × $0.002/1k = $0.0009 per conversation
- **Monthly projection**: 10k conversations × $0.0009 = **$9/month** (LLM only, very cheap!)
- **Business value**: Confirms LLM cost is not the bottleneck — we have budget for RAG, tools, embeddings

❌ **What we can't solve yet:**

**1. No grounding in Mamma Rosa's actual menu** → hallucinations everywhere:
- ❌ Claims pizzas exist that don't ("We have a Hawaiian BBQ pizza!")
- ❌ Makes up prices ("The Margherita is $12.99" — real price is $15.99)
- ❌ Invents calorie counts ("Around 550 calories" — real: 680 calories)
- **Business impact**: ~40% error rate → customer trust destroyed → conversion at 8%

**2. No structured output** → can't process orders reliably:
- ❌ Output format changes every query (sometimes JSON, sometimes prose)
- ❌ Can't extract: `{pizza: "Margherita", size: "large", quantity: 2}`
- ❌ Can't call `calculate_order_total()` API without structured data
- **Business impact**: Orders fail to process → manual intervention required → defeats automation goal

**3. No multi-step reasoning** → fails complex queries:
- ❌ "Cheapest gluten-free pizza under 600 calories" → picks wrong item or gives up
- ❌ Can't chain: filter gluten-free → filter by calories → sort by price → return cheapest
- **Business impact**: 30% of queries are multi-constraint → these users abandon immediately

**4. No business value** → 8% conversion kills the project:
- ❌ CEO's verdict: "Phone staff do better. Why are we building this?"
- ❌ Need to reach 25% conversion (3.1× improvement) to justify $300k investment
- ❌ Current ROI: Negative (8% conversion generates less revenue than phone baseline)

**Business metrics update:**
- **Order conversion**: 8% (baseline: 22% phone) — **❌ 14 points below target**
- **Average order value**: $36.20 (baseline: $38.50) — Slightly worse (no upselling)
- **Cost per conversation**: $0.001 LLM only (target <$0.08 total) — Good, but system doesn't work yet
- **Error rate**: ~40% (target <5%) — **❌ Catastrophic hallucination problem**

**Why the CEO should keep funding this (even though it doesn't work yet):**

1. **Low LLM cost ceiling**: Base API cost is $0.001/conv — plenty of budget left for RAG, tools, embeddings
2. **Clear path forward**: We know exactly what's broken (no grounding, no structure, no reasoning)
3. **Next 3 chapters fix the core problems**:
   - Ch.2 (Prompt Engineering): Structured output + system prompts → 12% conversion
   - Ch.3 (CoT Reasoning): Multi-step queries → 15% conversion
   - Ch.4 (RAG): Grounded menu answers → **18% conversion, <5% error rate**

**Next chapter**: [Prompt Engineering](../ch02_prompt_engineering) tackles the structured output problem with system prompts, few-shot examples, and JSON mode. We'll fix the format and bring conversion from 8% → 12%.

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

## 10 · Bridge

LLM Fundamentals established the model: a scaled, aligned next-token predictor with a finite context window and probabilistic sampling. The next document — `CoTReasoning.md` — shows how you exploit that predictor to produce step-by-step reasoning chains, and how those chains become the planning substrate for an agentic loop.

> *The model is the brain. It predicts tokens. Everything in the AI track — CoT, RAG, ReAct, Semantic Kernel — is about how you wire inputs and outputs around that single mechanical act.*

## Illustrations

![LLM fundamentals — BPE tokenisation, sampling, training stages, and the context window](img/LLM%20Fundamentals.png)
