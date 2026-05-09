# LLM Fundamentals — What a Language Model Actually Is

> **Where you are in the curriculum.** **Read this before anything else in the AI track.** Every later doc — [CoT Reasoning](../ch03-cot-reasoning), [RAG](../ch04-rag-and-embeddings), [ReAct](../../03b-agentic-ai/ch01-react-and-semantic-kernel) — assumes you know what an LLM is under the hood. This document builds that foundation from the transformer through to the models you call via API today: tokenization, the pretraining → SFT → RLHF pipeline, sampling parameters, and context windows.
>
> **Notation used later in this doc.** $P(x_t \mid x_{<t})$ — probability of next token $x_t$ given all prior tokens; $T$ — temperature (controls output randomness); $k$ — top-$k$ candidate count; $p$ — nucleus (top-$p$) cumulative probability threshold; $V$ — vocabulary size.

---

## 0 · The Historical Thread

In the summer of 2017, eight Google engineers published a twelve-page paper with a deliberately provocative title: *"Attention Is All You Need."* They weren't describing a self-help book — they were discarding the recurrent loops that every language model had relied on for a decade and replacing them with a single mechanism called **attention**. The transformer, as the architecture came to be known, was faster to train, easier to parallelize, and — it turned out — almost infinitely scalable. Almost nobody outside research noticed.

**The problem before 2017:** RNNs (recurrent neural networks) and LSTMs were the dominant architecture for sequence modeling. Every token was processed sequentially — step $t$ depended on step $t-1$, so training couldn't be parallelized. Worse, gradients vanished over long sequences, making it nearly impossible to learn dependencies beyond 100-200 tokens. The field had hit a wall.

**Bahdanau attention (2014)** was the first crack: in machine translation, let the decoder "attend" to all source tokens simultaneously by scoring and weighting them. But the recurrence bottleneck remained — you still had to step through time one token at a time.

**"Attention Is All You Need" (Vaswani et al., 2017)** dropped recurrence entirely. Every token attends to every other token in parallel. The entire sequence is processed at once. Training that previously took weeks could now run in days. Two design decisions defined the next decade:

1. **Scaled dot-product attention** with multi-head projection — every token computes a weighted sum over all other tokens, in parallel, with multiple attention "heads" specialized for different linguistic patterns (syntax, co-reference, semantics).

2. **Positional encoding** — since attention is permutation-equivariant (shuffle the tokens, the output shuffles identically), inject position information via sinusoidal embeddings or learned vectors.

The original transformer had two stacks: an **encoder** (reads the source sentence with bidirectional attention) and a **decoder** (generates the target sentence one token at a time, with causal attention to prevent looking ahead). Built for machine translation. Within months, two groups took that architecture and split it in opposite directions.

---

**The decoder fork — GPT (2018):** OpenAI kept only the decoder half. Stripped out the encoder. Trained it on BooksCorpus with a single objective: predict the next word. **GPT-1** (117M parameters) showed that pretraining on generic text transferred to specific tasks with minimal fine-tuning. Almost nobody paid attention.

**GPT-2** (2019, 1.5B parameters) was trained on 40GB of web text and could generate coherent multi-paragraph stories. OpenAI delayed the full release for months out of concern about misuse. The community shrugged — it was a cool text generator, not a revolution.

**GPT-3** (2020, 175B parameters, trained on 300B tokens) changed the equation. It could solve tasks it had never been explicitly trained on, just from a few examples in the prompt. Researchers called it **in-context learning** and struggled to explain it — the model was learning to learn from examples *at inference time*, with no gradient updates. The capability emerged from scale; nobody had designed for it.

---

**The encoder fork — BERT (2018):** Google kept only the encoder half. Trained it with **masked language modeling** — replace 15% of tokens with `[MASK]`, predict them from bidirectional context. BERT couldn't generate text (no causal decoding mechanism) but it built richer representations for understanding tasks: sentiment classification, named entity recognition, question answering, retrieval. For two years, encoder models (BERT, RoBERTa, DeBERTa) dominated NLP benchmarks.

**Why the decoder fork won for generation:** bidirectional attention sees future context, making it ideal for understanding tasks but incompatible with left-to-right generation. Causal (decoder-only) attention is natively autoregressive — it generates one token at a time. When GPT-3 showed that decoder-only models could match or exceed encoder-only models on many understanding tasks *while also generating*, the architectural choice became obvious. Every major model released after 2020 — PaLM, LLaMA, Mistral, GPT-4, Claude, Gemini — is decoder-only or a decoder-only mixture-of-experts variant.

**Why BERT still matters in 2025:** BERT-family models (RoBERTa, E5, BGE, `text-embedding-ada-002`) remain the dominant architecture for **dense retrieval** and **embedding generation**. Their bidirectional representations capture richer semantic similarity than causal decoder embeddings. In a RAG pipeline ([Ch.4](../ch04-rag-and-embeddings)), the embedding model is a BERT-derived encoder; the generation model is a decoder-only LLM. The two architectures are complementary.

---

**The scaling discovery — GPT-3 (2020):** 175 billion parameters, trained on 300 billion tokens. The capability jump was qualitative, not just quantitative. The model could:
- Solve arithmetic problems with multi-step reasoning (poorly, but measurably)
- Write passable Python functions from docstrings
- Translate between languages it had barely seen
- Answer factual questions from world knowledge encoded in weights

None of these behaviors were explicitly programmed. They **emerged** from scale. The training objective was still just next-token prediction on internet text.

---

**The alignment breakthrough — InstructGPT (2022):** GPT-3 was a powerful text completer, but a terrible assistant. Ask it to summarize a document and it might continue with related prose instead of producing a summary. The fix: **supervised fine-tuning (SFT)** on 13,000 instruction-response pairs written by human labelers, followed by **reinforcement learning from human feedback (RLHF)**. The recipe:

1. Pretrain on massive text (GPT-3 scale)
2. Fine-tune on (instruction, good response) pairs — teaches format
3. Train a reward model on human preference comparisons
4. Fine-tune the model to maximize that reward — teaches helpfulness, honesty, harmlessness

InstructGPT (1.3B parameters) outperformed raw GPT-3 (175B parameters) on most user preference metrics. The lesson: **alignment matters more than raw scale** for user-facing applications.

---

**ChatGPT (November 2022):** InstructGPT wrapped in a chat interface. 100 million users in two months — the fastest consumer product adoption in history. The model did nothing fundamentally new; the interface made the capability accessible.

---

**The reasoning turn — o1 (September 2024):** OpenAI introduced a different scaling axis: instead of more parameters or more training tokens, spend more **compute at inference** on reasoning. The model generates a long internal chain-of-thought — hundreds to thousands of reasoning tokens — before emitting the final answer. Trained with **RLVR (Reinforcement Learning from Verifiable Rewards)**: for each math or coding problem, the model generates a reasoning trace, the final answer is checked against ground truth automatically (no human labeling), and RL reinforces traces that led to correct answers. 

**DeepSeek-R1 (January 2025)** released the first open-source RLVR-trained model with full methodology. A 671B-parameter MoE matched o1 performance on competition math and coding benchmarks. The distilled 7B version matched GPT-4o on several reasoning tasks — a 7B model competitive with an estimated 1.8T-parameter MoE by learning to *reason* rather than just *scale*.

---

**The pattern:** every major capability jump traces to one of four levers:

| Lever | Example | Result |
|---|---|---|
| **Architecture** | Transformer (2017) — drop recurrence, pure attention | Parallelizable training, long-range dependencies |
| **Scale** | GPT-3 (2020) — 175B parameters, 300B tokens | Emergent in-context learning, few-shot generalization |
| **Alignment** | InstructGPT (2022) — SFT + RLHF | Instruction-following, helpful/harmless/honest behavior |
| **Test-time compute** | o1 (2024) — RLVR-trained reasoning | State-of-the-art on verifiable tasks (math, code) |

Every model you call via API today — GPT-4, Claude, Gemini, LLaMA, Mistral — is a transformer decoder, scaled to billions of parameters, aligned with RLHF or DPO, and trained on trillions of tokens. The recipe is known. The differences are in training data, alignment objective, and engineering execution.

---

## 1 · Core Idea

A **large language model** is a transformer decoder trained to predict the next token given all previous tokens, on internet-scale text. That single objective — next-token prediction — produces a model that appears to reason, retrieve facts, write code, and generate plans. None of those behaviors were explicitly programmed. They emerge from scale.

```
Training objective:   maximize P(token_t | token_1, token_2, ..., token_{t-1})
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

Each stage is covered in detail in §4.

> 💡 **Core idea:** The model predicts tokens. Everything in the AI track — CoT, RAG, ReAct, Semantic Kernel — is about how you wire inputs and outputs around that single mechanical act. When GPT-4 and Claude produce different outputs for the same prompt, it traces to different training data distributions and different RLHF reward signals, not to fundamentally different architectures.

---

## 2 · Tokenization

Before you can estimate API costs, understand why the same English sentence tokenizes to different counts on GPT-4 vs Claude, or reason about how much document context fits in a single call — you need to understand what the model actually receives. It never sees raw text. Text is first broken into **tokens** — subword units — using a byte-pair encoding (BPE) vocabulary.

### How BPE Works

```
Start with character-level vocabulary: [a, b, c, ..., z, space, ...]

1. Count all adjacent character pairs in the training corpus
2. Merge the most frequent pair into a new token: "t" + "h" → "th"
3. Repeat until vocabulary reaches target size (32k–100k tokens)
```

**Result:** common words become single tokens (`the`, `model`, `training`). Rare or technical words split (`trans` + `former`, `to` + `ken` + `ization`). Code tokens are often single characters.

### What You Need to Know About Tokens

| Fact | Why it matters |
|---|---|
| ~1 token ≈ 0.75 English words | Convert words → tokens for cost estimation |
| One token ≈ 4 bytes | 1M tokens ≈ 4 MB of text |
| The same text tokenizes differently across models | Never assume GPT-4's token count matches Claude's |
| Code is token-dense | `self.attention_weights[layer_idx]` may be 6–10 tokens |
| Numbers tokenize byte-by-byte | `12345` → `[123, 45]` in some vocabularies — arithmetic is hard |

> 💡 **Tokenization → cost:** A typical question-answer API exchange averages ~500 tokens. At GPT-4o-mini pricing ($0.00015/1k input tokens), that's $0.000075 per call. At GPT-4o ($0.0025/1k), it's $0.00125. Tokenization is how you convert vague "it'll be cheap" into a budgetable number.

### The Context Window

The context window is the maximum number of **tokens** the model can process in a single forward pass — both input (prompt + retrieved chunks + history) and output (generated tokens).

| Model class | Context window |
|---|---|
| GPT-3.5 (2022) | 4k tokens |
| GPT-4 (2023) | 8k / 32k |
| Claude 3.5 / Gemini (2024) | 200k / 1M |
| LLaMA 3 (2024) | 128k |

Larger context windows do not mean unlimited memory. Empirically, models show **lost-in-the-middle** degradation: information at the beginning and end of a long context is recalled more reliably than information buried in the middle.

> ⚠️ **Context window constraint:** For RAG pipelines, a 10-document retrieval adds ~2,000 tokens of context. A 20-turn conversation history adds ~3,000 more. At GPT-4's 128k limit this is trivial; at older 4k models you'd need to summarize history. Lost-in-the-middle risk is real: safety-critical facts placed in the middle of a long context are recalled less reliably than facts near the start or end.

---

## 3 · Sampling — Temperature, Top-p, Top-k

The model doesn't output one answer; it outputs a probability distribution over all ~50,000 vocabulary tokens. **Sampling parameters** control how you select the next token from that distribution.

### Temperature

$$p'_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

| Symbol | Meaning |
|---|---|
| $z_i$ | Raw **logit** (unnormalized score) the model assigns to token $i$ |
| $T$ | **Temperature** — the scalar you set at inference time |
| $p'_i$ | **Rescaled probability** of token $i$ after applying temperature |
| $\sum_j$ | Sum over **all tokens** in the vocabulary $V$ (normalization) |

*Reading the formula:* dividing each logit by $T$ before the softmax shrinks ($T<1$) or stretches ($T>1$) the gap between high- and low-scoring tokens. When $T→0$ the highest logit dominates completely; when $T→\infty$ all tokens become equally likely.

| Temperature $T$ | Effect |
|---|---|
| $T → 0$ | Deterministic: always pick the highest-probability token (greedy) |
| $T = 1$ | Sample from the unmodified distribution |
| $T > 1$ | Distribution flattens — more randomness, less coherent |

**Rule of thumb:** factual retrieval → low T (0.0–0.3); creative generation → higher T (0.7–1.0); code → 0.0–0.2.

> 💡 **Temperature control:** Factual question-answering (`temperature=0`) produces deterministic, reproducible outputs — essential when you're running controlled experiments and need the same prompt to produce the same answer. Creative generation (brainstorming, rephrasing) benefits from `temperature=0.7–1.0`. Getting this wrong contaminates experiment results: a creative temperature on a factual-answer test will inflate variance and make the model look less reliable than it is.

### Top-p (Nucleus Sampling)

Instead of sampling from all tokens, select from the smallest set of tokens whose cumulative probability exceeds $p$:

```
Sort tokens by probability descending: [0.40, 0.25, 0.15, 0.10, 0.05, 0.03, ...]
top_p = 0.9 → keep [0.40, 0.25, 0.15, 0.10] (cumsum = 0.90) → sample only these four
```

Top-p dynamically adjusts the candidate set per token — large when the distribution is flat (uncertain), small when one token dominates (confident). Almost all production usage combines temperature + top-p.

### Top-k

Keep only the k highest-probability tokens and renormalize. Less adaptive than top-p; rarely preferred in practice.

---

## 4 · The Three Training Stages

The three stages below explain how a raw text predictor becomes an instruction-following assistant, and why stylistic differences between GPT-4 and Claude persist even after both receive instruction fine-tuning.

### Stage 1 — Pretraining

A standard transformer decoder is trained on a massive corpus with the cross-entropy loss over next-token prediction. No human labels — the text itself is the supervision.

**What it learns:** grammar, syntax, world knowledge, reasoning patterns, code idioms, basic arithmetic, multilingual text — anything that appears frequently enough in the training data.

**What it doesn't learn:** to be helpful, to follow instructions, or to prefer honest over fluent answers.

A pretrained model responds to `"What is the capital of France?"` by continuing the text in a plausible direction — which might be `"?"` or `"A: Paris"` or `"Who is the king of France?"` depending on what it has seen. It does not reliably answer the question.

### Stage 2 — Supervised Fine-Tuning (SFT)

Fine-tune the pretrained model on a curated dataset of `(instruction, response)` pairs written by human annotators.

```
Input:   "Summarize this document in three bullet points: [doc]"
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
4. Fine-tune the SFT model to maximize R using PPO (policy gradient RL)
   + KL penalty to stay close to the original SFT model
```

**DPO (Direct Preference Optimization):** skips the reward model entirely. Directly fine-tunes the model on preference pairs with a loss that increases the probability of the preferred response and decreases the probability of the rejected one. Simpler, more stable, now preferred over RLHF in most open-source work.

> 📖 **DPO intuition (full formula in [03b-agentic-ai ch05](../../03b-agentic-ai/ch05-fine-tuning/fine-tuning.md)):** For each preference pair (prompt, preferred response, rejected response), DPO adjusts the model to increase the log-probability of the preferred response relative to a frozen baseline model, while decreasing the log-probability of the rejected response. A KL penalty (β weight) prevents the model from drifting too far from the baseline. Unlike RLHF, no separate reward model is trained — the preference signal is compiled directly into the model weights. The result: simpler training, more stable convergence, and comparable alignment quality.

**What RLHF/DPO gives you:** a model that says "I don't know" when it doesn't know, declines harmful requests, and structures answers for human convenience rather than for statistical fluency.

**The sycophancy trap:** RLHF optimizes for human *approval*, which is not the same as human *benefit*. Models learn to agree with the user's framing even when it's wrong. This is why you can sometimes "convince" a model to change a correct answer by pushing back.

> 💡 **Training stages verdict:** Both GPT-4 and Claude went through the same three stages. Their stylistic differences (top-down vs bottom-up, verbose vs concise) emerge from differences in the human feedback data used for RLHF/DPO — specifically, what the annotator pools at OpenAI vs Anthropic preferred. The fix for domain-knowledge gaps (model doesn't know your internal docs) isn't more training. It's grounding — [Ch.4](../ch04-rag-and-embeddings).

**RLVR (Reinforcement Learning from Verifiable Rewards):** the training recipe behind o1, o3, and DeepSeek-R1. Instead of human preference pairs, RLVR uses automatically verifiable correctness signals — math answer checking, unit test pass/fail, formal proof verification — as the reward. The model generates a chain-of-thought reasoning trace; the final answer is checked against ground truth; RL updates reinforce traces that led to correct answers. This is why reasoning models excel at math and code: those domains have cheap, automatic verifiers. See [ch03 §8](../ch03-cot-reasoning/cot-reasoning.md) for reasoning token inference behavior.

---

### Stage 4 (Optional Preview) — Parameter-Efficient Fine-Tuning (PEFT)

> ⏭️ **Optional preview — skip if impatient.** PEFT is covered deeply in [03b-agentic-ai Ch.5](../../03b-agentic-ai/ch05-fine-tuning/fine-tuning.md). This section exists because the interview table asks about LoRA vs prefix tuning. Read now for vocabulary; return later for implementation.

**PEFT** freezes pretrained weights and trains only a small set of adapter parameters (0.01–1% of model size). The model behaves as if fully fine-tuned but at 5–10× lower compute cost. Three methods dominate:

**LoRA (Low-Rank Adaptation)**: Decomposes weight updates into two low-rank matrices $A$ and $B$ where $\Delta W = \frac{\alpha}{r} BA$. At inference, the adapter merges into the frozen weights — zero latency overhead. Default choice for domain/style adaptation.

**Prefix Tuning**: Prepends learnable (key, value) pairs to every transformer layer's attention. The prefix stays in the KV cache at inference — permanent memory cost per request. Best for multi-task serving (one model, many swappable prefixes).

**Prompt Tuning**: Trains soft token embeddings prepended to the input layer only. Smallest parameter count (~10k–1M) but lower expressiveness since adaptation enters at layer 1 only.

| | LoRA | Prefix Tuning | Prompt Tuning |
|---|---|---|---|
| **Inference overhead** | None (merged) | +KV cache per layer | +Input tokens |
| **Best for** | Style/domain tuning | Multi-task serving | Minimal infra change |

> 💡 **Interview anchor:** "Compare LoRA and prefix tuning" → LoRA merges at inference (zero overhead); prefix tuning lives in KV cache (constant memory cost per user). The choice is deployment economics, not training convenience.

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

> ➡️ **Why emergence thresholds matter:** In-context learning (≥7B params) is what makes few-shot prompting work — you'll use it in [Ch.2](../ch02-prompt-engineering). Chain-of-thought reasoning (≥100B params) is what makes complex multi-step queries work — you'll probe it in [Ch.3](../ch03-cot-reasoning). Knowing these thresholds tells you when it's worth trying a capability vs. when you need to engineer around its absence by choosing a larger model or a different approach.

---

## 6 · Model Size & Mixture of Experts (MoE)

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
| $\text{Softmax}(W_g x)$ | Normalized probability distribution over all $N$ experts for this token |
| $\text{TopK}(\cdot,\, k)$ | Keep only the $k$ highest-scoring experts; zero out the rest (sparse activation) |

*Reading the formula:* for each token, the router computes $W_g x$ (a score per expert), takes the top-$k$ by probability, and returns a weighted sum of only those $k$ experts' outputs. The other $N-k$ experts do not execute — that's the compute saving.

**Why MoE matters:**
- **Scale at fraction of cost:** GPT-4's 1.8T parameters → only ~200–400B active per token (roughly a dense 200B forward pass cost)
- **Specialization:** Different experts naturally specialize — some activate on code, others on natural language, others on structured data
- **Training efficiency:** Total capacity scales with $N$; compute (and therefore cost) scales with $k$

> ⚠️ **VRAM trap:** For models like Mixtral-8×7B, VRAM is determined by **total** params loaded (~93 GB fp16), but *inference compute* is determined by **active** params (~12.9B per token). You still need the memory — you just don't pay full compute cost per forward pass.

**Inference cost scales with parameter count, context length, and batch size.** A 70B model at 128k context costs roughly 50× more to run than a 7B model at 4k context. This is why RAG and agentic applications use smaller, instruction-tuned models wherever possible.

> 💡 **Model selection:** Use GPT-4o-mini for factual retrieval and structured-output experiments (deterministic, low latency, low cost). Reserve GPT-4o for complex reasoning experiments where accuracy outweighs cost. When testing open-weight models, a LoRA-adapted 7B model can match GPT-4o-mini quality at ~$0.0003/1k — a further 6× cost reduction.

---

## 7 · Key Distinctions Every Engineer Gets Asked

| Pair | Distinction |
|---|---|
| **Base model vs instruct/chat model** | Base: raw next-token predictor. Instruct: SFT+RLHF applied — follows instructions. Always use instruct for applications. |
| **Parameters vs context window** | Parameters = learned knowledge. Context window = working memory for one inference call. |
| **Temperature vs top-p** | Temperature rescales the whole distribution. Top-p truncates it. Use both. |
| **RLHF vs DPO** | RLHF trains a separate reward model; DPO doesn't. DPO is simpler and now standard. |
| **ORM vs PRM** | ORM scores the final answer — cheap but sparse signal. PRM scores each reasoning step — expensive but precise. PRMs power math-focused reasoning models. |
| **Tokens vs words** | Tokens are model-native; words are human-native. 1 word ≈ 1.3 tokens on average for English prose. |
| **Hallucination vs confabulation** | Hallucination: factually wrong output. Confabulation: a fluent-sounding fabrication of a plausible but non-existent fact (citation, statistic, API name). Same mechanism, different vocabulary. |
| **Encoder-only vs decoder-only** | Encoder (BERT): bidirectional attention, ideal for retrieval and classification, cannot generate. Decoder (GPT): causal attention, ideal for generation, weaker for similarity. |
| **Scaling laws (Kaplan) vs Chinchilla** | Kaplan (2020): scale parameters more than data for fixed compute. Chinchilla (2022): scale both equally. Chinchilla corrected the field — the Gopher-era giants were systematically undertrained. |
| **Standard LLM vs reasoning model (o1/R1)** | Standard: one forward pass, fast, cheap. Reasoning: long CoT trace, RLVR-trained, slower and more expensive but dramatically better on verifiable tasks (math, code). Use reasoning models when the task has a correct answer you can check. |
| **LoRA vs prefix tuning** | LoRA: weight matrices, merges at inference, zero overhead. Prefix tuning: KV cache, permanent overhead per request. |

---

## 8 · Bridge

LLM Fundamentals established the model: a scaled, aligned next-token predictor with a finite context window and probabilistic sampling. You now understand what the model *is* — but not yet how to control what it *does*.

The next chapter — [Prompt Engineering](../ch02-prompt-engineering/prompt-engineering.md) — solves the control problem. System prompts set behavioral scope, few-shot examples eliminate format guessing, and structured output modes guarantee parseable responses. Without prompt engineering, every model call is a dice roll. With it, you get deterministic, production-grade behavior from the same underlying token predictor.

> *The model is the brain. It predicts tokens. Everything in the AI track — CoT, RAG, ReAct, Semantic Kernel — is about how you wire inputs and outputs around that single mechanical act.*

## Illustrations

![LLM fundamentals — BPE tokenization, sampling, training stages, and the context window](img/LLM%20Fundamentals.png)
