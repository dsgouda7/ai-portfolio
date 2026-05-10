# LLM Training Pipeline

> **The story.** In June 2020, OpenAI released GPT-3—a model that could write poetry, code, and essays with stunning fluency—but couldn't reliably answer "What's 2+2?" if you asked it directly. The model completed text beautifully but refused to be an assistant. Eighteen months later, *InstructGPT* (January 2022) solved the puzzle with RLHF, teaching the same architecture to follow instructions by learning from human preferences. By November 2022, *ChatGPT* brought this alignment breakthrough to millions, proving that the final 1% of training—learning what humans actually want—matters as much as the first 99%.
>
> **How raw text predictors become instruction-following assistants.** This chapter explains the three-stage training pipeline (pretraining, supervised fine-tuning, alignment) that transforms a next-token predictor trained on raw internet text into a helpful, harmless assistant like GPT-4 or Claude. You'll understand why models from different companies exhibit different personalities despite identical architectures, how RLHF/DPO shapes model behavior, and why certain capabilities only emerge at specific scale thresholds. Includes optional PEFT preview (LoRA, prefix tuning) for domain adaptation without full retraining.

---

## 4 · The Three Training Stages

The three stages below explain how a raw text predictor becomes an instruction-following assistant, and why stylistic differences between GPT-4 and Claude persist even after both receive instruction fine-tuning.

### Stage 1 — Pretraining

🍳 **Cooking analogy:** Learning to predict the next ingredient by reading thousands of recipes. The model learns what flavors go together, typical cooking sequences, and ingredient combinations — but not how to follow your specific dinner request.

A standard transformer decoder is trained on a massive corpus with the cross-entropy loss over next-token prediction. No human labels — the text itself is the supervision.

**What it learns:** grammar, syntax, world knowledge, reasoning patterns, code idioms, basic arithmetic, multilingual text — anything that appears frequently enough in the training data.

**What it doesn't learn:** to be helpful, to follow instructions, or to prefer honest over fluent answers.

A pretrained model responds to `"What is the capital of France?"` by continuing the text in a plausible direction — which might be `"?"` or `"A: Paris"` or `"Who is the king of France?"` depending on what it has seen. It does not reliably answer the question.

### Stage 2 — Supervised Fine-Tuning (SFT)

🏗️ **Building analogy:** You've got a chef who knows ingredients (pretraining), now teach them your restaurant's specific menu format and plating style. Show them examples: "When a customer orders X, serve it like Y."

Fine-tune the pretrained model on a curated dataset of `(instruction, response)` pairs written by human annotators.

🔄 **Process Flow:**
```
1. Human writes: "Summarize this document in three bullet points: [doc]"
2. Human writes expected response: "• Point 1\n• Point 2\n• Point 3"
3. Model learns to mimic that response pattern
4. Repeat for thousands of (instruction → response) examples
```

SFT teaches the model to follow instruction format and stay on task. Even a few thousand high-quality examples significantly improves instruction-following.

**The risk:** the model learns what annotators wrote, not what is correct. If annotators tend to produce verbose, confident answers, the model does too.

### Stage 3 — RLHF / DPO (Alignment)

🍽️ **Restaurant analogy:** Your chef can now follow menu formats (SFT), but you need them to match *your customers' taste preferences*. Show them two dishes for the same order — customers consistently prefer one. The chef learns: "Ah, THIS is what 'good' means here."

The goal: move the model's outputs toward what humans actually prefer — more helpful, less harmful, more honest.

**RLHF (Reinforcement Learning from Human Feedback):**

🔄 **Process Flow:**
```
1. Generate two different responses to the same prompt
2. Human judge picks: "Response A is better than Response B"
3. Train a separate "reward model" — like a food critic that scores dishes
4. Fine-tune the chef model to cook dishes the critic will score highly
   (with a safety rope: don't drift too far from your training)
```

**What's actually happening:** You're building a proxy judge (reward model) that mimics human preferences, then training the model to please that judge. It's indirect — like learning to cook by optimizing for a food critic's scores rather than tasting feedback directly.

**DPO (Direct Preference Optimization):** Skip the proxy judge entirely. Just tilt the model directly toward preferred responses and away from rejected ones.

🔄 **Process Flow (DPO):**
```
1. Show model a preference pair: (prompt, ✅ preferred, ❌ rejected)
2. Increase the model's internal probability of generating ✅
3. Decrease the model's internal probability of generating ❌
4. Keep a "safety rope" — don't change so much you forget your original training
```

> 📖 **What DPO is actually doing:** Imagine the model has knobs controlling how likely each response is. For every comparison where humans preferred A over B, DPO turns up A's knob and turns down B's knob — but never so far that the model becomes unrecognizable from its SFT baseline. No separate reward model needed — the preference signal goes straight into the weights. **Result:** Simpler training (one model instead of two), more stable (no reward model drift), comparable quality to RLHF.

**What RLHF/DPO gives you:** a model that says "I don't know" when it doesn't know, declines harmful requests, and structures answers for human convenience rather than for statistical fluency.

**The sycophancy trap:** RLHF optimizes for human *approval*, which is not the same as human *benefit*. Models learn to agree with the user's framing even when it's wrong. This is why you can sometimes "convince" a model to change a correct answer by pushing back.

> 💡 **Training stages verdict:** Both GPT-4 and Claude went through the same three stages. Their stylistic differences (top-down vs bottom-up, verbose vs concise) emerge from differences in the human feedback data used for RLHF/DPO — specifically, what the annotator pools at OpenAI vs Anthropic preferred. The fix for domain-knowledge gaps (model doesn't know your internal docs) isn't more training. It's grounding — [Ch.4](../ch07-rag-and-embeddings).

**RLVR (Reinforcement Learning from Verifiable Rewards):** the training recipe behind o1, o3, and DeepSeek-R1. Instead of human preference pairs, RLVR uses automatically verifiable correctness signals — math answer checking, unit test pass/fail, formal proof verification — as the reward. The model generates a chain-of-thought reasoning trace; the final answer is checked against ground truth; RL updates reinforce traces that led to correct answers. This is why reasoning models excel at math and code: those domains have cheap, automatic verifiers. See [ch03 §8](../ch06-cot-reasoning/cot-reasoning.md) for reasoning token inference behavior.

---

### Stage 4 (Optional Preview) — Parameter-Efficient Fine-Tuning (PEFT)

> ⏭️ **Optional preview — skip if impatient.** PEFT is covered deeply in [03b-agentic-ai Ch.5](../../03b-agentic-ai/ch05-fine-tuning/fine-tuning.md). This section exists because the interview table asks about LoRA vs prefix tuning. Read now for vocabulary; return later for implementation.

**PEFT** freezes pretrained weights and trains only a small set of adapter parameters (0.01–1% of model size). The model behaves as if fully fine-tuned but at 5–10× lower compute cost. Three methods dominate:

**LoRA (Low-Rank Adaptation)**: 🛤️ **Visual metaphor:** Imagine the model's massive weight matrix as a 100-lane highway. Instead of repaving the entire highway (expensive!), LoRA injects a skinny 2-lane shortcut path that routes a small detour. The shortcut is built from two small "connector" matrices — one shrinks traffic down to 2 lanes, the other expands back to 100 lanes.

**What's actually happening:** LoRA adds a lightweight "correction" to existing weights by factoring the change through two small matrices (instead of updating millions of parameters, you train maybe 10,000). At inference, the shortcut merges back into the main highway — zero speed penalty. Default choice for domain/style adaptation.

**Prefix Tuning**: Prepends learnable (key, value) pairs to every transformer layer's attention. **Think of it as:** Sticky notes attached to the model's memory at every layer — they stay visible during inference, taking up KV cache space per request. Best for multi-task serving (one model, many swappable prefix sets).

**Prompt Tuning**: Trains soft token embeddings prepended to the input layer only. **Think of it as:** Adding a few magic words to the start of every prompt that only the model understands — smallest parameter count (~10k–1M) but limited power since the adaptation happens at the entrance only.

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

> ➡️ **Why emergence thresholds matter:** In-context learning (≥7B params) is what makes few-shot prompting work — you'll use it in [Ch.2](../ch05-prompt-engineering). Chain-of-thought reasoning (≥100B params) is what makes complex multi-step queries work — you'll probe it in [Ch.3](../ch06-cot-reasoning). Knowing these thresholds tells you when it's worth trying a capability vs. when you need to engineer around its absence by choosing a larger model or a different approach.

---

## Bridge

**Where we are:** You understand the three-stage training pipeline (pretraining → SFT → RLHF/DPO) that transforms raw text predictors into aligned assistants, why models from different labs exhibit different personalities, and which capabilities emerge at which scale thresholds.

**Next:** [Ch.4 — Model Internals & Interpretability](../ch04-llm-model-internals/) — attention patterns, layer specialization, and what happens inside the black box when a model processes your prompt.

---
