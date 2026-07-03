# AI Track Primer — Before You Read Chapter 1

> This primer is for readers who have just completed the Transformers chapter in the ML track (ML ch10_transformers) and are about to start the AI/LLM track. It answers three questions: what changed from ML's transformer coverage, what this track adds, and how to use it efficiently.

## Part 1: What This Track Covers

The ML track's ch10_transformers gave you: the attention mechanism, positional encoding, encoder-decoder architecture, and a PyTorch transformer implementation.

This track picks up exactly there and adds:
- **ch00**: Bridge from sequence models to language models (what changes when the task is language, not classification)
- **ch01**: Transformer architecture in depth (tokenization/BPE, full scaled dot-product attention math, multi-head, positional encodings including RoPE, three architecture families)
- **ch02**: LLM inference mechanics (autoregressive generation, KV cache, sampling strategies, continuous batching, PagedAttention)
- **ch03**: LLM training pipeline (pretraining, SFT, RLHF, DPO, LoRA — concepts; implementation is in 05-agentic-ai)
- **ch04**: LLM model internals (parameter counting, VRAM budgeting, quantization, Flash Attention, MoE, GQA)
- **ch05**: Prompt engineering (base vs instruct, system prompts, few-shot, temperature, scope enforcement)
- **ch06**: Chain-of-thought reasoning (CoT, self-consistency, Tree-of-Thought, test-time compute, o1/DeepSeek-R1)
- **ch07**: RAG and embeddings (chunking, retrieval, hybrid search BM25+dense, HyDE, intent routing)
- **ch08**: Vector databases (distance metrics, IVF/HNSW/PQ/DiskANN, scaling ladder, production architectures)

## Part 2: What This Track Does NOT Cover (and Where Those Topics Live)

| Topic | Where it lives |
|---|---|
| LLM guardrails, jailbreak defense, PII detection | [05-agentic-ai/ch02-safety-and-hallucination](../05-agentic-ai/ch02-safety-and-hallucination/) |
| LLM-as-a-judge, eval pipeline, RAGAS | [05-agentic-ai/ch03-evaluating-ai-systems](../05-agentic-ai/ch03-evaluating-ai-systems/) |
| Cost optimization, LLM gateways, semantic caching | [05-agentic-ai/ch04-cost-and-latency](../05-agentic-ai/ch04-cost-and-latency/) |
| Fine-tuning implementation (LoRA/QLoRA running code) | [05-agentic-ai/ch05-fine-tuning](../05-agentic-ai/ch05-fine-tuning/) |
| Local LLM serving, vLLM, inference optimization | [07-ai-infrastructure](../07-ai-infrastructure/) |
| ReAct agents, tool use, multi-step reasoning | [05-agentic-ai/ch01-react-and-semantic-kernel](../05-agentic-ai/ch01-react-and-semantic-kernel/) |

## Part 3: Suggested Reading Order

1. If you completed ML ch10_transformers: start at ch01 (ch00 is optional context).
2. If you came from outside the ML track: read ch00 first.
3. After ch01-ch04: you have enough to do the prerequisites-test.md self-assessment.
4. ch05-ch08 can be read in any order after ch01-ch04.

---

## The Running Example — The Intelligence Audit

Every note — from LLM fundamentals to vector database scaling — refers back to this investigation framework. You are a **Staff Engineer** assigned to your company's AI Adoption Review. Your mandate: evaluate two leading models — **GPT-4o** and **Claude 3.5 Sonnet** — across a structured experiment suite and deliver an **AI Literacy Kit** to guide the engineering org's AI adoption decisions.

The investigation follows a **hypothesis -> experiment -> finding** cadence. Each chapter is one experiment type:

| Chapter | Hypothesis | Key Question |
|---|---|---|
| ch01 - LLM Fundamentals | Models are black boxes; let us open them | How does next-token prediction produce coherent, factual answers? |
| ch02 - Inference Mechanics | Generation strategy changes output quality | How does sampling temperature, KV cache, and batching affect quality and cost? |
| ch03 - Training Pipeline | Alignment methods produce qualitatively different models | When does DPO beat RLHF, and what does LoRA actually change? |
| ch04 - Model Internals | VRAM and compute are the real bottlenecks | How do quantization and MoE change deployment decisions? |
| ch05 - Prompt Engineering | We can control LLM behavior through prompts | How do system prompts, few-shot, and JSON mode affect output reliability? |
| ch06 - CoT Reasoning | Step-by-step prompting improves multi-step logic | When does CoT help, and at what cost? |
| ch07 - RAG & Embeddings | Grounding in private docs reduces hallucination | How much does retrieval reduce hallucination on internal data? |
| ch08 - Vector Databases | ANN indexes make retrieval viable at production scale | What is the recall/latency trade-off at 50k documents? |

---

## Document Map

| File | Purpose |
|---|---|
| [authoring-guide.md](authoring-guide.md) | Track conventions and style |
| [ch00-from-networks-to-language/README.md](ch00-from-networks-to-language/README.md) | ML foundations bridge: RNNs → attention → transformers → language modeling |
| [ch01-transformer-architecture/transformer-architecture.md](ch01-transformer-architecture/transformer-architecture.md) | Tokenization, BPE, attention mechanics, positional encoding, architecture families |
| [ch02-llm-inference-mechanics/inference-mechanics.md](ch02-llm-inference-mechanics/inference-mechanics.md) | Autoregressive generation, KV cache, sampling, production serving |
| [ch03-llm-training-pipeline/training-pipeline.md](ch03-llm-training-pipeline/training-pipeline.md) | Pretraining, SFT, RLHF, DPO, LoRA |
| [ch04-llm-model-internals/model-internals.md](ch04-llm-model-internals/model-internals.md) | Parameter counting, VRAM, quantization, MoE, GQA |
| [ch05-prompt-engineering/prompt-engineering.md](ch05-prompt-engineering/prompt-engineering.md) | Base vs instruct, system prompts, few-shot, structured output, injection defense |
| [ch06-cot-reasoning/cot-reasoning.md](ch06-cot-reasoning/cot-reasoning.md) | CoT, self-consistency, Tree-of-Thought, o1, DeepSeek-R1 |
| [ch07-rag-and-embeddings/rag-and-embeddings.md](ch07-rag-and-embeddings/rag-and-embeddings.md) | Embeddings, chunking, retrieval, hybrid search, HyDE |
| [ch08-vector-dbs/vector-dbs.md](ch08-vector-dbs/vector-dbs.md) | ANN index types, HNSW vs IVF vs DiskANN, production architecture |

---

## How to Use This Efficiently

**If you have 2 hours and want the essentials:**
Read ch01 (what an LLM is), ch05 (how to control it), ch07 (how to ground it in your data). That covers the practical stack for 90% of production LLM tasks.

**If you are preparing for a senior AI/ML interview:**
Read all chapters. Focus on the "Key Distinctions" section in each chapter. The [interview-guides/ai.md](../interview-guides/ai.md) guide covers the highest-frequency questions.

**If you are coming from the ML track and want to extend your transformer knowledge:**
Read ch01 -> ch02 -> ch04 in order. These three chapters bridge from "I understand attention" to "I can make deployment decisions."

**Security note:** All code examples that connect to LLM APIs load keys from environment variables only.
