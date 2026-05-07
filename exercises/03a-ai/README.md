# Exercise 03: PizzaBot AI — LLM Fine-Tuning, RAG, and Prompt Engineering

> **Grand Challenge:** Build production-quality AI models comparing LoRA fine-tuning, RAG pipelines, and few-shot prompting. Achieve BLEU 0.3+, retrieval accuracy 70%+, and perplexity <50.

**Scaffolding Level:** 🔴 Minimal (demonstrate independence with inline TODOs)

---

## 0. Grand Challenge: Mamma Rosa's PizzaBot

> 🎯 **The Mission**: Build **Mamma Rosa's PizzaBot** — a conversational AI ordering system that beats human phone staff on business metrics while maintaining safety and reliability.

**What PizzaBot is:**  
An AI ordering assistant that must outperform the existing phone staff. The CEO is skeptical that AI can deliver superior business value. This isn't a demo chatbot — you're proving AI can handle real production constraints: conversion rate, accuracy, latency, cost, safety, and uptime.

**Phone baseline (what we're competing against):**
- 22% conversion rate (orders per conversation)
- $38.50 average order value
- <2% error rate on menu facts (prices, ingredients, calories)
- ~2s average response time
- ~$0.08 per conversation cost (~$157k/year in labor)
- 99.2% uptime

---

### Current Constraint Status

| # | Constraint | Target | Current Status |
|---|------------|--------|----------------|
| **#1** | **BUSINESS VALUE** | >25% conversion + +$2.50 AOV | ❌ Not met (track unlocks 27% conv, +$2.80 AOV) |
| **#2** | **ACCURACY** | <5% error rate | ❌ Not met (track achieves <5% with RAG) |
| **#3** | **LATENCY** | <3s p95 response time | ❌ Not met (track achieves <2s p95) |
| **#4** | **COST** | <$0.08 per conversation | ❌ Not met (track achieves $0.06/conv) |
| **#5** | **SAFETY** | Zero successful attacks | ❌ Not met (track implements guardrails) |
| **#6** | **RELIABILITY** | >99% uptime | ❌ Not met (track achieves >99% with monitoring) |

---

### What's Blocking Us

**Without the techniques in this exercise track:**

1. **No grounding → hallucinations**: Base LLM has 40% error rate on menu facts (prices, specials, ingredients). Customers receive wrong information → trust destroyed → 8% conversion (vs 22% phone baseline).

2. **No domain adaptation → generic responses**: Without prompt engineering or fine-tuning, LLM doesn't understand pizza ordering patterns → sounds like a help desk, not an order-taker → low conversion.

3. **No tool orchestration → can't process orders**: LLM can chat but can't execute actions (check inventory, calculate totals, apply coupons, confirm orders) → conversational dead-end.

4. **No safety hardening → vulnerable to attacks**: Prompt injection can leak customer data, bypass business rules, or generate harmful content → PR disaster + regulatory risk.

5. **No optimization → too slow and expensive**: Naive API calls cost $0.15/conv (87% over budget) and take 5-8s → customers hang up, economics don't work.

6. **No monitoring → production blindness**: Can't measure reliability, detect regressions, or debug failures → can't ship to production with confidence.

---

### What This Exercise Unlocks

This track teaches 11 chapters of AI techniques, progressively unlocking each constraint:

| Chapters | What You'll Learn | Constraints Unlocked | Metrics Achieved |
|----------|-------------------|----------------------|------------------|
| **1-3** | LLM fundamentals, prompt engineering, chain-of-thought | Foundation knowledge | 15% conv, 10% errors |
| **4-5** | RAG & embeddings, vector databases | **#2 Accuracy ✅** | 18% conv, **<5% errors** |
| **6-7** | ReAct agents, safety & guardrails | **#1 Business Value ✅, #5 Safety ✅** | **27% conv, +$2.80 AOV**, zero attacks |
| **8-10** | Evaluation, cost/latency optimization, fine-tuning | **#3 Latency ✅, #4 Cost ✅, #6 Reliability ✅** | **<2s p95, $0.06/conv, >99% uptime** |
| **11** | Advanced agentic patterns | Edge case mastery | 0.7% edge case errors (11× improvement) |

**After completing this exercise track:**
- You'll understand **why RAG matters** (grounding prevents hallucination → trust → conversion)
- You'll compare **three approaches** (fine-tuning vs RAG vs few-shot prompting) on real metrics (BLEU >0.3, perplexity <50, retrieval accuracy >70%)
- You'll build **production-ready AI systems** that beat human baselines on business metrics
- You'll know **when to use** each technique (RAG for facts, fine-tuning for style, agents for actions)

**The CEO's verdict progression:**

```
Start:        8% conv, 40% errors → "This is embarrassing. Phone staff never give wrong info."
After Ch.2:  12% conv, 15% errors → "Better, but still unreliable. Can't deploy this."
After Ch.4:  18% conv,  5% errors → "Okay, we're approaching parity. What about upselling?"
After Ch.6:  27% conv, +$2.80 AOV → "Wait... this is BETTER than phone staff?!"
After Ch.9:  <2s p95, $0.06/conv → "And it's faster AND cheaper?! Let's deploy."
After Ch.10: >99% uptime         → 🎉 "Best decision we made. Scaling to all locations."
After Ch.11: 0.7% edge cases     → 🧠 "Handles complex scenarios that stump phone staff!"
```

---

**What's blocking us:**
1. **Knowledge grounding**: Without RAG, bot hallucinates menu prices (15% error rate) → customer trust breaks → 12% conversion
2. **Domain adaptation**: Base LLM doesn't understand pizza ordering patterns → generic responses → low conversion
3. **Prompt optimization**: No systematic approach to few-shot examples → inconsistent quality → unreliable performance

**What this exercise unlocks:**
- **Constraint #2 (Accuracy)**: RAG pipeline → <5% error rate (from current 15%)
- **Foundation for Constraint #1 (Business Value)**: Fine-tuned model + RAG → 18% conversion (from 12%), approaching 22% baseline
- **Technical capabilities**: Evaluation metrics (BLEU, perplexity, retrieval accuracy) to measure progress

**After completing this exercise:**
- You'll understand *why* RAG matters (grounding prevents hallucination → trust → conversion)
- You'll compare *three approaches* (fine-tuning vs RAG vs few-shot prompting) on real metrics
- You'll know *how to evaluate* AI systems (perplexity <50, BLEU >0.3, retrieval accuracy >70%)

---

## Overview

PizzaBot AI is an educational exercise teaching modern LLM techniques through hands-on implementation. You'll build three different AI approaches from scratch and compare them:

1. **LLM Fine-Tuning** with LoRA/QLoRA (parameter-efficient training)
2. **RAG Pipeline** with vector search and context injection
3. **Few-Shot Prompting** with example-based learning

All with **immediate console feedback**, **evaluation metrics**, and **plug-and-play comparison**.

### Key Learning Objectives

- 🧠 **LLM Fine-Tuning**: Implement LoRA (Low-Rank Adaptation) for domain adaptation
- 🔍 **RAG Pipeline**: Build retrieval-augmented generation with ChromaDB/FAISS
- 📝 **Prompt Engineering**: Master few-shot learning and prompt optimization
- 📊 **Evaluation**: Measure perplexity, BLEU, ROUGE, and retrieval accuracy
- 🏆 **Comparison**: Discover which approach works best for pizza ordering

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Text Input (User Query)                │
└────────────────────┬────────────────────────────────────┘
                     │
      ┌──────────────┴──────────────┬───────────────┐
      │                             │               │
      ▼                             ▼               ▼
┌──────────────┐          ┌──────────────┐   ┌─────────────┐
│  Fine-Tuned  │          │     RAG      │   │  Few-Shot   │
│  LLM (LoRA)  │          │   Pipeline   │   │  Prompting  │
└──────┬───────┘          └──────┬───────┘   └──────┬──────┘
       │                         │                    │
       │                    ┌────┴─────┐             │
       │                    │          │             │
       │                    ▼          ▼             │
       │              ┌──────────┐ ┌─────────┐      │
       │              │ Vector   │ │  LLM    │      │
       │              │ Database │ │  Base   │      │
       │              │(Chroma/  │ │ Model   │      │
       │              │ FAISS)   │ │         │      │
       │              └──────────┘ └─────────┘      │
       │                                             │
       └──────────────────┬──────────────────────────┘
                          ▼
              ┌─────────────────────┐
              │  Evaluation Metrics  │
              │  BLEU, ROUGE, PPL    │
              │  Retrieval Accuracy  │
              └─────────────────────┘
```

---

## Key Concepts

### 1. LLM Fine-Tuning with LoRA

**What is LoRA?**
- Low-Rank Adaptation: Only train 0.1% of model parameters
- Adds small trainable matrices to frozen LLM weights
- Quality comparable to full fine-tuning, 10-100x faster

**Why LoRA over Full Fine-Tuning?**
- **Memory**: 7B model needs 14GB RAM vs. 28GB for full fine-tuning
- **Speed**: Minutes vs. hours for domain adaptation
- **Modularity**: Swap LoRA adapters without reloading base model

**QLoRA Enhancement:**
- Uses 4-bit quantization for even lower memory
- 7B model runs on 6GB GPU (vs. 14GB with LoRA alone)
- Minimal quality loss with bitsandbytes quantization

**Implementation in `src/models.py`:**
```python
class LLMFineTuner(AIModel):
    def train(self, train_data, eval_data, config):
        # 1. Load base model (frozen)
        # 2. Apply LoRA config (only train q_proj, v_proj)
        # 3. Fine-tune on pizza ordering examples
        # 4. Evaluate with perplexity and BLEU
```

**Time Estimate:** 60-90 minutes  
**Metrics:** Perplexity (lower better), BLEU (0.3+ target)

---

### 2. RAG (Retrieval-Augmented Generation)

**What is RAG?**
1. **Retrieve**: Find relevant documents from vector database
2. **Augment**: Inject documents into prompt context
3. **Generate**: LLM generates response with grounded info

**Why RAG?**
- ✅ **No hallucinations**: Facts come from retrieved docs
- ✅ **No retraining**: Just update document collection
- ✅ **Explainable**: Can show which docs were used
- ❌ **Latency**: Retrieval adds 50-200ms
- ❌ **Context limits**: LLM context window constrains docs

**RAG Pipeline:**
```
User Query
  ↓
Text Embedding (sentence-transformers)
  ↓
Vector Search (ChromaDB/FAISS)
  ↓
Top-K Documents Retrieved
  ↓
Context Injection (prompt engineering)
  ↓
LLM Generation
  ↓
Response with Citations
```

**Vector Database Options:**
- **ChromaDB**: Easy API, persistent storage, metadata support
- **FAISS**: Faster for 1M+ docs, more memory efficient

**Implementation in `src/models.py` and `src/features.py`:**
```python
class RAGPipeline(AIModel):
    def train(self, train_data, eval_data, config):
        # 1. Load LLM and embedding model
        # 2. Build vector database from training docs
        # 3. Evaluate retrieval accuracy (70%+ target)
        # 4. Evaluate generation quality (BLEU, ROUGE)

class VectorDatabase:
    def create(self, embedding_dim):
        # ChromaDB or FAISS initialization
    def add_documents(self, documents, embeddings):
        # Index documents for semantic search
    def query(self, query_embedding, top_k):
        # Return top-k most similar documents
```

**Time Estimate:** 45-60 minutes  
**Metrics:** Retrieval accuracy (70%+ target), BLEU, ROUGE-L

---

### 3. Few-Shot Prompting

**What is Few-Shot Learning?**
- Provide 3-7 example pairs in prompt
- LLM learns pattern without training
- Often matches fine-tuning quality!

**Zero-Shot vs. Few-Shot vs. Many-Shot:**
```
Zero-Shot:
  "You are a pizza ordering assistant. User: What pizzas do you have?"

Few-Shot (3 examples):
  "Example 1: Input: What's on the menu? Output: We have...
   Example 2: Input: I want pepperoni. Output: Great! That's...
   Example 3: Input: Delivery time? Output: 30-45 minutes...
   
   Now respond: Input: What pizzas do you have?"

Many-Shot (10+ examples):
  [Same pattern with more examples - hits context limit]
```

**When to Use Few-Shot?**
- ✅ **Quick iteration**: No training needed
- ✅ **Small datasets**: Works with 3-10 examples
- ✅ **API-based LLMs**: Can't fine-tune GPT-4, use prompting
- ❌ **Context limits**: Long prompts reduce output space
- ❌ **Consistency**: Slight variations in phrasing matter

**Implementation in `src/models.py`:**
```python
class PromptEngineer(AIModel):
    def train(self, train_data, eval_data, config):
        # 1. Select diverse few-shot examples (random or clustering)
        # 2. Build prompt template with examples
        # 3. Evaluate on test queries (BLEU, ROUGE)

    def generate(self, prompt, ...):
        # Inject few-shot examples into prompt
        # Generate with base LLM (no training)
```

**Time Estimate:** 30-45 minutes  
**Metrics:** BLEU, ROUGE-L (compare to fine-tuning)

---

## Evaluation Metrics

### Perplexity (PPL)
- **What**: Measures language model "confusion"
- **Formula**: `PPL = exp(average_loss)`
- **Target**: <50 for domain-specific tasks, <30 is excellent
- **Used for**: Fine-tuned LLMs only (not RAG/few-shot)

### BLEU Score
- **What**: Precision-based metric (n-gram overlap)
- **Range**: 0-1 (higher better)
- **Target**: 0.3+ for conversational AI, 0.5+ for translation
- **Used for**: All models (fine-tuning, RAG, few-shot)
- **Limitation**: Favors exact matches, misses paraphrases

### ROUGE-L Score
- **What**: Recall-based metric (longest common subsequence)
- **Range**: 0-1 (higher better)
- **Target**: 0.4+ for summarization, 0.3+ for Q&A
- **Used for**: All models, better for longer outputs
- **Limitation**: Doesn't consider word order

### Retrieval Accuracy
- **What**: % of queries where expected doc is in top-k results
- **Target**: 70%+ at k=5, 85%+ at k=10
- **Used for**: RAG pipeline evaluation
- **Formula**: `hits / total_queries`

---

## Infrastructure Note

> **Note:** This exercise focuses on AI model implementation. Infrastructure files (Docker, Makefile, etc.) have been intentionally removed to keep the exercise focused on core LLM concepts. For production deployment patterns, see `exercises/01-ml/` which includes containerization and orchestration examples.

---

## Directory Structure

```
exercises/03-ai/
├── src/
│   ├── models.py           # LLMFineTuner, RAGPipeline, PromptEngineer (TODOs)
│   ├── features.py         # TextPreprocessor, EmbeddingGenerator, VectorDatabase (TODOs)
│   ├── utils.py            # Logging, config helpers
│   └── __init__.py         # Package exports
├── main.py                 # Interactive demo comparing all approaches
├── requirements.txt        # Dependencies (transformers, sentence-transformers, etc.)
├── README.md               # This file
├── data/
│   └── vector_db/          # Persisted ChromaDB/FAISS index
├── knowledge_base/
│   └── pizza_menu.txt      # Pizza ordering knowledge
└── _REFERENCE/
    ├── models_complete.py  # Solution reference (PizzaBot chatbot)
    └── api_complete.py     # API reference (Flask endpoints)
```

---

## Setup

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

**Key Dependencies:**
- `transformers` (4.30+): HuggingFace LLMs
- `peft` (0.4+): LoRA fine-tuning
- `sentence-transformers` (2.2+): Embeddings
- `chromadb` (0.4+): Vector database
- `nltk` (3.8+): BLEU score
- `rouge-score` (0.1+): ROUGE metrics
- `rich` (13.0+): Beautiful console output

### 2. Download NLTK Data

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

### 3. Run Interactive Demo

```bash
python main.py
```

---

## Implementation Guide

### Order of Implementation (Recommended)

#### Phase 1: Text Processing & Embeddings (1-1.5 hours)
1. `src/features.py` → `TextPreprocessor.preprocess()` [20 min]
2. `src/features.py` → `TextPreprocessor.preprocess_batch()` [10 min]
3. `src/features.py` → `EmbeddingGenerator.load()` [10 min]
4. `src/features.py` → `EmbeddingGenerator.encode()` [5 min]
5. `src/features.py` → `EmbeddingGenerator.encode_batch()` [10 min]

**Test:** Run `main.py` — should see preprocessing and embedding generation

#### Phase 2: Vector Database (1-1.5 hours)
6. `src/features.py` → `VectorDatabase.create()` [15 min]
7. `src/features.py` → `VectorDatabase.add_documents()` [20 min]
8. `src/features.py` → `VectorDatabase.query()` [20 min]
9. `src/features.py` → `VectorDatabase.query_text()` [5 min]
10. `src/features.py` → `VectorDatabase.evaluate_retrieval()` [25 min]

**Test:** Run `main.py` — should see retrieval accuracy

#### Phase 3: AI Models (2-3 hours)
11. `src/models.py` → `PromptEngineer.train()` [30 min] ← Start here (easiest)
12. `src/models.py` → `PromptEngineer.generate()` [10 min]
13. `src/models.py` → `RAGPipeline.train()` [45 min]
14. `src/models.py` → `RAGPipeline.generate()` [15 min]
15. `src/models.py` → `LLMFineTuner.train()` [60 min] ← Most complex
16. `src/models.py` → `LLMFineTuner.generate()` [10 min]

**Test:** Run `main.py` — should see each model train and generate

#### Phase 4: Experiment Framework (30 min)
17. `src/models.py` → `ExperimentRunner.run_experiment()` [10 min]
18. `src/models.py` → `ExperimentRunner.print_leaderboard()` [10 min]

**Test:** Run `main.py` — should see full pipeline with leaderboard

---

## Common Pitfalls & Hints

### Memory Issues (CUDA Out of Memory)
```python
# Solution 1: Reduce batch size
config = AIModelConfig(batch_size=4)  # Default: 8

# Solution 2: Use gradient checkpointing
model.gradient_checkpointing_enable()

# Solution 3: Use QLoRA instead of LoRA
use_qlora=True  # 4-bit quantization
```

### Slow Training
```python
# Use smaller model for testing
base_model="gpt2"  # 124M params vs. gpt2-large (774M)

# Reduce epochs
max_epochs=3  # Default: 5

# Use CPU for quick testing
use_gpu=False
```

### Low BLEU Scores
- **Expected**: BLEU 0.2-0.4 for conversational AI (not translation)
- **Improve**: More training data, longer training, better examples
- **Check**: Are outputs in right format? Tokenization correct?

### Poor Retrieval Accuracy
- **Expected**: 60-80% at k=5 for small datasets
- **Improve**: Better embedding model (all-mpnet-base-v2), more docs
- **Check**: Are test queries semantically similar to training docs?

---

## Success Criteria

**Minimum Viable Product (MVP):**
- ✅ All TODOs implemented and working
- ✅ BLEU ≥ 0.25 for at least one model
- ✅ Retrieval accuracy ≥ 60% (RAG)
- ✅ Leaderboard shows all 3 approaches

**Stretch Goals:**
- 🎯 BLEU ≥ 0.35 for fine-tuned model
- 🎯 Retrieval accuracy ≥ 75%
- 🎯 Perplexity < 30 for fine-tuned model
- 🎯 Interactive demo works smoothly

---

## References

### LLM Fine-Tuning
- [LoRA Paper (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314)
- [HuggingFace PEFT Documentation](https://huggingface.co/docs/peft)

### RAG
- [RAG Paper (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)

### Prompt Engineering
- [Few-Shot Learning (Brown et al., 2020)](https://arxiv.org/abs/2005.14165)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)

### Evaluation
- [BLEU Paper (Papineni et al., 2002)](https://aclanthology.org/P02-1040/)
- [ROUGE Paper (Lin, 2004)](https://aclanthology.org/W04-1013/)

---

## Support

- **Concepts from**: [notes/03-ai/](../../notes/03-ai/) Ch.1-12
- **Reference solution**: `_REFERENCE/models_complete.py` (chatbot implementation)
- **Issues**: Check TODO comments for step-by-step instructions
- **Stuck?**: Focus on one model first (start with `PromptEngineer`)

---

**Time Estimate:** 4-6 hours for complete implementation  
**Learning Outcome:** Deep understanding of LLM fine-tuning, RAG, prompt engineering, and evaluation metrics
