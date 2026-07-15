# LLM Concepts — Deep Dive Notebooks

Comprehensive educational notebooks covering advanced LLM concepts from first principles. Each notebook follows the pedagogical approach of building intuition through concrete examples, visualizations, and problem-solution narratives.

## 📚 Notebooks

### [rag-evaluation.ipynb](rag-evaluation.ipynb)
**RAG Evaluation Metrics from First Principles**

Learn how to evaluate Retrieval-Augmented Generation systems using metrics like Faithfulness, Answer Relevance, Context Precision, and Context Recall. Builds intuition through concrete examples and shows when metrics disagree.

**Key Topics:**
- The RAG evaluation challenge
- Metric definitions with toy examples
- Failure modes and edge cases
- Implementing metrics from scratch
- Trade-offs and best practices

**Prerequisites:** Basic understanding of embeddings and RAG architecture

---

### [hybrid-search.ipynb](hybrid-search.ipynb)
**Hybrid Search: Combining Semantic and Keyword Retrieval**

Understand when pure semantic search fails, when pure keyword search (BM25) fails, and how to combine them effectively using fusion strategies like Reciprocal Rank Fusion (RRF).

**Key Topics:**
- The search spectrum (semantic vs. keyword)
- Concrete failing examples for each approach
- BM25 intuition and math
- Fusion strategies (RRF, weighted, learned)
- Side-by-side comparisons

**Prerequisites:** Basic understanding of embeddings and vector similarity

---

### [llm-gateway.ipynb](llm-gateway.ipynb)
**LLM Gateway Patterns: Routing, Fallback, and Caching**

Learn production patterns for managing LLM requests efficiently: intelligent routing to different models, fallback strategies for reliability, caching for cost reduction, and load balancing.

**Key Topics:**
- Production LLM challenges (cost, latency, reliability)
- Routing patterns with decision trees
- Fallback strategies and circuit breakers
- Caching patterns (semantic, exact, approximate)
- Monitoring and best practices

**Prerequisites:** Experience calling LLM APIs, basic understanding of distributed systems

---

## 🎯 Pedagogical Approach

All notebooks follow a consistent learning framework:

1. **Start with the Problem** — Motivate why the concept is needed through concrete failure modes
2. **Build Intuition** — Use toy examples with actual numbers you can trace through
3. **Visualize** — Show concepts graphically before diving into code
4. **Implement from Scratch** — Build understanding by coding the core logic yourself
5. **Use Libraries** — Then show the production-ready way
6. **Show Trade-offs** — Discuss when to use each approach

## 📖 Recommended Reading Order

For learners new to LLM systems:

1. **Start here:** `hybrid-search.ipynb` — Builds foundation for retrieval
2. **Then:** `rag-evaluation.ipynb` — Learn to measure RAG quality
3. **Finally:** `llm-gateway.ipynb` — Production deployment patterns

For learners focused on production deployment:

1. **Start here:** `llm-gateway.ipynb` — Get systems patterns first
2. **Then:** `hybrid-search.ipynb` — Optimize retrieval
3. **Finally:** `rag-evaluation.ipynb` — Measure improvements

## 🔗 Related Content

- **Fine-Tuning:** See [../llm-tuning/llm_finetuning_deep_dive.ipynb](../llm-tuning/llm_finetuning_deep_dive.ipynb) for comprehensive coverage of 6 fine-tuning techniques
- **Transformers:** See [../transformers/transformers.ipynb](../transformers/transformers.ipynb) for the transformer architecture from first principles
- **Playground:** Original exploratory notebooks in `playground/af-advanced-ai/` (these notebooks are cleaned-up, educational versions)

## 💡 Learning Tips

- **Run cells incrementally** — Don't execute everything at once; trace through examples step-by-step
- **Modify the examples** — Change parameters and observe what breaks or improves
- **Draw diagrams** — Sketch the concepts on paper before reading the provided visualizations
- **Compare approaches** — Many notebooks show multiple solutions; understand when each is appropriate

## 🛠️ Setup

Each notebook includes a setup cell for dependencies. Generally requires:

```bash
pip install transformers sentence-transformers rank-bm25 ragas
```

Some notebooks use visualization libraries (matplotlib, seaborn, plotly) which are installed in the notebook setup cells.

---

**Feedback?** These notebooks are living documents. If you find concepts unclear, examples unconvincing, or have suggestions for additional topics, contributions are welcome.
