# Retrieval-Augmented Generation

This two-notebook sequence separates retrieval quality from answer quality.

1. [Hybrid Search](04-hybrid-search.ipynb) makes dense and lexical retrieval fail differently, fuses their rankings, evaluates retrieval, and checks unsupported-query and authorization boundaries.
2. [RAG Evaluation](05-rag-evaluation.ipynb) diagnoses retriever versus generator failures, tests proxy limits, uses a gold-context ablation, and defines citation/refusal release boundaries.

Run `setup.ps1` on Windows or `setup.sh` on Linux/macOS; both use the environment defined in [`../_llm-shared/`](../_llm-shared/README.md). The deeper general evaluation track remains in [`../05-llm-evaluation/`](../05-llm-evaluation/).
