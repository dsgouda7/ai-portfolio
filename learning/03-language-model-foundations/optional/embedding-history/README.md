# Optional Embedding History

**Contents:** [one-line map of this directory](CONTENTS.md)


This is optional adjacent background, not a required step toward attention. It follows the history of static word vectors; the main route continues without it.

[From Counts to Static Geometry](01-from-counts-to-static-geometry.ipynb) follows the representation lineage from one-hot and bag-of-words through Word2Vec and GloVe, then stops at the static/contextual boundary.

This branch intentionally excludes sentiment classification, pretrained pipeline demos, and semantic search. Sentiment is an application task; semantic search belongs in the RAG track where retrieval has a concrete job.

Run `setup.ps1` on Windows or `setup.sh` on Linux/macOS. The script creates or reuses this chapter's `.venv`, installs `requirements.txt`, registers the unique `stage03-language-model-optional-embedding-history` kernel, and assigns it to the notebook. Use `-SkipKernel` or `--skip-kernel` when only the local environment and dependencies are needed.
