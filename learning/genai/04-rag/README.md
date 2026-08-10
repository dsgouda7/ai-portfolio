# Retrieval-Augmented Generation

This two-notebook sequence separates retrieval quality from answer quality.

1. [Hybrid Search](04-hybrid-search.ipynb) · [Theory notes](04-hybrid-search-theory.md)
2. [RAG Evaluation](05-rag-evaluation.ipynb) · [Theory notes](05-rag-evaluation-theory.md)

Run `setup.ps1` on Windows or `setup.sh` on Linux/macOS; either script creates this chapter's `.venv`, installs `requirements.txt`, registers its Jupyter kernel, and assigns that kernel to both notebooks. The deeper general evaluation track remains in [`../05-llm-evaluation/`](../05-llm-evaluation/).

## Continue Into Operations

This chapter owns retrieval, citation, refusal, authorization, and retriever-versus-generator
diagnosis. Continue with:

- [FDE: Data Onboarding and Contracts](../../role-based-tracks/fde/03-data-onboarding-and-contracts/README.md)
	for source ownership, parsing, quality, ACL, lineage, sync, and deletion decisions;
- [RAG Knowledge Pipeline](../../../projects/rag-knowledge-pipeline/README.md) for the independently
	deployable local ingest, vectorization, and serving boundaries;
- [Databricks Index Operations](../../../projects/rag-knowledge-pipeline/databricks/indexing/OPERATIONS.md)
	for the remote governed-record and Direct Vector Access source assets;
- [Riverside architecture](../../../projects/riverside-ai-platform/docs/architecture.md) for the
	contract boundary between the Databricks data plane and Azure serving composition.

The local project and remote Databricks source assets are distinct paths. The remote assets exist,
but workspace identity, Unity Catalog, Delta merge, vector filtering, deletion, latency, and cost
remain **live-unvalidated** in Azure Databricks.
