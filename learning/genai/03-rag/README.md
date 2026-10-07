# Retrieval-Augmented Generation

This three-notebook sequence separates retrieval theory, pipeline implementation, and
retriever-versus-generator failure localization.

1. [Hybrid Retrieval Theory](01a-hybrid-retrieval-theory.ipynb) ·
   [Theory notes](01-hybrid-retrieval-theory.md)
2. [Hybrid Retrieval PyTorch Lab](01b-hybrid-retrieval-lab.ipynb) ·
   [Shared theory notes](01-hybrid-retrieval-theory.md)
3. [RAG Failure Localization and Evaluation](02-rag-failure-localization-and-evaluation.ipynb) ·
   [Theory notes](02-rag-failure-localization-and-evaluation-theory.md)

The theory notebook owns lexical-versus-dense failures, RRF intuition, candidate ceilings,
Recall@K/MRR, authorization ordering, and support boundaries. The lab owns the PyTorch encoder,
fusion sweeps, reranking, degraded mode, telemetry, and manifests. The final notebook owns
per-query RAG diagnosis and oracle-context ablation; general judge bias, hallucination detection,
and calibration remain in [`../04-llm-evaluation/`](../04-llm-evaluation/).

`rag_shared.py` centralizes deterministic fixtures, tokenization, seeding, and the shared teaching
caveat. The notebooks keep each teaching mechanism visible rather than hiding the pipeline behind
an end-to-end helper.

Run `setup.ps1` on Windows or `setup.sh` on Linux/macOS; either script creates this chapter's
`.venv`, installs `requirements.txt`, registers its Jupyter kernel, and assigns that kernel to all
three notebooks.

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
