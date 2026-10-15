# Stage 07: Retrieval-Augmented Generation

**Contents:** [one-line map of this directory](CONTENTS.md)

**Previous:** [Stage 06 · LLM Fine-Tuning](../06-llm-finetuning/README.md)

**Current:** Stage 07 · RAG

**Next:** [Stage 08 · LLM Evaluation](../08-llm-evaluation/README.md)

## Stage Boundary: Learned Behavior Is Not Request-Time Evidence

Stage 06 changed **learned behavior**: fine-tuning altered how the model tends to respond. Stage 07
adds **request-time evidence**: retrieval selects current, authorized passages for this particular
question. One cannot substitute for the other.

| Bridge case | What you observe | What it means |
| --- | --- | --- |
| Good model, bad retrieval | A capable model gives a fluent, well-formed answer from the wrong passage | The answer can be wrong even when generation behavior is healthy; repair retrieval before retraining the model |
| Same model, oracle context | Supplying the labeled gold passage repairs the answer | The oracle-context gain localizes the bottleneck to retrieval; it is diagnostic evidence, not a deployable retrieval strategy |
| Same model, oracle context, no repair | The answer stays unsupported or incorrect | Retrieval was not the controlling defect; inspect context use, answer construction, or the evaluator |

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
and calibration remain in [`../08-llm-evaluation/`](../08-llm-evaluation/).

## Handoff to Stage 08: Preserve the Evidence Chain

Do not hand Stage 08 one blended "RAG score." Preserve the causal chain:

| Evidence link | Question the next stage must answer |
| --- | --- |
| Retrieval recall | Did the required evidence enter the candidate set? |
| Context support | Does the selected context support the answer's atomic claims? |
| Claim attribution | Which source supports each claim, and which claims remain unsupported or contradictory? |
| Confidence and risk routing | Given calibrated confidence and claim risk, may the system answer, abstain, or require review? |

Healthy downstream signals cannot average away a broken upstream link. Stage 08 turns these
per-query records into versioned evaluators and decision thresholds.

`rag_shared.py` centralizes deterministic fixtures, tokenization, seeding, and the shared teaching
caveat. The notebooks keep each teaching mechanism visible rather than hiding the pipeline behind
an end-to-end helper.

Run `setup.ps1` on Windows or `setup.sh` on Linux/macOS; either script creates this chapter's
`.venv`, installs `requirements.txt`, registers its Jupyter kernel, and assigns that kernel to all
three notebooks.

## Continue Into Operations

This chapter owns retrieval, citation, refusal, authorization, and retriever-versus-generator
diagnosis. Continue with:

- [FDE: Data Onboarding and Contracts](../12-role-based-tracks/fde/03-data-onboarding-and-contracts/README.md)
	for source ownership, parsing, quality, ACL, lineage, sync, and deletion decisions;
- [RAG Knowledge Pipeline](../../projects/rag-knowledge-pipeline/README.md) for the independently
	deployable local ingest, vectorization, and serving boundaries;
- [Databricks Index Operations](../../projects/rag-knowledge-pipeline/databricks/indexing/OPERATIONS.md)
	for the remote governed-record and Direct Vector Access source assets;
- [Riverside architecture](../../projects/riverside-ai-platform/docs/architecture.md) for the
	contract boundary between the Databricks data plane and Azure serving composition.

The local project and remote Databricks source assets are distinct paths. The remote assets exist,
but workspace identity, Unity Catalog, Delta merge, vector filtering, deletion, latency, and cost
remain **live-unvalidated** in Azure Databricks.
