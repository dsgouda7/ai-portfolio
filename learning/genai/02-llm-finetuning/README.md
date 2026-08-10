# LLM Fine-Tuning

Riverside House uses each notebook for one layer of the fine-tuning story: objective intuition, parameter mechanics, then evaluation and decisions.

**Series anchor:** Parts 1 and 2 reuse the same Aria thread from *The Weight of Distant Light* so the objective and parameter choices stay directly comparable.

1. [What Should the Model Learn?](01-llm-finetuning-data-techniques.ipynb) · [Theory notes](01-llm-finetuning-data-techniques-theory.md)
2. [Where Should the Update Live?](02-llm-finetuning-parameter-techniques.ipynb) · [Theory notes](02-llm-finetuning-parameter-techniques-theory.md)
3. [Evaluation, Comparison & Decision](03-llm-finetuning-comparison-and-decision.ipynb) · [Theory notes](03-llm-finetuning-comparison-and-decision-theory.md)
4. [GPU Practice: Fine-Tune and Evaluate Every Riverside Novel](04-llm-finetuning-practice.ipynb) · [Theory notes](04-llm-finetuning-practice-theory.md)

The notebooks share the committed `content/` corpus, generated `data/`, calibration artifacts, and local teaching checkpoints in this directory. Run `setup.ps1` on Windows or `setup.sh` on Linux/macOS; either script creates this chapter's `.venv`, installs `requirements.txt`, registers its Jupyter kernel, and assigns that kernel to all four notebooks.

Parts 1-3 select a CPU or CUDA profile. Part 4 is intentionally CUDA-only and stops immediately when PyTorch cannot see a compatible GPU.

## Continue Into Operations

This chapter owns objective choice, parameter-update strategy, evaluation design, and training
provenance. Parts 1 and 2 are mechanism notebooks; Part 3 is the evaluation and decision boundary.
Continue with:

- [AI Engineer: Training Data Quality and Lineage](../../role-based-tracks/ai-engineer/01-training-data-quality-and-lineage/README.md)
	for pre-training gates and dataset fingerprints;
- [AI Engineer: Release Registry and Lineage](../../role-based-tracks/ai-engineer/04-release-registry-and-lineage/README.md)
	for the distinction between training provenance and a complete application release;
- [Azure Operational LLM Serving](../../ai-infrastructure/09-azure-operational-llm-serving/README.md)
	for the authored, unexecuted local serving bridge;
- [Riverside release contracts](../../../projects/riverside-ai-platform/contracts/README.md) and
	[evaluation assets](../../../projects/riverside-ai-platform/evaluations/README.md) for the
	production-oriented source surfaces.

The Riverside source assets exist, but no live Azure deployment or serving result is claimed.
Azure compatibility and behavior remain **live-unvalidated**.
