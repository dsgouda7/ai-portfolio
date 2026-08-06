# LLM Fine-Tuning

Riverside House uses each notebook for one layer of the fine-tuning story: objective intuition, parameter mechanics, then evaluation and decisions.

1. [What Should the Model Learn?](01-llm-finetuning-data-techniques.ipynb) builds intuition for continued pretraining, response-masked SFT, PPO, and DPO using the complete 40-chapter Aria corpus.
2. [Where Should the Update Live?](02-llm-finetuning-parameter-techniques.ipynb) opens full fine-tuning, partial freezing, LoRA, and the QLoRA path through parameter, tensor, and artifact exercises.
3. [Evaluation, Comparison & Decision](03-llm-finetuning-comparison-and-decision.ipynb) introduces all evaluation concepts, independent data boundaries, workload-aligned measurements, candidate comparison, release gates, and production decisions.
4. [GPU Practice: Fine-Tune and Evaluate Every Riverside Novel](04-llm-finetuning-practice.ipynb) runs independent, chapter-disjoint LoRA experiments across all eight novels with validation-selected checkpoints and untouched test sets.

The notebooks share the committed `content/` corpus, generated `data/`, calibration artifacts, and local teaching checkpoints in this directory. Run `setup.ps1` on Windows or `setup.sh` on Linux/macOS; either script creates this chapter's `.venv`, installs `requirements.txt`, registers its Jupyter kernel, and assigns that kernel to all four notebooks.

Parts 1-3 select a CPU or CUDA profile. Part 4 is intentionally CUDA-only and stops immediately when PyTorch cannot see a compatible GPU.

## Continue Into Operations

This chapter owns objective choice, parameter-update strategy, evaluation design, and training
provenance. Parts 1 and 2 are mechanism notebooks; Part 3 is the evaluation and decision boundary.
Continue with:

- [AI Engineer: Training Data Quality and Lineage](../../ai-engineer/01-training-data-quality-and-lineage/README.md)
	for pre-training gates and dataset fingerprints;
- [AI Engineer: Release Registry and Lineage](../../ai-engineer/04-release-registry-and-lineage/README.md)
	for the distinction between training provenance and a complete application release;
- [Azure Operational LLM Serving](../../ai-infrastructure/09-azure-operational-llm-serving/README.md)
	for the authored, unexecuted local serving bridge;
- [Riverside release contracts](../../../projects/riverside-ai-platform/contracts/README.md) and
	[evaluation assets](../../../projects/riverside-ai-platform/evaluations/README.md) for the
	production-oriented source surfaces.

The Riverside source assets exist, but no live Azure deployment or serving result is claimed.
Azure compatibility and behavior remain **live-unvalidated**.
