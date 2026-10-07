# LLM Fine-Tuning

Riverside House uses four notebooks for four decisions: what the examples should teach, where the
update should live, what independent evidence supports, and whether the complete CUDA evidence
contract survives across eight novels.

**Series anchor:** Parts 1 and 2 reuse the same Aria scene from *The Weight of Distant Light*. Part
3 keeps every candidate tied to one declared job. Part 4 starts with the same novel as a pilot
before it permits the other seven runs.

1. [What Should the Model Learn?](01-llm-finetuning-data-techniques.ipynb) · [Theory notes](01-llm-finetuning-data-techniques-theory.md)
2. [Where Should the Update Live?](02-llm-finetuning-parameter-techniques.ipynb) · [Theory notes](02-llm-finetuning-parameter-techniques-theory.md)
3. [Evaluation, Comparison & Decision](03-llm-finetuning-comparison-and-decision.ipynb) · [Theory notes](03-llm-finetuning-comparison-and-decision-theory.md)
4. [GPU Practice: Fine-Tune and Evaluate Every Riverside Novel](04-llm-finetuning-practice.ipynb) · [Theory notes](04-llm-finetuning-practice-theory.md)

The notebooks share the repository-owned `content/` corpus, generated `data/`, and local teaching
checkpoints. Run `setup.ps1` on Windows or `setup.sh` on Linux/macOS; either script creates this
chapter's `.venv`, installs `requirements.txt`, registers its Jupyter kernel, and assigns that
kernel to all four notebooks.

Parts 1-3 select a CPU or CUDA profile. Part 4 has two explicit modes:

- `FINETUNING_PRACTICE_MODE=preflight` is the default CPU-safe path. It audits all chronological
  splits, duplicate checks, stage ordering, checkpoint schedule, bootstrap size, and gates without
  loading a tokenizer or model.
- `FINETUNING_PRACTICE_MODE=cuda` enables model work and fails immediately if PyTorch cannot see a
  compatible GPU. Add `FINETUNING_RUN_ONE_NOVEL=1` for the required pilot. Add
  `FINETUNING_RUN_ALL_NOVELS=1` only after the pilot stages have been inspected. Destructive
  replacement remains a separate `FINETUNING_OVERWRITE_RUNS=1` opt-in.

The CUDA notebook order is fixed:

```text
environment and split audit
-> one-novel pilot
-> validation selection and reload parity
-> one-time test opening
-> paired bootstrap, retention, and gates
-> the other seven novels and final ledgers
```

No quick path bypasses test isolation.

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
