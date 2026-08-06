# Training Data Quality and Lineage

Riverside House has 13 structurally valid training rows and a release deadline. The problem is that structural validity is the least interesting fact about the file: duplicate leakage, invalid role order, PII, missing or unknown rights, split contamination, thin slices, contradictory preferences, and a length shortcut are all present.

[Open the notebook](training-data-quality-and-lineage.ipynb) to build a deterministic data gate that measures those failures, fingerprints the source and curated candidate, and writes a machine-readable promotion report. The notebook reads the shared fixtures in `../shared/training-data/` without modifying them.

## Setup

From this directory:

```powershell
.\setup.ps1
```

On Linux or macOS:

```bash
./setup.sh
```

Both scripts create a chapter-local `.venv`, install [requirements.txt](requirements.txt), register a Jupyter kernel, and assign it to the notebook. Use `-SkipKernel` or `--skip-kernel` to install dependencies without kernel registration. Setup does not execute notebook cells.

## Validated Local Workflow

The notebook was executed successfully in the unified FDE environment. Its outputs and execution counts were then cleared for reuse. The validated fixture workflow computes:

- source-file and canonical-record SHA-256 fingerprints;
- schema and semantic-template validation results;
- exact and cross-split near-duplicate groups;
- PII, provenance, rights, and split-policy findings;
- task, split, and slice distributions;
- preference disagreement and length-shortcut findings;
- row-level issue ledger;
- uncurated and curated promotion reports under `artifacts/`.

Expected fixture facts remain assertions in the cleared notebook rather than committed displayed outputs.

## Completion Evidence

You have finished this chapter when you have:

- retained the verified fixture version and source digest;
- retained schema, duplicate/leakage, template, PII, provenance/rights, contamination, slice, and preference findings;
- produced both the uncurated decision and a separately fingerprinted curated candidate report;
- recorded every curation action rather than silently deleting a blocker;
- stated what the fixture result does not prove about production data, legal approval, or model behavior;
- linked the report into a release or capstone evidence index as `LOCAL_FIXTURE`, not as a passing production data gate.

## Conceptual Owners

- [Fine-Tuning: What Should the Model Learn?](../../genai/03-llm-finetuning/01-llm-finetuning-data-techniques.ipynb) owns SFT, preference objectives, held-out evidence, and training manifests.
- [Fine-Tuning README](../../genai/03-llm-finetuning/README.md) owns the adaptation arc and hardware boundary.
- [LLM Evaluation](../../genai/05-llm-evaluation/README.md) owns evaluator validity, slice gates, and release thresholds.
- [Agent Safety, Human Control, and Governance](../../agentic-ai/06-safety-human-control-and-governance/06-safety-human-control-and-governance.ipynb) owns broader governance and human approval boundaries.
- [Production Feedback and Drift](../05-production-feedback-and-drift/production-feedback-and-drift.ipynb) owns the production-to-dataset feedback loop.
- [Release Registry and Lineage](../04-release-registry-and-lineage/release-registry-and-lineage.ipynb) owns cross-artifact release manifests.

## Scope Boundary

This chapter teaches deterministic pre-training data gates. It does not train a model, claim that regex is production DLP, make legal licensing decisions, or treat one similarity heuristic as universal semantic deduplication.
