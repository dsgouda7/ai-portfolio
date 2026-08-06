# Release Registry and Lineage

An artifact directory can exist while the application release is unknown, incompatible, or impossible to roll back. This chapter turns that operational gap into a local, inspectable release registry for the Riverside House system.

Open [release-registry-and-lineage.ipynb](release-registry-and-lineage.ipynb) to work through the failure chain:

1. an artifact path exists but proves no release identity;
2. a mutable alias cannot reproduce what served;
3. a base model and adapter do not belong together;
4. prompt, index, or evaluator lineage is missing;
5. a rollback target does not resolve to an accepted release;
6. schema and semantic compatibility gates produce an auditable decision.

The notebook reads the shared synthetic fixtures in `../shared/release-lineage/` unchanged. It also reads selected Riverside platform contracts and fine-tuning provenance as reference surfaces. It writes no registry records or artifacts.

## Learning Contract

By the end, you should be able to:

- distinguish a file, registered artifact, model release, and complete application release;
- resolve an immutable release ID to its base, adapter, dataset, prompt, index, evaluator evidence, and rollback target;
- explain why JSON Schema cannot prove cross-field equality, graph reachability, or promotion authority;
- reject a passing evaluation when base/adapter compatibility fails;
- trace a request's `release_id` back to the exact shared manifest;
- map local records to Azure ML and Microsoft Foundry without claiming unperformed cloud work.

## Setup

From this directory, create an isolated environment and install the one direct dependency:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Select that environment as the notebook kernel. Run the notebook from any working directory inside the checkout; its setup cell locates the repository root using `AUTHORING_GUIDE.md` and `learning/role-based-tracks/ai-engineer/shared`, then resolves all inputs from that root.

The notebook executed successfully in the unified FDE environment and was then cleared for reuse. Expected decisions remain documented in `../shared/release-lineage/EXPECTED_OUTCOMES.md`; retain a fresh, versioned result when building your own evidence package.

## Inputs

| Input | Role |
| --- | --- |
| `../shared/release-lineage/release-manifests.json` | Stable teaching release records |
| `../shared/release-lineage/release-manifests.schema.json` | Structural contract for the complete shared document |
| `../shared/release-lineage/EXPECTED_OUTCOMES.md` | Expected semantic decisions and lineage queries |
| `../../../checkpoints/instruction-lora/experiment-manifest.json` | Unchanged training provenance example |
| `../../../projects/riverside-ai-platform/contracts/v1/model-release-manifest.schema.json` | Stricter serving-model contract |
| `../../../projects/riverside-ai-platform/src/artifact_validation/verification.py` | Production-oriented compatibility reference |

## Evidence Boundary

The shared manifest proves the notebook's local registry mechanics. It does not contain the platform contract's tokenizer bundle, immutable serving URIs, runtime profile, precision, source commit, deployment slot, or complete eight-domain evaluation report.

Azure ML can register versioned model/data/environment assets and expose deployment metadata. Microsoft Foundry can organize projects, model deployments, evaluation assets, and traces. Prompt and retrieval-index versions may still live in separate stores. Neither product automatically turns those resources into the complete portable application-release proof taught here; your release workflow must bind their immutable identifiers and retained evidence.

No Azure or Foundry authentication, registration, deployment, endpoint invocation, or rollback is performed or validated by this chapter.

## Completion Evidence

You have finished this chapter when you have:

- retained the verified fixture version and the release/trace input digests;
- resolved one immutable release to model, adapter, data, prompt, index, evaluator, runtime, and rollback references;
- retained schema and semantic compatibility decisions, including the blocked incompatible release;
- demonstrated request-to-release attribution and a valid accepted rollback edge;
- mapped local fields to Riverside/Azure source contracts without relabeling source as cloud execution;
- linked the release manifest into a capstone evidence index as `LOCAL_FIXTURE` and kept deployment or rollback behavior `UNVALIDATED` unless separately proven.

## Files

- [requirements.txt](requirements.txt) pins the notebook's direct validation dependency.
- [release-registry-and-lineage.ipynb](release-registry-and-lineage.ipynb) contains the complete lesson, successfully executed and cleared for reuse.

No notebook-local schema is included because the shared schema is the owning teaching contract. The notebook adds semantic checks in Python where JSON Schema cannot express the invariant.
