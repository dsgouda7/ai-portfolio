# Data Onboarding and Contracts

Riverside House has six source systems and fourteen synthetic sample records.
The samples look ingestible until you inspect them: an old policy is still
searchable, a two-column PDF scrambles meaning, a rights schedule repeats a
page, an autosave needs a tombstone, an ERP null can widen rights, page two of
an API silently changes schema, and a disabled contractor survives in a stale
group snapshot.

This chapter turns those failures into a reviewable data-readiness decision. It
uses only the frozen synthetic fixtures in [`../shared/`](../shared/README.md)
and writes no customer or cloud data.

## What you produce

| Artifact | File | Decision supported |
|---|---|---|
| `DATA-01` | `templates/source-inventory.yaml` | Which sources, owners, purposes, refresh paths, and unknowns are in scope? |
| `DATA-02` | `templates/mapping-specification.yaml` | How does each source become a versioned document contract without silent coercion? |
| `DATA-03` | `templates/quality-report.json` | Which fixture checks passed, failed, or entered quarantine? |
| `DATA-04` | `templates/lineage-sync-delete-plan.yaml` | How do versions, ACL changes, watermarks, tombstones, and index deletes propagate? |
| `DATA-05` | `templates/retrieval-readiness-verdict.md` | Which sources are ready, conditional, blocked, or excluded, and why? |
| Run record | `templates/notebook-output-record.md` | What was actually observed, in which environment, with which limitations and external gaps? |

The notebook is a teaching lab, not the production data plane. For the
Databricks implementation, use the existing [RAG Knowledge Pipeline](../../../projects/rag-knowledge-pipeline/README.md):

- [remote ingestion contracts](../../../projects/rag-knowledge-pipeline/phase1-ingest/src/remote/contracts.py)
- [parsing and quarantine orchestration](../../../projects/rag-knowledge-pipeline/phase1-ingest/src/remote/pipeline.py)
- [durable ingestion quality gates](../../../projects/rag-knowledge-pipeline/phase1-ingest/src/remote/quality.py)
- [chunk and vector contracts](../../../projects/rag-knowledge-pipeline/phase2-vectorize/src/remote/contracts.py)
- [Databricks index operations](../../../projects/rag-knowledge-pipeline/databricks/indexing/OPERATIONS.md)

## Downstream integration path

Carry the chapter artifacts into implementation in this order:

1. Map `DATA-01` and `DATA-02` into the pipeline's remote ingestion and vector contracts; record every field that cannot be represented without coercion.
2. Translate `DATA-03` into quarantine and quality rules, preserving per-source failures rather than replacing them with one aggregate pass rate.
3. Map `DATA-04` into the Databricks ingestion and index-update jobs, including ACL refresh, watermark, tombstone, reindex, and delete behavior.
4. Use `DATA-05` as an input to a supervised validation plan for identity, filters, deletion, drift, latency, quota, region, cost, and rollback in the authorized target environment.

The linked source shows candidate implementation surfaces. It does not show that a job was deployed, a Databricks control worked, or customer data is ready.

## Evidence boundary

The notebook keeps four kinds of statements distinct:

| Label | Meaning in this chapter |
|---|---|
| `[Measured - local fixture]` | A deterministic check calculated from committed synthetic records after the learner runs it |
| `[Modeled]` | A projection from assumptions, never an observed production result |
| `[Customer-validated]` | A future scoped decision by an authorized customer owner; the notebook cannot create this evidence |
| `[External validation required]` | A Databricks, cloud, legal, security, or operational behavior that local fixtures cannot prove |

Supplied `customer_claim`, `policy_constraint`, `unknown`, and other frozen-case
labels are preserved. A local calculation over a customer claim does not promote
the claim to measured customer truth.

## Setup

Run the script for your platform:

```powershell
.\setup.ps1
```

```bash
./setup.sh
```

Then select `Python (FDE 03 Data Onboarding .venv)` and open
`data-onboarding-and-contracts.ipynb`.

The route setup environment was verified, and the notebook executed successfully
against the committed synthetic fixtures. Its outputs were then cleared. No
customer or cloud service was contacted by that validation.

## Scope and limitations

The local lab can prove that its fixture rules detect known synthetic failures.
It cannot prove representative parser accuracy, source completeness, customer
ownership, legal retention, identity freshness, Databricks RBAC, Delta merge
semantics, vector-index filters, regional behavior, deletion completion,
performance, cost, or operational readiness. Those items stay blocked or
conditional until their named owners provide evidence in the target environment.

The chapter also separates data readiness from answer quality. Passing these
gates means a bounded source can proceed to retrieval evaluation; it does not
mean retrieval relevance, citation quality, or generated answers are acceptable.
Use [Hybrid Search](../../genai/04-rag/04-hybrid-search.ipynb) and
[RAG Evaluation](../../genai/04-rag/05-rag-evaluation.ipynb) for those gates.

## Validation status

Validation covered the route setup environment and successful end-to-end notebook
execution against the committed synthetic records. The generated outputs were
cleared afterward, so the committed notebook remains empty. The run validates local
fixture logic only; representative customer data, Databricks behavior, cloud controls,
legal/compliance conclusions, and production readiness remain unvalidated.
