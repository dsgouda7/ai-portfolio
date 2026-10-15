# DATA-05 Retrieval Readiness Verdict

## Decision record

| Field | Value |
|---|---|
| Engagement | `ENG-RIV-FDE-001` |
| Artifact version | `0.1.0-example` |
| Fixture or customer sample version | `RIV-FDE-1.0.0` |
| Proposed decision | `BLOCKED` |
| Decision owner | `PENDING` |
| Recorded at | `PENDING` |
| Revalidate on | Source, schema, ACL, parser, retention, deletion, embedding, or index-version change |

This template is not a verdict until evidence references and authorized owners
replace every `PENDING` value.

## Evidence summary

| Claim class | Statement | Scope | Evidence reference | Limitations |
|---|---|---|---|---|
| `[Measured]` | `PENDING` | Environment, fixture/sample, version, population, and method | `PENDING` | `PENDING` |
| `[Modeled]` | `PENDING` | Assumptions and scenario range | `PENDING` | `PENDING` |
| `[Customer-validated]` | `PENDING` | Authorized role, workflow, sources, environment, date, and conditions | `PENDING` | This row must never be inferred from local checks. |
| `[External validation required]` | `PENDING` | Target platform or authority | `PENDING` | Exposure remains blocked where this evidence is material. |

## Per-source decision

Use exactly one decision per source:

- `READY_FOR_RETRIEVAL_EVALUATION`: contract, quality, lifecycle, ACL, lineage,
  and deletion gates pass for the bounded scope.
- `CONDITIONAL`: bounded evaluation may proceed behind named controls while a
  non-critical evidence item has an owner and due date.
- `BLOCKED`: a quality, authority, security, lifecycle, schema, or deletion gap
  can produce unsafe or misleading retrieval.
- `EXCLUDED`: the source or version is deliberately outside the current index.

| Source ID | Decision | Passed gates | Blocking evidence | Owner | Exposure limit |
|---|---|---|---|---|---|
| `SRC-EXAMPLE-001` | `BLOCKED` | None recorded | Owner, representative sample, ACL, and deletion behavior unvalidated | `PENDING` | No indexing |

## Required gates

- [ ] Purpose and source owner are approved for the stated scope.
- [ ] Required document-contract fields are complete without permissive defaults.
- [ ] Parser quality is measured on representative slices; failures quarantine.
- [ ] Stable identity, source versions, duplicates, and superseded records behave as specified.
- [ ] Schema drift fails closed and has a review owner.
- [ ] Tenant, region, classification, title scope, and ACL negative tests pass.
- [ ] Raw-to-index lineage is complete and versioned.
- [ ] Watermark, replay, reconciliation, and idempotency behavior are evidenced.
- [ ] Deletion or access revocation removes derived and indexed records with a receipt or negative query.
- [ ] Retrieval relevance and citation evaluation is still required after data readiness.

## Decision rationale

`PENDING: state why the weakest material gate determines this verdict.`

## Honest limitations

1. `PENDING`
2. `PENDING`
3. `PENDING`

## Approval and expiry

| Role | Name or stable role ID | Decision | Date | Conditions / expiry |
|---|---|---|---|---|
| Data owner | `PENDING` | `PENDING` | `PENDING` | `PENDING` |
| Security / identity owner | `PENDING` | `PENDING` | `PENDING` | `PENDING` |
| Retrieval evaluation owner | `PENDING` | `PENDING` | `PENDING` | `PENDING` |
