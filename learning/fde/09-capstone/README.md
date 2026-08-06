# FDE Capstone: Riverside Engagement Package

> **Evidence banner:** `FROZEN SYNTHETIC CASE`, `EXECUTION VERIFIED THEN CLEARED`,
> `NO CUSTOMER OR CLOUD VALIDATION`, `NO PRODUCTION-READINESS CLAIM`.

Riverside House has asked for an editorial assistant before its autumn catalog
cycle. The request sounds specific until you try to turn it into a commitment:
stakeholders disagree about workflow writes, regional processing, latency,
support coverage, volume, quality, deletion, and budget. The production project
contains substantial source assets, but its own documentation records unresolved
integration gaps and no retained live Azure or Databricks evidence.

Your capstone is to turn that uncertainty into one bounded, reviewable engagement
package. You are not rewarded for adding services, agents, or training runs. You
are rewarded for choosing the smallest defensible scope, preserving authority
boundaries, connecting every claim to evidence, and making the next decision
clear.

## Capstone outcome

Submit one package that lets an independent customer architect, security reviewer,
operations owner, and commercial owner answer:

1. What workflow and outcome are actually in scope?
2. Which claims are measured, modeled, customer-validated, unknown, or externally
   unvalidated?
3. What is the smallest architecture supported by the evidence?
4. Which data, identity, residency, threat, and control boundaries block exposure?
5. What quality, safety, latency, capacity, cost, and support evidence gates each
   rollout stage?
6. How will the team contain an incident, communicate known facts, recover, and
   obtain re-enablement approval?
7. Can customer operations run and change the system without private FDE context?
8. Which project assets can be reused, which gaps remain, and what should happen
   next?

## Package map

| File | Use |
|---|---|
| [customer-brief.md](customer-brief.md) | Intentionally incomplete and conflicting customer input |
| [capstone-workbook.ipynb](capstone-workbook.ipynb) | Failure-first workbook; synthetic execution verified, then cleared |
| [deliverable-index.md](deliverable-index.md) | Required artifacts, stable IDs, dependencies, and review order |
| [pre-capstone-readiness-checklist.md](pre-capstone-readiness-checklist.md) | Stop/go check before investing in the package |
| [templates/](templates/) | Blank review-oriented deliverable templates |
| [examples/teaching-only/](examples/teaching-only/README.md) | Small filled excerpt that ends at an evidence-backed `HOLD` |
| [optional-multi-tenant-variant.md](optional-multi-tenant-variant.md) | Optional isolation, shared-resource, deletion, cost, rollout, and incident overlay |
| [assessment-rubric.md](assessment-rubric.md) | Scoring, critical gates, and unsupported-claim penalties |

The frozen source of truth remains
[the shared Riverside case](../shared/README.md), especially
`riverside-engagement-v1.json`, `riverside-source-samples-v1.json`, and
`expected-facts-v1.json`. Do not edit those fixtures or silently resolve their
conflicts inside the workbook.

## Before you start

Complete the [pre-capstone readiness checklist](pre-capstone-readiness-checklist.md). A failed item is a study or mentorship target, not a field to waive. Passing the checklist means you are ready to attempt this static package; it does not establish independent production-execution competence.

The brief is intentionally incomplete and conflicted. That is the work, not a defect to repair. For each conflict or material unknown:

1. quote or link the competing statements without choosing the convenient one;
2. separate known input, customer claim, modeled assumption, policy constraint, and missing evidence;
3. name the resolution owner and the authority that owner must hold;
4. specify the evidence or decision needed, needed-by gate, and exposure blocked while unresolved;
5. record the escalation path and a safe default such as `HOLD`, `NARROW SCOPE`, manual operation, or disabled capability;
6. supersede the record when authorized evidence arrives; do not rewrite the history as if the ambiguity never existed.

A high-quality package may preserve unresolved conflicts. It loses quality when it invents certainty, hides the blocked consequence, or lets a calendar date resolve an authority gap.

## Required deliverables

Use the stable artifact families already established by the route:

| Package area | Required evidence |
|---|---|
| Engagement control | `ENG-CAP-01` charter, evidence inventory, claim register, decision log |
| Discovery and success | `DSC-CAP-01` report, `DSC-CAP-02` baseline, `DSC-CAP-03` acceptance matrix and golden workflow set |
| Architecture | `ARC-CAP-01` option matrix, `ARC-CAP-02` diagrams and boundary register, `ADR-CAP-*` decisions |
| Data | `DATA-CAP-01` source inventory, `DATA-CAP-02` contract/mapping, `DATA-CAP-03` onboarding and sync/delete plan, `DATA-CAP-04` readiness verdict |
| Identity and controls | `SEC-CAP-01` identity flow/RBAC, `SEC-CAP-02` threat model, `SEC-CAP-03` residency map, `SEC-CAP-04` controls and isolation evidence plan |
| Service envelope | `CAP-CAP-01` scenarios/sensitivity, `COST-CAP-01` cost attribution, `SVC-CAP-01` proposed SLO/SLA/support terms |
| Evaluation and rollout | `EVAL-CAP-01` versioned evaluation plan, `ROL-CAP-01` staged rollout, `ROL-CAP-02` rollback/compensation and communication plan |
| Incident response | `INC-CAP-01` simulation record, `INC-CAP-02` customer update, `INC-CAP-03` re-enablement decision and follow-up actions |
| Operations and handoff | `OPS-CAP-01` signal/alert/runbook map, `HOF-CAP-01` readiness and training package, `HOF-CAP-02` acceptance/exit/health-review record |
| Project mapping | `MAP-CAP-01` trace from requirements to Riverside platform and RAG data-plane assets, limitations, and validation owners |
| Product iteration | `GAP-CAP-01` product-gap ledger and `NEXT-CAP-01` evidence-backed next-iteration recommendation |

The [deliverable index](deliverable-index.md) defines the minimum fields and
cross-artifact traceability for each item.

## Quantitative scope guidance

Use these ranges to calibrate review depth for the base case. They are guidance, not quotas and not a reason to split one fact into filler rows. Expand only when a distinct workflow, authority, failure mode, or decision needs its own record.

| Surface | Base-case guidance | Quality test |
|---|---:|---|
| Claim register | 25-50 material claims | Includes every external, quantitative, normative, readiness, cost, SLA, security, compliance, and customer-acceptance statement used in the recommendation |
| Conflicts and unknowns | All 11 supplied conflicts and all material frozen-case unknowns | Each has authority, evidence needed, needed-by gate, blocked exposure, and escalation/default path |
| Acceptance criteria | 8-15 across the in-scope workflows | Each has a slice, method, decision rule or unresolved threshold, owner, evidence class, and non-compensating failures |
| Golden workflow set | 12-24 cases, including positive, negative, boundary, abstention, stale/deleted, and degraded-mode cases | Covers critical slices without pretending the set is statistically representative |
| Architecture decisions | 3-6 ADRs plus one option matrix | Every selected component maps to a criterion; every complex option has a named failed smaller option or remains deferred |
| Data and control records | All six supplied sources; 8-15 high-risk control or negative-test cases | Reports source-specific readiness and fail-closed behavior rather than one aggregate pass |
| Evaluation plan | 20-40 planned cases across retrieval, generation, authorization/policy, safety, latency/cost, and rollout domains | Marks cases `NOT RUN` until retained execution exists and keeps critical domains separate |
| Capacity and cost | Low, expected, high, and at least one stress scenario | Shows source-dated inputs, sensitivity, headroom, attribution, and replacement-by-measurement owners |
| Rollout and recovery | Shadow plus at least two bounded cohort stages; one rollback/compensation rehearsal plan; one incident simulation | Names gates, observation windows, owners, stop conditions, communication, and evidence locations |
| Operations and handoff | 5-8 highest-risk runbooks; 3-5 operator drill plans | Mark drills `NOT RUN` until those drills are actually performed and retained; the verified workbook run does not demonstrate operator capability |

The base package should be reviewable as one coherent decision record. Completeness is coverage and traceability, not page count.

After the base case, the [optional multi-tenant variant](optional-multi-tenant-variant.md) adds shared-resource contention, tenant transfer, deletion, entitlement, cost-attribution, rollout, and incident edge cases without changing the frozen Riverside fixture.

## Working sequence

```mermaid
flowchart LR
    B["Conflicting brief"] --> D["Discovery and success gates"]
    D --> A["Smallest architecture and ADRs"]
    A --> C["Data, identity, threat, residency controls"]
    C --> S["Evaluation and service envelope"]
    S --> R["Shadow, canary, rollback, communication"]
    R --> I["Incident simulation and re-enablement"]
    I --> H["Operations, handoff, and health review"]
    H --> M["Project mapping, gaps, next iteration"]

    style B fill:#b91c1c,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style D fill:#1d4ed8,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style A fill:#1d4ed8,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style C fill:#b45309,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style S fill:#b45309,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style R fill:#b45309,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style I fill:#b91c1c,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style H fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style M fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
```

Do not advance because a section is filled in. Advance only when the prior gate
has an artifact, evidence, owner, limitations, and recorded decision. A later
finding may reopen an earlier gate.

## Claim discipline

Use the route claim classes exactly:

- `[Measured]` requires an executed test, trace, query, drill, or observation with
  environment, release/configuration, population, method, result, evidence link,
  and limitations.
- `[Modeled]` requires formulas or logic, source-dated inputs, low/expected/high or
  stress scenarios, sensitivity, limitations, and a replacement-by-measurement
  owner.
- `[Customer-validated]` requires an authorized customer role, artifact/version,
  scope, decision, conditions, date, and revalidation trigger.
- `[External validation required]` marks Azure, Databricks, identity-provider,
  networking, quota, price, residency, legal, compliance, security, and operational
  claims that the local capstone cannot close.
- `[Unknown]` is valid when it has an owner, needed-by gate, and consequence of
  remaining unresolved.

A local calculation over modeled inputs remains modeled. An accepted ADR is a
design decision, not implementation proof. A customer preference is not legal,
security, or compliance approval unless the representative has that authority.

## Hard rules

1. Do not invent customer approval, live measurements, cloud behavior, prices,
   quota, availability, compliance, security, residency, or support commitments.
2. Do not describe source assets or test files as passing evidence without retained
   execution output.
3. Do not select an agent, fine-tuning job, Azure service, Databricks feature, or
   multi-region design until a smaller option fails a named acceptance criterion.
4. Do not use client-supplied tenant or ACL filters as authority.
5. Do not average away a critical authorization, policy, safety, deletion, or
   committed-action failure.
6. Do not confuse deployment rollback with compensation or reconciliation for an
   action already committed.
7. Do not call the engagement complete while a recurring task, alert, decision, or
   revalidation trigger lacks an accepted owner.

Unsupported AI, cloud, security, compliance, residency, cost, SLA, or production
claims receive explicit deductions and may trigger an automatic cap in the
[assessment rubric](assessment-rubric.md).

## Project references

Map to existing assets; do not copy their implementation into the capstone:

- [Riverside platform architecture](../../../projects/riverside-ai-platform/docs/architecture.md)
- [Promise versus evidence ledger](../../../projects/riverside-ai-platform/docs/promise-vs-evidence.md)
- [Current platform limitations](../../../projects/riverside-ai-platform/docs/limitations.md)
- [Versioned platform contracts](../../../projects/riverside-ai-platform/contracts/README.md)
- [Evaluation assets](../../../projects/riverside-ai-platform/evaluations/README.md)
- [RAG Knowledge Pipeline](../../../projects/rag-knowledge-pipeline/README.md)
- [Databricks indexing operations](../../../projects/rag-knowledge-pipeline/databricks/indexing/OPERATIONS.md)

The capstone must call out the documented region, endpoint-name, timeout, packaging,
deployment-composition, APIM publication, cloud-test, rollback, and retained-evidence
gaps where they affect the recommendation.

## Completion gate

The package is review-ready only when:

- every required artifact is present and indexed;
- every acceptance criterion traces through architecture, data/control,
  evaluation, rollout, operations, recovery, and handoff ownership;
- every selected component maps back to a requirement and every unmapped component
  is removed or justified;
- all conflicts and unknowns have dispositions or explicit blockers;
- every external claim has a valid class, evidence reference, limitation, owner,
  and revalidation trigger;
- product gaps are ranked by customer outcome and risk, not novelty;
- the final recommendation may be `HOLD`, `NARROW SCOPE`, or `DO NOT PROCEED`.

Passing this gate and the rubric supports entry into a supervised practicum. It does not establish that the learner can independently deploy controls, run a production incident, negotiate binding commercial terms, or complete a customer handoff. Record those as practicum objectives and require observed evidence from authorized environments and owners.

## Validation and evidence limitation

The route setup environment was verified, and this workbook executed successfully
end to end against the committed synthetic fixtures, producing its documented `HOLD`
and `REJECT` outcomes. Generated outputs were then cleared, so every committed code
cell retains `execution_count: null` and empty outputs. This validates local workbook
logic only. No model, cloud, Databricks, Azure, deployment, live incident, customer
operation, legal/compliance review, customer approval, or learner practicum evidence
is established by the run.
