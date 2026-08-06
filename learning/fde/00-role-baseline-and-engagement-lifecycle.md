# FDE Role Baseline and Engagement Lifecycle

An FDE engagement is a sequence of evidence gates, not a slide sequence. The role begins with uncertainty: the customer has a workflow problem, partial data, multiple owners, and constraints that may conflict. Your job is to reduce that uncertainty without turning assumptions into commitments.

This document defines the role boundary, baseline challenge, lifecycle gates, artifact traceability, claim discipline, and handoff standard used by the [FDE learning route](README.md).

## Role boundary

The FDE owns the technical path from discovery through operational handoff. That includes requirements translation, architecture, integration, evaluation, rollout evidence, incident participation, and knowledge transfer. Ownership does not mean unilateral authority.

| Decision | FDE responsibility | Required authority outside the FDE role |
|---|---|---|
| Business outcome and workflow acceptance | Make the workflow and criterion testable; expose contradictions and missing baselines | Customer workflow/product owner accepts scope and priority |
| Architecture and implementation | Recommend the smallest design, implement bounded proofs, and record tradeoffs | Service owners accept dependencies; architecture owner accepts material platform changes |
| Data use | Inventory, map, test, minimize, trace, and enforce approved use | Data owner approves source, purpose, retention, and deletion terms |
| Identity and security | Design propagation and controls; run negative tests; record residual risk | Customer security/identity owners approve policy intent and exceptions |
| Compliance | Map technical controls and evidence to stated requirements | Authorized legal/compliance owner determines applicability and approval |
| SLA and commercial commitments | Model capacity, cost, support, and failure assumptions; identify evidence gaps | Authorized business/service owners commit contract terms and support coverage |
| Production exposure | Supply release evidence, cohort design, rollback conditions, and live signals | Named go/no-go authority approves exposure and ramp |
| Incident communication | Establish facts, preserve evidence, draft scoped technical updates | Incident commander and communications owner approve severity and external messages |
| Handoff | Prove runbooks, dashboards, drills, and change paths work; train owners | Customer operations accepts ownership and unresolved limitations |

An FDE does not certify that a system is secure, compliant, production-ready, or valuable by assertion. The role produces evidence and routes decisions to the people with authority.

## Claim discipline

Use three claim classes everywhere: discovery reports, ADRs, evaluation summaries, status updates, rollout decisions, incident communications, and handoff packs.

### Measured

`[Measured]` means the result was observed. Record:

- claim text and metric definition;
- environment and release/configuration identifiers;
- dataset, traffic population, or test cohort;
- sample size and time window;
- test or query method;
- result and uncertainty where applicable;
- evidence location;
- exclusions and known limitations.

Example: `[Measured] Release r17 denied 20 of 20 seeded cross-tenant retrieval attempts in the staging identity harness on 2026-08-05. This does not cover production identity-provider misconfiguration.`

### Modeled

`[Modeled]` means the result was calculated from assumptions rather than observed end to end. Record:

- formula or simulation model;
- every material input, source, and date;
- base, low, high, and stress scenarios;
- sensitivity to uncertain inputs;
- confidence limits or qualitative confidence;
- owner and event that will replace the model with measurement.

Example: `[Modeled] At 6 requests per second, 1.7 model calls per request, and the stated token distribution, estimated monthly model cost is $8,100-$12,600. Validate with a two-week shadow cohort before any budget commitment.`

### Customer-validated

`[Customer-validated]` means an authorized customer representative accepted an artifact, workflow, criterion, control intent, or result for a stated scope. Record:

- person or role and the authority they hold;
- artifact and version reviewed;
- environment, users, workflow, and data scope;
- decision, conditions, and unresolved exceptions;
- date, expiry, and revalidation trigger.

Example: `[Customer-validated] Editorial Operations accepted acceptance criteria AC-01 through AC-07 for the acquisitions workflow in review CR-12. Finance exceptions and non-English manuscripts remain outside scope.`

Customer validation is not a measurement method. If a customer accepts a modeled forecast, keep both records: the forecast remains `[Modeled]`; acceptance of its use for a decision is `[Customer-validated]`. Legal, compliance, security, finance, and production approvals require representatives with the relevant authority.

### Claim register

Maintain one append-only register with at least these fields:

| Field | Purpose |
|---|---|
| `claim_id` | Stable identifier used by status, release, incident, and handoff artifacts |
| `statement` | Exact bounded claim; avoid adjectives without a metric or decision |
| `class` | `Measured`, `Modeled`, or `Customer-validated` |
| `scope` | Workflow, users, data, environment, release, and excluded cases |
| `evidence_ref` | Test report, trace query, model workbook, or approval record |
| `owner` | Person accountable for the evidence and correction |
| `recorded_at` | Observation, model, or decision date |
| `limitations` | Conditions under which the statement should not be reused |
| `revalidate_on` | Time limit or material change that expires the claim |
| `status` | Proposed, active, superseded, or withdrawn |

Never silently rewrite a claim after contradictory evidence appears. Supersede it, link the new evidence, and preserve the decision history.

## Engagement lifecycle

```mermaid
flowchart LR
    B["0. Baseline and qualify"] --> D["1. Discover workflow"]
    D --> A["2. Translate architecture"]
    A --> T["3. Prove data readiness"]
    T --> I["4. Prove identity and controls"]
    I --> E["5. Set evaluation and service envelope"]
    E --> R["6. Roll out in stages"]
    R --> O["7. Operate and recover"]
    O --> H["8. Hand off ownership"]
    H --> C["Recurring health and change review"]

    style B fill:#1e3a8a,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style D fill:#1d4ed8,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style A fill:#1d4ed8,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style T fill:#b45309,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style I fill:#b45309,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style E fill:#1d4ed8,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style R fill:#b45309,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style O fill:#b91c1c,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style H fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style C fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
```

Stages can overlap for bounded discovery spikes, but their gates cannot be bypassed. A prototype may start before every source is ready; production data may not be used before data purpose, access, retention, and identity controls are approved. A canary may not start before evaluation and rollback gates exist.

### Lifecycle gates

| Stage | Work required | Core artifacts | Exit gate |
|---:|---|---|---|
| 0. Baseline and qualify | Establish role fit, sponsor, decision owner, problem boundary, access constraints, and evidence gaps | `ENG-00` engagement charter; baseline response; existing-evidence inventory; initial claim register | A named sponsor and decision owner exist; the problem is bounded enough for discovery; unsupported pre-engagement claims are marked |
| 1. Discover workflow | Observe or reconstruct the current workflow; interview users and exception owners; establish volumes, delays, error costs, constraints, non-goals, and unknowns | `DSC-01` stakeholder map; `DSC-02` current-state workflow; `DSC-03` baseline; `DSC-04` acceptance matrix; `DSC-05` assumption/risk log and backlog | Workflow owner confirms current state and non-goals; each acceptance criterion has a method, slice, threshold or decision rule, owner, and validation class |
| 2. Translate architecture | Compare no-AI, deterministic, search, RAG, prompt-only, fine-tuned, workflow, single-agent, and multi-agent options; mark human, policy, data, identity, state, and side-effect boundaries | `ARC-01` option matrix; `ARC-02` architecture and sequences; `ADR-*` decisions; `ARC-03` customer-readable explanation; product-gap log | Selected design is the smallest option supported by evidence; each material dependency and residual risk has an owner; customer technical owner accepts the explanation |
| 3. Prove data readiness | Inventory sources and owners; sample real shapes safely; define schemas, parsing, deduplication, quality, ACL, provenance, refresh, deletion, and failure handling | `DATA-01` source inventory; `DATA-02` mapping contracts; `DATA-03` quality report; `DATA-04` lineage and sync/delete plan; `DATA-05` readiness verdict | Data owner approves purpose and sources; quality thresholds and quarantine paths are testable; ACL and deletion behavior have negative tests; unresolved sources are excluded explicitly |
| 4. Prove identity and controls | Trace tenant, user, role, region, purpose, and correlation context through ingress, retrieval, tools, logs, and response; test fail-closed behavior | `SEC-01` identity flow; `SEC-02` RBAC matrix; `SEC-03` threat model; `SEC-04` data-flow/residency map; `SEC-05` controls matrix; `SEC-06` isolation report | Seeded cross-tenant and over-privileged attempts fail; logs are useful without exposing prohibited data; security, identity, data, and compliance owners record decisions or open gaps |
| 5. Set evaluation and service envelope | Build representative workflow slices; separate retrieval, generation, trajectory, policy, safety, latency, cost, and recovery measures; model workload, quota, headroom, support, and cost | `EVAL-01` versioned set and rubric; `EVAL-02` baseline/candidate report; `CAP-01` workload model; `CAP-02` sensitivity/cost report; `SVC-01` proposed SLO/SLA and support map | Release gates identify must-pass slices and uncertainty; every service/cost number is labeled measured or modeled; commitment authorities accept, reject, or condition the proposal |
| 6. Roll out in stages | Establish historical/human baseline; run shadow comparison; review disagreements; select canary cohorts; define ramp, pause, rollback, compensation, and communications | `ROL-01` rollout plan; `ROL-02` cohort and disagreement report; `ROL-03` go/no-go record; `ROL-04` rollback/compensation drill; `CHG-*` change records | Candidate passes offline and shadow gates; rollback is timed and tested; actions already committed have compensation or escalation paths; named authority approves the canary |
| 7. Operate and recover | Monitor service, quality, policy, cost, and customer-specific signals; classify and contain incidents; preserve evidence; communicate known facts; remediate and revalidate | `OPS-01` signal/alert catalog; `OPS-02` on-call map; `INC-*` timeline, evidence, communication, remediation, revalidation, and postmortem | Alerts have owners and runbooks; incident drills demonstrate containment and evidence preservation; re-enablement requires the same or stronger gates as release |
| 8. Hand off ownership | Run operational readiness review; train owners; verify dashboards and runbooks; document support, policy changes, data operations, retirement, limitations, and backlog | `HOF-01` readiness review; `HOF-02` runbook set; `HOF-03` support/escalation matrix; `HOF-04` training/drill record; `HOF-05` acceptance and limitations; health-review cadence | Every recurring task and alert has an owner; customer operators complete agreed drills; acceptance conditions and exclusions are signed; FDE-only access and undocumented knowledge are removed |

### Gate rules

1. A gate needs an artifact, evidence, owner, and recorded decision. Meeting notes that say "looks good" are not a gate.
2. Conditional approval names the condition, owner, due date, exposure limit, and automatic response if the condition is missed.
3. A later-stage failure can reopen an earlier gate. Schema drift reopens data readiness; a new role model reopens identity; a model change reopens evaluation and rollout.
4. Deployment rollback stops a release from receiving new work. It does not undo side effects already committed. Those require compensation, correction, or escalation.
5. Production re-enablement after an incident requires evidence that containment remains in place, the fix passes relevant regression gates, and the authorized owner accepts residual risk.
6. Schedule pressure changes scope or exposure. It does not change the evidence standard silently.

## Traceability contract

The engagement package must answer both directions of traceability:

- For any acceptance criterion, show the workflow need, architecture component, data/control dependency, evaluation case, rollout gate, operating signal, and handoff owner.
- For any component or service, show the acceptance criterion it supports. Components without a mapped need are candidates for removal.

Use stable IDs and maintain this minimum chain:

```text
DSC-04 acceptance criterion
  -> ADR-* architecture decision
  -> DATA-* source contract and SEC-* control
  -> EVAL-* test and CAP-* service assumption
  -> ROL-* release gate
  -> OPS-* signal and INC-* recovery path
  -> HOF-* operating owner
```

| Trace question | Required answer |
|---|---|
| Why does this feature exist? | Acceptance criterion and named workflow/user |
| What evidence supports this design? | ADR inputs with measured or modeled claim references |
| Which data can it use? | Approved source, purpose, schema, ACL, retention, residency, and deletion contract |
| Who may trigger or approve it? | Identity context, policy, RBAC rule, and audit event |
| How is failure detected? | Evaluation case, telemetry signal, threshold, and alert owner |
| What limits exposure? | Cohort, quota, budget, timeout, approval, circuit breaker, or feature flag |
| How is it stopped or corrected? | Deployment rollback plus action compensation/correction where needed |
| Who owns it after delivery? | Runbook, support boundary, change authority, and accepted handoff record |

## Baseline challenge

The baseline challenge determines study placement and whether prior evidence can replace a route exercise. It is not a conversational self-rating.

Use the frozen, synthetic [Shared Riverside FDE Case](shared/README.md) when the fixture package is available. The abbreviated case below preserves the same decision problem for a route-level review.

### Case

A publisher asks: "Help editors find answers and continue manuscripts faster." Editors work across multiple imprints. Some manuscripts are embargoed. Rights and contract records live in a separate ERP. The customer has not supplied a baseline, target, identity design, approved source list, support model, or definition of an acceptable continuation. A stakeholder has already described the desired system as a multi-agent platform.

### Deliverables

Produce a bounded response that includes:

1. The first discovery workshop agenda, stakeholder/authority map, and ten highest-value questions.
2. A current-state hypothesis clearly separated from known facts, plus a plan to establish the baseline.
3. Draft acceptance criteria with slices, methods, owners, and explicit non-goals.
4. An option matrix comparing no-AI/process change, deterministic software, search, RAG, prompt-only generation, fine-tuning, workflow, single-agent, and multi-agent approaches.
5. A first-pass data and identity flow that marks authorization, residency, audit, and deletion unknowns.
6. A claim register containing at least one correctly scoped measured example, modeled example, and customer-validation request. Do not invent customer approval.
7. An evaluation, rollout, incident, and handoff outline with named gates and missing owners.
8. A gap plan mapping weak areas to exact artifacts in the [FDE route](README.md#ai-engineer-technical-core).

### Baseline decision

Use the route's [honest assessment rubric](README.md#honest-assessment-rubric). The baseline can challenge out study only when the submitted artifact already meets the target competency score and includes provenance, limitations, reviewer, and revalidation trigger.

Automatic failure conditions:

- accepting "multi-agent" as a requirement without comparing smaller options;
- presenting targets, savings, latency, quality, availability, or cost as observed without measurement;
- inventing customer validation or treating stakeholder enthusiasm as acceptance;
- using production data before purpose, access, retention, and identity decisions;
- omitting cross-tenant negative tests, rollback, incident communication, or handoff ownership;
- claiming compliance or security approval without the authorized owner and scope.

## Decision and evidence reviews

Run reviews at the points where evidence changes exposure, not merely at fixed reporting intervals.

| Review | Minimum participants | Decision |
|---|---|---|
| Discovery review | FDE, workflow owner, affected user representative, data/technical owners | Confirm problem, baseline plan, non-goals, acceptance criteria, and unknowns |
| Architecture review | FDE, customer technical owner, service/data/security owners as applicable | Select the smallest viable design and assign residual risks |
| Data/control readiness | FDE, data owner, identity/security owner, compliance owner where applicable | Approve bounded data use and control evidence or block exposure |
| Evaluation and service review | FDE, workflow owner, evaluation owner, operations/finance owners | Set release gates and distinguish measured results from forecasts/commitments |
| Go/no-go | Release owner, incident/operations owner, customer decision owner | Approve cohort and exposure only after rollback and communication paths are ready |
| Re-enablement | Incident commander, affected service/control owners, customer decision owner | Confirm containment, regression evidence, residual risk, and monitoring before restore |
| Handoff acceptance | FDE, customer operations, support, product/workflow, data, security owners | Accept ownership, limitations, change process, and health-review cadence |

## Handoff standard

Handoff is complete only when the receiving team can operate and change the system without relying on private FDE context.

The package must include:

1. Service and dependency inventory with versions, owners, quotas, credentials process, data flows, and support contacts.
2. Dashboard guide connecting availability, latency, quality, retrieval, safety/policy, cost, and customer-specific signals to decisions.
3. Alert catalog with severity, threshold, owner, first action, escalation path, and false-positive handling.
4. Runbooks for degraded providers, stale or unauthorized retrieval, identity failure, quota/budget exhaustion, tool failure, data sync/deletion failure, rollback, compensation, and re-enablement.
5. Release and policy-change process with versioning, evaluation gates, approvals, canary rules, rollback target, and retained evidence.
6. Data operations for onboarding, schema change, reindexing, access changes, retention, deletion, lineage, and audit requests.
7. Support matrix defining hours, severity, response target, communication owner, vendor dependency, and what remains outside support.
8. Training record and drills showing that named operators can interpret dashboards, execute high-risk runbooks, and escalate correctly.
9. Acceptance record with measured results, modeled limits, customer-validated decisions, unresolved risks, exclusions, and revalidation triggers.
10. Recurring health review covering workflow changes, data drift, quality regressions, policy changes, cost/capacity, incidents, user feedback, and retirement criteria.

Minimum handoff drills:

- locate a failing request across gateway, retrieval/model, tool, policy, and audit records;
- disable or roll back a release without confusing deployment rollback with action compensation;
- process an access revocation or deletion request and verify downstream propagation;
- respond to a seeded cross-tenant retrieval attempt and preserve the evidence;
- change an evaluation threshold through the approved process and show the retained decision record.

If a drill fails, handoff remains open. Record the failure as measured evidence, repair the package or training, and repeat the drill. A signed document does not override an operator's inability to execute the accepted responsibilities.

## Honest engagement close

Close the engagement with four explicit lists:

1. What was measured, where, on which release and population, with what limitations.
2. What remains modeled, which assumptions dominate the range, and when production evidence will replace it.
3. What the customer validated, who had authority, the accepted scope, and what was excluded.
4. What is unknown, who owns the next evidence, and what exposure remains blocked until it exists.

The final status is not "done" when code is deployed. It is complete for the agreed scope when acceptance is traceable, exposure is bounded, operations owns the system, and every remaining uncertainty has an owner or an explicit decision not to proceed.
