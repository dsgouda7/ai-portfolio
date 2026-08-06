# Riverside Editorial Assistant: Customer Brief

> **Brief status:** `INCOMPLETE`, `CONFLICTING`, `PRE-DISCOVERY`, `NOT APPROVED FOR BUILD`.
>
> This is a synthetic capstone input derived from frozen fixture version
> `RIV-FDE-1.0.0`. It deliberately omits decisions and preserves contradictory
> stakeholder statements. Do not repair the brief by inventing facts.

## Request from the sponsor

Riverside House wants an editorial assistant ready for the autumn catalog cycle.
Editors should be able to find the right policy or rights answer, continue a
selected manuscript in Riverside style, and avoid updating another system by hand.
The experience should feel instant, use the content Riverside already owns, and
work for both UK/EU and US imprints.

Unpublished manuscripts cannot be sent to a public generative AI service. The
sponsor has heard that a multi-agent platform and fine-tuning could make the
assistant more capable, but no architecture has been approved. The sponsor expects
an implementation proposal, a launch plan, and a monthly operating estimate at the
next steering meeting.

## Outcomes people say they want

- Editors find current, authorized answers faster and can inspect supporting
  evidence.
- Editors prepare bounded continuations faster while retaining final editorial
  control.
- Riverside avoids autonomous publication, rights, payment, contract, royalty, or
  distribution decisions.
- Workflow friction decreases without creating an unsafe or unmaintainable
  integration.
- Customer operations can own the service after a 30-day hypercare period.

These are desired outcomes, not accepted metrics. The frozen case contains limited
synthetic baselines, customer claims, modeled planning inputs, policy constraints,
and unresolved unknowns. Preserve those evidence classes.

## What Riverside says is already decided

1. The first release must support cited policy answers and bounded manuscript
   continuations.
2. A human remains accountable for editorial acceptance.
3. The assistant must not autonomously publish, distribute, grant rights, change
   contracts, change payments, or infer missing rights.
4. Tenant, role, region, purpose, title assignment, and deletion state must fail
   closed.
5. The system should support UK/EU and US editorial teams.

Treat each statement as an input to confirm against the frozen case and the named
authority. It is not implementation, customer acceptance, or proof that a cloud
service can satisfy the requirement.

## Conflicting instructions

The steering group has not resolved the following tensions:

| Conflict | Statement A | Statement B | Required decision artifact |
|---|---|---|---|
| `CON-RIV-001` latency | The sponsor says the experience should feel instant. | Operations proposes 8-second policy p95 and 18-second continuation p95 until cloud tests exist. | Acceptance criteria by request type |
| `CON-RIV-002` write authority | Editorial wants confirmed steps updated automatically. | Legal requires human confirmation for consequential system-of-record changes. | Proposal/approval/commit boundary |
| `CON-RIV-003` quality | Editorial wants Riverside style and time savings. | Rights requires every rights answer to use current title-specific evidence. | Separate retrieval, generation, citation, and review gates |
| `CON-RIV-004` volume | The sponsor wants all editors supported during deadlines. | Finance asks the team to budget for 900 daily sessions rather than the 620 expected case. | Low/expected/high workload scenarios |
| `CON-RIV-005` region | UK/EU manuscript content must remain in its primary region. | The US imprint accepts approved managed services in its own region. | Tenant-aware residency map and service review |
| `CON-RIV-006` deletion | Contract evidence must be retained. | Withdrawn manuscript drafts should disappear from assistant results quickly. | Record-class retention, revocation, and deletion contract |
| `CON-RIV-007` telemetry | Security prohibits manuscript or user content in operational telemetry. | Support needs enough evidence to reproduce failures. | Content-free telemetry and controlled evidence workflow |
| `CON-RIV-008` integration | IT wants the first release to avoid PageTurn writes. | Editorial says a launch without workflow updates creates another tool. | Phased architecture ADR |
| `CON-RIV-009` budget | Finance gives a 14,000 USD monthly target and 18,000 USD planning ceiling. | The sponsor says not to reduce untested quality to hit an estimate. | Sensitivity model and explicit quality floor |
| `CON-RIV-010` support | The sponsor wants coverage whenever deadlines hit. | Riverside support covers only weekday UK business hours. | Service-hours and severity agreement |
| `CON-RIV-011` policy scope | US editors must not see UK-only guidance. | Editorial wants shared house style across imprints. | Applicability metadata and retrieval filters |

Do not choose the convenient statement. Record the resolution owner, authority,
evidence needed, needed-by gate, exposure blocked while unresolved, and the decision
or escalation path.

## Available source material

You may use:

- the frozen engagement and source-sample fixtures in `../shared/fixtures/`;
- the artifact methods and templates from FDE modules 01 through 08;
- AI Engineer evidence patterns for data quality, prompt releases, latency/cost,
  release lineage, and production feedback;
- the Riverside platform contracts, source assets, ADRs, evaluation assets,
  operations documents, promise/evidence ledger, and limitations;
- the RAG Knowledge Pipeline's local prototype and unexecuted remote Databricks
  ingestion/indexing source.

Source presence is not passing evidence. The platform documentation records no
retained live Azure or Databricks validation and several unresolved integration
and configuration mismatches.

## Missing decisions and evidence

The customer has not supplied or approved:

- a representative golden workflow set and quality slices;
- minimum time saving, accepted-output rate, citation floor, quality floor, or
  broad-rollout decision rule;
- deletion propagation timing for withdrawn drafts and derived records;
- acceptable entitlement staleness for employees and contractors;
- PageTurn idempotency/reconciliation support;
- an approved service/region/SKU/quota matrix or regional failover behavior;
- an approved out-of-hours support model;
- a named post-hypercare owner for model and retrieval quality;
- a current price, discount, tax, exchange-rate, quota, or billing export;
- a committed SLO, SLA, RTO, RPO, error budget, service-credit term, or legal
  interpretation;
- a security, privacy, compliance, threat-model, penetration-test, or residency
  approval;
- executed release, isolation, deletion, load, rollback, incident, or operator
  drill evidence.

## Product and project tensions

Your proposal must inspect and disposition at least these documented project facts:

1. The Riverside platform has no root README, package manifest, environment
   profiles, integrated test orchestration, deployable RAG-orchestrator host, `azd`
   service entries, cloud tests, APIM deployment pipeline, release-gate CLI/job, or
   load-result download pipeline.
2. The Bicep endpoint name and Azure ML YAML endpoint name differ.
3. The production parameter example selects `uksouth`, while deployment metadata
   hardcodes `eastus2`.
4. The application timeout contract caps requests at 120 seconds, while Azure ML
   deployment definitions use 180 seconds.
5. The Databricks Direct Vector Access path is implemented in source but not live
   validated for identity, filters, deletion, latency, quota, region, cost, or
   rollback.
6. APIM policy source exists without a deterministic publish and restore pipeline.
7. Blue/green and load-test assets exist without retained cloud execution or
   rollback-rehearsal evidence.
8. The first release is single-region and has no regional serving failover claim.

For each item, decide whether it is a launch blocker, pre-canary requirement,
post-canary improvement, accepted limitation, or out of scope. Name the owner and
evidence that would change the disposition.

## Requested steering-meeting output

Bring a recommendation that includes:

1. A corrected problem statement, in-scope workflows, non-goals, and discovery
   backlog.
2. Measurable success criteria and a proposed golden workflow set with slices,
   methods, owners, and evidence classes.
3. The smallest supported architecture, rejected alternatives, ADRs, boundary
   diagrams, and phased delivery scope.
4. Source inventory, data contract, onboarding/quality gates, ACL and deletion
   behavior, lineage, and readiness verdict.
5. Identity flow, RBAC, threat model, residency map, controls matrix, negative-test
   plan, and external approval owners.
6. Low/expected/high capacity and cost scenarios, sensitivity analysis, proposed
   service envelope, support assumptions, and evidence needed before a commitment.
7. Evaluation, shadow, canary, ramp, rollback, compensation, and customer change
   plan.
8. One incident simulation with a redacted customer update, regression gates,
   residual risk, and re-enablement authority.
9. Operations and handoff package with dashboards, alerts, runbooks, support,
   training drills, acceptance, exit, and recurring health review.
10. A requirements-to-project map, product-gap ledger, and evidence-backed next
    iteration recommendation.

The steering group will accept a narrower scope or a `HOLD` recommendation when
the evidence supports it. It will not accept fabricated certainty.

## Constraints on your response

- Use no real customer data, credentials, secrets, live endpoints, or paid-service
  calls.
- Do not execute cloud, model, notebook, test, deployment, or customer operations
  as part of the authored capstone.
- Keep all external behavior explicitly unvalidated until retained evidence exists.
- Do not call Riverside secure, compliant, resident, production-ready, highly
  available, scalable, cost-optimized, or within SLA without exact scoped evidence
  and authorized review.
- Preserve an anti-AI/manual path and explain what remains usable when generation
  or workflow writes are disabled.
