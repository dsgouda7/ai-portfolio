# Identity, Isolation, and Compliance

Riverside's assistant can answer only after seven pieces of context survive every
authority boundary: tenant, actor, roles, region, purpose, title assignments, and
trace ID. A valid sign-in is not enough. The local lab starts with the failure that
matters most: caller-controlled filters make an EU editor look authorized for a US
manuscript, and stale group membership makes a disabled contractor look active.

This chapter uses only synthetic, committed fixtures. It does not contact Azure,
an identity provider, a model, a vector store, or a customer system. The notebook
executed successfully against those fixtures during route validation and was then
cleared, so its committed code cells retain null execution counts and empty outputs.

## Learning outcomes

By the end, you can:

1. Separate authentication, request purpose, authorization, and response minimization.
2. Derive tenant, actor, and effective roles from a trusted identity record rather than request fields.
3. Build retrieval filters from tenant, role, region, purpose, title, classification, and deletion state.
4. Re-check returned records so a backend filter defect does not become disclosure.
5. Authorize a concrete tool and payload independently of model intent or human-friendly prose.
6. Audit allow and deny branches without placing content or high-cardinality identity in metrics.
7. Design and report negative tests across tenant, role, region, purpose, and title boundaries.
8. State exactly what local evidence cannot prove about cloud enforcement, residency, privacy, or law.

## Chapter map

| Item | Use |
|---|---|
| [Notebook](identity-isolation-and-compliance.ipynb) | Failure-first implementation and artifact rendering |
| [Scenario fixture](fixtures/identity-scenarios-v1.json) | Deterministic positive and negative expected outcomes |
| [Identity flow](templates/identity-flow.md) | End-to-end authority and minimization path |
| [RBAC matrix](templates/rbac-matrix.md) | Human and service authority by action/resource |
| [Data-flow and residency map](templates/data-flow-residency-map.md) | Data category, location, transfer, and validation owner |
| [Threat model](templates/threat-model.md) | Assets, boundaries, abuse cases, controls, and residual risk |
| [Controls matrix](templates/controls-matrix.md) | Control objective, implementation, evidence, owner, and status |
| [Isolation report](templates/isolation-test-report.md) | Expected decisions, execution state, and release verdict |
| [Notebook output record](templates/notebook-output-record.md) | `NOT RUN`-by-default observations, evidence references, false allows/denies, and external gaps |
| [Local proof checklist](checklists/local-proof-checklist.md) | Review before calling local evidence complete |
| [External validation checklist](checklists/external-validation-checklist.md) | Cloud/customer/legal gates local work cannot close |
| [Setup](SETUP.md) | Optional local Jupyter environment; no cloud authentication |

## Evidence boundary

| Label | Meaning here | Example |
|---|---|---|
| `[Local-static]` | Inspected source or expected result; no mechanism was run | The scenario ledger contains a cross-tenant denial case |
| `[Local-measured]` | A named local run produced a result | Seven of seven local scenarios matched expected decisions after a learner runs the notebook |
| `[Modeled]` | Expected design behavior based on supplied policy | EU manuscript requests should remain in `REG-UKS` |
| `[External validation required]` | Needs authorized environment and named approver | Deployed tokens, RBAC, private DNS, backups, diagnostics, and subprocessors preserve the approved boundary |

Do not write `secure`, `compliant`, `resident`, or `least privilege` as an
unqualified conclusion. Record scope, environment, evidence, reviewer, date, and
revalidation trigger.

## Failure ledger

The local fixture seeds:

- `ISO-RIV-002`: cross-tenant manuscript request;
- `ISO-RIV-003`: disabled contractor with a stale nested editor role;
- `ISO-RIV-004`: requested role escalation plus a prohibited rights write;
- `ISO-RIV-005`: EU content routed to a disallowed region;
- `ISO-RIV-006`: valid identity with no declared purpose;
- `ISO-RIV-007`: title assignment mismatch;
- `ISO-RIV-008`: valid role denied a prohibited rights tool.

`ISO-RIV-001` is the retrieval positive control. `ISO-RIV-009` is the bounded
tool positive control: an editor may propose, but not commit, a workflow change.

Any successful forbidden access is a stop-ship result and maps to `RISK-RIV-003`.
The contractor case maps to `INC-RIV-003`, whose frozen severity is `SEV-1`.

## Before you run it

The route setup and synthetic notebook execution have been verified. For your own
recorded run, follow [SETUP.md](SETUP.md), inspect the fixture first, run cells in
order, and preserve the generated report separately from the committed `NOT RUN`
example. Never substitute production exports or real tokens into the notebook.

## Production bridge and downstream integration path

The notebook teaches the mechanism; the Riverside platform owns the production
contracts and Azure-shaped source. Compare the local flow with:

- [Security and data boundaries](../../../projects/riverside-ai-platform/docs/security-and-data-boundaries.md)
- [Data residency](../../../projects/riverside-ai-platform/docs/data-residency.md)
- [APIM gateway boundary](../../../projects/riverside-ai-platform/apim/README.md)
- [APIM client authentication policy](../../../projects/riverside-ai-platform/apim/policies/fragments/client-auth.xml)
- [Managed identity and RBAC infrastructure](../../../projects/riverside-ai-platform/infra/README.md)
- [Versioned contracts](../../../projects/riverside-ai-platform/contracts/README.md)
- [Governance, guardrails, and security](../../agentic-ai-system-design/11-governance-guardrails-and-security.md)

Source policy and a local pass are inputs to a security review. They are not a
deployed-control attestation or a legal conclusion.

Carry `SEC-01` through `SEC-06` into a supervised implementation review by mapping each trusted-context field and deny rule to the APIM, application, retrieval, tool, telemetry, and infrastructure boundary that enforces it. The practicum must retain negative-test evidence for tenant, role, region, purpose, title, stale entitlement, and deleted-content paths. Entra configuration, private networking, deployed RBAC, diagnostic destinations, residency, and legal or compliance conclusions remain with their authorized owners.
