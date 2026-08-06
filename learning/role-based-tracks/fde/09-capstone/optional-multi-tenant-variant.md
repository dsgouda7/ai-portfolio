# Optional Capstone Variant: Multi-Tenant and Edge Cases

> **Variant status:** `OPTIONAL`, `SYNTHETIC`, `UNEXECUTED`, `NOT CUSTOMER VALIDATED`.

Use this overlay only after completing the base Riverside package. Do not edit the frozen `RIV-FDE-1.0.0` fixtures or present the variant as a new Riverside fact. Add variant records in your own package with the prefix `VAR-MT-*` and label every input `[Modeled]` or `[Unknown]`.

## Scenario overlay

Riverside considers offering the editorial assistant to two separately governed imprints:

| Tenant | Modeled constraint | Open authority question |
|---|---|---|
| `VAR-MT-UK` | UK/EU primary processing; unpublished manuscripts; weekday UK support | Which services and support paths are approved for each data class? |
| `VAR-MT-US` | US primary processing; separate editorial and rights roles; deadline-week support request | Who can approve expanded support and its commercial terms? |

The product team proposes shared control-plane components and a shared vector service to reduce cost. No owner has approved shared indexes, shared caches, cross-tenant failover, pooled quotas, or shared operational evidence.

## Required edge cases

Add at least these cases to the base package:

1. The same public policy appears in both tenants, but tenant-specific applicability metadata differs.
2. A manuscript moves from the UK imprint to the US imprint while an older chunk remains indexed in the UK scope.
3. A contractor belongs to groups in both tenants and is disabled in only one identity source snapshot.
4. A deletion request covers a source document, derived chunks, cached responses, evaluation examples, and incident evidence with different retention authorities.
5. One tenant exhausts a shared quota during the other tenant's deadline window.
6. A rollback restores a model release but not the tenant-specific index or policy version.
7. A regional incident affects one tenant while status communication and SLA assumptions differ between tenants.
8. Cost allocation is disputed because shared infrastructure, retries, and observability cannot be attributed from averages alone.

## Additional deliverables

Extend the base artifacts with:

- a tenant/resource isolation matrix covering control plane, data plane, indexes, caches, queues, telemetry, secrets, identities, quotas, and support evidence;
- cross-tenant negative tests and expected deny behavior;
- per-tenant deletion and transfer-state diagrams;
- noisy-neighbor capacity and cost-attribution scenarios;
- tenant-specific rollout, rollback, incident communication, and re-enablement authorities;
- a decision on shared versus dedicated resources for each boundary, with revisit triggers;
- an explicit manual or isolated degraded mode when a shared dependency is unsafe.

## Variant gate

A strong response does not assume that sharing is efficient or that dedication is safer. It names the evidence and authority required for each choice, preserves unresolved constraints, and blocks cross-tenant exposure when the boundary cannot be proved. This variant can deepen the static capstone score, but it cannot establish deployed isolation, real quota behavior, contractual commitments, or production incident competence.
