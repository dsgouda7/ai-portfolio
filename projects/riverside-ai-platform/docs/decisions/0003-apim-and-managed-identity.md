# ADR-0003: APIM and Managed Identity

- **Status:** Accepted
- **Date:** 2026-08-05

## Context

The editing application needs one stable contract while backend deployments,
capacity, and releases change. Static backend keys would create secret distribution
and rotation work and would weaken attribution.

## Decision

Use Azure API Management as the application gateway. Authenticate clients with
Microsoft Entra ID, authorize at the API boundary, and authenticate APIM to Azure
backends with a least-privilege managed identity. Apply request/token bounds,
deadlines, bounded retries, routing, policy rejection, and bounded telemetry at the
gateway.

Tenant and authorization context come from trusted identity claims. Clients cannot
submit tenant or ACL filters. API-key authentication is outside the configuration
contract.

## Consequences

- APIM policy-edit permission becomes security-sensitive because a policy editor
  can use the service identity to reach allowed backends. Limit and audit it.
- Each backend requires an explicit audience/resource and scoped data-plane role.
- Identity token acquisition, role propagation, policy behavior, private network
  reachability, and negative authorization paths require live Azure tests.

## Evidence state

The config/API contracts, OpenAPI document, backend Bicep, composed policy,
fragments, named-value contract, and static policy tests are implemented source
assets. They were not executed in this task. No imported API, published policy,
managed-identity token, role assignment, private path, circuit-breaker observation,
or negative cloud authorization evidence is linked.
