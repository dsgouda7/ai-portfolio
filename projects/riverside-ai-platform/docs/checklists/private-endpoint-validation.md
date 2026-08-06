# Private Endpoint Positive and Negative Validation

> **Status:** Blank live-validation checklist. Source topology and private endpoint
> resources are not proof of reachability or isolation, and this checklist has not
> been executed.

Run from named approved and denied origins for Storage Blob/DFS, Key Vault, Azure
ML API/scoring, the internal orchestrator backend, and any separately approved
Databricks or Azure Monitor private path.

## Capture configuration

- [ ] Record environment, region, subscription/resource group, VNet/subnets,
  private endpoint IDs/states, NIC private IPs, private DNS zones/links/records,
  route tables, NSGs/firewalls, service public-access flags, APIM network mode, and
  client origin identities.
- [ ] Confirm names resolve to expected private IPs from approved origins; retain
  resolver, answer, TTL, and timestamp.
- [ ] Confirm denied/public origins do not resolve a usable private path and cannot
  fall back to an enabled public endpoint.

## Positive tests

- [ ] APIM reaches the orchestrator with valid managed identity and trusted backend
  audience; health/readiness and bounded chat smoke behave as designed.
- [ ] Orchestrator identity reaches only required Storage, Key Vault, Azure ML,
  Databricks, and telemetry operations over the approved path.
- [ ] TLS name/chain validation succeeds and the observed remote address/path is
  retained without tokens or content.
- [ ] Required diagnostics arrive through the approved destination.

## Negative tests

- [ ] The same FQDN/operation fails from a public host and a VNet/subnet without an
  allowed route; record DNS, TCP/TLS, HTTP status, and Azure deny evidence.
- [ ] Missing/invalid token, wrong audience/tenant, wrong managed identity, and
  excessive data-plane operation fail from an otherwise approved network origin.
- [ ] Direct orchestrator invocation cannot inject trusted tenant, actor, tier, or
  group headers.
- [ ] Public endpoint and service bypass settings are disabled or have an approved,
  tested exception. Private DNS removal/mislink does not silently use public DNS.
- [ ] Cross-tenant and forbidden-document application requests remain denied after
  network access succeeds.

## Result

| Field | Value |
|---|---|
| Validation ID and UTC window | `<values>` |
| Approved/denied origin inventory | `<references>` |
| Configuration export hashes | `<references>` |
| Positive results | `<references>` |
| Negative results | `<references>` |
| Exceptions, owners, and expiry | `<references>` |
| Network/security reviewers | `<identities and decision>` |
