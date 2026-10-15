# Local Proof Checklist

Use this before describing a local result as evidence. A checked item proves only
the named local property and environment.

## Fixture and contract integrity

- [ ] The frozen fixture version is `RIV-FDE-1.0.0`; no shared fixture was edited.
- [ ] Every join uses stable IDs, not array positions or display names.
- [ ] Required context is exactly tenant, actor, roles, region, purpose, titles, and trace ID.
- [ ] The active editor and disabled contractor records are both inspected.
- [ ] Expected facts for `FDE-04` retain their supplied evidence classes.
- [ ] No credential, token value, connection string, live endpoint, or customer identifier is present.

## Boundary behavior

- [ ] Gateway rejects missing context and derives effective roles from the identity record.
- [ ] Requested roles can narrow authority but cannot add authority.
- [ ] Tenant and region are validated against the identity and tenant policy.
- [ ] Purpose is validated independently because the frozen IdP record does not supply it.
- [ ] Retrieval filters are server-derived and include tenant, region, purpose, titles, ACL, classification, and deletion state.
- [ ] Returned resources are authorized again after retrieval.
- [ ] Tool authorization checks the concrete tool, action, payload, purpose, role, and human-confirmation state.
- [ ] Policy failure or missing context denies by default.
- [ ] Allow and deny branches both produce content-free audit decisions.
- [ ] Metrics exclude actor, tenant, trace, request, document, prompt, and completion identifiers.
- [ ] Response assembly consumes trusted context but the public response does not echo roles or internal filters.

## Negative coverage

- [ ] Same-tenant positive control is present.
- [ ] Cross-tenant resource access is denied.
- [ ] Disabled contractor access is denied despite stale group membership.
- [ ] Requested role escalation is denied.
- [ ] Prohibited rights/payment mutation is denied even with human confirmation.
- [ ] Disallowed region is denied.
- [ ] Missing purpose is denied.
- [ ] Unassigned title access is denied.
- [ ] Any false allow is treated as stop-ship and mapped to `RISK-RIV-003` / `INC-RIV-003`.

## Evidence statement

- [ ] Result states whether it is `[Local-static]`, `[Local-measured]`, or `[Modeled]`.
- [ ] A measured result names date, environment, fixture/version, code/source commit, scenario count, result, and limitations.
- [ ] The report does not use `secure`, `compliant`, `resident`, or `least privilege` without scope and reviewer.
- [ ] External validation gaps remain open in the controls and residency matrices.
