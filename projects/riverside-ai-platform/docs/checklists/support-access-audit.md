# Support-Access Audit

> **Status:** Blank review procedure. It does not establish that support access is
> approved, configured, or reviewed.

## Scope and population

- [ ] Record review ID, period, environments, subscriptions/resource groups,
  Databricks workspace/catalog/schema, APIM, Azure ML, Storage, Key Vault,
  Container Apps, ACR, monitoring, CI/CD, and evidence stores.
- [ ] Export direct and group-based role assignments, privileged identity
  eligibility/activation, service principals, managed identities, access packages,
  local/service-specific permissions, break-glass paths, and support-vendor access.
- [ ] Map every principal to owner, employer/vendor, purpose, ticket/change,
  approved scope, role, start/expiry, and last use. Do not place customer content
  or secrets in the audit record.

## Review tests

- [ ] No standing support access exists where just-in-time, time-bound access is
  feasible.
- [ ] Support cannot edit APIM policy, RBAC, production data, model/index artifacts,
  telemetry retention, or evidence without explicit scoped approval.
- [ ] Data-plane access is separated from management-plane diagnosis and from
  deployment approval.
- [ ] Activation requires MFA, reason/ticket, approval, bounded duration, and
  audit logging; emergency paths have retrospective review.
- [ ] Logs, traces, source URIs, and operational exports expose only the minimum
  approved fields. Content access requires a separate legal/purpose approval.
- [ ] Positive test proves an approved support task; negative tests prove denied
  customer-content access, denied cross-tenant access, and denied privilege
  escalation.
- [ ] Access, activation, query, export/download, policy/RBAC change, and evidence
  access events reach the approved audit destination and retention policy.
- [ ] Dormant, orphaned, duplicate, excessive, expired, and vendor-offboarded
  access is removed through change control and re-exported for confirmation.

## Evidence and sign-off

| Field | Value |
|---|---|
| Review ID and UTC period | `<values>` |
| Population/export queries and hashes | `<references>` |
| Exceptions and expiry | `<owner/date/reference>` |
| Positive/negative test evidence | `<references>` |
| Removed access and confirmation | `<references>` |
| Security/data/service reviewers | `<identities and decisions>` |
| Next review date | `<UTC date>` |
