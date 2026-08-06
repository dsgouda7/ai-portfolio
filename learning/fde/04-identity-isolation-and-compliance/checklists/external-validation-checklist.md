# External Validation Checklist

Local source and fixture tests cannot close these gates. Record evidence in the
approved customer or cloud evidence system; do not paste tokens, content, or live
resource exports into the notebook.

## Customer identity and policy

- [ ] Identity owner confirms issuer, audience, signing-key rotation, token lifetime, and revocation behavior.
- [ ] Security owner approves tenant, role, title-assignment, purpose, and exception semantics.
- [ ] Positive and negative token tests cover direct groups, nested groups, disabled users, guests, service identities, and stale caches.
- [ ] Entitlement freshness and full reconciliation meet an approved bound.
- [ ] Break-glass access is time-bound, approved, monitored, and independently reviewed.

## Cloud authorization and network

- [ ] Deployed RBAC assignments are exported and reviewed at exact scopes.
- [ ] Positive and negative data-plane calls prove each managed/workload identity path.
- [ ] Management-plane deploy rights are separated from data-plane read/write rights.
- [ ] Private endpoints, DNS resolution, firewalls, public-access flags, and denied-origin tests are retained.
- [ ] Gateway policy editors and identity administrators have constrained, audited privileges.

## Residency and privacy

- [ ] Legal/privacy owners define data categories, purpose/legal basis, retention, deletion, and data-subject handling.
- [ ] Customer security/legal approve allowed storage and processing regions for each category.
- [ ] Resource inventory records service, SKU, region, replication, backup, diagnostics, and support paths.
- [ ] Databricks and other processors are reviewed for control-plane, classic/serverless plane, cross-Geo, and subprocessor behavior.
- [ ] Evaluation, build, CI/CD, telemetry, incident evidence, and support access appear in the residency map.
- [ ] Representative request, indexing, deletion, backup, restore, and incident flows are traced in the authorized environment.

## Audit and operations

- [ ] Audit destination access, encryption, immutability, retention, deletion exceptions, and legal holds are approved.
- [ ] Sampled metrics, logs, and traces under success and failure contain no customer content or forbidden labels.
- [ ] Alerting detects forbidden access, entitlement staleness, policy failure, and route disablement.
- [ ] Tenant-route containment and security-owned re-enablement are rehearsed in staging.
- [ ] Incident notification, evidence preservation, and regulator/customer escalation obligations are approved by authorized owners.

## Decision record

| Field | Required value |
|---|---|
| Environment | Exact non-production or production environment |
| Source commit and release | Immutable identifiers |
| Evidence window | Start/end timestamps |
| Reviewers | Named identity, security, privacy/legal, and operations authorities |
| Exceptions | Scope, approver, expiry, compensating controls |
| Decision | Approved, conditionally approved, or rejected |
| Revalidation trigger | Identity, policy, region, service, network, retention, or architecture change |
