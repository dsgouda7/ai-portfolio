# Incident and Rollback Tabletop Evidence

> **Status:** Scenario and blank evidence template. Completing prose is not live
> recovery evidence. Distinguish discussion, staging execution, and production
> observation in every result.

## Scenario definition

| Field | Value |
|---|---|
| Exercise ID/version/date | `<values>` |
| Participants and assigned roles | `<identities>` |
| Environment and evidence class | `<discussion-only/staging exercise>` |
| Scenario/injects and hidden assumptions | `<description>` |
| Active release/slot/index/policy | `<values>` |
| Expected detection and severity | `<values>` |
| Customer/security/residency impact | `<hypothesis>` |
| Known-good rollback targets | `<immutable references>` |
| Success/abort criteria | `<approved criteria>` |

Include at least one dependency failure, misleading or missing telemetry, retry
amplification risk, authorization negative, deletion/legal-hold constraint, capacity
decision, communications decision, and rollback-versus-roll-forward choice.

## Timeline evidence

| UTC or simulated time | Inject/signal | Role/action/decision | Evidence used | Approval | Outcome/gap |
|---|---|---|---|---|---|
| `<time>` | `<inject>` | `<action>` | `<reference>` | `<identity>` | `<result>` |

## Required decisions

- [ ] Incident declaration, severity, roles, change freeze, and update cadence.
- [ ] Failing boundary and applicable
  [dependency policy](../dependency-failure-policy.md).
- [ ] Containment that preserves identity, ACL, deletion, safety, and evidence.
- [ ] Exact release/index/APIM/IaC rollback target and compatibility check.
- [ ] Commands are reviewed or run only in the declared evidence class; outputs and
  failures are retained and redacted.
- [ ] Recovery checks include readiness, contracts, authorization negatives,
  retrieval/citations/refusal, telemetry redaction/freshness, and observation.
- [ ] Customer/legal/security communications and closure authority are identified.

## After-action record

| Measure | Result |
|---|---|
| Time to detect/declare/contain/decide/recover | `<discussion estimate or measured staging value>` |
| Decision points lacking approved data | `<gaps>` |
| Procedure/command defects | `<gaps>` |
| Access or separation-of-duties defects | `<gaps>` |
| Evidence/redaction defects | `<gaps>` |
| Corrective action, owner, due date, proof test | `<values>` |
| Exercise reviewers and acceptance | `<identities/decision>` |
