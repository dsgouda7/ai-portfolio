# Draft SLO and Error-Budget Decision

> **UNAPPROVED TEMPLATE:** Riverside has no approved SLO, error budget, burn-rate
> alert, or policy window. Blank values and examples in source are not targets.
> Approval requires business, operations, security/privacy, and service owners
> plus representative live evidence.

## Decision context

| Field | Proposed value |
|---|---|
| Decision ID and version | `<id/version>` |
| Service boundary and customer journey | `<scope>` |
| Environments/regions/tiers | `<scope>` |
| Measurement source and query version | `<reference>` |
| Evaluation window and minimum volume | `<window/volume>` |
| Exclusions and maintenance treatment | `<rules>` |
| Business owner | `<owner>` |
| Technical owner | `<owner>` |

## Candidate indicators

| SLI | Proposed objective | Error event | Required dimensions | Evidence basis |
|---|---|---|---|---|
| Availability | `<unapproved>` | Eligible request lacks a valid bounded response | environment, route, tier, release, deployment, region | `<representative result>` |
| Deadline success | `<unapproved>` | Eligible request exceeds the approved end-to-end deadline | same bounded dimensions | `<representative result>` |
| Authorization safety | `100% proposed; unapproved` | Any false allow or forbidden-content result | test case and evidence ID, never customer identity/content labels | `<positive/negative result>` |
| Citation/refusal correctness | `<unapproved>` | Approved evaluation case breaches its release threshold | dataset/report version | `<release evidence>` |

Define eligibility, planned maintenance, client faults, policy rejections, overload,
dependency faults, and missing telemetry explicitly. Missing data is an evidence
gap, not an automatic exclusion.

## Error-budget policy proposal

| Decision | Unapproved proposal |
|---|---|
| Budget window | `<rolling/calendar window>` |
| Fast/slow burn thresholds | `<rates and windows>` |
| Alert and decision owners | `<owners>` |
| Rollout freeze threshold | `<threshold>` |
| Mandatory rollback review | `<threshold>` |
| Capacity/degradation authority | `<role and bounds>` |
| Budget reset/carry policy | `<rule>` |
| Security false-allow treatment | `<outside availability budget; immediate incident proposed>` |

## Approval record

- [ ] Queries were independently reviewed and tested against delayed/missing data.
- [ ] Representative peak, burst, dependency failure, and recovery evidence is
  linked.
- [ ] Customer and contractual consequences were reviewed.
- [ ] Security/privacy confirms safety events cannot be traded for availability.
- [ ] Cost/capacity confirms the objective is supportable.
- [ ] Incident, rollout, and rollback procedures use the approved thresholds.

| Approver role | Name/identity | Decision | UTC date | Evidence/comment |
|---|---|---|---|---|
| Business owner | `<pending>` | `unapproved` | `<pending>` | `<reference>` |
| Operations owner | `<pending>` | `unapproved` | `<pending>` | `<reference>` |
| Security/privacy owner | `<pending>` | `unapproved` | `<pending>` | `<reference>` |
| Service owner | `<pending>` | `unapproved` | `<pending>` | `<reference>` |
