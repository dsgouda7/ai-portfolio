# Recurring Customer Health Review Template

> The review is a decision forum, not a dashboard recital. Every section ends in continue, change, investigate, reduce exposure, or retire.

## Review control

| Field | Value |
|---|---|
| Artifact ID | `HOF-HEALTH-<period>` |
| Review period | `<start/end UTC>` |
| Cadence | `<weekly during hypercare; monthly after acceptance>` |
| Chair/decision owner | `<role>` |
| Required attendees | Workflow/product, operations, support, data, security, finance, quality/model owner |
| Release/index/policy scope | `<versions>` |
| Previous actions reviewed | `<references>` |
| Next review | `<date>` |

## Decision dashboard

| Domain | Evidence reviewed | Trend/slice | Claim class | Decision | Owner/due date | Revalidation trigger |
|---|---|---|---|---|---|---|
| Workflow value | Cycle time, accepted output, active use, structured rejection reasons | By approved use case/cohort | `<class>` | `<decision>` | `<owner/date>` | Workflow or cohort change |
| Retrieval/data | Current-guidance rate, no-result, stale/duplicate/deletion/ACL events | By source/index/workflow slice | `<class>` | `<decision>` | `<owner/date>` | Source/schema/index change |
| Generation/quality | Evaluation trend, abstention, citation use, critical slices | By release/use case | `<class>` | `<decision>` | `<owner/date>` | Model/prompt/policy change |
| Identity/policy | Forbidden/unauthorized events, entitlement freshness, policy denials | By region/tier and protected audit scope | `<class>` | `<decision>` | `<owner/date>` | Identity/RBAC/policy change |
| Reliability/support | Incidents, deadline success, runbook use, alert noise, support load | By severity/service/cohort | `<class>` | `<decision>` | `<owner/date>` | Rota/service/dependency change |
| Capacity/cost | Queue, tokens/quota, tool/checkpoint/trace pressure, unit cost, budget burn | By region/tier/release | `<class>` | `<decision>` | `<owner/date>` | Traffic/pricing/quota change |
| Change health | Releases, indexes, policies, thresholds, emergency changes | Change success/failure/rework | `<class>` | `<decision>` | `<owner/date>` | Process or authority change |
| Training/ownership | Drill freshness, new owners, FDE dependencies, knowledge gaps | By role/runbook | `<class>` | `<decision>` | `<owner/date>` | Owner/runbook change |

## Incident and feedback learning

| Input ID | What changed our belief? | Evidence | Affected criterion/control | Action or backlog ID | Owner | Due date |
|---|---|---|---|---|---|---|
| `<INC/feedback/review ID>` | `<bounded statement>` | `<reference>` | `<ID>` | `<ID>` | `<owner>` | `<date>` |

## Assumptions, limitations, and expiry

- Which modeled assumptions now have production measurements?
- Which evidence expired or no longer matches the active release, index, policy, region, or population?
- Which customer-validated decision needs revalidation because workflow or ownership changed?
- Which unknown still blocks exposure, including `UNK-RIV-006` and `UNK-RIV-010`?
- Which limitation is being tolerated, by whom, until when, and with what exposure bound?

## Retirement check

Evaluate retirement or scope reduction when any condition persists:

- workflow value remains below the customer-approved floor after agreed remediation;
- operating cost/support burden exceeds the accepted envelope;
- critical controls cannot be evidenced or maintained;
- the source workflow, policy, or data authority is retired;
- a deterministic or manual alternative now meets the need with lower risk;
- the customer cannot sustain named ownership after hypercare.

Record the retirement decision authority, data/artifact retention and deletion path, user communication, disabled integrations, access removal, and final evidence package.

## Review outcome

| Decision | Scope | Evidence basis | Conditions | Owner | Next checkpoint |
|---|---|---|---|---|---|
| `continue / change / investigate / reduce exposure / retire` | `<scope>` | `<references>` | `<conditions>` | `<owner>` | `<date/event>` |

## Quick health check

- [ ] The review includes workflow value, not only technical uptime.
- [ ] Per-customer slices do not leak customer identifiers into metric labels.
- [ ] Every action is triggered by evidence or an explicitly labeled unknown.
- [ ] Expired evidence is not reused silently.
- [ ] Model, retrieval, policy, and ownership changes reopen the right gates.
- [ ] Retirement remains an available decision.
