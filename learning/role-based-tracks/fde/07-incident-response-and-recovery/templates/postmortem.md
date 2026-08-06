# Blameless Postmortem: `<INCIDENT-ID>`

> Status: `DRAFT | REVIEWED | ACTIONS ACCEPTED | CLOSED`
> Classification: `<approved handling label>`
> The purpose is to improve controls and decisions, not assign personal blame.

## Review boundary

| Field | Record |
|---|---|
| Incident window (UTC) | `<start/end>` |
| Customer impact window (UTC) | `<start/end/UNKNOWN>` |
| Review owner | `<role>` |
| Participants | `<roles>` |
| Approved audience | `<roles/groups>` |
| Evidence set | `<immutable manifest reference>` |
| Related release/index/policy/config | `<stable IDs>` |
| Legal/privacy handling instructions | `<authorized reference or NONE>` |

## Executive summary

`<What happened, bounded impact, containment, recovery, and current residual
risk. Use verified facts; omit raw content and blame.>`

## Impact

| Dimension | Confirmed result | Population/time | Evidence | Limitation |
|---|---|---|---|---|
| Customer/workflow | `<result>` | `<scope>` | `<reference>` | `<unknown/exclusion>` |
| Confidentiality/isolation | `<result or NONE FOUND in bounded review>` | `<scope>` | `<reference>` | `<what was not reviewed>` |
| Integrity/policy | `<result>` | `<scope>` | `<reference>` | `<limitation>` |
| Availability/deadline | `<result>` | `<scope>` | `<reference>` | `<limitation>` |
| Side effects/cost | `<result>` | `<scope>` | `<reference>` | `<limitation>` |

Avoid “no impact” unless the evidence supports the full stated scope. Prefer
“no additional impact found in `<bounded review>`.”

## Blameless postmortem checklist

### Facts: what we know

- [ ] The UTC timeline is append-only; corrections link to the superseded entry.
- [ ] Every material fact names a log, query, test, decision, or other approved evidence reference.
- [ ] Scope and limitations are explicit; unknowns are not rewritten as facts.
- [ ] The review records decision context and system conditions without personal blame.

### Hypotheses: what we tested

| Hypothesis | Test performed | Result | Evidence | Implication |
|---|---|---|---|---|
| Policy override typo | Code and policy-version review | `FALSIFIED` | `<illustrative reference>` | The reviewed code is correct; test the next causal boundary |
| Index included a disallowed version | Bounded provenance scan | `SUPPORTED` | `<illustrative reference>` | Investigate and remediate the promotion control |
| Disabled identity retained an active token | Authorized revoke-operation review | `SUPPORTED` | `<illustrative reference>` | Repair revocation and add a negative authorization gate |

Use `SUPPORTED | FALSIFIED | UNKNOWN`. These rows are illustrative patterns, not
Riverside incident findings. A supported hypothesis identifies a mechanism worth
remediating; it does not by itself prove a sole root cause.

### Corrective actions: how we prevent recurrence

An acceptable action names the control change, owner, due date, completion evidence,
monitoring period, and expiry when temporary.

| Action | Owner | Due date | Completion evidence | Monitor until | Expiry |
|---|---|---|---|---|---|
| Add negative test: disabled contractor cannot access the EU route | `<role>` | `<date>` | Green CI/CD or approved gate run | `PERMANENT` | `N/A` |
| Validate index promotion against approved document versions | `<role>` | `<date>` | Versioned release and bounded provenance scan | `PERMANENT` | `N/A` |
| Alert when the revoke job fails | `<role>` | `<date>` | Production rule reference and alert drill | `<date>` | `<date or N/A>` |

| Label | Candidate action | Decision |
|---|---|---|
| [Avoid] | “We will be more careful.” | Reject: no control, owner, or completion evidence. |
| [Avoid] | “TBD” or “Investigate further.” | Reject: no bounded action, owner, or deadline. |
| [Use] | “Run negative test `<ID>` on main before merge; alert owner `<role>`; due `<date>`.” | Accept when the named evidence and authority are recorded. |

## UTC timeline

| Event time | Observed time | Decision/observation | Evidence/reference | Decision quality note |
|---|---|---|---|---|
| `<timestamp>` | `<timestamp>` | `<fact or decision>` | `<reference>` | `<signal available, uncertainty, tradeoff>` |

## Causal analysis

### Trigger

`<The event that exposed the latent conditions; not automatically the root cause.>`

### Direct cause

`<The mechanism that produced the observed failure, supported by evidence.>`

### Contributing conditions

| Condition | Why it existed | Control expected | Why the control did not prevent/detect/limit it | Evidence |
|---|---|---|---|---|
| `<condition>` | `<organizational/technical context>` | `<control>` | `<gap>` | `<reference>` |

### Detection and response analysis

| Question | Finding | Evidence | Improvement needed? |
|---|---|---|---|
| Why did the first signal fire when it did? | `<finding>` | `<reference>` | `<YES/NO>` |
| What earlier signal could have detected it? | `<finding>` | `<reference>` | `<YES/NO>` |
| Did containment reduce exposure without weakening controls? | `<finding>` | `<reference>` | `<YES/NO>` |
| Was evidence preserved and access-controlled? | `<finding>` | `<reference>` | `<YES/NO>` |
| Were severity and communications timely and accurate? | `<finding>` | `<reference>` | `<YES/NO>` |
| Did handoff or ownership gaps affect response? | `<finding>` | `<reference>` | `<YES/NO>` |

### Five-whys caution

Use repeated “why” only while each answer remains evidence-backed. Stop before
the chain becomes a story about individual carelessness. Complex incidents may
have multiple interacting causes; a single root-cause label is not required.

## What went well, poorly, and luckily

| Category | Observation | Evidence | Keep/change action |
|---|---|---|---|
| Went well | `<control or decision>` | `<reference>` | `<action>` |
| Went poorly | `<gap>` | `<reference>` | `<action>` |
| Was lucky | `<uncontrolled condition that limited impact>` | `<reference>` | `<action that removes reliance on luck>` |

## Corrective actions

| Action ID | Cause/gap addressed | Action type | Owner | Due | Priority | Completion evidence | Regression/alert/drill | Status |
|---|---|---|---|---|---|---|---|---|
| `ACT-01` | `<cause/gap>` | `CONTROL | TEST | ALERT | RUNBOOK | TRAINING | DESIGN | PROCESS` | `<role>` | `<date>` | `<P0-P3>` | `<required artifact>` | `<reference/criterion>` | `OPEN` |

Rules:

- “Document,” “be careful,” and “monitor” are incomplete unless paired with a
  specific control, signal, decision rule, owner, and completion evidence.
- Every high-severity cause, contributing condition, and detection gap maps to
  an action or a recorded authorized risk acceptance.
- Closing an issue tracker item is not completion evidence by itself.

## Temporary controls and residual risk

| Control/risk | Owner | Expiry | Removal or revalidation gate | Escalation if missed |
|---|---|---|---|---|
| `<control/risk>` | `<role>` | `<date/time>` | `<evidence>` | `<automatic response>` |

## Organizational and legal follow-up

| Question | Authorized owner | Decision/reference | Status |
|---|---|---|---|
| Contract/customer notification | `<role>` | `<reference>` | `<status>` |
| Privacy/security/regulatory assessment | `<role>` | `<reference>` | `<status>` |
| Evidence retention/legal hold | `<role>` | `<reference>` | `<status>` |
| Vendor/insurer/partner notice | `<role>` | `<reference>` | `<status>` |

The postmortem records these decisions; it does not make them on behalf of the
authorized owner.

## Verification and closure

| Review date | Actions sampled | Evidence reviewed | Recurrence test/drill result | Residual gaps | Closure owner |
|---|---|---|---|---|---|
| `<date>` | `<IDs>` | `<references>` | `<result>` | `<gaps>` | `<role>` |

### Incident closure checklist

- [ ] All facts are bounded, evidence-backed, and recorded in the immutable UTC timeline.
- [ ] Every hypothesis is `SUPPORTED`, `FALSIFIED`, or `UNKNOWN` with an owner and decision consequence.
- [ ] Every corrective action is owned, scheduled, and has verifiable completion evidence.
- [ ] Re-enablement gates passed for the approved scope, or remaining exposure is contained and tracked.
- [ ] Approved customer and internal communications are archived by immutable reference.
- [ ] No placeholder ownership, due dates, `TBD`, or `TODO` items remain.
- [ ] The incident commander and required service, control, customer, security, privacy, or legal owners approve closure.

Closure does not erase unresolved risk. An `UNKNOWN` may remain only when an authorized
owner accepts the bounded residual risk, records monitoring and expiry/revalidation,
and keeps any unsupported exposure contained.

## Health check

- [ ] Impact statements are bounded and evidence-backed.
- [ ] Trigger, direct cause, contributing conditions, and detection gaps are distinct.
- [ ] The analysis focuses on system conditions and decision context, not blame.
- [ ] Every material gap maps to an action or authorized risk acceptance.
- [ ] Actions have owners, dates, and verifiable completion evidence.
- [ ] Temporary controls and residual risks have expiry/revalidation gates.
- [ ] Legal, privacy, contract, regulator, vendor, and insurer decisions remain with authorized owners.
- [ ] Closure includes a recurrence test, alert verification, or drill where applicable.
