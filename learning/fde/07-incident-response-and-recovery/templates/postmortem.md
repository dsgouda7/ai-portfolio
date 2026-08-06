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

## Health check

- [ ] Impact statements are bounded and evidence-backed.
- [ ] Trigger, direct cause, contributing conditions, and detection gaps are distinct.
- [ ] The analysis focuses on system conditions and decision context, not blame.
- [ ] Every material gap maps to an action or authorized risk acceptance.
- [ ] Actions have owners, dates, and verifiable completion evidence.
- [ ] Temporary controls and residual risks have expiry/revalidation gates.
- [ ] Legal, privacy, contract, regulator, vendor, and insurer decisions remain with authorized owners.
- [ ] Closure includes a recurrence test, alert verification, or drill where applicable.
