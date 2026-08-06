# Riverside Capstone Decision Record

## Decision Scope

| Field | Value |
|---|---|
| Candidate release ID/version | `<immutable ID/version>` |
| Baseline release | `<accepted ID/version>` |
| Rollback target | `<accepted earlier release>` |
| Environment/region | `<scope>` |
| Workload/dataset | `<version and digest>` |
| Requested next stage | `<offline/staging/shadow/canary/broad>` |
| Source commit | `<commit>` |
| Decision time UTC | `<timestamp>` |
| Decision | `PROMOTE ONE STAGE / HOLD / REJECT` |

## Gate Record

| Gate | Status | Evidence link and class | Failed/unproven condition | Next discriminating test | Owner |
|---|---|---|---|---|---|
| Data quality | `<status>` | `<link/class>` | `<condition>` | `<test>` | `<owner>` |
| Retrieval | `<status>` | `<link/class>` | `<condition>` | `<test>` | `<owner>` |
| Generation/citation | `<status>` | `<link/class>` | `<condition>` | `<test>` | `<owner>` |
| Safety/authorization | `<status>` | `<link/class>` | `<condition>` | `<test>` | `<owner>` |
| Prompt comparison | `<status>` | `<link/class>` | `<condition>` | `<test>` | `<owner>` |
| Local operations/cost | `<status>` | `<link/class>` | `<condition>` | `<test>` | `<owner>` |
| Artifact compatibility/rollback | `<status>` | `<link/class>` | `<condition>` | `<test>` | `<owner>` |
| Azure evidence for requested stage | `<status>` | `<link/class>` | `<condition>` | `<test>` | `<owner>` |

## Decision Rationale

State why the evidence supports this decision. Name every non-compensating gate and explain why an aggregate improvement cannot override it.

## Intervention and Follow-Up

| Action | Why selected or rejected first | Owner | Evidence required | Follow-up window | Rollback/containment |
|---|---|---|---|---|---|
| `<prompt/retrieval/guardrail/fine-tune/no-action/operations>` | `<reason>` | `<owner>` | `<test>` | `<window>` | `<target>` |

## Claim Boundary

### Strongest Supported Claims

1. `<claim with evidence link and scope>`
2. `<claim with evidence link and scope>`
3. `<claim with evidence link and scope>`
4. `<claim with evidence link and scope>`
5. `<claim with evidence link and scope>`

### Most Important Unsupported Claims

1. `<ledger link and evidence-safe wording>`
2. `<ledger link and evidence-safe wording>`
3. `<ledger link and evidence-safe wording>`
4. `<ledger link and evidence-safe wording>`
5. `<ledger link and evidence-safe wording>`

## Review and Approval

| Role | Reviewer | Decision | Conditions/residual risk | Timestamp |
|---|---|---|---|---|
| Data owner | `<name>` | `<decision>` | `<conditions>` | `<time>` |
| Security owner | `<name>` | `<decision>` | `<conditions>` | `<time>` |
| Operations owner | `<name>` | `<decision>` | `<conditions>` | `<time>` |
| Release approver | `<name>` | `<decision>` | `<conditions>` | `<time>` |

## Rubric Result

- Raw score: `<0-100>`
- Applicable score caps: `<caps or none>`
- Final score: `<0-100>`
- Material deductions: `<evidence gaps>`

A favorable decision does not increase the score. Trustworthy evidence and honest limits do.
