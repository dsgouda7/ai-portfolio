# Discovery Review

## Review record

| Field | Value |
|---|---|
| Engagement/artifact versions | `<IDs and versions>` |
| Review date | `<date>` |
| Participants and authority | `<roles>` |
| Workflow scope | `<included>` |
| Explicit exclusions | `<excluded>` |
| Decision | `<pass/conditional/block>` |

## Gate evidence

| Gate question | Artifact/evidence ref | Owner | Result | Conditions or gaps |
|---|---|---|---|---|
| Is current state confirmed? | `DSC-02/DSC-03` | `<workflow owner>` | `<result>` | `<conditions>` |
| Are non-goals explicit? | `DSC-04` | `<business/legal owner>` | `<result>` | `<conditions>` |
| Is every criterion testable and sliced? | `DSC-04` | `<acceptance owner>` | `<result>` | `<conditions>` |
| Are action boundaries explicit? | `<action inventory>` | `<workflow/security owner>` | `<result>` | `<conditions>` |
| Are data uses and unknowns owned? | `<data inventory/DSC-05>` | `<data owners>` | `<result>` | `<conditions>` |
| Are claims correctly classified? | `<claim register>` | `<FDE/reviewer>` | `<result>` | `<conditions>` |

## Conditional approval

| Condition | Owner | Due date/gate | Exposure limit | Automatic response if missed |
|---|---|---|---|---|
| `<condition>` | `<owner>` | `<milestone>` | `<what remains blocked>` | `<block/reopen gate>` |

## Architecture handoff

| Handoff input | Version/ref | Open decisions retained | Architecture question enabled |
|---|---|---|---|
| `<artifact>` | `<ref>` | `<IDs>` | `<smallest-solution question>` |

Record acceptance as a scoped customer-validation event only when the participant has the required authority. Meeting attendance is not approval.
