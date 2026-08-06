# Cohort Communication

## Invitation / pre-change notice

**Subject:** Riverside Editorial Copilot: [cohort name] starts [date]

You are part of the `[cohort]` evaluation for `[approved workflows]`. During this
window, the assistant may `[allowed behavior]`. It will not `[prohibited behavior]`.
Your current manual workflow remains available at `[location/process]`.

Before access starts:

- complete `[training]` by `[date]`;
- use only `[approved titles/data/use cases]`;
- review citations and every proposed action before acceptance;
- report incorrect, unsafe, or confusing results through `[feedback route]`;
- contact `[support route and hours]` for service problems.

We will evaluate `[outcome, quality, safety, adoption, latency, cost, support]` over
`[observation window]`. Participation does not mean the release is broadly approved.
The named business owner is `[owner]`; the next decision is `[date/time UTC]`.

## Training record

Use an approved participant reference rather than a raw user ID.

| Participant reference / role | Required module | Completed UTC | Workflow understood | Fallback tested | Feedback route tested | Support route tested | Exception owner |
|---|---|---|---|---|---|---|---|
|  |  |  | `YES | NO` | `YES | NO` | `YES | NO` | `YES | NO` |  |

### Participant acknowledgement

> I acknowledge the approved workflow boundary, prohibited uses, required review,
> manual fallback, feedback route, and support route for this cohort. I understand
> that access may be paused or removed when a gate is not met.

| Participant reference | Role | Acknowledged by | Signed at (UTC) | Training evidence | Exception/expiry |
|---|---|---|---|---|---|
|  |  |  |  |  | `NONE or approved reference/date` |

Training is complete only when the required checks and acknowledgement have retained
evidence. Attendance or module assignment alone is not proof of workflow readiness.

## Feedback contract

Collect structured reason, task type, tenant/imprint slice, release ID, timestamp,
and redacted evidence reference. Do not place manuscript text, prompts, completions,
or user IDs in metric labels or broad communication channels.

## Feedback collection

| Timestamp (UTC) | Workflow/task | Outcome | Reason | Evidence reference | Suggested action |
|---|---|---|---|---|---|
| `<timestamp>` | Policy search and draft | `WORKED` | Cycle time increased 15% in the bounded exercise | `FDB-RIV-001` | Review retrieval and review-step timing before the next gate |
| `<timestamp>` | Citation review | `CONFUSING` | Generated section was incomplete | `EV-RIV-002` in the approved evidence system | Add the case to disagreement review; do not copy content here |
| `<timestamp>` | Guidance | `BLOCKED` | Proposed guidance contradicted policy | `FDB-RIV-003` | Preserve the block and route to policy review |

Feedback informs quality, adoption, support, and safety gates; it does not waive a
threshold or authorize promotion. Keep raw content, prompts, responses, customer or
user IDs, and sensitive attributes in approved systems, not labels or this record.

## Structured cohort exit

Allowed reason categories: `TRAINING INCOMPLETE`, `WORKFLOW CHANGE`, or
`POLICY CONSTRAINT`. Add an approved bounded reason when none applies; do not place
personal or sensitive detail in this record.

| Participant reference | Exit reason | Feedback provided | Owner approval | Rollback/access removal date (UTC) | Evidence reference |
|---|---|---|---|---|---|
|  |  | `YES | NO` | `<role/reference>` |  |  |

An exit is complete only when the owner records the decision, access or cohort routing
is restored to the approved state, and any unresolved safety feedback is routed to the
current gate or incident process.
