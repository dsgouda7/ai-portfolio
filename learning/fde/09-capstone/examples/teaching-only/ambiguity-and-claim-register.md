# Teaching Example: Ambiguity and Claims

## Ambiguity record

| Field | Example value |
|---|---|
| Conflict ID | `CON-RIV-001` |
| Statement A | Sponsor says the experience should feel instant. |
| Statement B | Operations proposes an 8-second policy p95 until cloud tests exist. |
| What is known | The statements use different decision forms: qualitative expectation versus modeled planning threshold. |
| What is not known | Representative request mix, accepted wait by workflow step, target-environment latency, and who can accept the criterion. |
| Resolution owner and authority | Riverside editorial workflow owner proposes acceptable experience; service owner confirms operational feasibility; authorized steering owner records the scoped decision. |
| Evidence needed | Workflow observation/interview, representative policy request set, target-environment latency distribution, and review record. |
| Needed-by gate | Before `EVAL-CAP-01` thresholds are frozen and before any canary approval. |
| Exposure blocked | Customer cohort exposure and any latency/SLA commitment. |
| Escalation path | Steering review records `HOLD` or narrows scope if owners cannot agree before the gate. |

The author does not choose 8 seconds merely because it is numeric. The conflict remains open until the named authorities review evidence.

## Claim register excerpt

| Claim ID | Statement | Class | Evidence reference | Limitations | Owner | Revalidate on |
|---|---|---|---|---|---|---|
| `CLM-EX-001` | The sponsor described the desired experience as "instant." | `[Unknown]` (source fixture class: `customer_claim`) | `CON-RIV-001` in the frozen brief | Not an accepted metric or measurement | Discovery owner | Workflow-owner review |
| `CLM-EX-002` | An 8-second policy p95 is the current planning threshold. | `[Modeled]` | `CON-RIV-001`; assumption record pending | No target-environment run; no customer acceptance | Service-envelope owner | Representative latency evidence or changed workload |
| `CLM-EX-003` | The target environment meets the accepted policy-answer latency criterion. | `[External validation required]` | `VAL-EX-001` | No execution or accepted criterion exists | Platform validation owner | Before canary decision |

`CLM-EX-003` is a validation request, not a positive claim. No row is labeled `[Measured]` or `[Customer-validated]` because this teaching excerpt has neither retained execution output nor authorized approval.
