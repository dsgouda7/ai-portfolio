# Teaching Example: Traceability and Recommendation

## Acceptance thread

| Field | Example value |
|---|---|
| Acceptance ID | `AC-EX-001` |
| Workflow and user | Authorized editor requests a current policy answer with inspectable citations. |
| Proposed decision rule | Threshold remains `TBD`; it must be sliced by request type and agreed before scoring. Critical authorization or citation failures are non-compensating. |
| Architecture | `ADR-EX-001`: retain manual/current-source search; evaluate bounded cited RAG only after data and identity gates. |
| Data/control dependencies | Approved current policy source, applicability metadata, tenant/role/region/purpose filters, deleted/stale exclusion. |
| Evaluation | Versioned representative policy questions; retrieval, citation, answer-support, authorization, abstention, and latency reported separately. |
| Service assumption | `CLM-EX-002` remains modeled; `VAL-EX-001` must replace it with retained target-environment evidence. |
| Rollout/rollback | No cohort until criterion and owners are accepted; disable generation and return to authorized source search on gate failure. |
| Signal/incident path | Slice-level latency and unsupported-answer signals route to service and evaluation owners; authorization failure stops exposure and enters the incident path. |
| Handoff owner | `TBD`; lack of an accepted quality/service owner blocks handoff. |

## Architecture excerpt

`ADR-EX-001` selects the smallest next proof, not a production design:

1. preserve the manual and authorized source-search path;
2. prepare a bounded cited-answer evaluation over approved current policies;
3. keep workflow writes, manuscript continuation, fine-tuning, and agents outside this example scope;
4. require identity/data gates and accepted evaluation criteria before any cohort;
5. revisit only when a named criterion fails under retained evidence.

## Recommendation

**Recommendation: `HOLD` before customer cohort exposure.**

The package may proceed to a supervised non-production validation plan after the workflow owner and service owner define the policy-answer criterion, the data and identity owners approve the bounded source/control plan, and `VAL-EX-001` names the target environment, dataset, method, reviewer, and retained evidence location.

This recommendation does not say that Riverside is secure, compliant, resident, within SLA, production-ready, or unable to meet the goal. It says the current authored evidence cannot support exposure or a service commitment. The next useful work is to close a named decision and measurement gap, not to add architecture.
