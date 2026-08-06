# DSC-04 Outcomes and Acceptance Matrix

## Outcomes and non-goals

| ID | Type | Statement | Beneficiary/owner | Evidence class | Source IDs | Status |
|---|---|---|---|---|---|---|
| `<OUT/NG-...>` | `<outcome/non-goal>` | `<technology-neutral statement>` | `<role>` | `<class>` | `<IDs>` | `<draft/validated/blocked>` |

## Acceptance criteria

| Criterion ID | Outcome | Metric or decision rule | Baseline | Proposed target | Slice | Method/test set | Acceptance owner | Evidence class | Source IDs | Status/revalidation |
|---|---|---|---|---|---|---|---|---|---|---|
| `<AC-...>` | `<outcome ID>` | `<precise definition>` | `<value + class>` | `<threshold/range + class>` | `<tenant/role/task/risk/...>` | `<how evaluated>` | `<authorized owner>` | `<class>` | `<IDs>` | `<draft/validated/blocked>` |

## Golden workflow set

| Case ID | Workflow/use case | Slice | Expected safe behavior | Failure cost | Label owner | Review cadence |
|---|---|---|---|---|---|---|
| `<CASE-...>` | `<workflow>` | `<slice>` | `<answer/action/abstention/escalation>` | `<impact>` | `<owner>` | `<trigger>` |

## Quality floor and stop conditions

| Gate ID | Must-pass condition | Scope | Evidence required | Decision owner | Automatic response |
|---|---|---|---|---|---|
| `<GATE-...>` | `<zero-tolerance or threshold>` | `<slice>` | `<test/report>` | `<owner>` | `<block/pause/rollback>` |

## Health check

- [ ] Outcomes are technology-neutral and linked to a user or workflow need.
- [ ] Every criterion has a slice, method, owner, evidence class, and status.
- [ ] Policy constraints and zero-tolerance failures are not hidden in averages.
- [ ] Proposed targets are not labeled measured or customer-validated.
- [ ] Non-goals exclude consequential actions and unsupported scope explicitly.
