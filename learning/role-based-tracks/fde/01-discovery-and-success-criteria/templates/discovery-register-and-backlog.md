# DSC-05 Discovery Register and Backlog

## Assumptions

| ID | Statement/input | Range | Basis/source | Evidence class | Sensitivity | Validation owner | Replace with evidence by | Status |
|---|---|---|---|---|---|---|---|---|
| `<ASM-...>` | `<planning input>` | `<low/base/high>` | `<source/date>` | `modeled_assumption` | `<decision affected>` | `<owner>` | `<gate>` | `<open/validated/superseded>` |

## Conflicts

| ID | Topic | Statements/source IDs | Decision owner | Required artifact | Needed by | Status/decision ref |
|---|---|---|---|---|---|---|
| `<CON-...>` | `<topic>` | `<bounded conflicting statements>` | `<authority>` | `<artifact>` | `<gate>` | `<open/decided>` |

## Risks

| ID | Statement | Likelihood | Impact | Owner | Mitigation | Trigger | Linked criteria/unknowns | Status |
|---|---|---|---|---|---|---|---|---|
| `<RISK-...>` | `<failure condition>` | `<rating>` | `<rating>` | `<owner>` | `<control/evidence work>` | `<observable trigger>` | `<IDs>` | `<open/accepted/mitigated>` |

## Unknowns and evidence backlog

| ID | Question | Type | Priority | Owner | Evidence action | Needed by | Blocking artifact/decision | Evidence class | Status |
|---|---|---|---:|---|---|---|---|---|---|
| `<UNK-...>` | `<one answerable question>` | `<workflow/data/security/commercial/...>` | `<1..n>` | `<owner>` | `<interview/sample/test/external validation>` | `<gate>` | `<artifact>` | `<unknown/external_validation_required>` | `<open/in progress/closed>` |

## Prioritization rule

Prioritize work that can change scope or stop unsafe exposure: authority, forbidden access, consequential actions, representative workflow slices, deletion, support coverage, and architecture-changing integration constraints. Ease of answering is not priority.
