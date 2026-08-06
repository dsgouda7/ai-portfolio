# Decision and Action Inventory

| ID | Use case/workflow | Decision or action | Actor | Assistant role | Proposal allowed | Approval authority | Commit authority/system | Required context | Audit evidence | Retry/idempotency | Compensation/correction | Prohibited behavior | Status/source IDs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `<ACT-...>` | `<UC/WF ID>` | `<decision/action>` | `<role>` | `<retrieve/draft/propose/none>` | `<yes/no>` | `<human/role>` | `<service/system>` | `<tenant, role, purpose, title...>` | `<event fields>` | `<rule>` | `<path>` | `<hard boundary>` | `<status + IDs>` |

## Review questions

1. What is the smallest reversible assistant contribution?
2. Which exact payload does a human approve?
3. Can a timeout occur after a committed side effect?
4. How is committed state reconciled before retry?
5. What requires correction or compensation rather than deployment rollback?
6. Which actions remain prohibited even if a stakeholder asks for automation?
