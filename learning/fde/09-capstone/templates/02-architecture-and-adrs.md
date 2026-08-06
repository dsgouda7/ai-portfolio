# Architecture Option Analysis, Boundaries, and ADRs

## Document control

| Field | Value |
|---|---|
| Artifact IDs | `ARC-CAP-01`, `ARC-CAP-02`, `ADR-CAP-*` |
| Version / status | `[TODO] / DRAFT` |
| Owner / architecture authority / reviewers | `[TODO]` |
| Discovery inputs | `[TODO: AC, NG, CON, UNK, CLM IDs]` |
| Scope / exclusions / revalidation trigger | `[TODO]` |

## Option matrix

Complete every row before selecting a design.

| Option | Criterion it can satisfy | Named failure/limit | Authority and recovery risk | Evidence available | Evidence gap/revisit trigger | Disposition |
|---|---|---|---|---|---|---|
| No AI / process repair | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Deterministic software | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Search | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| RAG | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Prompt-only generation | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Fine-tuning | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Deterministic workflow | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Single agent | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Multi-agent system | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |

If no frozen use case requires an unenumerable runtime branch, an agent has not
earned its control-loop risk. Fine-tuning cannot own current facts or authorization.

## Selected phased design

| Phase | In-scope capability | Components/boundaries | Entry evidence | Explicitly disabled | Exit/revisit trigger |
|---|---|---|---|---|---|
| 0 | `[manual/process repair]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| 1 | `[read/search/cited answer]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| 2 | `[bounded draft]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| 3 | `[optional confirmed workflow proposal]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |

## Boundary register

| Boundary | Trusted input | Decision/control | Prohibited input/action | Output/evidence | Owner | Failure/rollback path |
|---|---|---|---|---|---|---|
| Client to gateway | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Gateway to orchestrator | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Orchestrator to retrieval | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Orchestrator to model | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Orchestrator to tool | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Components to telemetry | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |

Attach context, container, sequence, data-flow, and failure/rollback diagrams. Mark
model, retrieval, identity, policy, human, state, side-effect, data, and external
validation boundaries.

## ADR template

### `ADR-CAP-[NNN]: [Decision]`

- **Status/date/owner/approver:** `[TODO]`
- **Decision question:** `[TODO]`
- **Acceptance criteria and constraints:** `[TODO: stable IDs]`
- **Options considered:** `[TODO]`
- **Evidence and claim IDs:** `[TODO]`
- **Decision:** `[TODO]`
- **Consequences and residual risks:** `[TODO]`
- **Rejected alternatives:** `[TODO: named failure, not preference]`
- **External validation required:** `[TODO]`
- **Rollback/fallback:** `[TODO]`
- **Revisit trigger:** `[TODO]`

An accepted ADR records a design decision. It does not prove implementation or
cloud behavior.

## Architecture gate

- [ ] Every selected component maps to an acceptance criterion or control.
- [ ] Every rejected option has a Riverside-specific reason and revisit trigger.
- [ ] Manual and generation-disabled paths remain usable.
- [ ] Authorization and consequential actions are deterministic and fail closed.
- [ ] PageTurn writes stay disabled while `UNK-RIV-005` is open.
- [ ] Cloud/service assumptions have external validation owners.

Decision and conditions: `[TODO]`
