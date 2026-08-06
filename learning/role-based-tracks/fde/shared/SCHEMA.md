# Riverside Shared-Case Schema

## Contract Boundary

The shared case is a versioned data contract, not notebook-owned prose. The
canonical engagement, source samples, and expected facts all use fixture version
`RIV-FDE-1.0.0`. Consumers join records by stable ID and preserve the supplied
evidence class.

The schemas use JSON Schema draft 2020-12. They validate structure and required
fields; they do not prove that a modeled assumption is correct, that a cloud
service is available, or that a legal interpretation is approved.

## Files And Roots

| Fixture | Root | Schema |
|---|---|---|
| `riverside-engagement-v1.json` | Customer narrative and planning contract | `riverside-engagement.schema.json` |
| `riverside-source-samples-v1.json` | Deterministic source-shaped records | `riverside-source-samples.schema.json` |
| `expected-facts-v1.json` | Cross-notebook fact ledger | `expected-facts.schema.json` |

The expected-facts ledger uses `riverside-engagement-v1.json` as its default
pointer root. A fact that targets a source sample declares
`source_fixture: riverside-source-samples-v1.json`.

## Stable ID Prefixes

| Prefix | Entity |
|---|---|
| `ORG` | Organization |
| `ENG` | Engagement |
| `BRF` | Ambiguous brief |
| `PER` | Synthetic person or persona |
| `ROLE` | Human or service role |
| `TEN` | Tenant or isolation boundary |
| `REG` | Planned processing region |
| `SYS` | Source or operational system |
| `SRC` | Source inventory entry |
| `REC` | Source-sample envelope |
| `DOC`, `ERP`, `API` | Source-native document or record |
| `UC`, `NG` | Use case or non-goal |
| `WF`, `WFSTEP` | Workflow and workflow step |
| `MET` | Measured baseline metric |
| `ASM` | Modeled planning assumption |
| `SLA` | Provisional service or safety target |
| `SEC` | Security or data constraint |
| `CON` | Intentional conflict |
| `RISK` | Seeded risk |
| `INC` | Seeded incident |
| `ROL`, `COH` | Rollout and cohort |
| `SUP` | Support model or ownership record |
| `UNK` | Discovery unknown |
| `FACT` | Expected fact |

An ID is immutable within a fixture version. Display names, owners, status, and
versions may change; the ID does not. Never join on list position. Array positions
appear only in expected-fact JSON Pointers and are protected by the frozen
fixture contract.

## Evidence Classes

| Value | Required interpretation |
|---|---|
| `measured_baseline` | Treat as a supplied synthetic observation with stated population and limitation. |
| `modeled_assumption` | Use in scenarios and sensitivity analysis; do not call it measured. |
| `customer_claim` | Add corroboration or an owner before turning it into an acceptance commitment. |
| `policy_constraint` | Enforce until the named owner approves a change. |
| `external_validation_required` | Name the cloud, legal, security, or operational validation owner. |
| `intentional_conflict` | Produce a decision artifact; do not choose one side silently. |
| `unknown` | Add to discovery backlog with owner and needed-by milestone. |

## Canonical Engagement Domains

| Domain | Key semantics |
|---|---|
| `personas_and_stakeholders` | Goals, concerns, decision rights, tenant context, and statements are separate. Influence does not imply data access. |
| `ambiguous_brief` | Preserves the original customer wording and lists what it does not define. |
| `current_workflow` | Steps describe the current human process. Timing is baseline evidence, not a future SLA. |
| `baseline_metrics` | Every value includes unit, population, window, source, evidence class, and limitations. |
| `source_inventory` | Describes ownership, format, update mode, ACL model, region, lifecycle risk, and deletion behavior. |
| `identity_and_data_constraints` | Separates tenant, role, region, purpose, title assignment, service identity, and tool authority. |
| `demand_cost_and_sla_assumptions` | Carries ranges and bases so the capacity notebook can avoid false precision. |
| `intentional_conflicts` | Stores both statements, resolution owner, and required artifact. |
| `seeded_risks` | Every risk has likelihood, impact, owner, mitigation, and trigger. |
| `seeded_incidents` | Every incident has containment, evidence preservation, affected IDs, and re-enablement owner. |
| `rollout` | Cohorts have entry, exit, abort, rollback, business owner, and technical owner. |
| `support_and_handoff` | Defines service hours, severity response, ownership, acceptance, and required artifacts. |
| `unknowns` | Keeps missing decisions visible through delivery and handoff. |

## Source Record Envelope

Every source sample has the contract fields needed to bridge into the Riverside
data plane:

- `tenant_id`, `document_id`, `source_uri`, and `source_version` identify the
  source and isolation scope.
- `content_hash`, `ingested_at`, and `pipeline_version` preserve lineage.
- `acl`, `region`, and `classification` support authorization and residency.
- `lifecycle` and `deletion_state` distinguish current, superseded, retained,
  deleted, and stale records.
- `payload_type` and `payload` hold a deterministic source-shaped example.
- `expected_onboarding` names the correct local disposition and the failure a
  naive pipeline should expose.

The committed `content_hash` values are stable synthetic fingerprints, not
digests to use for security decisions or to recompute against a real file.

## Expected Facts

Each expected fact declares:

| Field | Meaning |
|---|---|
| `fact_id` | Stable identity of the check |
| `subject_id` | Entity the fact describes |
| `source_fixture` | Optional override of the default pointer root |
| `json_pointer` | RFC 6901-style location in the source fixture |
| `comparison` | `equals` for direct comparison or `summary` for a documented aggregate |
| `expected_value` | Expected scalar, array, object, or null |
| `evidence_class` | How the result may be described |
| `notebook_ids` | Planned notebooks that depend on the fact |

For `summary` comparisons, the object states the aggregate to derive. Examples
include array count, first and last stable IDs, or the set of source types. A
summary fact never authorizes replacing the underlying records with only the
summary.

## Intentional Failure Semantics

The source fixture must retain the failures below until the relevant notebook
demonstrates its control:

1. Superseded policy remains available in the source but is excluded from the
   current-policy index.
2. Two-column extraction can scramble heading relationships.
3. A duplicate PDF page and low-confidence OCR force quarantine.
4. A deleted autosave requires a tombstone and must not survive in the index.
5. ERP null territory remains unknown; it never defaults to worldwide.
6. API pagination must continue past 100 records.
7. API schema drift is quarantined rather than converted to silent nulls.
8. A timeout after a committed write requires a business idempotency key and
   reconciliation.
9. A disabled contractor is denied even when a stale nested group still lists an
   editor role.

## Versioning And Extensions

- Documentation-only clarification: no fixture version change if values and
  semantics are unchanged.
- Additive optional field: increment the schema patch version and update schema
  documentation.
- Added or changed case fact: increment the fixture minor version and update the
  expected-facts ledger.
- Changed meaning, removed field, reused ID, or incompatible authority boundary:
  create a new major fixture and file set.

Do not add notebook outputs, discovered decisions, credentials, live endpoints,
or environment-specific deployment identifiers to these frozen fixtures.
