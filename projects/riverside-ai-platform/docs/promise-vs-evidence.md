# Promise Versus Evidence

## Capability ledger

This ledger is the claim control for Riverside documentation. It prevents a design,
file, or command from being described as a validated capability.

**Status vocabulary:**

- **Implemented source asset:** inspectable repository content exists.
- **Static validation pending/completed:** a non-cloud check is defined or has
  retained output.
- **Modeled assumption:** architecture/operations reasoning without measurement.
- **Live Azure validation required/completed:** authorized cloud evidence is absent
  or retained for one exact environment/release.
- **Deferred:** intentionally outside the first release.

As of 2026-08-05, no static validation command in this project and no live Azure
validation has been run as part of this documentation task.

| Capability or promise | Source evidence now | Static state | Live Azure state | Allowed wording now |
|---|---|---|---|---|
| Versioned integration contracts exist | `contracts/v1/*.schema.json` and contract README | Pending; fixtures say unexecuted | Not applicable to schema existence | "v1 schemas are implemented source assets" |
| Fixtures prove contract behavior | Valid/invalid fixtures and test source | Pending; no result linked | Not applicable | "fixtures and tests document expected outcomes" |
| Config excludes local and API-key profiles | `config/schema.json` | Pending | Identity behavior untested | "schema permits dev/staging/production and managed/workload identity" |
| Training artifacts are deployable releases | Artifact verifier/service and Azure ML package contract source | Tests present, result pending; registration workflow absent | No registry/deployment | "release verification source exists; registration and deployment evidence are required" |
| Azure ML custom model serving works | Scoring source, endpoint/environment/deployment/rollout YAML | Asset/unit tests present, result pending | Required | "Azure ML serving assets are implemented source" |
| APIM protects one stable API | OpenAPI, policies, backend Bicep, parameter contract, tests | Tests present, result pending; publish pipeline absent | Required | "APIM gateway controls are implemented source" |
| Entra and managed identity enforce least privilege | APIM auth/identity policies and identity/RBAC Bicep | Review/tests pending | Required | "identity-based controls are implemented source" |
| RAG uses governed production data | Databricks remote ingestion/indexing and Riverside orchestrator source | Cloud-free tests present, result pending | Required | "v1 data-plane and consumption adapters are implemented source" |
| Tenant/ACL isolation prevents leakage | Fail-closed data/index/orchestrator/APIM source and negative tests | Test result pending | Required | "tenant/ACL enforcement has source and test coverage" |
| Deleted content stops being retrievable | Deletion-state contract, remote tombstone/index deletion source and tests | Test result pending | Required including backups/index | "deletion propagation is implemented in source" |
| Citations are grounded and content-free | Citation/response schemas | Fixture validation pending; no evaluator | Required on deployed path | "citation shape excludes source excerpts" |
| Release promotion is evidence-driven | Gate engine, eight-domain datasets, report source and tests | Test result pending; no composed release job | Required for smoke/load/rollout domains | "evidence-gate source is implemented" |
| Blue/green rollback works | Azure ML blue/green/traffic YAML, asset tests, procedure | Test result pending; cross-file profile mismatches remain | Required rehearsal | "blue/green assets are implemented source" |
| Telemetry is bounded and content-free | Allowlist, instrumentation, sanitizer, APIM metric and tests | Test result pending | Required under success/error/load | "bounded telemetry controls are implemented source" |
| SLOs are met | None | No thresholds/results | Required | "SLOs are unset" |
| Capacity handles peak demand | Staged load assets, criteria, parser, tests, and capacity model | Test result pending; criteria are illustrative | Required with quota/SKU | "load-test source is implemented; capacity is unproven" |
| Costs are within budget | Cost categories/formulas only | No pricing/billing result | Required | "no budget estimate is supported" |
| Data stays in an approved region | Region metadata, parameter examples, IaC | Cross-file region mismatch; no approval | Required service-by-service | "residency is undecided and unvalidated" |
| Private networking blocks public paths | Private-mode Bicep and security design | Build/review result pending | Required positive/negative tests | "private-connectivity source is implemented" |
| Incident response and rollback are operable | Runbooks | No tabletop/rehearsal | Required | "procedures are documented but unproven" |
| Multi-region active-active is available | None | None | None | "deferred" |
| Triton/vLLM production runtime is available | None | None | None | "deferred" |
| Foundry Models backend is available | Optional architecture note | None | None | "optional future backend" |
| Platform is production ready | None | None | None | "not supported" |

## Evidence package rules

A capability moves to completed only when the ledger links an immutable evidence
package containing:

- capability, acceptance criteria, environment, region, release/index/runtime, and
  source commit;
- exact command/test/policy/query and tool versions;
- input dataset/workload/evaluator versions and digests;
- timestamp, operator, reviewer, and approval;
- raw permitted output plus normalized machine-readable result/digest;
- negative tests, uncertainty/sample size, deviations, expiry/revalidation trigger;
- no secrets or customer content outside approved governed storage.

Evidence is scoped. A staging load test does not prove production capacity; one
region does not prove another; one release does not permanently prove future
releases; static Bicep build does not prove deployed networking.

## Claim review

Before a README, release note, customer statement, or architecture review uses a
capability claim:

1. Find the ledger row and evidence package.
2. Confirm environment/release/region/time scope matches the claim.
3. Confirm no unresolved failed gate or expired evidence.
4. Use the narrowest supported wording.
5. Link the retained evidence internally without exposing sensitive details.
6. Downgrade the status when implementation/config changes invalidate evidence.

## Wording examples

| Unsupported | Evidence-safe replacement |
|---|---|
| "Production-ready RAG platform" | "Azure production architecture with implemented v1 schemas; runtime and cloud validation remain" |
| "Secure multi-tenant retrieval" | "Tenant/ACL fields and negative-test requirements are defined; deployed isolation is unvalidated" |
| "Highly available" | "Single-region design; no availability SLO or regional failover evidence" |
| "Scales automatically" | "Autoscale policy and safe operating envelope are not yet measured" |
| "Data remains in-region" | "Residency requires a selected service/region matrix and live configuration review" |
| "Rollback supported" | "Blue/green rollback procedure is documented; staging rehearsal evidence is required" |
| "Cost optimized" | "Cost categories and normalization are defined; no SKU or billing evidence exists" |

## Ownership

Component owners attach source/static evidence. Release and operations owners attach
environment/release evidence. Security/privacy approve security and residency claims.
The integration owner reconciles paths and commands. Independent reviewers challenge
scope and wording. No owner may self-promote a modeled assumption to completed live
validation without retained evidence and review.
