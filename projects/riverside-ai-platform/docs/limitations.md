# Limitations

## Current implementation limitations

As of 2026-08-05, Riverside contains implemented source assets for schemas and
fixtures; artifact validation and Azure ML scoring; endpoint clients and RAG
orchestration; release gates and versioned evaluation datasets; bounded telemetry;
APIM policies; staged load testing and result normalization; modular Bicep and
`azure.yaml`; an internal Azure Container Apps/ACR host; Azure ML blue/green
definitions; and focused unit/contract/integration/cloud test source.
The separate RAG project contains Databricks remote ingestion and Direct Vector
Access indexing source, bundles, and tests.

The Riverside non-cloud suite passed 142 tests with 5 cloud tests deselected, and
the offline preflight passed 9 tests. No cloud test, Azure/Databricks command,
deployment, or live validation was performed. Source presence and local passing
tests are not evidence of production readiness or cloud behavior.

The Riverside project now has its root `README.md`, `pyproject.toml`, Azure-only
YAML profiles, test orchestration, deployable RAG-orchestrator source, `azd` service
wiring, guarded cloud tests, and release-gate CLI. It still lacks an APIM deployment
pipeline, immutable image build/provenance gate, release-gate deployment job, and
load-result download pipeline.

## Architecture limitations

- Databricks Direct Vector Access is the implemented initial backend. Its live
  filter, identity, deletion, latency, quota, region, and cost behavior is unproven.
- No production region, subscription layout, resource naming, SKU, quota,
  networking topology, or retention policy is selected.
- The first release is single-region; multi-region active-active and regional
  failover are explicitly deferred.
- Custom Triton/vLLM serving is deferred. Azure ML managed online endpoint source
  is implemented, but no endpoint/deployment result exists.
- Microsoft Foundry Models are an optional future backend, not a validated path.
- No SLO, error budget, RTO, RPO, capacity target, or budget is committed.

## Artifact limitations

Committed fine-tuning experiment manifests provide training provenance but are not
release manifests. They lack immutable serving URIs/digests, complete runtime
compatibility, release decision, and bound evaluation report. The serving project
must consume artifacts; it must not infer promotion from their presence or rewrite
their source manifests.

## Data-plane limitations

The separate RAG knowledge pipeline has two distinct states:

- remote ingestion and vectorization source now emits v1 governed records, quality
  reports, tombstone/stale deletion handling, and Direct Vector Access payloads;
- those remote jobs and bundles were not executed in a Databricks workspace;
- the preserved local path still stores insufficient production metadata in Chroma,
  binds generation to Watsonx, exposes `/query` rather than the v1 application API,
  logs query prefixes, and enables permissive CORS;
- the serving project must use contracts/adapters, not import the remote pipeline.

Riverside documentation does not upgrade that prototype into production evidence.

## Evaluation limitations

- Versioned datasets, evaluators, gate policy source, comparison helpers, report
  generation, and focused tests are present; the non-cloud suite passed locally.
- No approved production threshold policy, real candidate result, calibrated human
  review process, or cloud operational/cost evidence is present.
- JSON Schema cannot enforce all cross-field invariants listed in the contracts.
- Automated groundedness/judge scores have evaluator bias and calibration limits.
- Offline quality cannot prove authorization, latency, overload recovery, cost, or
  Azure service behavior.
- A canary cannot establish rare-slice safety without sufficient representative
  samples and observation time.

## Security and privacy limitations

- Contract fields, filter/redaction source, APIM policy, RBAC/network Bicep, and
  tests express and implement controls in source; no deployed control is proven.
- Data residency is uncommitted and unvalidated.
- No threat-model approval, penetration test, access review, secret-history scan,
  SBOM/provenance pipeline, vulnerability gate, or policy compliance report exists.
- Incident severity, notification obligations, evidence retention, and legal hold
  require organizational approval.
- No content capture for telemetry/debugging is approved.

## Operational limitations

- Blue/green/canary YAML, staged overload/recovery load assets, rollback commands,
  and bounded Container Apps HTTP autoscale are implemented source; shadow
  mirroring, graceful drain, autoscale validation, APIM restore, and atomic
  index-version rollback remain integration gaps.
- The Bicep-provisioned endpoint name (`riverside-<environment>`) does not match the
  Azure ML YAML endpoint name (`riverside-smollm2-chat`).
- The production parameter example selects `uksouth`, while Azure ML blue/green
  deployment metadata hardcodes `eastus2`.
- Profiles now enforce model dependency < application < APIM deadlines. Azure ML's
  180-second container timeout remains a separate outer kill boundary and is not
  yet reconciled by an automated Azure ML publication pipeline.
- The Container Apps environment is internal and requires a delegated subnet, but
  APIM-to-host private DNS/routing, ACR pull, diagnostics, zone redundancy, probes,
  and autoscale behavior have not been deployed or validated.
- Azure ML YAML references registered model/environment versions and a model-package
  layout without a committed registration/package workflow.
- APIM policy source has no deterministic import/publish/restore pipeline.
- Load tests have an Azure test definition and parser but no result-download pipeline
  or retained cloud output.
- Example SKUs, counts, thresholds, regions, retention, and network modes are modeled
  source inputs, not approved commitments.
- No command in these documents was executed during authoring.
- Azure CLI/Azure service behavior and regional availability can change; operators
  must verify current official documentation and installed tool versions.

## Learning boundary

The Azure-operational learning bridge executed all 13 code cells successfully
locally and was then cleared. Its p95 gates returned `HOLD_LOCAL_RELEASE` and
`HOLD_AZURE_PROMOTION`. It labels its loopback/synthetic mechanisms
`LOCAL`/`SUBSTITUTED` and Azure behavior `UNVALIDATED`.
Fine-tuning, RAG, evaluation, gateway, and AI-infrastructure tracks own the concepts;
none provides a local Riverside production profile or Azure control-plane proof.

## Deferred claims

Do not claim production readiness, high availability, disaster recovery, compliant
residency, least privilege, private networking, safe autoscaling, bounded cost,
quality/safety targets, or rollback capability until the capability ledger links
specific retained evidence for the exact environment and release.
