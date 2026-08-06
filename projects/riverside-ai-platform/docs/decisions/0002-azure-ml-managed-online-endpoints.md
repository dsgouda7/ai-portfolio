# ADR-0002: Azure ML Managed Online Endpoints

- **Status:** Accepted
- **Date:** 2026-08-05

## Context

Riverside's committed training outputs are custom SmolLM2 base-plus-adapter
artifacts. The serving boundary needs artifact verification, custom scoring,
readiness after warm-up, managed identity, deployment logs, and blue/green traffic.

## Decision

Use Azure Machine Learning managed online endpoints as the default model-serving
backend. Address the model through the stable `riverside-editor` alias and keep
physical blue/green deployment selection outside the client contract.

Microsoft Foundry Models may later implement the same application-facing contract
for a managed-model use case. It is not the default backend for the custom artifact.
Custom Triton or vLLM hosting is deferred.

## Consequences

- Release manifests must bind model profile, precision, tokenizer, adapter,
  runtime interface, and immutable digests before readiness succeeds.
- Azure ML regional SKU availability, quota, autoscale, identity, networking,
  startup, and cost must be validated in the chosen environment.
- The endpoint abstraction does not remove backend-specific operational work.

## Evidence state

The release schemas, Pydantic verifier/service, scoring source, endpoint/environment
definitions, blue/green deployments, sample requests, rollout profiles, and asset
contract tests are implemented source assets. They were not executed in this task;
no registered model/environment, managed endpoint, deployment log, invocation,
streaming result, identity/network test, or cloud smoke evidence is linked.
