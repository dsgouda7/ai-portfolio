# Riverside AI Platform Documentation

These documents describe an Azure production profile for Riverside House. They
do not claim that the profile has been deployed or proven production ready.

## Evidence vocabulary

Every capability claim uses one of four classes:

- **Implemented source asset:** the named repository file exists and is inspectable.
- **Static validation:** a non-cloud check ran and its command, commit, and output
  were retained.
- **Modeled assumption:** a design input awaiting measurement or service review.
- **Live Azure validation required:** an authorized Azure environment must produce
  retained evidence before the claim is accepted.

As of 2026-08-05, implemented source assets include the Riverside schemas and
fixtures; artifact verification, endpoint client, RAG orchestration, release-gate,
and telemetry libraries; evaluation datasets; Azure ML endpoint/deployment/scoring
assets; APIM policies; staged load tests; Bicep/`azd` infrastructure; and unit or
contract tests for those surfaces. The Databricks data project also contains remote
ingestion and Direct Vector Access indexing source. This documentation task ran no
cloud commands. The Riverside non-cloud suite passed 142 tests with 5 cloud tests
deselected, and its offline preflight passed 9 tests. No Azure or Databricks live
validation was performed. Source presence and local test results must not be
reported as cloud validation or production readiness.

## Document map

- [Architecture](architecture.md)
- [Architecture decisions](decisions/README.md)
- [Deployment](deployment.md)
- [Evaluation strategy](evaluation-strategy.md)
- [Operations runbook](operations-runbook.md)
- [Incident response](incident-response.md)
- [Rollback](rollback.md)
- [Security and data boundaries](security-and-data-boundaries.md)
- [Data residency](data-residency.md)
- [Cost and capacity assumptions](cost-and-capacity-assumptions.md)
- [Limitations](limitations.md)
- [Promise versus evidence](promise-vs-evidence.md)

## Learning relationship

The learning tracks own the concepts and local failure-first experiments:

- [`../../../learning/genai/02-llm-finetuning/`](../../../learning/genai/02-llm-finetuning/)
  owns adaptation objectives and held-out evidence.
- [`../../../learning/genai/03-rag/`](../../../learning/genai/03-rag/) owns retrieval,
  citation, refusal, and authorization evaluation concepts.
- [`../../../learning/genai/04-llm-evaluation/`](../../../learning/genai/04-llm-evaluation/)
  owns evaluator design and uncertainty.
- [`../../../learning/genai/05-llm-gateway/`](../../../learning/genai/05-llm-gateway/)
  owns normalization, routing, resilience, caching, cost control, and observability
  concepts through deterministic simulations.
- [`../../../learning/ai-infrastructure/`](../../../learning/ai-infrastructure/)
  owns model memory, quantization, and inference systems. Its
  [`09-azure-operational-llm-serving/`](../../../learning/ai-infrastructure/09-azure-operational-llm-serving/)
  bridge owns the local, substituted operational experiment and labels Azure
  behavior unvalidated.

This production documentation may refer to those concepts. It does not provide a
local deployment profile, copy notebook code into production, or treat a local
measurement as Azure evidence.

## Command policy

Command blocks are exact expected operator commands with placeholders. They were
not executed while these documents were authored. A command is not evidence until
an authorized operator records the environment, commit, tool versions, output,
timestamp, and reviewer. Never place credentials in command history, arguments,
documentation, or retained output.
