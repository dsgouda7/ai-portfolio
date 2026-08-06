# ADR-0001: Versioned Contracts Are Integration Boundaries

- **Status:** Accepted
- **Date:** 2026-08-05

## Context

Fine-tuning, governed ingestion/indexing, and Azure serving have different owners
and release cadences. Importing another project's implementation would couple
runtime behavior to notebook or pipeline internals and make lineage migrations
implicit.

## Decision

Exchange model releases, documents, chunks, vector records, API messages,
evaluation reports, deployment metadata, and telemetry attributes through the
versioned schemas in [`../../contracts/v1/`](../../contracts/v1/). The serving
project does not import ingestion implementation, and it does not rewrite source
training manifests.

Closed root objects make unreviewed additions fail. Renames, removals, semantic
changes, new required fields, or relaxed security boundaries require a new version
directory and an explicit migration.

## Consequences

- Producers and consumers can validate independently.
- An artifact or record is rejected when its contract version or runtime invariants
  are unsupported.
- JSON Schema cannot express every cross-field invariant, so application validators
  still have work to do.
- Contract existence is not proof that any producer or consumer implements it.

## Evidence state

The v1 schemas and fixtures are implemented source assets. The fixture matrix says
they have not been executed or schema-validated. Runtime conformance and live data
flow remain unproven.
