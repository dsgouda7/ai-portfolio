# ADR-0005: Single-Region First Release

- **Status:** Accepted
- **Date:** 2026-08-05

## Context

Multi-region active-active adds data replication, identity, index consistency,
traffic management, model-capacity, cost, and residency decisions. No concrete SLA,
approved geography, or measured regional failure requirement is committed.

## Decision

Model the first production profile in one approved Azure region. Co-locate the data
plane, vector index, model serving, gateway dependencies, telemetry destinations,
and evaluation artifacts where service availability and policy permit. Do not claim
regional failover or disaster-recovery objectives.

The final region is a deployment input selected only after residency, service
availability, GPU/SKU quota, latency, compliance, and cost review.

## Consequences

- A regional outage may make the service unavailable.
- Backup and restore can protect durable artifacts but do not create active serving
  continuity.
- Recovery-time and recovery-point objectives remain uncommitted until a business
  impact analysis and rehearsal support them.

## Evidence state

This is a modeled architecture decision. No region, subscription, quota, backup,
restore, or outage-recovery evidence exists yet.
