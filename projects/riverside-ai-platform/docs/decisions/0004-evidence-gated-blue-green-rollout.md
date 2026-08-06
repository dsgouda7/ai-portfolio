# ADR-0004: Evidence-Gated Blue/Green Rollout

- **Status:** Accepted
- **Date:** 2026-08-05

## Context

Artifact presence does not show that a release is compatible, grounded, authorized,
fast enough, affordable, or recoverable. In-place replacement also removes the
fastest rollback target.

## Decision

Deploy a candidate to the inactive blue or green slot and keep the prior known-good
deployment available. Promotion proceeds through offline evaluation, cloud smoke,
bounded load, shadow, canary, and broad rollout. Every stage has an owner,
observation window, thresholds, abort criteria, rollback target, and retained
evidence.

The release report must contain all eight contract domains and agree with the
release manifest. `hold`, `reject`, a failed metric, missing evidence, or a digest
mismatch blocks promotion.

## Consequences

- Two slots temporarily increase serving cost and quota needs.
- Database/index changes must remain backward compatible with both active releases.
- Traffic changes and deployment retention need explicit cleanup policy.
- A successful local or offline test cannot authorize production traffic.

## Evidence state

The schemas, release-gate source, eight-domain evaluation datasets, Azure ML
blue/green definitions and rollout files, staged load assets, result normalizer,
and relevant tests are implemented source assets. They were not executed in this
task. No release report produced from a real candidate, Azure deployment, load
result, traffic change, observation window, or rollback rehearsal is linked.
