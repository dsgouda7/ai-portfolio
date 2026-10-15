# Systems Composition: From Model Server to Reliable Service

**Contents:** [one-line map of this directory](CONTENTS.md)

**Previous:** [Inference Systems](../07-inference-systems/inference-systems.ipynb) and [LLM Gateway](../../09-llm-gateway/README.md)  
**Current:** Required systems-composition bridge  
**Next:** [Azure Operational LLM Serving](../10-azure-operational-llm-serving/README.md)

Adding replicas sounds like a serving fix. It is not a fix when an upstream concurrency gate still
serializes requests, when queues grow without admission control, or when no layer owns the signal
that should trigger scaling.

This short chapter composes the mechanisms that earlier chapters taught separately:

```text
model artifact
-> model server
-> replica and process topology
-> scheduling and batching
-> KV-memory capacity
-> admission and backpressure
-> autoscaling
-> gateway interaction
-> observability and SLOs
-> deployment and rollback
```

## Chapter Contract

| Contract | Requirement |
|---|---|
| Failure being repaired | Replica count changes, but tail latency and queue growth do not improve |
| Prerequisites | Stage 09 gateway contracts and Stage 11 inference-system mechanics |
| Predict | Decide whether three replicas help while gateway concurrency remains one |
| Implement | Run the deterministic queue fixture under four topology and admission policies |
| Inspect | Compare accepted work, p50/p95 latency, queue wait, throughput, and utilization |
| Prove | Show that replica count is ineffective behind a serializing gate, then bound accepted queue wait |
| Evidence class | `LOCAL_FIXTURE`: mechanics demonstrated; production capability and live behavior remain unvalidated |

## Required Artifacts

1. [Systems-composition lab](systems-composition-lab.ipynb)
2. [Systems-composition theory notes](systems-composition-theory.md)

The lab uses a deterministic synthetic burst and Python's standard library. It does not benchmark a
real model server, reproduce continuous batching, predict a cloud autoscaler, or establish a
production SLO.

## Completion Evidence

Retain:

- your prediction for the constrained three-replica case;
- the four-scenario metric table;
- the ownership decision for gateway, model server, platform, and release control;
- the transfer run with one changed burst size;
- one sentence explaining which conclusion the fixture supports and which production claim it does
  not support.

Only then continue to the Azure-shaped operational chapter, where the same ownership boundaries are
exercised through readiness, deadlines, idempotency, telemetry, and release gates.
