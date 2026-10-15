# Systems Composition — Handwritten Theory Notes

## 1. The complaint: three replicas, same p95

You already know the ingredients. The gateway normalizes requests and applies policy. The inference
server schedules token work and manages the KV cache. The platform can add replicas. The release
system can replace one model artifact with another.

The remaining problem is composition.

Imagine Riverside House adds two model-server replicas after a burst causes long waits. The
deployment dashboard now says three healthy replicas, but p95 latency is unchanged. Nothing is
wrong with the replicas. A gateway concurrency gate still allows only one request into the serving
pool at a time. Three checkout lanes were opened behind one locked turnstile.

That is the durable lesson:

> Capacity only helps when every earlier boundary can feed it safely.

The deterministic lab makes the failure visible. One replica with effective concurrency one and
three replicas behind a gateway limit of one produce the same latency. The extra capacity exists,
but the request path cannot use it.

## 2. A service is a chain of owners

Treat the request path as a chain rather than one large “serving layer”:

```text
gateway
-> admission queue
-> model scheduler
-> replica
-> token generation
-> response and trace
```

The gateway owns application-facing contracts: authentication context, tenant eligibility, budgets,
route selection, retry and fallback policy, and stable error semantics. It may reject work before
the model server sees it.

The model server owns model-execution mechanics: tokenization compatibility, request scheduling,
continuous batching, KV-cache allocation, decoding, and model-level telemetry. The gateway should
not attempt to reproduce those mechanisms.

The platform or autoscaler owns process and replica capacity. It observes signals such as queue
delay, active sequences, token throughput, memory headroom, and replica readiness. CPU utilization
alone is rarely a sufficient scaling signal for an accelerator-backed token service.

Release control owns immutable model identity, rollout cohorts, health gates, rollback targets, and
the evidence that authorizes a traffic shift.

If two layers both own a mechanism, retries multiply, queues hide inside queues, and a timeout in one
layer becomes unexplained load in another. If no layer owns it, the system waits politely until the
incident review.

## 3. Why averages lie

Average latency can look acceptable while burst traffic traps a small group of requests behind a
long queue. Those users experience the tail, not the average.

For a token service, inspect at least:

- queue wait before execution;
- time to first token;
- time per output token;
- end-to-end p50 and p95 latency;
- accepted and rejected request counts;
- token throughput;
- active sequences and KV-memory headroom.

These numbers answer different questions. A long queue wait points toward admission, capacity, or
scheduling. A normal queue wait with poor time per output token points toward model execution. Good
server metrics with poor end-to-end latency point back toward the gateway, network, or client.

The lab intentionally reports accepted latency and rejection separately. Dropping slow requests
from a latency calculation without counting them as rejected would manufacture a prettier SLO.

## 4. Admission is not failure

When arrivals exceed bounded capacity, every system must choose where overload becomes visible.

Without admission control, all requests enter the queue. Throughput may remain fixed while queue
wait grows until callers time out. Retries then add more work, creating a retry storm.

With a queue-wait budget, the service rejects work that cannot begin within the promised envelope.
That does not increase raw capacity. It protects accepted requests, gives the gateway an explicit
retryable-overload signal, and prevents hidden queues from consuming the whole latency budget.

This creates a trade-off:

```text
accept everything -> impressive acceptance count, uncontrolled tail
reject beyond capacity -> bounded accepted experience, explicit denied work
```

Neither choice is automatically correct. The workload, idempotency contract, fallback path, and
business consequence determine the policy. The important part is that the trade-off is measured and
owned.

## 5. Batching belongs near execution

The model server sees active sequences, token positions, KV-cache pressure, and decode readiness. It
therefore owns continuous batching.

The gateway may batch independent application operations only when the API contract permits it, but
it should not imitate token scheduling. A request-level batch and a decode-step batch solve different
problems.

Adding replicas changes parallel capacity. Batching changes how efficiently each replica uses its
capacity. Quantization changes the artifact's memory and execution profile. Admission bounds demand.
Autoscaling changes capacity after a delayed observation. These mechanisms interact, but they are
not interchangeable.

## 6. Scale on the complaint

Choose a scaling signal that corresponds to the failure:

| Visible complaint | Useful signal |
|---|---|
| Requests wait before execution | Queue delay and queued-request count |
| Decode capacity is saturated | Active sequences and token throughput |
| New sequences cannot enter | KV-memory headroom or cache allocation failures |
| Replicas exist but receive no work | Gateway concurrency and routing distribution |
| A rollout degrades one cohort | Release-scoped latency, errors, and quality gates |

Scaling from average CPU while ignoring queue delay is like measuring the temperature outside to
decide how many checkout lanes to open.

## 7. Rollback needs more than weights

A deployable model identity includes the checkpoint, tokenizer, model configuration, decoding
defaults, serving runtime, quantization format, and compatibility contract. Rollback must restore a
compatible set, not merely an older weights file.

Operational state also matters. Idempotency records, in-flight requests, budgets, trace continuity,
and release-scoped metrics must survive or reconcile across a rollout. A deployment rollback cannot
undo a side effect that a tool or external system already committed.

## 8. Evidence boundary

The notebook is `LOCAL_FIXTURE` evidence. It proves that the queue model behaves as specified and
that an upstream concurrency gate can neutralize extra replicas. It does not prove:

- real model-server throughput;
- a particular batching gain;
- cloud autoscaling behavior;
- Azure capacity or quota;
- production reliability;
- a customer-acceptable SLO.

Those require workload-specific local measurements, authorized live-environment evidence, and
independent review.

The durable summary is:

> Do not scale a box. Trace the queue, name each owner, and scale the boundary that is actually
> withholding capacity.
