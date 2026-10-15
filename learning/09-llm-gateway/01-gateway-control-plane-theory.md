# Gateway Control-Plane Theory: Handwritten Notes

## 1. The control-plane boundary

Riverside House has three ways to answer an editing request: a fast hosted deployment, a premium
hosted deployment, and a local private deployment. The application can call each one, but it
should not know every provider SDK, price, quota, region, or failure shape.

The gateway is the **request control plane** around those deployments. It authenticates and
normalizes a request, applies policy, chooses among eligible routes, performs bounded recovery,
normalizes the result, and records what happened.

It is not an inference server. Continuous batching, token generation, GPU scheduling, KV-cache
management, and serving backpressure live inside a deployment. The gateway can reject overload
before dispatch or choose another eligible route. It cannot repair a saturated GPU scheduler.

Memorable picture:

**application -> gateway controls -> eligible route -> deployment -> normalized result**

## 2. Normalize once, but preserve evidence

The first failed attempt lets application code call providers directly. One provider nests text
under `choices`, another returns content blocks, and the local server uses `generated_text`.
Token counts disagree too. A provider swap now changes every caller.

The smallest fix is one adapter per provider and one application-facing contract. A normalized
response carries:

- text and finish status;
- provider, deployment, and pinned model version;
- input and output tokens;
- latency and cost;
- request ID and attempt number;
- normalized error code and whether it is retryable.

Normalization should remove syntax differences, not operational evidence. If the gateway returns
only "success," operators cannot see that a fallback route answered every request.

The local alias `local-llama-8b` is routing metadata preserved from the earlier lesson. In the lab
it points to simulated `HuggingFaceTB/SmolLM2-135M-Instruct` metadata. No transformer is loaded,
and the alias does not prove that an 8B model ran.

## 3. Eligibility before routing

The second failed attempt asks a cost router to inspect every deployment. It chooses the globally
cheapest route. That route is outside the required region and may not receive unpublished
manuscripts.

The complaint is precise: **the optimizer was allowed to compare an illegal choice.**

First remove routes that fail hard rules:

- required capability and context length;
- tenant and data-sensitivity policy;
- region and residency;
- health and deployment state.

Then optimize inside the approved set. Round-robin, least-busy, latency-aware, weighted, and
cost-aware routing are soft objectives. None may trade away a hard rule. "Cheapest capable and
approved" is valid. "Cheapest, then check whether it was allowed" is not.

Least-busy routing uses in-flight work or queue depth rather than equal turns. It helps when service
times vary, but only if queue state is timely and shared. Stale per-process counters can make every
gateway replica chase the same apparently idle route. Capacity weights and hysteresis can reduce
that oscillation.

## 4. Lifecycle order makes the controls meaningful

The third failed attempt has all the right pieces in the wrong order. It spends a rate-limit slot
and reserves money before discovering an exact cache hit.

A durable request trace is:

1. Authenticate, assign a request ID, and normalize.
2. Check a complete versioned exact-cache key.
3. On a miss, estimate input and maximum output tokens.
4. Check request, token, and concurrency limits.
5. Atomically reserve estimated cost.
6. Filter eligibility, then apply the routing objective.
7. Dispatch and perform bounded recovery.
8. Normalize output and actual usage.
9. Settle the reservation against actual cost.
10. Run output policy, cache only approved output, emit telemetry, and return.

The exact key includes tenant, normalized input, pinned model, generation parameters, corpus
version, policy version, and relevant tool state. A corpus update therefore cannot reuse an old
chapter count. Hits still pass current output policy, and sensitive prompts are not cached by
default.

Request limits are not spending limits. Routes have different prices, outputs have different
lengths, and failed attempts may still cost money. Reserve before uncertain work. Settle after
observed work. The lab measures both a refund when actual cost is lower than estimated and an
overage when it is higher.

## 5. Retry is not fallback

The fourth failed attempt calls every second attempt a retry. That hides a policy difference.

A **retry** calls the same deployment again because a timeout, connection failure, selected
throttle, or selected server error may be transient. A **fallback** changes to another eligible
deployment after the route's retry budget is exhausted.

Do not retry invalid input, policy denial, or deterministic context overflow. Use a small attempt
limit, per-attempt timeouts, jittered backoff, and one overall deadline. Fallback never broadens
policy. If no approved route succeeds, return a clear normalized failure.

The caller sees one result. Operators need the attempt story: route, attempt kind, normalized
error, tokens, latency, cost, fallback position, settled spend, and remaining deadline. Redact
prompts, secrets, and personal identifiers. Track tail latency, cache hits, rejection reasons,
retry amplification, fallback share, cost per acceptable result, and quality by route.

Circuit breakers are a forward boundary here. They keep traffic away from a route already known
to be unhealthy, but they should not be claimed until open, half-open, and recovery behavior is
measured. Serving backpressure remains a separate inference-layer concern.

## 6. Practical failure modes

| Symptom | Likely control-plane mistake |
|---|---|
| Cheap route violates residency | Optimized before eligibility |
| Cache returns 21 after corpus has 23 chapters | Key omitted corpus version |
| Negated question reuses a cached answer | Similarity was treated as equivalence |
| Spend exceeds a request quota | No atomic budget reserve/settle |
| Tail latency grows during incidents | Retry budget or deadline is too loose |
| Every replica chooses the same route | Least-busy state is stale or local |
| Success rises while complaints rise | Telemetry lost route-level quality |

**Durable summary:** normalize once, filter before optimizing, order controls around real work,
reserve before spending, settle actual usage, and keep retry distinct from policy-safe fallback.
