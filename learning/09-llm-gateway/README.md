# Stage 09: LLM Gateway

**Previous:** [Stage 08 · LLM Evaluation](../08-llm-evaluation/README.md)

**Current:** Stage 09 · LLM Gateway

**Next:** [Stage 10 · Agentic AI](../10-agentic-ai/README.md)

## Input Contract from Stage 08

The gateway does not invent quality thresholds. It enforces this versioned evaluation matrix:

| Evaluation evidence | Gateway control | Runtime decision |
| --- | --- | --- |
| Capability/slice benchmark passes its release threshold | Route eligibility for that matching request class | Include the deployment in the approved set; a failing or unmeasured slice is ineligible |
| Safety gate passes | Hard policy eligibility | Dispatch only to releases that passed the required safety gate; never average a safety failure into quality |
| Atomic claim support and attribution pass | Output policy | Return attributed output; otherwise refuse, abstain, or send to bounded review |
| Calibrated confidence is above the risk-cost threshold | Confidence/risk route | Answer on the eligible route; below threshold, abstain, fall back only to another eligible route, or escalate |
| Judge disagreement or evaluator uncertainty exceeds its review threshold | Review gate | Do not convert an uncertain offline verdict into automatic runtime authority |
| Versioned regression or drift gate fails | Release/route state | Disable or roll back that evaluated release and retain the triggering evidence version in the trace |

Latency and cost optimize **inside** the approved set; they cannot buy their way around a failed
quality, safety, support, confidence, or authority gate.

[Gateway Control-Plane Theory](01-gateway-control-plane-theory.ipynb) ·
[Handwritten theory notes](01-gateway-control-plane-theory.md) ·
[Gateway Routing and Resilience Lab](02-gateway-routing-resilience-lab.ipynb)

The theory notebook is a 45-60 minute failure-first lesson: gateway versus inference-server
boundaries, one normalized contract, eligibility before routing, lifecycle ordering, and the
difference between retry and fallback. The 75-90 minute lab resumes the same Riverside House
request and measures deterministic adapters, round-robin/least-busy/latency/weighted/cost routing,
request-token-concurrency controls, token-bucket burst shaping, budget reserve/settle, exact and
unsafe semantic caches, recovery, telemetry, cost, and latency.

The simulations keep provider behavior reproducible so the notebooks can isolate systems decisions.
No transformer is loaded and no provider reliability, answer quality, cloud-policy behavior, or
savings are claimed. Circuit breakers remain a forward boundary until measured. Continuous
batching, KV-cache management, GPU scheduling, and serving backpressure continue in the AI
Infrastructure track.

## Simulation Labels

| Simulation | Mechanics built and measured | Capability earned | Evidence boundary |
| --- | --- | --- | --- |
| Control-plane theory | Response normalization, eligibility-before-routing, lifecycle-order consequences, deterministic retry/fallback trace | Explain where a request control belongs and why ordering changes policy meaning | Deterministic control mechanics, not provider or model behavior |
| Routing and resilience lab | Adapters, routing, multi-axis admission, reserve/settle, exact-cache keys, bounded recovery, telemetry | Build and diagnose a provider-neutral policy path without widening eligibility | Simulated latency/cost/failure evidence, not cloud reliability, answer quality, or savings |

## Handoff to Stage 10: From Calls to Commitments

Stage 10 reuses each gateway contract but raises the side-effect stakes:

| Stage 09 contract | Stage 10 agent-system interpretation |
| --- | --- |
| Normalized model calls | Normalized calls plus typed tool contracts with explicit inputs, outputs, and errors |
| Route eligibility | Tool/action eligibility plus authority: the agent may consider only actions it is permitted to commit |
| Retry, fallback, and bounded recovery | Bounded tool/agent recovery that never widens authority or silently changes the requested action |
| Request and attempt traces | End-to-end traces and audit records for model choices, tool calls, approvals, results, and failures |
| Budget reserve/settle | Reserve before financial or scarce-resource side effects; settle observed cost and release unused reservation |
| Request IDs and exact cache identity | Idempotency keys and commitment records: retries must not duplicate payments, bookings, messages, or other irreversible commitments |

A normalized successful call is not yet a safe action. Stage 10 adds authority, side-effect
semantics, approval, and no-duplicate-commitment guarantees.

Run `setup.ps1` on Windows or `setup.sh` on Linux/macOS; either script creates this chapter's `.venv`, installs `requirements.txt`, registers its Jupyter kernel, and assigns that kernel to both notebooks.
Both notebooks execute without credentials. The lab's LiteLLM bridge is disabled unless
`RUN_LIVE_LLM_DEMO=1`, `LITELLM_MODEL`, and provider credentials are set explicitly.

## Continue Into Operations

This chapter owns provider-neutral normalization, routing, rate limiting, fallback, caching, cost
control, and observability concepts through deterministic simulations. Continue with:

- [AI Engineer: Application Latency and Cost](../12-role-based-tracks/ai-engineer/03-application-latency-and-cost/README.md)
	for stage attribution, TTFT/TPOT, retry amplification, cache savings, and cost denominators;
- [Azure Operational LLM Serving](../11-ai-infrastructure/10-azure-operational-llm-serving/README.md)
	for local readiness, admission, deadlines, idempotency, release, and tail-latency failures;
- [Riverside APIM Gateway](../../projects/riverside-ai-platform/apim/README.md) for the static
	Azure policy mapping and [Riverside documentation](../../projects/riverside-ai-platform/docs/README.md)
	for the composed production profile.

The APIM and Riverside assets are implemented source, not deployment evidence. Authentication,
managed identity, policy behavior, circuit breaking, quota, networking, telemetry, and cost remain
**live-unvalidated** in Azure.
