# LLM Gateway

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

Run `setup.ps1` on Windows or `setup.sh` on Linux/macOS; either script creates this chapter's `.venv`, installs `requirements.txt`, registers its Jupyter kernel, and assigns that kernel to both notebooks.
Both notebooks execute without credentials. The lab's LiteLLM bridge is disabled unless
`RUN_LIVE_LLM_DEMO=1`, `LITELLM_MODEL`, and provider credentials are set explicitly.

## Continue Into Operations

This chapter owns provider-neutral normalization, routing, rate limiting, fallback, caching, cost
control, and observability concepts through deterministic simulations. Continue with:

- [AI Engineer: Application Latency and Cost](../../role-based-tracks/ai-engineer/03-application-latency-and-cost/README.md)
	for stage attribution, TTFT/TPOT, retry amplification, cache savings, and cost denominators;
- [Azure Operational LLM Serving](../../ai-infrastructure/09-azure-operational-llm-serving/README.md)
	for local readiness, admission, deadlines, idempotency, release, and tail-latency failures;
- [Riverside APIM Gateway](../../../projects/riverside-ai-platform/apim/README.md) for the static
	Azure policy mapping and [Riverside documentation](../../../projects/riverside-ai-platform/docs/README.md)
	for the composed production profile.

The APIM and Riverside assets are implemented source, not deployment evidence. Authentication,
managed identity, policy behavior, circuit breaking, quota, networking, telemetry, and cost remain
**live-unvalidated** in Azure.
