# LLM Gateway

[LLM Gateways: Routing, Resilience, and Cost Control](06-llm-gateway.ipynb) · [Theory notes](06-llm-gateway-theory.md)

The notebook builds an application-facing request control plane over deterministic provider simulations: normalization, routing, rate limiting, fallback, caching, cost control, and observability.

The simulation keeps provider behavior reproducible so the notebook can isolate systems decisions. Production serving internals such as continuous batching, KV-cache management, and backpressure continue in the AI Infrastructure track.

Run `setup.ps1` on Windows or `setup.sh` on Linux/macOS; either script creates this chapter's `.venv`, installs `requirements.txt`, registers its Jupyter kernel, and assigns that kernel to the notebook.

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
