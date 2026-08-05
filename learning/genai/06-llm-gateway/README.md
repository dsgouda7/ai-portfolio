# LLM Gateway

[LLM Gateways: Routing, Resilience, and Cost Control](06-llm-gateway.ipynb) builds an application-facing request control plane over deterministic provider simulations: normalization, routing, rate limiting, fallback, caching, cost control, and observability.

The simulation keeps provider behavior reproducible so the notebook can isolate systems decisions. Production serving internals such as continuous batching, KV-cache management, and backpressure continue in the AI Infrastructure track.

Run the shared environment setup from [`../_llm-shared/`](../_llm-shared/README.md).
