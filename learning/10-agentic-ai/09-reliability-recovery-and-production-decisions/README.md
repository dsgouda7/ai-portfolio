# Chapter 09 — Reliability, Recovery, and Production Decisions

**Failure being repaired:** Retrying an ambiguous financial action can duplicate a commitment, while partial failure leaves no trustworthy resume or compensation path.

**Prerequisites:** [Chapter 08](../08-multi-agent-communication-and-coordination/README.md), durable checkpoints/events, idempotent tool surfaces, and trajectory evaluation.

## Local Learning Contract

- **Predict:** Decide when to retry, inspect durable state, dead-letter, resume, or compensate after each injected failure.
- **Implement:** Add idempotency keys, bounded backoff, a circuit breaker, durable events, replay, dead-letter states, saga compensation, and a chaos decision gate.
- **Inspect:** Read commitment identity, attempt history, breaker state, event sequence, replay cursor, dead-letter reason, compensation record, and decision scorecard.
- **Prove:** Produce zero duplicate commitments and show every injected failure either resumes safely or compensates explicitly.
- **Required evidence:** The notebook's idempotency, breaker, replay, dead-letter, saga, chaos-suite, and production-decision assertions.
- **Bounded transfer:** Replay one already committed fixture a second time with the same idempotency key; prove the commitment count stays unchanged and the duplicate is observable.
- **Relevant system-design reference:** [10 — Recoverability, Rollbacks & Saga](../system-design/10-recoverability-rollbacks-and-saga.md) and [12 — Production Scale & Capacity](../system-design/12-production-scale-and-capacity.md), both **reference-only depth**.

**Previous:** [Chapter 08 — Multi-Agent Coordination](../08-multi-agent-communication-and-coordination/README.md) · **Next:** [Stage 11 — AI Infrastructure](../../11-ai-infrastructure/README.md)

[Open the notebook](09-reliability-recovery-and-production-decisions.ipynb)
