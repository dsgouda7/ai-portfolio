# AI Engineer Learning Route

> **Track intent:** This is a role-based practice track, not a standalone subject area or a replacement for the chapters under `learning/`. Use it alongside GenAI, agentic AI, and infrastructure content to practice connecting those concepts into an observable production loop.

You can finish every notebook in `learning/` and still be unable to answer the question that matters in production: what failed, which layer owns the failure, and what evidence supports the next release?

This route reorganizes the existing curriculum around that job. You learn enough model mechanics to debug behavior, enough application engineering to control it, and enough infrastructure to explain its latency, memory, reliability, and cost. You do not get credit for attendance. You get credit for evidence.

Start with [Role Baseline and Route](00-role-baseline-and-route.md). It contains the skip gates, complete artifact sequence, competency matrix, milestones, time and compute notes, and completion checklist.

The current production-loop source is present: five local fixture chapters, the [Azure Operational LLM Serving tutorial](../../ai-infrastructure/09-azure-operational-llm-serving/README.md), the [capstone](06-capstone/README.md), and the [Riverside AI Platform](../../../projects/riverside-ai-platform/README.md). Chapters 01-05 were executed successfully in the unified FDE environment, then their notebook outputs and execution counts were cleared for reuse. This local validation does not establish cloud behavior, and the platform has no retained live Azure result in this route.

## Route Rules

| Label | What it means |
|---|---|
| Required | You must demonstrate the competency. Complete the linked artifact or pass its prior-experience evidence gate. |
| Optional | Study it when it matches your target role, workload, or hardware. It does not block the general route. |
| Skip with evidence | You may bypass the study material only when you can attach equivalent, reproducible evidence. Familiarity is not evidence. |
| Planned | The repository has identified the gap, but the linked artifact is not yet available as completion evidence. |

## The Route

| Phase | Failure you must be able to diagnose | Required learning surface | Exit evidence |
|---|---|---|---|
| 0. Baseline | You cannot tell which fundamentals you actually need | [Role baseline and skip gates](00-role-baseline-and-route.md) | A route decision for every competency, with evidence links for every skip |
| 1. Foundations | Framework calls work, but tensor shape, gradient, and tokenization failures are guesswork | [GenAI prerequisites](../../genai-prerequisites/) and [PyTorch fundamentals](../../genai/00-pytorch-fundamentals/01-keras-to-pytorch-antarctic-field-guide.ipynb) | A small reproducible training and reload proof |
| 2. Model mechanics | Generation is wrong, but you cannot localize masking, attention, objective, or decoding | [RNN bridge](../../genai/01-rnns/01-pytorch-rnn-bridge.ipynb) and [Transformer foundations](../../genai/02-transformers/README.md) | Mechanism-level diagnosis with measured checks |
| 3. Adaptation | A training run completes, but you cannot justify the objective, update strategy, or checkpoint | [LLM fine-tuning](../../genai/03-llm-finetuning/README.md) | Held-out comparison, artifact lineage, and a release or reject decision |
| 4. Retrieval and evaluation | One aggregate score hides retrieval, generation, slice, or evaluator failure | [RAG](../../genai/04-rag/README.md) and [LLM evaluation](../../genai/05-llm-evaluation/README.md) | Separate retrieval and generation reports with explicit release thresholds |
| 5. Application control | Provider behavior leaks into the app and failures cannot be attributed | [LLM gateway](../../genai/06-llm-gateway/06-llm-gateway.ipynb) | A stable request contract with routing, limits, fallback, caching, cost, and traces |
| 6. Production mechanics | You tune software without knowing whether memory, compute, data movement, or serving policy is limiting it | [AI Infrastructure](../../ai-infrastructure/README.md) | A measured bottleneck diagnosis and a workload-specific serving decision |
| 7. Specialization | A general route does not prove agent orchestration or deep systems expertise | [Agentic AI](../../agentic-ai/README.md), [Agentic AI System Design](../../agentic-ai-system-design/system-design.md), and optional infrastructure depth | Branch-specific evidence tied to your target role |
| 8. Production loop | Data, prompt, release, cost, and feedback decisions are not connected end to end | [Operational chapters and integration surfaces](#operational-and-integration-surfaces) | Source is available; complete locally with retained evidence, and defer live Azure claims until separately validated |

The dependency chain is:

```text
baseline
  -> foundations
  -> language-model mechanics
  -> adaptation
  -> retrieval + evaluation
  -> gateway
  -> infrastructure core
  -> capstone

After retrieval + evaluation:
  -> agentic specialization          (when agents are in scope)
  -> system-design reference depth   (when architecture depth is in scope)
```

Agentic AI and infrastructure can proceed in parallel after the shared GenAI core. The system-design track is a reference surface, not a prerequisite for the hands-on agent notebooks.

### Cross-Track Return Points

Use these links to return to the operational route after completing a concept-owning track:

- after fine-tuning objective and parameter work, continue with [Training Data Quality and Lineage](01-training-data-quality-and-lineage/README.md) before admitting a new data candidate;
- after RAG, evaluation, and gateway work, continue with [Prompt Release and Experimentation](02-prompt-release-and-experimentation/README.md);
- after gateway, profiling, and inference systems, continue with [Application Latency and Cost](03-application-latency-and-cost/README.md) and [Azure Operational LLM Serving](../../ai-infrastructure/09-azure-operational-llm-serving/README.md);
- after producing model, prompt, index, and evaluator artifacts, bind them in [Release Registry and Lineage](04-release-registry-and-lineage/README.md);
- after a release has trace evidence, continue with [Production Feedback and Drift](05-production-feedback-and-drift/README.md), then assemble the [capstone](06-capstone/README.md).

## Choose Your Entry

### You are new to ML or PyTorch

Start at Phase 0 and assume nothing is skippable until you pass the evidence gate. Do not rush past tokenization, tensor shape, or gradient checks. Those failures return later with more expensive names.

### You already build LLM applications

Run the Phase 0 gates first. You will often skip some foundations but fail the mechanism, evaluation, or infrastructure gates. Start at the first failed gate, not at the first unfamiliar title.

### You already train or serve models

Attach evidence for the mechanics and infrastructure gates, then test the application layer. A profiler trace does not prove retrieval quality, release safety, or rollback behavior.

### You are targeting agentic systems

Complete the shared route through retrieval and evaluation, then take the complete numbered [Agentic AI sequence](../../agentic-ai/README.md). Use the [system-design reference](../../agentic-ai-system-design/system-design.md) to deepen architecture decisions; do not substitute reading architecture documents for workflow, safety, evaluation, and recovery evidence.

## Operational and Integration Surfaces

These artifacts exist in source. Chapters 01-05 were executed successfully in the unified FDE environment and then cleared; the committed notebooks therefore show no outputs or execution counts. The Azure-shaped tutorial and Riverside platform do not establish deployed cloud behavior.

| Artifact | Gap it closes in source |
|---|---|
| [Training Data Quality and Lineage](01-training-data-quality-and-lineage/training-data-quality-and-lineage.ipynb) | Duplicate leakage, malformed training rows, preference shortcuts, PII, provenance, and dataset fingerprints |
| [Prompt Release and Experimentation](02-prompt-release-and-experimentation/prompt-release-and-experimentation.ipynb) | Versioned prompt/config candidates, offline gates, paired or shadow comparison, uncertainty, and rollback |
| [Application Latency and Cost](03-application-latency-and-cost/application-latency-and-cost.ipynb) | Stage-level p50/p95 latency, TTFT, TPOT, retry amplification, cache savings, and cost per successful request |
| [Release Registry and Lineage](04-release-registry-and-lineage/release-registry-and-lineage.ipynb) | One release object tying model, adapter, tokenizer, prompt, index, evaluator, deployment, and rollback target together |
| [Production Feedback and Drift](05-production-feedback-and-drift/production-feedback-and-drift.ipynb) | Trace sampling, drift detection, failure clustering, reviewed feedback, evaluation updates, and action selection |
| [Azure Operational LLM Serving](../../ai-infrastructure/09-azure-operational-llm-serving/azure-operational-llm-serving.ipynb) | Local Azure-shaped serving mechanics plus an explicit list of cloud-only claims that still require live validation |
| [AI Engineer Capstone](06-capstone/README.md) and [assessment rubric](06-capstone/assessment-rubric.md) | Integrated release evidence across data, retrieval, generation, control, infrastructure, and feedback |
| [Riverside AI Platform](../../../projects/riverside-ai-platform/) | A project surface for the capstone's production-oriented evidence |

Use the five fixture chapters for deterministic local mechanisms, the Azure tutorial for local serving mechanics and cloud revalidation requirements, the platform for production-shaped contracts and source mapping, and the capstone to join retained evidence. The successful local validation of chapters 01-05 is scoped to their fixture workflows; do not replace missing evidence for any other scope with a diagram, source file, expected outcome, or aspirational claim.

## Troubleshooting

| Symptom | Likely boundary | Recovery |
|---|---|---|
| Fixture path is not found | Notebook started outside the repository tree or checkout is incomplete | Open the repository workspace, confirm `learning/role-based-tracks/ai-engineer/shared/` exists, and restart from the chapter README |
| `Stale or modified fixture` | A pinned fixture, schema, or expected-outcome file differs from [`shared/fixture-manifest.json`](shared/fixture-manifest.json) | Restore the pinned bytes or intentionally version the complete fixture contract; never edit a digest just to silence the check |
| Schema or expected-outcome assertion fails after the digest check passes | Notebook logic, dependency behavior, or documented expectation may be stale | Stop, retain the failure, compare it with the owning `EXPECTED_OUTCOMES.md`, and review before changing source |
| Import or kernel error | Chapter environment was not selected or setup was not completed | Follow the chapter setup instructions; setup creates a local environment but does not execute the notebook |
| Mermaid does not render | Frontend renderer support, not evidence arithmetic | Read the adjacent text and code; report the rendering issue separately from the lesson result |
| Local result is being described as Azure or production proof | Evidence-class error | Downgrade the claim and use the capstone evidence-class guide before review |

## Extending the Route

The default path is local, deterministic, and credential-free. To add a real provider, model, service, dataset, or Azure environment:

1. Keep shared v1 fixtures unchanged. Create a new versioned input set outside v1 and record its schema, digest, provenance, rights, and expected decision policy.
2. Preserve the chapter's logical contract at the adapter boundary; do not splice network calls into fixture arithmetic.
3. Predeclare thresholds, workload, environment, cost ceiling, stop conditions, and teardown before running an integration.
4. Retain commands, versions, inputs, raw permitted output, normalized result, reviewer, and limitations.
5. Classify the result by what actually ran. Source remains `IMPLEMENTED_SOURCE`; deterministic fixtures become `LOCAL_FIXTURE`; local workloads may become `LOCAL_MEASURED`; only authorized retained cloud evidence may become `LIVE_AZURE`.
6. Map production-shaped work to the Riverside contracts and claim ledger, then add the result to a capstone package without rewriting the original teaching evidence.

Optional integrations may require network access, credentials, paid services, quota, data approval, and teardown. None is required to learn the local mechanism, and none should be attempted from the committed notebooks without an explicit execution decision.

## Completion Standard

You have completed the current route when:

- every current required competency is backed by a linked evidence artifact or a passed prior-experience gate;
- every optional branch is marked completed, deferred, or not relevant to your stated target role;
- every unavailable or live-unvalidated competency is explicitly marked deferred rather than silently credited;
- each operational notebook has retained its fixture version, input digests, environment, and generated evidence outside the shared fixture directory;
- your evidence separates measured local results, modeled results, reference numbers, and cloud-validated results;
- you can localize a regression to data, retrieval, generation, orchestration, infrastructure, or policy and name the next discriminating test.

Use the detailed [evidence checklist and milestones](00-role-baseline-and-route.md#evidence-checklist) to make that standard auditable.
