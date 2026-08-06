# AI Engineer Role Baseline and Route

An AI Engineer owns the path from a user need to an operated AI capability. The common failure is to optimize one layer in isolation: train a better checkpoint for a retrieval problem, tune a prompt for a policy problem, or add GPUs before measuring where latency lives.

This route trains a different habit. You expose the failure, run the cheapest test that separates plausible causes, change the owning layer, and retain evidence for the decision.

## Role Outcome

When you complete the route, you can:

1. Explain and modify transformer and LLM behavior at the token, objective, and parameter levels.
2. Build and evaluate fine-tuned, retrieval-augmented, and, when relevant, agentic applications.
3. Select models and serving strategies from quality, latency, memory, safety, and cost evidence.
4. Operate models behind a stable gateway contract with tracing, limits, fallback, and release gates.
5. Version prompts, datasets, models, adapters, indexes, evaluators, and deployment configuration.
6. Diagnose whether a regression belongs to data, retrieval, generation, orchestration, infrastructure, or policy.

The current repository fully teaches some of these outcomes and only partially teaches others. The [competency matrix](#competency-matrix) names that boundary instead of hiding it.

## Baseline Decision

For every competency, choose exactly one route state:

- **Study:** You do not yet have the evidence, so complete the linked material and produce it.
- **Skip with evidence:** You already have equivalent evidence and can explain it under review.
- **Optional:** The capability is outside your current role target.
- **Deferred:** The capability is required for your target, but hardware, cloud access, or a planned artifact is not available.

Do not use "I have used this before" as a skip condition. A skip needs an inspectable artifact, measured result, and explanation of what would falsify your conclusion.

### Prior-Experience Gates

| Gate | You may skip the study material only if you can show all of this | If the gate fails |
|---|---|---|
| Math and ML foundations | Derive and numerically check a gradient; explain train/validation/test separation; choose a metric that matches a stated failure; identify leakage in a proposed split | Start with the five required [GenAI prerequisite notebooks](#phase-1-foundations) |
| PyTorch | Build a small `nn.Module`; explain batch, sequence, and feature dimensions; run `zero_grad`, forward, backward, and `step`; save and reload a `state_dict`; verify inference parity | Complete [PyTorch Fundamentals](../../genai/00-pytorch-fundamentals/01-keras-to-pytorch-antarctic-field-guide.ipynb) |
| Language-model mechanics | Trace one token through embedding, position, attention, residual, normalization, and output projection; explain causal masking; distinguish decoder-only from encoder-decoder behavior; isolate a decoding failure from a training-objective failure | Complete the [RNN and Transformer route](#phase-2-language-model-mechanics) |
| Adaptation | Given a behavior failure, choose among continued pretraining, SFT, preference alignment, full tuning, freezing, LoRA, and QLoRA; produce held-out evidence and artifact lineage; reject a run when the evidence is weak | Complete [LLM Fine-Tuning](#phase-3-adaptation) |
| Retrieval | Construct separate lexical and semantic failure cases; fuse or rerank results; measure retrieval independently of generation; test authorization and unsupported-query boundaries | Complete [RAG](#phase-4-retrieval-and-evaluation) |
| Evaluation | Define a versioned evaluation set with slices; use task-appropriate metrics; audit judge agreement and bias; detect hallucination; calibrate or qualify confidence; state a release threshold before seeing candidate results | Complete the [LLM Evaluation sequence](#phase-4-retrieval-and-evaluation) |
| Gateway and control plane | Normalize at least two provider contracts; demonstrate bounded retries, fallback, rate limiting, cache semantics, cost accounting, and a trace that attributes the final response | Complete the [LLM Gateway](#phase-5-application-control) |
| Infrastructure | Estimate model and KV-cache memory; use a profiler to identify a bottleneck; explain when quantization changes the decision; choose a batching and serving strategy from workload evidence | Complete the [infrastructure core](#phase-6-infrastructure-core) |
| Agentic systems | Demonstrate typed tool contracts, bounded control, durable state, retrieval, trajectory evaluation, safety boundaries, interoperability, coordination, and idempotent recovery under injected failure | Complete the full [Agentic AI branch](#phase-7-optional-specializations) when agents are in scope |
| Production operations | Tie data, prompt, model, index, evaluator, deployment, cost, trace, feedback, and rollback evidence to one release identifier | Complete the [authored production-loop artifacts](#phase-8-production-loop) and capstone with retained evidence, or attach equivalent external evidence; defer only the execution classes you cannot support |

Your external evidence may come from a work project, open-source contribution, or independent portfolio project. Remove secrets and proprietary data, but retain enough structure, configuration, metrics, and failure analysis for another engineer to review the claim.

## Phase 1: Foundations

The required competency is the ability to reason about the operations. The notebooks are required when you do not pass the corresponding gate.

### Required or Skip with Evidence

1. [Math Foundations for ML](../../genai-prerequisites/00-math-foundations/math-foundations-for-ml.ipynb)
2. [ML Basics](../../genai-prerequisites/01-ml-basics/ml-basics.ipynb)
3. [Neural Networks and Backpropagation](../../genai-prerequisites/02-neural-networks/neural-networks-and-backprop.ipynb)
4. [RNN Sequence Modeling](../../genai-prerequisites/04-rnn-sequence-modeling/rnn-sequence-modeling.ipynb)
5. [Tokenization and Embeddings](../../genai-prerequisites/05-tokenization/tokenization-and-embeddings.ipynb)
6. [Keras to PyTorch Antarctic Field Guide](../../genai/00-pytorch-fundamentals/01-keras-to-pytorch-antarctic-field-guide.ipynb)

### Optional

- [Convolutional Neural Networks](../../genai-prerequisites/03-cnns/convolutional-neural-networks.ipynb) is optional for the general AI Engineer route. Make it required for yourself when multimodal, vision, or convolution-heavy workloads are part of your target role.

### Exit Evidence

- One small deterministic training run with the data split, seed, environment, metric, and result recorded.
- A save/reload check showing that the reloaded model reproduces the expected inference within a stated tolerance.
- A short failure note showing one broken shape, gradient, split, or tokenization assumption and the check that exposed it.

## Phase 2: Language-Model Mechanics

If you cannot localize a model failure below the API level, complete this phase. Prompt iteration is not a substitute for understanding the computation you are changing.

### Required or Skip with Evidence

1. [PyTorch RNN Bridge](../../genai/01-rnns/01-pytorch-rnn-bridge.ipynb)
2. [Attention and Transformer Blocks](../../genai/02-transformers/01-attention-and-transformer-blocks.ipynb)
3. [Decoder-Only Language Models](../../genai/02-transformers/02-decoder-only-language-model.ipynb)
4. [Encoder-Decoder and Cross-Attention](../../genai/02-transformers/03-encoder-decoder-and-cross-attention.ipynb)

The [Transformer Foundations README](../../genai/02-transformers/README.md) defines the three-part contract and fresh-kernel behavior.

### Exit Evidence

- A diagram or trace that follows tensor shapes through attention and names where each mask applies.
- A controlled comparison in which one variable changes and the measured behavior changes for the reason you predicted.
- A diagnosis note that distinguishes architecture, objective, context, and decoding failures.

## Phase 3: Adaptation

A falling training loss is not release evidence. You must show that the intended behavior changed, retained capabilities did not regress beyond your threshold, and the exact artifact can be reproduced.

### Required or Skip with Evidence

1. [Fine-Tuning Data Techniques](../../genai/03-llm-finetuning/01-llm-finetuning-data-techniques.ipynb)
2. [Fine-Tuning Parameter Techniques](../../genai/03-llm-finetuning/02-llm-finetuning-parameter-techniques.ipynb)
3. [Fine-Tuning Comparison and Decision](../../genai/03-llm-finetuning/03-llm-finetuning-comparison-and-decision.ipynb)

### Required for Full GPU Training Evidence, Otherwise Deferred

4. [Fine-Tuning Practice](../../genai/03-llm-finetuning/04-llm-finetuning-practice.ipynb) is intentionally CUDA-only. Complete it when you claim end-to-end GPU fine-tuning evidence across the full Riverside corpus. If compatible CUDA hardware is unavailable, mark this artifact deferred; do not report its expected results as your measurements.

Parts 1-3 select a CPU or CUDA profile. The [Fine-Tuning README](../../genai/03-llm-finetuning/README.md) records that boundary.

### Exit Evidence

- A written mapping from observed behavior failure to training objective and parameter-update strategy.
- Train, validation, and untouched test definitions with leakage checks.
- A base-versus-candidate comparison on task and retention slices.
- A manifest containing base revision, dataset hashes, seed, arguments, environment, metrics, and artifact paths.
- A release, reject, or collect-more-data decision that follows the measured result.

## Phase 4: Retrieval and Evaluation

An answer score cannot tell you whether the retriever missed the evidence or the generator ignored it. Split those failures before changing either system.

### Required or Skip with Evidence

1. [Hybrid Search](../../genai/04-rag/04-hybrid-search.ipynb)
2. [RAG Evaluation](../../genai/04-rag/05-rag-evaluation.ipynb)
3. [Metrics and Benchmarks](../../genai/05-llm-evaluation/01-llm-evaluation-metrics-and-benchmarks.ipynb)
4. [LLM-as-Judge, Safety, and Pipeline](../../genai/05-llm-evaluation/02-llm-as-judge-safety-and-pipeline.ipynb)
5. [Hallucination Detection](../../genai/05-llm-evaluation/03-hallucination-detection.ipynb)
6. [Calibration and Confidence](../../genai/05-llm-evaluation/04-calibration-and-confidence.ipynb)

Use the [RAG README](../../genai/04-rag/README.md) and [LLM Evaluation README](../../genai/05-llm-evaluation/README.md) for the chapter contracts.

### Exit Evidence

- A versioned query set with lexical, semantic, authorization, unsupported, and answerability slices.
- Retrieval metrics reported separately from generation metrics.
- At least one gold-context ablation that distinguishes retriever from generator failure.
- Judge or evaluator validity evidence, including disagreement, bias, or calibration limits.
- Release thresholds written before candidate scoring, plus a slice-level pass or fail report.

## Phase 5: Application Control

Without a gateway contract, every provider change becomes an application change and every retry can become an unpriced reliability experiment.

### Required or Skip with Evidence

1. [LLM Gateways: Routing, Resilience, and Cost Control](../../genai/06-llm-gateway/06-llm-gateway.ipynb)

The notebook uses deterministic provider simulations so you can isolate systems behavior. The [Gateway README](../../genai/06-llm-gateway/README.md) explicitly hands production serving internals to the infrastructure track.

### Exit Evidence

- One application-facing request and response schema across at least two provider implementations or doubles.
- Traces for success, rate limit, retry, fallback, cache hit, and terminal failure.
- A cost calculation that includes retries and distinguishes attempted from successful requests.
- A test showing the fallback path preserves the contract and remains bounded.

## Phase 6: Infrastructure Core

You do not need to become a kernel engineer to be an AI Engineer. You do need to know when hardware, precision, memory, profiling, or serving policy owns the failure.

### Required or Skip with Evidence

1. [GPU Hardware Foundations](../../ai-infrastructure/01-gpu-hardware/gpu-hardware-foundations.ipynb)
2. [Mixed Precision and Memory](../../ai-infrastructure/02-mixed-precision/mixed-precision-and-memory.ipynb)
3. [PyTorch Profiling](../../ai-infrastructure/03-profiling/pytorch-profiling.ipynb)
4. [Quantization in Depth](../../ai-infrastructure/06-quantization/quantization-in-depth.ipynb)
5. [Inference Systems](../../ai-infrastructure/07-inference-systems/inference-systems.ipynb)

### Optional Specialization

- [FlashAttention Internals](../../ai-infrastructure/04-flash-attention/flash-attention-internals.ipynb)
- [Distributed Training](../../ai-infrastructure/05-distributed-training/distributed-training.ipynb)
- [Custom Kernels with Triton](../../ai-infrastructure/08-triton-kernels/triton-kernels.ipynb)

The [AI Infrastructure README](../../ai-infrastructure/README.md) states that all current notebooks run on CPU with graceful fallbacks. A fallback or literature-reference number teaches the mechanism; it does not prove your GPU's live performance.

### Exit Evidence

- A model, activation, optimizer, and KV-cache memory estimate with assumptions shown.
- A profiler trace and a bottleneck classification: compute, memory, data movement, synchronization, or scheduling.
- A quality, memory, and latency comparison for at least one quantization decision.
- A serving recommendation tied to workload shape, batching, context length, and latency target.
- A clear label on every number: measured on your hardware, modeled, or taken from a reference.

## Phase 7: Optional Specializations

These branches become required when your target role makes the capability part of the job.

### Agentic AI

If agents are in scope, complete the sequence in order. Later chapters depend on contracts and failure controls established earlier.

1. [Agent Foundations and Tool Contracts](../../agentic-ai/00-agent-foundations-and-tool-contracts/00-agent-foundations-and-tool-contracts.ipynb)
2. [Reasoning, Planning, and Bounded Control](../../agentic-ai/01-reasoning-planning-and-control/01-reasoning-planning-and-control.ipynb)
3. [State, Context, and Memory](../../agentic-ai/02-state-context-and-memory/02-state-context-and-memory.ipynb)
4. [Durable Workflows with LangGraph](../../agentic-ai/03-durable-workflows-with-langgraph/03-durable-workflows-with-langgraph.ipynb)
5. [Agentic RAG and Self-Correction](../../agentic-ai/04-agentic-rag-and-self-correction/04-agentic-rag-and-self-correction.ipynb)
6. [Agent Evaluation and Observability](../../agentic-ai/05-agent-evaluation-and-observability/05-agent-evaluation-and-observability.ipynb)
7. [Safety, Human Control, and Governance](../../agentic-ai/06-safety-human-control-and-governance/06-safety-human-control-and-governance.ipynb)
8. [MCP and Agent Interoperability](../../agentic-ai/07-mcp-and-agent-interoperability/07-mcp-and-agent-interoperability.ipynb)
9. [Multi-Agent Communication and Coordination](../../agentic-ai/08-multi-agent-communication-and-coordination/08-multi-agent-communication-and-coordination.ipynb)
10. [Reliability, Recovery, and Production Decisions](../../agentic-ai/09-reliability-recovery-and-production-decisions/09-reliability-recovery-and-production-decisions.ipynb)

The [Agentic AI README](../../agentic-ai/README.md) distinguishes local fixture measurements from modeled production targets. Preserve that distinction in your evidence.

### Agentic System Design

Use [Designing Agentic AI Systems](../../agentic-ai-system-design/system-design.md) as the master architecture. Read component documents when the corresponding design decision appears in your work:

1. [Foundations of Agentic Systems](../../agentic-ai-system-design/01-foundations-of-agentic-systems.md)
2. [Agent Lifecycle and Runtime](../../agentic-ai-system-design/02-agent-lifecycle-and-runtime.md)
3. [Tool, MCP, and Skill Registry](../../agentic-ai-system-design/03-tool-mcp-and-skill-registry.md)
4. [Model Gateway and LLM Providers](../../agentic-ai-system-design/04-model-gateway-and-llm-providers.md)
5. [State Management and Memory](../../agentic-ai-system-design/05-state-management-and-memory.md)
6. [Non-Determinism, Loops, and Termination](../../agentic-ai-system-design/06-non-determinism-loops-and-termination.md)
7. [Agent Evaluation Frameworks](../../agentic-ai-system-design/07-agent-evaluation-frameworks.md)
8. [Observability, Tracing, and Health](../../agentic-ai-system-design/08-observability-tracing-and-health.md)
9. [Multi-Agent Communication Patterns](../../agentic-ai-system-design/09-multi-agent-communication-patterns.md)
10. [Recoverability, Rollbacks, and Saga](../../agentic-ai-system-design/10-recoverability-rollbacks-and-saga.md)
11. [Governance, Guardrails, and Security](../../agentic-ai-system-design/11-governance-guardrails-and-security.md)
12. [Production Scale and Capacity](../../agentic-ai-system-design/12-production-scale-and-capacity.md)
13. [Semantic Kernel vs. LangGraph](../../agentic-ai-system-design/13-semantic-kernel-vs-langgraph.md)

Reading these documents is architecture preparation. Completion evidence still needs a decision record, failure analysis, capacity model, threat boundary, or tested implementation.

## Phase 8: Production Loop

The five local production-loop notebooks below were executed successfully in the unified FDE environment, then cleared for reuse. Complete their evidence contracts in order; treat expected-outcome documents as source and retain the outputs required for your own evidence package.

| Capability | Authored artifact | Required completion evidence |
|---|---|---|
| Training-data quality and lineage | [Training Data Quality and Lineage](01-training-data-quality-and-lineage/training-data-quality-and-lineage.ipynb) | Duplicate leakage, contamination, schema/template validity, label agreement, slice balance, PII flags, provenance coverage, and dataset digest |
| Prompt and application release | [Prompt Release and Experimentation](02-prompt-release-and-experimentation/prompt-release-and-experimentation.ipynb) | Versioned candidate, deterministic offline gate, paired or shadow comparison, uncertainty, rollback target, and retained config |
| End-to-end latency and cost | [Application Latency and Cost](03-application-latency-and-cost/application-latency-and-cost.ipynb) | Stage p50/p95, TTFT, TPOT, throughput, token use, retries, cache savings, and cost per successful request |
| Release registry and lineage | [Release Registry and Lineage](04-release-registry-and-lineage/release-registry-and-lineage.ipynb) | One machine-readable release manifest with compatibility checks and rollback target |
| Production feedback and drift | [Production Feedback and Drift](05-production-feedback-and-drift/production-feedback-and-drift.ipynb) | Privacy-safe trace sample, drift report, failure clusters, reviewed cases, evaluation candidate, and action decision |
| Cloud operations | [Azure Operational LLM Serving](../../ai-infrastructure/09-azure-operational-llm-serving/azure-operational-llm-serving.ipynb) | Deployed configuration, live service measurements, operational checks, cost boundary, and explicit teardown evidence |
| Integrated assessment | [AI Engineer Capstone](06-capstone/README.md), [assessment rubric](06-capstone/assessment-rubric.md), and [Riverside AI Platform](../../../projects/riverside-ai-platform/) | One traceable release package spanning data, model, retrieval, evaluation, control, infrastructure, and feedback |

All linked targets now exist in source. Chapters 01-05 completed a successful local fixture run in the unified FDE environment and were then cleared. The Azure serving tutorial and Riverside platform remain separate evidence surfaces, and no cloud behavior was validated by the chapter run. Mark each result by its actual evidence class and keep live cloud behavior deferred unless you provide authorized, release-scoped evidence.

## Competency Matrix

| ID | Competency | Existing evidence source | Minimum evidence | Current status |
|---|---|---|---|---|
| C1 | ML, neural-network, tokenization, and PyTorch foundations | [Prerequisites](../../genai-prerequisites/) and [PyTorch Fundamentals](../../genai/00-pytorch-fundamentals/01-keras-to-pytorch-antarctic-field-guide.ipynb) | Deterministic train, evaluate, save, reload, and failure diagnosis | Available |
| C2 | Transformer and language-model mechanics | [RNNs](../../genai/01-rnns/README.md) and [Transformers](../../genai/02-transformers/README.md) | Shape and mask trace plus controlled mechanism comparison | Available |
| C3 | Objective and parameter-efficient adaptation | [LLM Fine-Tuning](../../genai/03-llm-finetuning/README.md) | Held-out task/retention comparison, lineage, and release decision | Available; full-corpus practice requires CUDA |
| C4 | Retrieval, grounding, and boundary checks | [RAG](../../genai/04-rag/README.md) | Retrieval report, generation report, gold-context ablation, authorization test | Available |
| C5 | Evaluation, safety, hallucination, and confidence | [LLM Evaluation](../../genai/05-llm-evaluation/README.md) | Versioned slices, evaluator audit, thresholds, regression report | Available |
| C6 | Provider-neutral application control | [LLM Gateway](../../genai/06-llm-gateway/README.md) | Contract, bounded failure traces, cache and cost evidence | Available through deterministic simulation |
| C7 | Hardware-aware training and inference decisions | [AI Infrastructure](../../ai-infrastructure/README.md) | Memory model, profile, quantization comparison, serving decision | Available; CPU fallbacks are not live GPU benchmarks |
| C8 | Agent orchestration, evaluation, safety, and recovery | [Agentic AI](../../agentic-ai/README.md) | Injected-failure traces and measured fixture outcomes across the full sequence | Available as an optional branch; production targets are modeled |
| C9 | Architecture and capacity reasoning | [Agentic System Design](../../agentic-ai-system-design/system-design.md) | Decision records, threat boundaries, recovery design, and capacity assumptions | Available as reference material |
| C10 | Data, prompt, release, latency, lineage, and feedback operations | [Production loop](#phase-8-production-loop) | Five retained operational reports tied to one release | Local fixture workflows validated in the unified FDE environment; notebooks cleared for reuse |
| C11 | Cloud serving and integrated capstone | [Azure serving notebook](../../ai-infrastructure/09-azure-operational-llm-serving/azure-operational-llm-serving.ipynb), [capstone](06-capstone/README.md), and [Riverside platform](../../../projects/riverside-ai-platform/) | Integrated release package; live cloud evidence only for claims or rollout stages that require it | Source available; local and static evidence possible, live Azure deferred |

## Milestones

### M0: Route Contract

You have a state for every competency and a link for every claimed skip. Any unsupported claim moves back to **Study**.

### M1: Mechanism-Level Debugger

You have completed or bypassed C1-C2 with evidence. You can trace tensors and tokens, isolate mask/objective/decoding failures, and reproduce a small model artifact.

### M2: Adaptation Decision

You have completed C3. You can reject a training run despite a falling training loss and can identify the exact data, base revision, configuration, and held-out result behind a candidate.

### M3: Evidence-Backed LLM Application

You have completed C4-C5. You can separate retrieval from generation failure, evaluate slices, qualify evaluator limits, and make a release decision against predeclared thresholds.

### M4: Controlled Application Surface

You have completed C6. Provider changes, retries, fallbacks, cache behavior, and costs are visible behind one contract.

### M5: Hardware-Aware Operator

You have completed C7. You can classify the bottleneck before optimizing and label modeled or reference numbers honestly.

### M6: Role Specialization

You have completed C8 and C9 only when your target requires agentic or Staff-level architecture depth. Your output is tested evidence or a reviewable decision record, not a reading log.

### M7: Production Loop

You have completed C10 through retained results from the five operational chapters or equivalent external work, and you have assembled the capstone package. You may complete the source, static, fixture, local-measured, and modeled portions of C11 without Azure. Keep any rollout stage or capability that requires live Azure evidence deferred until an authorized run produces a scoped evidence package. The repository source alone does not support an end-to-end production claim.

## Time and Compute Notes

The repository does not publish measured learner-completion times. The ranges below are planning estimates derived from artifact counts, not benchmarks or promises. They exclude dependency troubleshooting, model downloads, cloud approval, and queue time.

| Route segment | Artifact count | Planning range | Compute boundary |
|---|---:|---:|---|
| Foundations | 6 required, 1 optional | 18-36 focused hours | CPU is sufficient; package and model downloads still take disk and network |
| Language-model mechanics | 4 | 16-32 focused hours | CPU is sufficient for the teaching workloads; larger pretrained-model paths may download weights |
| Adaptation | 3 core plus 1 CUDA practice | 18-36 hours for core; add 8-20 hours for practice | Parts 1-3 select CPU or CUDA profiles; Part 4 requires compatible CUDA hardware |
| Retrieval, evaluation, and gateway | 7 | 24-45 focused hours | Local execution is the default learning path; optional model or provider paths may add downloads, credentials, cost, or nondeterminism |
| Infrastructure core | 5 | 18-35 focused hours | CPU fallbacks teach mechanisms; credible live GPU latency, memory, and kernel claims require suitable GPU hardware |
| Agentic specialization | 10 | 30-50 focused hours | The default path is deterministic, CPU-only, offline, and fixture-based; production targets remain modeled unless a notebook says otherwise |
| System-design reference | 14 documents including the master | 10-22 focused hours | No special compute; evidence requires design work beyond reading |
| Operational loop and capstone | 7 authored surfaces | Reserve 30-60 focused hours for local evidence assembly and review | Cloud extensions may require an Azure subscription, quota, credentials, spend limits, and teardown; source presence is not a result |

A new learner taking the current general route through infrastructure should plan roughly 95-185 focused hours. Strong prior evidence can reduce study time, but it does not reduce the number of competencies you must prove. Add roughly 30-50 hours for the hands-on agentic branch. Treat all ranges as scheduling aids and record your actual time if you want a defensible estimate for the next learner.

### Compute Honesty Rules

- Record hardware, software versions, precision, batch size, sequence length, and warm-up policy beside performance numbers.
- Label fixture measurements, simulations, modeled capacity, literature references, and live measurements differently.
- Do not compare CPU fallback numbers with GPU numbers as though only the technique changed.
- Do not call a local simulation cloud validation.
- Include retries, failed requests, cache effects, and setup overhead when the claim is end-to-end cost or latency.
- Record paid-service boundaries before you run them and retain teardown evidence afterward.

## Evidence Checklist

Use this checklist for milestone review. A notebook with all cells executed is an input to the evidence, not the evidence package itself.

### Route and Scope

- [ ] Your target role and workload are stated.
- [ ] Every competency is marked Study, Skip with evidence, Optional, or Deferred.
- [ ] Every skip links to an inspectable artifact and names the reviewer-relevant result.
- [ ] Every unavailable or live-unvalidated gap remains visible and is not credited as completed.

### Reproducibility

- [ ] Source revision, environment, dependency versions, random seeds, and hardware are recorded.
- [ ] Inputs and generated artifacts have stable identifiers or hashes.
- [ ] Train, validation, test, and evaluation-set boundaries are explicit.
- [ ] A clean reload or reconstruction check succeeds without notebook kernel state.

### Data and Adaptation

- [ ] The observed behavior failure is tied to the chosen data objective.
- [ ] Leakage, duplication, contamination, schema, template, PII, and provenance checks are reported where applicable.
- [ ] Base and candidate models are compared on task and retention slices.
- [ ] The decision branches on actual results and may end in reject or collect more data.

### Retrieval and Evaluation

- [ ] Retrieval and generation metrics are reported separately.
- [ ] Lexical, semantic, unsupported, authorization, and answerability cases are represented.
- [ ] A gold-context or equivalent ablation localizes at least one failure.
- [ ] Evaluator agreement, bias, uncertainty, or calibration limitations are measured or explicitly bounded.
- [ ] Release thresholds were fixed before candidate scoring.

### Application and Agents

- [ ] Provider or model behavior is hidden behind a versioned application contract.
- [ ] Retry, fallback, rate-limit, cache, and terminal-failure paths are bounded and traced.
- [ ] Cost is reported per successful request and includes retry amplification.
- [ ] Agent tool calls are typed, authority is bounded, and high-impact actions have the required human control.
- [ ] Agent evaluation includes the trajectory, not only the final answer.
- [ ] Recovery tests prove idempotency, resume, or compensation under injected failure.

### Infrastructure and Operations

- [ ] Memory and latency assumptions are written before optimization.
- [ ] A profiler or stage trace identifies the bottleneck.
- [ ] Quality, latency, memory, and cost trade-offs are compared on the same workload.
- [ ] Every operational number is labeled measured, simulated, modeled, or reference.
- [ ] The release has a rollback target and enough lineage to reconstruct its model, prompt, data, index, evaluator, and deployment configuration.
- [ ] Production feedback is privacy-safe, reviewed, versioned, and tied to a specific action or no-action decision.

### Final Review

- [ ] You can name the first discriminating test for a regression in data, retrieval, generation, orchestration, infrastructure, or policy.
- [ ] You can state what your evidence does not prove.
- [ ] Your strongest claim is no stronger than your weakest required evidence source.
- [ ] A reviewer can reproduce or audit the decision without relying on your notebook's live memory.
