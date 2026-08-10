# Forward Deployed Engineer Learning Route

> **Track intent:** This is a role-based practice track, not a standalone subject area or a replacement for the chapters under `learning/`. Use it alongside the technical chapters needed by the customer problem, then practice applying them through Riverside discovery, design, rollout, incident, and handoff decisions.

This route is for engineers who must turn an ambiguous customer workflow into a bounded AI system proposal, define what evidence would prove it, and route implementation and operating decisions to named owners. It reuses the repository's AI Engineer technical core and adds the engagement work that begins before implementation and continues through handoff planning.

The authored route builds discovery, architecture translation, scope control, and evidence discipline. It does not establish independent production-execution competence. The setup environment was verified, and all nine FDE notebooks, including the capstone, executed successfully end to end against committed synthetic fixtures. The validation runs produced the documented fail-closed outcomes, including `BLOCKED`, `HOLD`, `ABORT`, and `REJECT`, and were then cleared so the committed notebooks retain null execution counts and empty outputs. This local synthetic validation supports entry into a supervised practicum; it is not evidence that the learner can independently onboard customer data, deploy identity controls, run a live rollout or incident, negotiate contract terms, or accept a production handoff. Those capabilities require observed practice with authorized systems and customer owners under an experienced FDE or equivalent mentor.

The route does not treat customer contact, a demo, a completed template, or a plausible architecture as evidence of role readiness. Completion requires traceable artifacts from discovery through handoff planning. Start with [Role Baseline and Engagement Lifecycle](00-role-baseline-and-engagement-lifecycle.md).

## Route contract

By the end of the route, you must be able to:

1. Convert an ambiguous request into a workflow baseline, explicit non-goals, testable acceptance criteria, and a discovery backlog.
2. Select the smallest valid intervention: deterministic software, search, RAG, a prompt-only LLM call, fine-tuning, a workflow, an agent, multiple agents, or no AI.
3. Onboard customer data with source ownership, schemas, quality gates, ACL propagation, lineage, deletion, and residency decisions.
4. Carry identity and authorization context through ingress, retrieval, tools, audit, and response without cross-tenant leakage.
5. Tie quality, latency, availability, capacity, support, and cost statements to measured evidence, modeled assumptions, or named customer validation.
6. Run shadow, canary, rollback, containment, recovery, revalidation, and re-enablement workflows.
7. Hand over dashboards, runbooks, change procedures, support boundaries, known limitations, and ownership.

This is a role route, not a replacement for the technical tracks. Complete or challenge out the [AI Engineer Learning Route](../ai-engineer/README.md) first, using its [role baseline and evidence gates](../ai-engineer/00-role-baseline-and-route.md). The focused list below identifies the subset used most often in FDE engagements and links directly to existing learning artifacts. Planned operational notebooks remain uncredited until their files and evidence exist.

## Setup

Use one shared Python 3.11-3.13 environment for the complete route. From `learning/role-based-tracks/fde`, run:

```powershell
.\setup.ps1
```

```bash
./setup.sh
```

The setup creates `learning/role-based-tracks/fde/.venv`, installs Jupyter plus every FDE chapter dependency and the local Azure operational lab dependencies, installs the Riverside AI Platform in editable mode with `test` and `telemetry` extras, and installs the RAG shared/ingestion editable artifacts plus vectorization, serving, Databricks-client, and local-test dependencies. Phase 1 and phase 2 both contain a top-level `remote` package, so setup does not install both editable into the same interpreter; their tests and import checks resolve each phase's source tree explicitly. It registers one `fde` kernel and assigns it to every FDE notebook and the Azure operational tutorial notebook. Chapter setup scripts delegate to this root setup and accept the same options.

| PowerShell | Bash | Effect |
|---|---|---|
| `-SkipKernel` | `--skip-kernel` | Install and verify without registering a kernel or changing notebook metadata |
| `-SkipProjects` | `--skip-projects` | Skip the editable Riverside platform and its test/telemetry extras |
| `-SkipRag` | `--skip-rag` | Skip RAG phase 1/2/3 local validation dependencies |
| `-IncludeAzureML` | `--include-azureml` | Also install Riverside's heavy `azureml` model-serving extra |

The default is intentionally comprehensive and can be large: local RAG validation installs PyTorch, Sentence Transformers, ChromaDB, Delta Lake, MLflow, and Databricks clients. Use `-SkipRag` or `--skip-rag` when you need only the FDE notebooks, Azure tutorial, and Riverside tests. The Azure ML model extra is opt-in because it additionally installs Transformers, PEFT, Safetensors, and the Azure ML inference server.

These are package-install costs only; the setup does not create billable cloud resources. It does not install or start Docker, provision infrastructure, authenticate to Azure or Databricks, download a model during setup, or start a service. Later notebook cells or tests that download datasets/models, and any separately authorized Azure or Databricks run, can incur bandwidth, storage, compute, or service charges.

For a lightweight preflight that does not create the environment or install packages, run:

```powershell
$errors = $null; [System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path .\setup.ps1), [ref]$null, [ref]$errors) > $null; if ($errors) { $errors; exit 1 }
```

```bash
bash -n ./setup.sh
```

The full setup commands for the main validation pass remain `.\setup.ps1 -SkipKernel` on PowerShell or `./setup.sh --skip-kernel` on Bash. They perform the install and the import checks, including `langchain_core`, Databricks AI Search/SDK clients, Riverside, the editable RAG shared/ingestion packages, and the phase-specific RAG source trees, without changing notebook kernel metadata.

## Prerequisites

### Engineering baseline

You should already be able to work with Python, Git, HTTP/JSON APIs, typed data contracts, automated tests, and basic distributed-system failure modes. You also need working knowledge of authentication versus authorization, least privilege, logs, metrics, traces, and deployment rollback.

Use the following rule throughout: skipping study never skips evidence. Prior experience changes what you read; it does not remove an exit criterion.

### AI Engineer technical core

#### Language-model mechanics and adaptation

Required at decision depth:

1. [Attention and Transformer Blocks](../../genai/01-transformers/01-attention-and-transformer-blocks.ipynb)
2. [Decoder-only Language Model](../../genai/01-transformers/02-decoder-only-language-model.ipynb)
3. [Encoder-decoder and Cross-attention](../../genai/01-transformers/03-encoder-decoder-and-cross-attention.ipynb)
4. [Fine-tuning Data Techniques](../../genai/02-llm-finetuning/01-llm-finetuning-data-techniques.ipynb)
5. [Fine-tuning Parameter Techniques](../../genai/02-llm-finetuning/02-llm-finetuning-parameter-techniques.ipynb)
6. [Fine-tuning Comparison and Decision](../../genai/02-llm-finetuning/03-llm-finetuning-comparison-and-decision.ipynb)

[GPU Fine-tuning Practice](../../genai/02-llm-finetuning/04-llm-finetuning-practice.ipynb) is optional unless the engagement includes training execution or checkpoint promotion.

#### Retrieval, evaluation, and application control

Required:

1. [Hybrid Search](../../genai/03-rag/01-hybrid-search.ipynb)
2. [RAG Evaluation](../../genai/03-rag/02-rag-evaluation.ipynb)
3. [Evaluation Metrics and Benchmarks](../../genai/04-llm-evaluation/01-llm-evaluation-metrics-and-benchmarks.ipynb)
4. [LLM-as-Judge, Safety, and Pipeline](../../genai/04-llm-evaluation/02-llm-as-judge-safety-and-pipeline.ipynb)
5. [Hallucination Detection](../../genai/04-llm-evaluation/03-hallucination-detection.ipynb)
6. [Calibration and Confidence](../../genai/04-llm-evaluation/04-calibration-and-confidence.ipynb)
7. [LLM Gateway](../../genai/05-llm-gateway/01-llm-gateway.ipynb)

Use the [RAG Knowledge Pipeline project](../../../projects/rag-knowledge-pipeline/README.md) as a reference for independently deployable ingest, vectorization, and serving boundaries. Its static Wikipedia snapshot, local model, lack of incremental updates, and absence of customer ACL integration are explicit limitations, not capabilities to inherit by assumption.

#### Agent and workflow systems

Required when the proposed system uses tools, long-running state, or agentic control:

1. [Agent Foundations and Tool Contracts](../../agentic-ai/00-agent-foundations-and-tool-contracts/00-agent-foundations-and-tool-contracts.ipynb)
2. [Durable Workflows with LangGraph](../../agentic-ai/03-durable-workflows-with-langgraph/03-durable-workflows-with-langgraph.ipynb)
3. [Agent Evaluation and Observability](../../agentic-ai/05-agent-evaluation-and-observability/05-agent-evaluation-and-observability.ipynb)
4. [Safety, Human Control, and Governance](../../agentic-ai/06-safety-human-control-and-governance/06-safety-human-control-and-governance.ipynb)
5. [MCP and Agent Interoperability](../../agentic-ai/07-mcp-and-agent-interoperability/07-mcp-and-agent-interoperability.ipynb)
6. [Reliability, Recovery, and Production Decisions](../../agentic-ai/09-reliability-recovery-and-production-decisions/09-reliability-recovery-and-production-decisions.ipynb)

The [Agentic AI Platform system design](../../agentic-ai-system-design/system-design.md) is the architecture reference. Use these focused components during the engagement:

- [Agent Lifecycle and Runtime](../../agentic-ai-system-design/02-agent-lifecycle-and-runtime.md)
- [Model Gateway and LLM Providers](../../agentic-ai-system-design/04-model-gateway-and-llm-providers.md)
- [Agent Evaluation Frameworks](../../agentic-ai-system-design/07-agent-evaluation-frameworks.md)
- [Observability, Tracing, and Agent Health](../../agentic-ai-system-design/08-observability-tracing-and-health.md)
- [Recoverability, Rollbacks, and Saga](../../agentic-ai-system-design/10-recoverability-rollbacks-and-saga.md)
- [Governance, Guardrails, and Security](../../agentic-ai-system-design/11-governance-guardrails-and-security.md)
- [Production Scale and Capacity](../../agentic-ai-system-design/12-production-scale-and-capacity.md)

#### Serving and infrastructure decisions

Required at decision depth when latency, memory, self-hosting, or serving cost can change the architecture:

1. [Mixed Precision and Memory](../../ai-infrastructure/02-mixed-precision/mixed-precision-and-memory.ipynb)
2. [Quantization in Depth](../../ai-infrastructure/06-quantization/quantization-in-depth.ipynb)
3. [Inference Systems](../../ai-infrastructure/07-inference-systems/inference-systems.ipynb)

GPU hardware, profiling, FlashAttention, distributed training, and Triton are specializations. Require them only when the chosen deployment makes their tradeoffs material.

## Skip rules

| Area | You may skip the study sequence when you can produce | You may not skip |
|---|---|---|
| Math, ML, and PyTorch foundations | A working implementation plus an explanation of gradients, held-out evaluation, tokenization, and train/inference behavior | Any concept you cannot use to diagnose the proposed system |
| Transformer mechanics | A reviewable explanation of attention, token limits, objective choice, decoder-only versus encoder-decoder behavior, and likely failure surfaces | The ability to explain why the selected model family fits the workflow |
| Fine-tuning implementation | A decision record comparing prompt, retrieval, tool, workflow, and adaptation options using data volume, privacy, quality, cost, and rollback evidence | The release-evidence decision; CUDA practice is optional unless training is in scope |
| RAG | Evidence that the engagement has no changing private knowledge, retrieval requirement, citation requirement, or document-level authorization boundary | Authorization and unsupported-query analysis when customer data is retrieved |
| Agentic systems | A justified deterministic workflow with no model-selected tools, long-running agent state, or model-controlled side effects | Tool contracts, authorization, audit, and recovery when any model-selected action remains |
| Multi-agent design | Evidence that one bounded workflow or one agent meets context, ownership, and throughput constraints | A coordination design merely because multiple business teams exist |
| Infrastructure internals | A managed-service architecture whose quotas, latency, residency, observability, and cost have been tested or assigned external validation owners | Capacity, failure, quota, and cost analysis for the actual deployment choice |
| Lifecycle stage | Sanitized prior evidence that satisfies every artifact and exit criterion for that stage | Discovery, identity/isolation, evaluation, rollback, incident, or handoff based only on self-reported familiarity |

A challenge-out review records the evidence location, reviewer, date, limitations, and revalidation trigger. A verbal explanation alone can guide study placement but cannot waive a route artifact.

## Discovery-to-handoff route

| Stage | Decision to earn | Required engagement evidence | Reused technical content |
|---:|---|---|---|
| 0. Baseline | Is the learner ready to enter a supervised engagement, and which technical study can be challenged out? | Baseline case response, evidence inventory, claim register, gap plan | [Role Baseline](00-role-baseline-and-engagement-lifecycle.md#baseline-challenge) |
| 1. Discovery | What workflow, users, baseline, failure cost, constraints, non-goals, and unknowns define the problem? | Stakeholder map, current-state workflow, baseline, acceptance criteria, assumption/risk log, discovery backlog | [Engagement Lifecycle](00-role-baseline-and-engagement-lifecycle.md#engagement-lifecycle) |
| 2. Architecture translation | What is the smallest valid intervention, and where are the model, retrieval, policy, human, data, and side-effect boundaries? | Option matrix, architecture, ADRs, threat assumptions, customer-readable explanation | [System Design](../../agentic-ai-system-design/system-design.md), [Agent Foundations](../../agentic-ai/00-agent-foundations-and-tool-contracts/00-agent-foundations-and-tool-contracts.ipynb) |
| 3. Data onboarding | Can each source be ingested, mapped, authorized, refreshed, deleted, and traced at acceptable quality? | Source inventory, owners, schema mappings, quality gates, ACL tests, lineage, sync/delete plan, readiness verdict | [Hybrid Search](../../genai/03-rag/01-hybrid-search.ipynb), [RAG Evaluation](../../genai/03-rag/02-rag-evaluation.ipynb), [RAG Knowledge Pipeline](../../../projects/rag-knowledge-pipeline/README.md) |
| 4. Identity, isolation, and compliance | Does tenant, user, role, region, and purpose context survive every boundary and fail closed? | Identity flow, RBAC matrix, data-flow/residency map, threat model, controls matrix, isolation test report, external compliance owners | [Safety and Governance](../../agentic-ai/06-safety-human-control-and-governance/06-safety-human-control-and-governance.ipynb), [Governance and Security](../../agentic-ai-system-design/11-governance-guardrails-and-security.md) |
| 5. Evaluation, SLA, capacity, and cost | Which quality slices and operational constraints gate a release, and which numbers are observed versus projected? | Versioned evaluation set, rubric, regression gates, workload assumptions, sensitivity model, quota/headroom plan, cost attribution, proposed SLO/SLA | [LLM Evaluation](../../genai/04-llm-evaluation/README.md), [LLM Gateway](../../genai/05-llm-gateway/01-llm-gateway.ipynb), [Production Scale](../../agentic-ai-system-design/12-production-scale-and-capacity.md) |
| 6. Rollout and change | Who can approve exposure, what does shadow traffic prove, which cohort enters canary, and what triggers rollback? | Baseline comparison, disagreement review, rollout plan, go/no-go record, rollback/compensation plan, communications, change log | [Agent Lifecycle](../../agentic-ai-system-design/02-agent-lifecycle-and-runtime.md), [Agent Evaluation](../../agentic-ai/05-agent-evaluation-and-observability/05-agent-evaluation-and-observability.ipynb) |
| 7. Operate and recover | How are faults contained, evidence preserved, customers informed, fixes revalidated, and service re-enabled? | Severity model, on-call ownership, incident timeline, containment record, redacted communication, revalidation evidence, postmortem actions | [Reliability and Recovery](../../agentic-ai/09-reliability-recovery-and-production-decisions/09-reliability-recovery-and-production-decisions.ipynb), [Saga Recovery](../../agentic-ai-system-design/10-recoverability-rollbacks-and-saga.md), [Observability](../../agentic-ai-system-design/08-observability-tracing-and-health.md) |
| 8. Handoff | Can customer operations run, change, monitor, escalate, and retire the system without relying on undocumented FDE knowledge? | Operational readiness review, dashboards, runbooks, support matrix, training record, policy/change process, acceptance sign-off, known limitations, backlog | [Handoff Standard](00-role-baseline-and-engagement-lifecycle.md#handoff-standard) |
| 9. Capstone | Can the learner preserve ambiguity, bound scope, and assemble one traceable recommendation without inventing execution or approval? | Indexed engagement package, ambiguity register, bidirectional traceability, rubric review, practicum gap plan | [Riverside Capstone](09-capstone/README.md) |

Do not advance because a calendar milestone arrived. Advance when the stage exit gate in the lifecycle document has evidence and a named approver.

## Claim classes

Every number or assurance in an engagement artifact must carry one of these labels. Keep the original class when a claim is copied into a status report, architecture review, or handoff pack.

| Label | Meaning | Minimum record | What it does not mean |
|---|---|---|---|
| `[Measured]` | Observed from an executed test, trace, benchmark, audit query, or sample | Environment, configuration/release, dataset or population, sample size, time window, method, result, artifact link, limitations | A local fixture result is representative of customer production |
| `[Modeled]` | Calculated or projected from explicit assumptions | Model/formula, input assumptions, source and date for each input, scenario range, sensitivity, confidence/limitations, owner for validation | Forecast latency, capacity, availability, savings, or cost was observed |
| `[Customer-validated]` | A named authorized customer representative accepted a workflow, criterion, result, control, or handoff artifact for a stated scope | Person or role, authority, artifact/version, scope, date, decision, conditions, expiry or revalidation trigger | The claim is technically measured, legally approved, compliant, or valid for another customer |

One statement can reference more than one class, but the records remain separate. For example, a latency result may be `[Measured]` in a customer test environment and its suitability may be `[Customer-validated]` by the workflow owner. Customer acceptance does not convert a model into a measurement. A customer cannot self-certify a legal or regulatory conclusion unless that authority is explicitly established.

Unlabeled numbers, adjectives such as "production-ready," and statements such as "compliant" or "secure" without scope and authority are unsupported assertions. They are not a fourth claim class.

## Competency and evidence matrix

| Competency | Observable performance | Minimum evidence | Claim classes expected |
|---|---|---|---|
| Discovery | Elicits users, decisions, baseline, exceptions, constraints, owners, and unknowns without leading with a preferred solution | Interview plan/notes, stakeholder map, current-state workflow, discovery backlog | Measured baseline where available; customer-validated workflow |
| Success criteria | Turns desired outcomes into sliced metrics, thresholds, test methods, non-goals, and acceptance owners | Acceptance matrix and golden workflow set | Measured baseline, modeled target where necessary, customer-validated criteria |
| Scope control | Selects the smallest intervention and records why more complex options are rejected | Option matrix, ADRs, non-goals, product-gap log | Measured feasibility evidence and modeled tradeoffs |
| Architecture translation | Maps model, retrieval, tool, policy, human, data, identity, state, and side-effect boundaries | Context/container diagrams, sequence/data flow, ADRs, threat assumptions | Modeled constraints plus measured spike results |
| Data onboarding | Defines source ownership, schemas, quality, provenance, ACL, refresh, deletion, and failure handling | Source inventory, mappings, quality/lineage reports, readiness verdict | Measured samples and pipeline checks; customer-validated source ownership |
| Identity and isolation | Propagates identity context and proves denied access across tenant, role, region, and purpose boundaries | Identity flow, RBAC matrix, isolation tests, audit records, controls gaps | Measured negative tests; customer/security validation of policy intent |
| Evaluation | Separates retrieval, generation, trajectory, policy, safety, latency, and cost failures with regression gates | Versioned dataset, rubric, evaluator calibration, slice report, gate decision | Measured scores with uncertainty; customer-validated critical slices |
| SLA, capacity, and cost | Converts workload and support assumptions into ranges, headroom, quotas, architecture, and escalation commitments | Workload model, sensitivity analysis, cost allocation, proposed SLO/SLA | Modeled scenarios until production measurement; customer-validated commitments |
| Rollout and change | Designs shadow/canary cohorts, disagreement review, approvals, ramp criteria, rollback, compensation, and communication | Rollout plan, gate record, rollback drill, change log | Measured shadow/canary evidence; customer-validated go/no-go |
| Incident response | Contains first, preserves evidence, classifies severity, communicates scope, revalidates, and obtains re-enablement approval | Incident simulation or record, timeline, communications, revalidation, postmortem | Measured incident facts; customer-validated re-enablement where applicable |
| Observability and operations | Connects service health, quality, policy, cost, and customer-specific ownership without leaking sensitive data | Signal catalog, dashboards, alerts, trace examples, on-call map, retention/redaction rules | Measured telemetry; modeled alert capacity; customer-validated operating thresholds |
| Handoff | Transfers operational, change, support, data, security, and retirement knowledge to named owners | Readiness review, runbooks, training, support matrix, sign-off, limitations/backlog | Customer-validated ownership and acceptance; measured drill results |

## Honest assessment rubric

Score each competency separately. Do not use an average to hide a critical gap.

| Score | Evidence standard |
|---:|---|
| 0 - Absent or unsafe | No artifact, an unsupported assertion, or a design that ignores a material authority, data, identity, or recovery boundary |
| 1 - Described | Can discuss the topic and name an approach, but evidence is incomplete, untraceable, or not testable |
| 2 - Constructed | Produces internally consistent artifacts and explicit assumptions using synthetic, replayed, or modeled evidence; important failure paths remain untested |
| 3 - Demonstrated | Produces reproducible measured evidence, exercises failure paths, records limitations, and passes independent review in a representative non-production environment |
| 4 - Customer-validated | Meets score 3 and has scoped acceptance from authorized customer owners in the target environment, including operating ownership and revalidation conditions |

Assessment outcomes for this authored route:

- **Not ready to enter a practicum:** any competency scores 0, any claim is knowingly misclassified, or the learner cannot preserve unresolved conflicts without inventing facts.
- **Ready to enter a supervised practicum:** every competency scores at least 2 in the static package, the capstone passes its rubric, and a reviewer confirms that discovery, architecture, and evidence boundaries are coherent.
- **Production execution not established by this route:** data integration, deployed identity enforcement, live rollout, incident command, and customer handoff require later observed evidence in an authorized environment.
- **Independent ownership is engagement-specific:** it can be considered only after supervised practice produces the required score 3 or 4 evidence for the exact workflow, environment, authorities, and operating conditions.

The verified local runs establish that the authored notebooks execute against their synthetic fixtures and exercise their documented decision paths. Because the generated outputs were cleared and no independent review artifact was retained, route completion still establishes score 2 through construction rather than score 3 demonstrated competence. A learner or practicum run may contribute to score 3 only when reproducible output, failure-path evidence, and independent review are retained. Synthetic work cannot establish customer validation. A score 4 is engagement-specific and expires when material workflow, data, policy, model, architecture, or ownership assumptions change.

## Supervised practicum and human skills

After the capstone, pair with an experienced FDE, customer engineer, or platform owner for a supervised real or quasi-real engagement. The practicum should include authorized data onboarding, deployed identity and isolation checks, a bounded release or rollback rehearsal, an incident simulation with approved communications, and an operator handoff drill. The supervisor observes the work, records feedback against the evidence matrix, and retains the final authority for production changes until the learner has demonstrated the relevant competency in context.

Use live coaching for the human work that a static notebook cannot assess reliably:

- run a discovery workshop without leading participants toward a preferred architecture;
- summarize disagreement neutrally and ask each authority to confirm or correct the record;
- state a fail-closed boundary in customer language, including the blocked outcome and the evidence needed to reopen it;
- escalate conflicting instructions without choosing the most convenient stakeholder;
- rehearse difficult updates with a mentor before customer delivery and request feedback on clarity, listening, authority mapping, and unsupported certainty;
- keep a reflection log with one communication decision, one piece of mentor feedback, and one changed behavior after each workshop, review, or drill.

The mentor is not a substitute approval authority. Customer, legal, security, commercial, incident, and operations decisions still belong to the named authorized owners.

## Completion package

A route review expects one traceable package, not a folder of disconnected templates:

1. Discovery report, current-state baseline, assumptions, risks, and acceptance criteria.
2. Architecture option analysis, selected design, ADRs, and customer-readable explanation.
3. Data/source inventory, onboarding contract, quality evidence, ACL/deletion tests, and lineage.
4. Identity flow, threat model, controls matrix, residency decisions, and isolation evidence.
5. Evaluation plan and results, workload/capacity/cost model, and proposed service commitments.
6. Shadow/canary evidence, rollout decision, rollback/compensation drill, and change record.
7. Incident exercise, communications, revalidation, and postmortem actions.
8. Operational readiness review, dashboards, runbooks, support boundaries, training, acceptance, known limitations, and iteration backlog.
9. Claim register linking every external statement to `[Measured]`, `[Modeled]`, or `[Customer-validated]` evidence.

All authored work follows the repository [Authoring Guide](../../../AUTHORING_GUIDE.md). Do not execute a notebook merely to claim completion; use its committed and reproducible evidence according to the track's setup and validation contract.
