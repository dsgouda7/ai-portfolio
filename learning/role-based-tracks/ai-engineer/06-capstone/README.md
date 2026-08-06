# AI Engineer Capstone: Riverside Release Review

You are the release owner for a Riverside House AI application candidate. Your job is not to make every score green. Your job is to assemble enough traceable evidence for another engineer to decide whether the candidate should be promoted, held, rejected, or sent back for a narrower experiment.

This capstone integrates existing work through versioned contracts, stable IDs, digests, and links. Do not copy notebook functions, fixture rows, or platform implementation into this directory. A result may be locally measured, fixture-derived, modeled, or cloud-validated; those classes are not interchangeable.

## Start Here

1. Read the [candidate brief](candidate-brief.md).
2. Review the [expected deliverables](EXPECTED_DELIVERABLES.md) and create your evidence package from the files in [`templates/`](templates/).
3. Use the [capstone evidence workbook](capstone-evidence-workbook.ipynb) to record checkpoints and review decisions. It is Markdown-only and structurally validated; it does not contain executable analysis code or reproduce upstream analysis logic.
4. Read the [worked teaching example](worked-teaching-example.md) to see why source-only evidence ends in `hold`. It is deliberately non-passing and is not submission evidence.
5. Grade the final package with the [assessment rubric](assessment-rubric.md).

## Mission

Produce one reviewable release package that answers all of these questions:

- Which immutable release is under review, and what is its rollback target?
- Did the data candidate pass quality, lineage, rights, leakage, and safety gates?
- What failed in retrieval, and what failed after correct evidence reached generation?
- Did the prompt candidate improve the right slices without regressing a critical slice?
- What do local latency, TTFT, TPOT, retry, cache, throughput, and cost measurements support?
- How would the local components map to Azure, and which parts still require live validation?
- Does production feedback support a prompt, retrieval/index, guardrail, fine-tuning, or no-action decision?
- Which attractive claims remain unsupported?

The final decision may be **promote**, **hold**, or **reject**. A well-supported hold or rejection can earn full credit. A promotion supported by mislabeled, missing, or contradictory evidence cannot.

## Evidence Classes

Use exactly one class for every result and claim:

| Class | Meaning | Allowed conclusion |
|---|---|---|
| `IMPLEMENTED_SOURCE` | An inspectable contract, fixture, configuration, or source asset exists | The asset exists; it has not necessarily run |
| `STATIC_VALIDATION` | A non-cloud check ran and retained command, commit, inputs, output, and reviewer | The checked behavior passed in that static scope |
| `LOCAL_FIXTURE` | A local deterministic fixture or simulation produced retained output | The mechanism worked for the named fixture and environment |
| `LOCAL_MEASURED` | A local workload produced retained measurements | The result applies to the named local hardware, software, and workload |
| `MODELED` | A calculation or architecture assumption has not been measured | The value is a planning input, not observed behavior |
| `LIVE_AZURE` | An authorized Azure environment produced retained release-scoped evidence | The result applies only to the named subscription, region, environment, release, workload, and time window |
| `UNVALIDATED` | Required evidence is absent, stale, contradictory, or out of scope | The claim must remain in the unsupported-claims ledger |

### Evidence-Class Decision Guide

Use the first matching row. Classify the evidence-producing action, not the file type or the conclusion you hoped to reach.

| Question | Class | Required boundary |
|---|---|---|
| Does only inspectable source, configuration, a schema, a fixture, or an expected-outcome document exist? | `IMPLEMENTED_SOURCE` | Do not imply it ran |
| Did a non-cloud validator, compiler, schema check, or test run with command, commit, inputs, output, and reviewer retained? | `STATIC_VALIDATION` | Scope the claim to that exact check |
| Did an unchanged deterministic teaching fixture or simulation run and retain output? | `LOCAL_FIXTURE` | Name fixture version, digests, environment, and limits |
| Did a real local workload run on named hardware/software? | `LOCAL_MEASURED` | Name workload, warm-up, population, method, and machine |
| Is the value calculated or architected without observation? | `MODELED` | State assumptions and sensitivity; do not call it measured |
| Did an authorized Azure environment produce retained release-scoped evidence? | `LIVE_AZURE` | Name subscription, region, environment, release, workload, time, operator, and teardown/expiry |
| Is any required provenance missing, stale, contradictory, or outside the tested scope? | `UNVALIDATED` | Put the claim in the unsupported-claims ledger |

Common distinctions:

- Running a notebook over frozen synthetic rows is `LOCAL_FIXTURE`, not `LOCAL_MEASURED` merely because it ran on your laptop.
- Reading an expected-outcome document is `IMPLEMENTED_SOURCE`, not `LOCAL_FIXTURE`.
- A Bicep build or schema validation may be `STATIC_VALIDATION`; it is not `LIVE_AZURE`.
- An architecture mapping without a run is `MODELED` when it makes a design assumption, and `IMPLEMENTED_SOURCE` when it only points to existing source.
- When evidence has mixed classes, classify each claim separately and let the required gate use the weakest necessary class.

## Contract Map

| Capstone evidence | Owning learning contract | Platform boundary |
|---|---|---|
| Data-quality report | [Training Data Quality and Lineage](../01-training-data-quality-and-lineage/README.md) and [shared fixture contract](../shared/training-data/EXPECTED_OUTCOMES.md) | [Evaluation data-quality domain](../../../../projects/riverside-ai-platform/contracts/v1/evaluation-release-report.schema.json) |
| Prompt comparison | [Prompt Release and Experimentation](../02-prompt-release-and-experimentation/README.md) and [shared fixture contract](../shared/prompt-release/EXPECTED_OUTCOMES.md) | Evaluation `rollout_comparison` domain and immutable release evidence |
| Local operational SLO report | [Application Latency and Cost](../03-application-latency-and-cost/README.md) and [shared trace contract](../shared/latency-cost/EXPECTED_OUTCOMES.md) | Evaluation `operational_slos` and `cost` domains plus [bounded telemetry](../../../../projects/riverside-ai-platform/contracts/v1/telemetry-attributes.schema.json) |
| Release manifest | [Release Registry and Lineage](../04-release-registry-and-lineage/README.md) and [shared release contract](../shared/release-lineage/release-manifests.schema.json) | [Model release manifest](../../../../projects/riverside-ai-platform/contracts/v1/model-release-manifest.schema.json), [deployment metadata](../../../../projects/riverside-ai-platform/contracts/v1/deployment-metadata.schema.json), and [evaluation release report](../../../../projects/riverside-ai-platform/contracts/v1/evaluation-release-report.schema.json) |
| Drift iteration decision | [Production Feedback and Drift](../05-production-feedback-and-drift/README.md) and [shared feedback contract](../shared/feedback-drift/EXPECTED_OUTCOMES.md) | Versioned evaluation evidence, release lineage, and operational follow-up |
| Retrieval and generation evaluation | [RAG](../../../genai/04-rag/README.md) and [LLM Evaluation](../../../genai/05-llm-evaluation/README.md) | Retrieval, generation/citation, and safety/authorization domains |
| Azure mapping and unsupported claims | [Riverside architecture](../../../../projects/riverside-ai-platform/docs/architecture.md) and [promise-versus-evidence ledger](../../../../projects/riverside-ai-platform/docs/promise-vs-evidence.md) | Azure ML, APIM, managed identity, Databricks Direct Vector Access, Azure Monitor, and optional Foundry assets |

## Staged Checkpoints

| Checkpoint | Required review outcome | Stop condition |
|---|---|---|
| C0 - Scope and evidence inventory | One candidate release ID, baseline, rollback target, environment, workload, and evidence index | Mutable IDs, unknown provenance, or unlabeled evidence |
| C1 - Data gate | Data-quality report with source/candidate digests, issue ledger, curation actions, and gate decision | Blocking leakage, PII, rights, template, provenance, or disagreement remains |
| C2 - Retrieval and generation gate | Separate retrieval and generation reports, slices, thresholds, and a gold-context ablation | Failures cannot be localized or critical authorization cases are absent |
| C3 - Prompt gate | Paired baseline/candidate comparison with changed fields, critical slices, uncertainty, and rollback evidence | Critical slice regresses or comparison is confounded |
| C4 - Local operations gate | Reconciled local latency/cost ledger with named populations, SLO status, bottleneck, and next test | Totals do not reconcile or local evidence is described as Azure evidence |
| C5 - Release and Azure map | Complete release manifest plus service, identity, network, telemetry, rollout, and live-validation mapping | Artifact compatibility, report binding, or rollback graph fails |
| C6 - Drift decision and claim control | Reviewed feedback lineage, intervention decision, follow-up test, and unsupported-claims ledger | The selected action lacks evidence or an unsupported claim is presented as validated |
| C7 - Final defense | Promote, hold, or reject with every reason linked to retained evidence | Decision contradicts a failed non-compensating gate |

## Final Review Rule

The release decision is a logical AND across required gates. Aggregate improvements cannot compensate for a failed critical slice, incompatible artifact, unresolved rollback target, missing evidence domain, or known authorization failure.

Use the narrowest wording your evidence supports. Source presence is not execution, local execution is not Azure validation, and one successful release does not permanently validate the next one.

## Generalizing Beyond Riverside

After completing the guided candidate, repeat the review with one controlled change:

1. Choose a different immutable release, application, or domain. Do not rename `rel-riv-002` while retaining its evidence.
2. Create a new candidate brief and versioned input set with provenance, rights, privacy, schemas, and digests.
3. Re-declare critical slices, thresholds, SLOs, identity boundaries, and rollback policy before inspecting candidate results.
4. Map every logical artifact to the new system's owning contract. Keep Riverside platform files as references only unless the new system actually uses them.
5. Re-run the full non-compensating gate sequence and retain fresh evidence. Prior fixture outcomes may teach the method but cannot validate the new release.
6. Compare the two decisions: identify which controls generalized unchanged, which thresholds were domain-specific, and which new failure required a new test.

A generalization submission is graded by the same rubric. It may end in `hold` or `reject`; novelty never compensates for missing lineage or inflated evidence classes.
