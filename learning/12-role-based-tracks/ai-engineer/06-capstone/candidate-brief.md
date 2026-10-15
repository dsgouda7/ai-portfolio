# Candidate Brief: Riverside Release Candidate

## Your Role

You are the AI Engineer accountable for the release decision on one Riverside House application candidate. The system answers editorial and policy questions over authorized content. It combines a versioned model/adaptation artifact, prompt bundle, retrieval index, evaluator suite, gateway behavior, and operational configuration.

Your reviewers are the release approver, security owner, data owner, and operations owner. They need an inspectable decision package, not a notebook tour.

## The Incident

Riverside has evidence spread across learning artifacts and platform contracts:

- a training-data candidate may be structurally valid while containing leakage, invalid chat templates, PII, missing provenance, unknown rights, contamination, preference disagreement, and thin slices;
- a prompt candidate may improve aggregate quality while regressing a security slice;
- local request traces may show acceptable average latency while hiding tail latency, retry spend, cache effects, and paid failure;
- release records may contain a passing evaluator but an incompatible base/adapter pair or invalid rollback edge;
- feedback may show retrieval, policy, latency, and cost failures at the same time.

You must decide what the evidence supports now and what remains unvalidated.

## Constraints

- Integrate through links, IDs, digests, and versioned contracts. Do not copy upstream analysis code or fixture rows.
- Keep the candidate package privacy-safe. Do not add prompts, completions, manuscript text, secrets, tokens, customer identifiers, or credentials.
- Separate retrieval from generation evaluation. Include at least one gold-context or equivalent ablation.
- Predeclare thresholds and critical slices before presenting candidate results.
- Keep critical safety and authorization gates non-compensating.
- Label every value and claim with an evidence class from the capstone README.
- Azure architecture mapping is required. Azure deployment is not required and must not be implied.
- Record contradiction and missing evidence. Do not repair an upstream artifact silently in the capstone.
- A `hold` or `reject` decision is valid and can earn full credit.

## Required Decision

Choose exactly one final state:

| Decision | Use when |
|---|---|
| `promote` | Every required gate passes with evidence strong enough for the named next rollout stage |
| `hold` | Evidence is incomplete, underpowered, stale, contradictory, or awaiting review |
| `reject` | A release boundary fails and the current candidate should not advance |

Your decision must identify:

1. the candidate, baseline, and rollback target;
2. every failed or unproven gate;
3. the next discriminating test for each unresolved issue;
4. the smallest justified intervention;
5. the claims you refuse to make.

## Starting Sources

Use the five AI Engineer operational modules under the parent directory, their shared fixtures, the GenAI RAG/evaluation material, and the Riverside platform contracts/docs. The [expected deliverables](EXPECTED_DELIVERABLES.md) lists the exact handoff files.

Do not assume a README promise means its notebook currently emits the promised artifact. If an upstream source stops short, record the limitation and provide a contract-compatible candidate artifact or mark the gate `hold`.

## Review Format

Submit the package described in [EXPECTED_DELIVERABLES.md](EXPECTED_DELIVERABLES.md). During review, be prepared to trace any claim backward from the final decision to:

- a stable artifact ID or path;
- a digest or immutable version;
- the producing environment and workload;
- the exact threshold and observed value;
- an evidence class;
- limitations and reviewer ownership.

You are not graded on the number of green checks. You are graded on whether another engineer can trust your decision.

## Optional Generalization Pass

Once the guided review is complete, apply the package to a different immutable release or a different application/domain. Create a new brief and evidence package; do not reuse Riverside IDs, thresholds, fixture results, or Azure mappings as though they were observations for the new system. Re-declare slices, policies, workloads, owners, and rollout stage before evaluating the candidate, then explain which controls transferred and which had to change.
