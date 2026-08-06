# Prompt Release and Experimentation

> **Evidence banner:** `LOCAL FIXTURES`, `DETERMINISTIC ANALYSIS`, `EXECUTED SUCCESSFULLY`, `OUTPUTS CLEARED`.

Riverside House has a prompt candidate that looks better in aggregate: four of
five cases pass instead of three. It is still unsafe to release. The one
security case regresses from pass to fail, and five paired cases cannot support
a confident population-level claim.

This chapter turns that failure into a release workflow. You version the prompt,
tool schema, retrieval configuration, model alias, and evaluator as one pinned
application bundle; compare candidate and baseline on the same cases; inspect
critical slices; quantify finite-sample uncertainty; design shadow and A/B
assignment; and require canary and rollback evidence before promotion.

## Start Here

1. Complete the [RAG evaluation](../../genai/04-rag/05-rag-evaluation.ipynb),
   [LLM evaluation](../../genai/05-llm-evaluation/README.md), and
   [LLM gateway](../../genai/06-llm-gateway/06-llm-gateway.ipynb) prerequisites,
   or attach equivalent evidence.
2. Run `setup.ps1` on Windows or `setup.sh` on macOS/Linux when execution is
   authorized.
3. Open `prompt-release-and-experimentation.ipynb` and select
   `Python (AI Engineer Prompt Release .venv)`.
4. Run from the top. The default path reads local fixtures only and makes no
   network, provider SDK, model, or LLM call.

The chapter setup completed as part of the unified FDE validation run.

## Failure-First Route

| Step | Failure exposed | Minimal release control |
|---|---|---|
| 1 | A changed prompt string cannot reproduce application behavior | Immutable bundle of prompt, tools, retrieval, model, and evaluator pins |
| 2 | Aggregate pass rate improves from 60% to 80% | Slice-aware regression policy |
| 3 | The security slice falls from 100% to 0% | Non-compensating critical-slice gate |
| 4 | A 20-point gain sounds conclusive on five cases | Paired deltas plus explicitly underpowered uncertainty |
| 5 | Different traffic confounds an online comparison | Stable exposure assignment and mirrored shadow replay |
| 6 | A candidate reaches users without exit evidence | Canary entry/exit gates, stop conditions, and retained rollback target |

## Expected Decision

`prompt-riv-002` must be rejected and `prompt-riv-001` retained as the active
release. The candidate wins two paired cases, loses one, ties two, and improves
aggregate pass rate by 20 percentage points. None of that compensates for the
security regression.

The chapter deliberately separates three conclusions:

- **Fixture conclusion:** the candidate fails Riverside's stated release policy.
- **Mechanism conclusion:** paired and slice-aware evaluation catches a failure
  hidden by the aggregate.
- **Population conclusion:** unavailable; five cases are underpowered and are
  not representative production evidence.

## Files

| Path | Purpose |
|---|---|
| `prompt-release-and-experimentation.ipynb` | Complete failure-first tutorial, executed successfully and cleared for reuse |
| `requirements.txt` | Small local analysis and notebook dependency set |
| `setup.ps1`, `setup.sh` | Isolated environment and kernel registration |
| `../shared/prompt-release/` | Immutable Riverside release and paired-evaluation fixtures |

## Fixture Boundary

The notebook reads the shared Riverside fixtures in place. It never copies,
rewrites, relabels, or enriches them. IDs are used as joins; row order is not an
identity. Expected outcomes are derived from fixture fields and checked against
the fixture contract.

Any later experiment that needs more cases must create a new versioned fixture
set outside the shared v1 directory. Editing the five rows to make a release
pass would erase the regression evidence the chapter exists to teach.

## Completion Evidence

You have finished this chapter when you have:

- retained the verified fixture version and all input digests;
- produced immutable baseline and candidate bundle digests with a field-level diff;
- retained paired, aggregate, critical-slice, and uncertainty results with populations shown;
- recorded the candidate rejection and valid rollback target without deleting the failed security case;
- labeled shadow, A/B, and canary material as design or local simulation unless a real integration ran;
- linked the prompt comparison into a capstone evidence index as `LOCAL_FIXTURE` with its underpowered-sample limitation.

## Honest Limits

- The responses are pre-recorded deterministic fixtures, not fresh generations.
- The pass labels are supplied ground truth; evaluator validity is not measured
  here.
- The paired bootstrap and exact sign test demonstrate mechanics on five cases;
  they do not establish statistical significance or production effect size.
- Shadow and A/B sections build assignment and evidence logic; they do not run a
  live service, expose users, or observe behavior change.
- Canary logic evaluates entry and rollback contracts; it does not prove
  deployment health, model capacity, latency, cost, or cloud rollback.
- Prompt rollback stops new exposure. It does not undo side effects already
  committed by a tool or agent run.

## Validation Status

The notebook executed successfully in the unified FDE environment. Outputs were
then cleared and execution counts reset for reuse. This validates the local
deterministic fixture workflow only; no provider, production, or cloud behavior
was validated.
