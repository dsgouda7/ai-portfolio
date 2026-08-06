# Discovery and Success Criteria

This chapter starts with Riverside House's request to "help editors find answers and continue manuscripts faster." You will watch a solution-first workshop turn that sentence into an unsupported architecture, then rebuild discovery around users, workflow evidence, decision rights, constraints, and testable outcomes.

The output is the Stage 1 discovery pack from the [FDE engagement lifecycle](../00-role-baseline-and-engagement-lifecycle.md): `DSC-01` through `DSC-05`. It is an input to architecture translation, not permission to build or deploy.

## Learning contract

You will produce:

- a stakeholder, user, and authority map;
- a current-state workflow and bounded baseline;
- desired outcomes and explicit non-goals;
- a decision/action inventory;
- a data and integration inventory;
- a constraints register;
- sliced acceptance criteria with methods and owners;
- an assumptions, risks, unknowns, and discovery backlog register.

You will not select an architecture, invent missing customer decisions, use live customer data, or claim that local fixture work proves cloud, security, legal, compliance, or production behavior.

## Shared case

The notebook reads the frozen synthetic case at [../shared/README.md](../shared/README.md). `riverside-engagement-v1.json` is the source of truth. Preserve its stable IDs and evidence classes:

| Fixture class | How to use it |
|---|---|
| `measured_baseline` | Treat as a supplied synthetic observation with its population and limitations |
| `modeled_assumption` | Keep as a scenario input; do not relabel it as measured |
| `customer_claim` | Corroborate before using it as an acceptance commitment |
| `policy_constraint` | Enforce until the named authority changes it |
| `external_validation_required` | Assign an external validation owner and gate |
| `intentional_conflict` | Create a decision record; never choose a side silently |
| `unknown` | Put it in the backlog with an owner and needed-by milestone |

## Setup

The notebook uses Python's standard library plus `jsonschema` for local contract validation. It has no network or paid-service dependency.

PowerShell:

```powershell
.\setup.ps1
```

Bash:

```bash
./setup.sh
```

Pass `-SkipKernel` or `--skip-kernel` to install dependencies without registering and assigning the chapter kernel.

## Notebook

Open [discovery-and-success-criteria.ipynb](discovery-and-success-criteria.ipynb). The notebook executed successfully against the committed synthetic fixtures during route validation and produced the documented `BLOCKED` gate. Its outputs were then cleared, so all code cells remain committed with empty outputs and null execution counts. Run the cells in order for your own recorded exercise; kernels do not share state with other notebooks.

The code reads committed fixtures and builds in-memory teaching artifacts. It does not write into `shared/`, mutate source facts, contact external services, or claim customer validation.

## Reusable templates

The [templates](templates/README.md) directory contains blank, review-oriented forms for each artifact. The notebook demonstrates how to populate the same fields from Riverside evidence; the templates are the version you can reuse on another synthetic or authorized engagement.

Use the [notebook output record](templates/notebook-output-record.md) for a later authorized run. It defaults to `NOT RUN` and keeps observed results, supplied evidence classes, limitations, and external validation separate.

## Discovery exit gate

Do not advance to architecture translation until:

1. the workflow owner has confirmed current state and explicit non-goals;
2. each acceptance criterion has a metric or decision rule, slice, method, owner, evidence class, and status;
3. every conflict and unknown has an owner and needed-by gate;
4. consequential actions have proposal, approval, commit, and audit boundaries;
5. unsupported claims remain marked as claims, models, unknowns, or external validation work.

Next: `../02-architecture-translation/architecture-translation.ipynb` consumes these discovery artifacts when that chapter is available.
