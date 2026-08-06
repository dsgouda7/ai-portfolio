# Setup and Validation Boundary

## Authored state

The notebook was authored without execution. Do not infer that a blank output means a check passed. Every code cell has `execution_count: null` and an empty `outputs` array.

The notebook uses only the Python standard library. `requirements.txt` intentionally declares no third-party package.

## Prerequisites for a future learner run

- Python 3.11 or later
- JupyterLab, Jupyter Notebook, or VS Code notebook support supplied by the learner's environment
- A checkout that preserves the relative paths to `learning/role-based-tracks/fde/shared/` and `projects/riverside-ai-platform/`
- No Azure subscription, customer data, credentials, network access, or paid service

The notebook operates on in-memory synthetic records. It does not deploy, query, or validate Riverside platform services.

## Suggested future setup

From this directory, a learner may create an isolated environment and install only notebook tooling supplied by their organization. The module itself adds no runtime dependency.

Do not add credentials to the notebook, templates, environment variables printed by cells, or committed configuration. Do not replace synthetic records with customer data.

## What to do when it fails

| Symptom | Likely cause | First safe action |
|---|---|---|
| `Could not locate the ai-portfolio repository` | The notebook started outside the checkout or the expected folders moved | Open the checked-out repository root, confirm `AUTHORING_GUIDE.md` exists, then run the notebook from this module directory |
| A fixture path is missing | The checkout does not preserve the shared Riverside fixture layout | Restore the repository layout; do not copy fixture contents into the notebook or invent replacements |
| A cell depends on a name that is not defined | Cells ran out of order or the kernel contains stale state | Restart the kernel and run from Cell 1 in order |
| A local check reports `BLOCKED` or `REJECT` | The exercise intentionally found missing evidence, ownership, or drill results | Read the named blockers and repair the exercise record; do not change the validator to force a pass |
| An import requests a third-party package | The selected kernel or notebook has drifted from the authored standard-library version | Stop and inspect the diff before installing anything; this module should not require third-party runtime packages |
| An output appears to prove Riverside readiness | Local synthetic results are being interpreted too broadly | Label the result as local exercise evidence and name the live owner, environment, and test needed to replace it |

If setup still fails, record the exact cell number, error, working directory, Python version, and whether the repository paths exist. That gives the module owner enough evidence to distinguish environment trouble from an intentional readiness blocker.

## Future validation order

When execution is explicitly authorized in a later task:

1. inspect notebook JSON and cell metadata;
2. confirm all inputs are synthetic and local;
3. execute from a fresh kernel in cell order;
4. retain the source-level diff to prove execution did not rewrite source unexpectedly;
5. treat outputs as local exercise evidence only;
6. clear outputs before publishing unless the repository owner explicitly requests retained results.

## What a local run cannot prove

A successful local run cannot prove:

- live Azure behavior, regional availability, quotas, latency, capacity, cost, or retention;
- deployed identity, tenant isolation, ACL, deletion, private networking, or telemetry redaction;
- SLO attainment, alert delivery, rollback viability, compensation behavior, or re-enablement;
- operator competence without observed drills;
- customer acceptance, legal/compliance approval, or funded support coverage.

Those claims require scoped retained evidence and authorized reviewers under the FDE claim contract.
