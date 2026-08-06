# Setup and Validation Boundary

## Authored state

The notebook was authored without execution. Do not infer that a blank output means a check passed. Every code cell has `execution_count: null` and an empty `outputs` array.

The notebook uses only the Python standard library. `requirements.txt` intentionally declares no third-party package.

## Prerequisites for a future learner run

- Python 3.11 or later
- JupyterLab, Jupyter Notebook, or VS Code notebook support supplied by the learner's environment
- A checkout that preserves the relative paths to `learning/fde/shared/` and `projects/riverside-ai-platform/`
- No Azure subscription, customer data, credentials, network access, or paid service

The notebook operates on in-memory synthetic records. It does not deploy, query, or validate Riverside platform services.

## Suggested future setup

From this directory, a learner may create an isolated environment and install only notebook tooling supplied by their organization. The module itself adds no runtime dependency.

Do not add credentials to the notebook, templates, environment variables printed by cells, or committed configuration. Do not replace synthetic records with customer data.

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
