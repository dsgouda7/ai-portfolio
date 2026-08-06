# Riverside staged load test

This directory contains the Locust plan and Azure Load Testing `v0.1`
configuration for five ordered stages: warm, steady, target, overload, and
recovery. The request fixture is synthetic and contains no customer content or
identifiers.

Azure Load Testing runs Locust in `LocalRunner` mode on every engine. The shape
therefore divides each aggregate stage target by `RIVERSIDE_ENGINE_INSTANCES`.
Keep that environment value equal to `engineInstances` in
`azure-load-test.yaml`; the default profile uses two engines and remains below
the recommended 500 users per engine.

Before a cloud run, materialize the target HTTPS origin, Entra `/.default`
scope, test ID, engine count, and managed-identity selection with
`../scripts/Materialize-AzureLoadTest.ps1`. The Locust engine obtains a token
with `ManagedIdentityCredential`; bearer-token secrets and Key Vault token
parameters are not accepted. The selected identity must already be assigned to
the Azure Load Testing resource and authorized for the target API. The Locust
script records stage-specific synthetic samplers named
`chat.completions.{total|ttft|tpot}.{stage}` and never retains response content.

Azure fail criteria provide the immediate client-side gate. After the run,
download every per-engine result CSV and collect the four engine-health fields
shown by Azure Load Testing: average CPU, average memory, average network bytes
per second, and maximum virtual users. Normalize that evidence with
`result_parser.py`. A run is not valid when an expected sampler or engine-health
record is missing, and it fails when any engine's average CPU or memory is at
least 75 percent.

The normalizer also enforces stage-specific thresholds and verifies that
recovery latency and error rate return close to the steady-state baseline. Its
output is deterministic JSON suitable for a later release-gate integration.

`../scripts/Start-AzureLoadTest.ps1` defaults to dry run and requires `-Apply` to
create/update the test and start one explicit lowercase run ID.
`../scripts/Export-AzureLoadTestEvidence.ps1` refuses a non-terminal run and an
existing evidence directory, downloads result/report/log archives, exports both
`LoadTestRunMetrics` and `EngineHealthMetrics`, and writes a SHA-256 evidence
inventory. A status other than `Passed` is retained but fails the script.

The service's preview metric response has not been normalized into the
`engine-health.json` shape consumed by `result_parser.py`; that transformation
remains a reviewed integration step until one real response shape is retained.
No load test, managed-identity token request, result download, metric query, or
parser run was executed while this workflow was authored. See
`../scripts/README.md` for exact inputs and limitations.
