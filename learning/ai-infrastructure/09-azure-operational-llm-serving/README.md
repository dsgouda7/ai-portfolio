# Azure Operational LLM Serving

> **Evidence banner:** `LOCAL` mechanisms, `SUBSTITUTED` model work, `UNVALIDATED` Azure behavior.

This chapter turns Riverside House's fine-tuned adapter into a bounded service
contract, then exposes the failures that appear under concurrency, retries,
duplicate requests, release changes, and tail latency. The notebook uses a
loopback HTTP server and deterministic synthetic model work so every operational
path is measurable without loading a model or contacting a cloud service.

The local lab is Azure-shaped. It is **not an Azure emulator**. It does not
emulate Azure ML, API Management, Entra ID, managed identity, Azure Monitor,
autoscaling, quota, regional capacity, private networking, throttling, or cost.

## Start Here

1. Run `setup.ps1` on Windows or `setup.sh` on macOS/Linux.
2. Open `azure-operational-llm-serving.ipynb`.
3. Select `Python (azure-operational-serving .venv)`.
4. Run from the top. The notebook starts only an ephemeral loopback server.

The setup scripts are provided for later use. They were not run while this
chapter was authored.

## Failure-First Route

| Step | Failure exposed | Minimal operational fix |
|---|---|---|
| 1 | An adapter works in-process but has no stable service boundary | Frozen request, response, error, and release contracts |
| 2 | An HTTP process accepts traffic before validation and warm-up | Separate liveness from readiness |
| 3 | Concurrent work creates queue delay and timeout storms | Bounded admission and explicit retryable overload |
| 4 | Repeated requests duplicate expensive work and hide token use | Idempotency, single-flight caching, and token accounting |
| 5 | A slow or failed backend consumes the whole latency budget | Absolute deadlines, bounded retries, circuit breaking, explicit fallback |
| 6 | A changed release mutates behavior without rollback evidence | Immutable manifest, contract tests, blue/green routing |
| 7 | Average latency hides user-visible failures | TTFT, TPOT, p95, traces, and SLO gates |
| 8 | Local success is mistaken for Azure proof | Per-mechanism mapping plus mandatory cloud revalidation |

## Exact Conceptual Owners

This chapter composes mechanisms owned elsewhere; it does not reteach them:

- [Fine-tuning comparison and release evidence](../../genai/09-llm-finetuning/03-llm-finetuning-comparison-and-decision.ipynb)
- [Gateway routing, rate limiting, fallback, and caching](../../genai/12-llm-gateway/01-llm-gateway.ipynb)
- [Quantized artifacts and backend compatibility](../06-quantization/quantization-in-depth.ipynb)
- [Inference scheduling, TTFT, TPOT, and admission](../07-inference-systems/inference-systems.ipynb)
- [Frozen Riverside v1 contracts](../../../projects/riverside-ai-platform/contracts/README.md)
- [Contract fixture expectations](../../../projects/riverside-ai-platform/tests/fixtures/README.md)

## Files

| Path | Purpose |
|---|---|
| `azure-operational-llm-serving.ipynb` | Validated failure-first tutorial with outputs cleared |
| `requirements.txt` | JSON Schema validation dependencies only |
| `setup.ps1`, `setup.sh` | Isolated local environment and kernel setup |
| `fixtures/deployment.local.json` | Default local endpoint-adapter profile |
| `fixtures/deployment.azureml-apim.json` | Network-blocked Azure ML/APIM production shape |
| `fixtures/model-release-manifest.json` | Contract-valid shape with fake tutorial digests |
| `fixtures/privacy-safe-request.json` | Public style-guide prompt; no manuscript text |
| `fixtures/operational-slo-policy.json` | Illustrative local thresholds and cloud revalidation list |
| `scripts/serving_lab.py` | Local HTTP, lifecycle, resilience, metrics, and adapter mechanisms |

## Configuration Boundary

The application code calls `build_endpoint_adapter(config)`. With
`deployment.local.json`, the factory returns a loopback HTTP adapter. With
`deployment.azureml-apim.json`, it returns an Azure ML/APIM request planner that
refuses network traffic. The service policy is identical in both fixtures; only
the adapter block changes.

That switch demonstrates configuration ownership, not Azure compatibility.
Production still requires authenticated cloud smoke tests, APIM policy tests,
Azure ML deployment validation, load tests, monitoring verification, and cost
measurement.

## Privacy and Security Boundary

- No fixture contains manuscript text, a real customer identifier, a tenant ID,
  a user ID, a request ID, a subscription ID, a resource ID, or a credential.
- Metric labels use the frozen bounded-cardinality telemetry allowlist. Trace IDs
  remain trace context and are not metric labels.
- The Azure-shaped endpoint uses the reserved `.invalid` domain and workload
  identity placeholder text. No API-key field exists.
- The model release digests are synthetic repeated digits. They demonstrate
  shape only and do not attest to checkpoint integrity.

## Validation Status

Validated locally on 2026-08-05 with the unified `Python (FDE .venv)` kernel:
all 13 code cells executed without exceptions, and the notebook was cleared
afterward. The designed tail-latency experiment produced `HOLD_LOCAL_RELEASE`
and `HOLD_AZURE_PROMOTION` because the applicable p95 gates failed; those safety
decisions are expected results, not execution errors. No Azure or Databricks live
validation or cloud test was performed. Shell portability, production readiness,
and all Azure claims remain unvalidated until the documented checks are run in an
authorized environment.
