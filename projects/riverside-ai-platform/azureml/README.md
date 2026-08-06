# Azure ML serving assets

These files define an Azure ML managed online endpoint for an immutable, registered
SmolLM2 base-plus-LoRA model package. The package mounted at `AZUREML_MODEL_DIR`
must contain the release manifest, all `repo://` paths referenced by that manifest,
the adjacent `adapter_config.json`, and a pinned `base-model/` snapshot.

## Materialization and registration

The committed endpoint, environment, deployment, and traffic YAML files are
templates, not direct `az ml` inputs. Their required endpoint, environment,
region, slot, model/environment version, image digest, code path, package digest,
index version, deadline, and timestamp values are supplied through
`../scripts/Materialize-AzureML.ps1`. A schema-valid uppercase sentinel is used
where Azure ML's editor schema rejects underscore-prefixed placeholders.

`Materialize-AzureML.ps1` verifies the approved SHA-256 digests of the code,
model packages, conda file, and environment template. It also verifies every
adapter, tokenizer, training-manifest, and evaluation-report digest declared by
each model release manifest. The base image must use an `@sha256:` reference.
Generated deployment tags bind release-manifest, model-package, code-package,
environment-image, conda, and environment-template digests to the exact model
and environment asset names and versions. It emits materialized YAML plus a
digest manifest.

`../scripts/Register-AzureMLAssets.ps1` registers immutable model and environment
versions and refuses an existing version with different digest tags. In apply
mode, it also validates the exact returned resource ID and name/version fields,
then writes `registration-manifest.json` as registration evidence.
`../scripts/Deploy-AzureML.ps1` creates or updates the endpoint/deployments without
changing traffic. `../scripts/Set-AzureMLTraffic.ps1` changes and reads back an
explicit blue/green allocation. All mutating scripts default to dry run, require
`-Apply`, verify the explicit tenant/subscription against the current Azure CLI
identity, and accept no secret.

The Azure ML `request_timeout_ms: 180000` value is intentionally retained as the
outer container timeout. It is not the application deadline. The materializer
requires a lower 1-120 second application deadline from the selected profile and
records it in deployment tags; APIM and the application must enforce their own
lower budgets.

See `../scripts/README.md` for the JSON input contract and evidence limits. None
of these scripts or generated assets was executed or cloud-validated while this
workflow was authored.

Create the endpoint before either deployment. Apply a rollout profile only after
both blue and green exist and the candidate has passed its smoke, load, canary, and
observation-window gates. Endpoint creation intentionally has no `traffic` field
because Azure ML rejects traffic assignments before deployments exist.

The endpoint uses Microsoft Entra authentication (`aad_token`), a system-assigned
identity, disabled public network access, Application Insights, and Azure ML probe
settings. Workspace managed-network, private DNS, RBAC, diagnostic settings to Log
Analytics, autoscale rules, and alert rules remain infrastructure responsibilities.

The scoring entry point uses the preview raw-HTTP compatibility surface to preserve
HTTP status codes. Streaming is deliberately classified as `buffered-sse`:
generation completes before SSE events are emitted, so this interface does not
claim token-by-token delivery or a reduced time to first token. Responses include
`X-Riverside-Streaming-Mode: buffered-sse` and the advisory
`X-Accel-Buffering: no`, but Azure ML, gateways, and clients can still buffer the
body. Promote streaming only after an end-to-end timing test proves the required
flush behavior. Event ordering and chunking are deterministic; generation is
deterministic only for requests such as the samples with `temperature: 0`. No
credential, token, key, or connection string belongs in these files or in the
registered model package.
