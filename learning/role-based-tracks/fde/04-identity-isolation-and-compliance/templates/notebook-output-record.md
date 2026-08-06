# Identity and Isolation Notebook Output Record

## Run status

| Field | Value |
|---|---|
| Execution status | `NOT RUN` |
| Evidence class | `[Modeled]` expected decisions plus `[Local-static]` source inspection |
| Engagement fixture | `RIV-FDE-1.0.0` |
| Scenario fixture | `RIV-FDE-04-LOCAL-1.0.0` |
| Environment | Not recorded |
| Source commit | Not recorded |
| Executed by / time | Not recorded |
| Release verdict | `NOT EVALUATED` |

## Boundary observations

| Boundary | Expected proof | Observed result | Evidence reference | External gap |
|---|---|---|---|---|
| Gateway | Active identity plus seven required fields; requested authority only shrinks | Not recorded | Pending | Token and IdP enforcement |
| Retrieval | Mandatory pre-filter and independent post-check | Not recorded | Pending | Index, cache, and adapter negatives |
| Tool | Role, purpose, prohibited action, and exact approval checked outside model | Not recorded | Pending | Service identity and data-plane RBAC |
| Audit | Allow and deny events contain decision facts but no customer content or credentials | Not recorded | Pending | Destination RBAC, retention, legal hold |
| Response | Public envelope omits internal roles, filters, and token claims | Not recorded | Pending | Deployed API and telemetry sampling |

## Scenario result

| Metric | Value |
|---|---|
| Scenarios evaluated | Not recorded |
| Expected matches | Not recorded |
| Mismatches | Not recorded |
| False allows | Not recorded |
| False denies | Not recorded |
| Missing audit events | Not recorded |
| Local verdict | `NOT EVALUATED` |

## External validation still required

Record authorized evidence references for Azure RBAC, managed identity, private networking, index enforcement, cache partitioning, IdP revocation, processing and backup regions, telemetry, deletion, retention, privacy, legal basis, incident response, and customer security approval. Never paste credentials, tokens, or customer content here.
