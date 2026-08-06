# Data Residency

## Decision state

No production region, geography, customer residency commitment, backup location,
or cross-region recovery design has been approved. The production parameter example
uses `uksouth`, while both Azure ML deployment definitions currently emit
`RIVERSIDE_REGION=eastus2` and target a separately named endpoint. These are
illustrative source profiles, not a coherent residency decision. Resolve them
before preview or deployment. The first release is modeled as single-region, but
that is an architectural constraint, not evidence that every service stores and
processes data only there.

The `region` field in the v1 records is policy and lineage metadata. It does not
configure Azure resources, prevent cross-region processing, or prove compliance.

## Data categories

| Category | Examples | Residency review |
|---|---|---|
| Customer content | manuscripts, parsed text, chunks, retrieved context, prompts, completions | storage, processing, backup, support/diagnostic paths |
| Security metadata | tenant, ACL, principals/groups, classification, deletion state | governed data plane, index filters, audit access |
| Model artifacts | base/adapter/tokenizer, containers, manifests | registry/storage replication and build locations |
| Evaluation data | test queries, labels, judge inputs/outputs, human review | governed dataset location and evaluator processing location |
| Operational data | logs, traces, metrics, alerts, incident evidence | diagnostic destinations, retention, export/support paths |
| Control-plane metadata | resource/deployment names, identities, configuration | each service's control-plane and support commitments |

## Service review matrix

Complete this matrix with exact service SKU/feature and official terms for the
selected date. Regional availability can change.

| Service boundary | Required decision | Live evidence |
|---|---|---|
| ADLS Gen2/Storage | primary/replication region, redundancy, soft delete/versioning, public access, key model | resource export, replication and access tests |
| Azure Databricks | workspace region/Geo, classic/serverless plane, Unity Catalog storage, Designated Services, cross-Geo setting | account/workspace configuration and job data-flow review |
| Production vector index | backend, region, replicas/backups, query/log processing | resource config and positive/negative residency tests |
| Azure ML | workspace/endpoint/storage/registry region, compute SKU, diagnostics, image/model stores | deployed resource graph and scoring data-flow review |
| API Management | region/tier, gateways, backups, diagnostics, developer portal if used | deployed config and request path review |
| Application Insights/Monitor | workspace region, ingestion/retention/export/archive | resource config and sampled telemetry review |
| Key Vault/Container Registry | region, replication, private access, retention | resource config and access tests |
| CI/CD and evaluation runners | runner/build region, artifact cache, logs, third-party services | pipeline provenance and processor inventory |

Azure Databricks has region-specific control/classic/serverless planes and
Geography-level Designated Services. Customer content may be processed across Geos
when cross-Geo processing is enabled. The production review must verify the
workspace's actual setting and every selected feature; the repository does not.

## Region selection gate

Select a region only when all of these are true:

1. Customer contracts, classification policy, legal/privacy, and threat model name
   allowed storage and processing locations.
2. Every required service, tier, feature, private-network option, and GPU/SKU is
   available with sufficient quota.
3. Data, evaluation, telemetry, backups, artifacts, and build/operations flows are
   mapped, including vendor support and diagnostic paths.
4. Cross-Geo/cross-region features are disabled unless explicitly approved.
5. Redundancy and backup settings match the approved boundary; a geo-redundant
   default is not accepted by accident.
6. Cost, latency, and single-region availability risk are accepted.
7. A reviewer records official documentation versions/date and live resource
   configuration evidence.

## Deployment controls

- Use one approved `region` in the environment profile and fail provisioning when
  a module attempts an unapproved location.
- Tag resources with environment, data classification, owner, and residency policy
  identifiers without placing customer identifiers in tags.
- Route diagnostics only to approved regional destinations.
- Deny unapproved public access, locations, replication modes, and cross-Geo
  processing through IaC/policy where feasible.
- Keep customer content and evaluation datasets out of source control and operator
  workstations.
- Treat movement to another region as a data migration with new approval,
  re-indexing/evaluation, deletion verification, and incident/rollback planning.

## Validation procedure

Static validation can check allowed-region parameters, policy definitions, closed
config, absence of local profiles, and the current cross-file region mismatch. It
cannot establish residency.

Live validation must retain:

- Azure resource inventory with resource IDs, regions, SKUs, and replication;
- Azure Policy assignments/compliance and reviewed exceptions;
- Databricks workspace/Geo and cross-Geo processing configuration;
- private endpoints, DNS, firewall/public-access, and diagnostic settings;
- storage/index/model/telemetry/backup destination inventory;
- representative request, indexing, evaluation, deletion, backup, and restore data
  flows;
- processor/subprocessor and support-access review required by policy;
- reviewer, timestamp, environment, source commit, and evidence digest.

## Recovery implications

The initial single-region design has no regional serving failover claim. Backups
may improve durability but can violate residency if their replication location is
unapproved. Recovery-time and recovery-point objectives remain unset until backup,
restore, index rebuild, and model redeployment are rehearsed inside the approved
boundary.

## Customer promise

Do not promise "data stays in region X" from a resource `location` field or this
document alone. A residency promise names data categories, storage and processing
scope, control-plane/diagnostic/support exceptions, replication/backup behavior,
third-party processors, deletion, and the evidence/review date.
