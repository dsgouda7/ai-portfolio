# Architecture Decision Records

| ADR | Status | Decision |
|---|---|---|
| [ADR-0001](0001-versioned-contracts-are-integration-boundaries.md) | Accepted | Integrate through versioned contracts, not implementation imports. |
| [ADR-0002](0002-azure-ml-managed-online-endpoints.md) | Accepted | Use Azure ML managed online endpoints as the default custom-artifact backend. |
| [ADR-0003](0003-apim-and-managed-identity.md) | Accepted | Put APIM at the application boundary and use Entra plus managed identity. |
| [ADR-0004](0004-evidence-gated-blue-green-rollout.md) | Accepted | Promote immutable releases through evidence-gated blue/green rollout. |
| [ADR-0005](0005-single-region-first-release.md) | Accepted | Start with one approved region; do not claim multi-region resilience. |
| [ADR-0006](0006-content-free-operational-telemetry.md) | Accepted | Exclude customer content and high-cardinality identifiers from operational metrics. |
| [ADR-0007](0007-databricks-direct-vector-index.md) | Accepted | Use the Databricks Direct Vector Access adapter as the initial production index target. |

An accepted ADR records a design decision, not implementation or Azure validation.
If implementation diverges, supersede the ADR; do not quietly edit its outcome.
