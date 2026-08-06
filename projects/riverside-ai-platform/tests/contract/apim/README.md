# APIM Static Contract Tests

This directory is the no-cloud inspection toolkit owned by the Riverside
project. Its assertions inspect the project APIM assets and frozen v1 contracts
without changing component source.

## Contents

| File | Purpose |
|---|---|
| `policy_static.py` | Standard-library loaders and inspectors for JSON, XML, named values, fragment composition, URLs, and token metric dimensions |
| `policy-cases.json` | Behavior-oriented scenarios with expected status/error intent and required static evidence |
| `test_policy_assets.py` | `unittest` and pytest-compatible assertions over the static assets |

The cases cover Entra authentication, trusted tenant-tier derivation, request
and token limits, managed identity, pool routing, bounded retry and
`Retry-After`, stream replay prevention, circuit breaking, correlation-safe
errors, extension defaults, cache partitioning, and telemetry dimensions.

## What these tests establish

- JSON and XML are parseable.
- Every policy fragment has a deployment ID and is composed in the intended
  section and order.
- Every policy named-value reference is declared.
- OpenAPI preserves the stable route, model alias, and v1 token bounds.
- Backend resources contain pool priority/weight and breaker configuration.
- Token metric dimensions are allowlisted and stay within APIM's five-custom-
  dimension limit.
- No live URL or secret-like named value is committed.

## What remains external

These tests do not execute APIM's C# policy-expression subset, validate Bicep
against the Azure resource provider, publish fragments, exercise Entra tokens,
trip a distributed circuit breaker, or prove managed-identity RBAC. Those checks
belong in an authorized non-production APIM validation stage. No such command,
test, or deployment was run while creating these assets.
