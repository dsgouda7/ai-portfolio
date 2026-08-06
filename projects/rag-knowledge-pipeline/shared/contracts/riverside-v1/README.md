# Vendored Riverside vector record contract

Phase 3 vendors the minimum frozen Riverside contract resources required to
validate retrieval results without importing the Riverside implementation.

- Contract: `vector-index-record`
- Contract version: `1.0.0`
- JSON Schema dialect: Draft 2020-12
- Source: `projects/riverside-ai-platform/contracts/v1/`
- Vendored: 2026-08-05

`common.schema.json` and `vector-index-record.schema.json` are exact contract
resources at this pin. Update them only as an explicit contract migration; do
not follow a mutable Riverside source directory at runtime.
