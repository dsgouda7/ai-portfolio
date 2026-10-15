# Fixture Pointer

This chapter consumes the immutable shared fixture at
`../../shared/feedback-drift/`. Do not copy, rewrite, reorder, or enrich those
records in this directory.

The shared package contains:

- `production-feedback.jsonl`: 12 synthetic privacy-safe feedback traces;
- `production-feedback.schema.json`: the per-line structural contract;
- `EXPECTED_OUTCOMES.md`: independent arithmetic, cluster, review, and decision
  expectations.

Notebook-generated evaluation candidates belong under the chapter's ignored or
reviewed output location when the notebook is run later. They are derived
artifacts, not replacement fixtures.
