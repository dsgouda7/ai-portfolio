# Training Data Expected Outcomes

The fixture contains 13 structurally valid rows. Structural validity is deliberate: the exercise is to show that schema-valid data can still be unsafe or unsuitable for training.

## Ground Truth

| Check | Expected result | Evidence IDs |
| --- | --- | --- |
| Exact duplicate groups | 1 group, 2 rows | `td-001`, `td-002` |
| Cross-split near-duplicate leakage | 1 train/validation cluster, 3 rows | `td-001`, `td-002`, `td-003` |
| Invalid chat templates | 2 rows | `td-005` has assistant before user; `td-006` has no assistant response |
| PII detections | 1 row, 2 findings | `td-007`: one reserved-domain email and one fictional `555` phone number |
| Missing provenance | 1 row | `td-008` |
| Unapproved or unknown rights | 1 row | `td-009` |
| Split-policy contamination | 1 row | `td-013` is in `train` but its source is `eval_reserved` |
| Preference-label disagreement | 1 comparison group, 2 rows | `cg-001`: `td-010` and `td-011` reverse chosen/rejected labels |
| Preference length shortcut | 1 row | `td-012` chooses the needlessly longer answer over an equivalent concise answer |

## Deterministic Counts

- Task mix: 10 SFT rows and 3 preference rows.
- Declared split mix: 12 train rows and 1 validation row.
- Slice mix: catalog 4, finance 3, editorial 2, rights 3, security 1. A notebook should report this imbalance rather than silently treating the data as uniform.
- Provenance present: 12 of 13 rows, or 92.3%.
- Approved provenance: 11 of 13 rows, or 84.6%. This denominator includes the missing-provenance row.
- Rows with at least one known blocking issue: 12 of 13 after comparison-group checks. The only clean row is `td-004`.

## Promotion Rule

A release candidate must fail its data gate when any of these conditions is true: cross-split duplicate leakage exists, template-invalid rows remain, PII findings are unredacted, provenance is missing or unapproved, an eval-reserved source appears in training, or a preference comparison group has unresolved disagreement. Under that rule, the uncurated fixture is **blocked**.

The fixture does not prescribe one automatic repair for every issue. A defensible curation report should identify which rows were removed, redacted, relabeled, or quarantined and then compute a new digest for the curated dataset.
