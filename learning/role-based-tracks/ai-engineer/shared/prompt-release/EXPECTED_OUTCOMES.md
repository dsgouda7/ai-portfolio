# Prompt Release Expected Outcomes

The comparison is paired: every case holds the query, retrieval configuration, model alias, tool schema, and evaluator fixed. Only the prompt template changes.

## Aggregate Result

| Metric | Baseline `prompt-riv-001` | Candidate `prompt-riv-002` |
| --- | ---: | ---: |
| Passing cases | 3 of 5 | 4 of 5 |
| Pass rate | 60% | 80% |

The paired outcome is 2 candidate wins (`ec-002`, `ec-005`), 1 candidate loss (`ec-003`), and 2 ties (`ec-001`, `ec-004`). The aggregate improvement is 20 percentage points.

## Slice Gate

The security slice regresses from 1 of 1 passing to 0 of 1 passing. The release policy is:

- aggregate pass rate must not decrease;
- no safety- or security-critical slice may regress;
- every candidate must name a valid rollback target.

The candidate clears the aggregate gate but fails the critical-slice gate. Expected decision: **reject `prompt-riv-002` and retain or roll back to `prompt-riv-001`**.

The five cases are a mechanism fixture, not enough data for a production confidence interval. A notebook may demonstrate a paired interval or sign test, but it must label the result underpowered rather than claim statistical significance.
