# Capstone Templates

Copy these templates into a separate candidate evidence directory and replace every angle-bracket placeholder. Do not edit the templates in place to represent a submission.

The templates are capstone integration contracts, not replacements for the owning upstream or Riverside platform schemas. They reference evidence by stable path/URI, ID, version, and digest. They do not contain copied notebook code or fixture rows.

## Files

| Template | Purpose |
|---|---|
| `evidence-index.template.json` | Package identity, artifact inventory, reviewers, and contradictions |
| `release-manifest.template.json` | End-to-end release graph, gates, rollback, and decision |
| `data-quality-report.template.json` | Data findings, curation, digests, and gate |
| `retrieval-generation-evaluation.template.json` | Separate retrieval/generation evidence and ablation |
| `prompt-comparison.template.json` | Pinned bundle diff, paired/slice evidence, and release control |
| `local-operational-slo-report.template.json` | Local stage, SLO, throughput, retry, cache, and cost evidence |
| `azure-mapping.template.json` | Contract-to-service map and live-validation gaps |
| `drift-iteration-decision.template.json` | Seven drift lenses, reviewed feedback, and action selection |
| `unsupported-claims-ledger.template.json` | Evidence-safe wording for every unproven claim |
| `decision-record.template.md` | Final promote, hold, or reject decision with gate links and reviewers |

Every JSON file is intentionally valid JSON before replacement. Placeholder values are not evidence and may not satisfy the stricter platform schemas.
