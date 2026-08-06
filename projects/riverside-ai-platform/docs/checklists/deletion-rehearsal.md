# Deletion Propagation Rehearsal

> **Status:** Authorized non-production rehearsal procedure and blank evidence
> record. It has not been run. Use synthetic, non-customer content unless a
> separately approved test authorizes otherwise.

## Prepare

- [ ] Record exercise/change ID, environment, region, owner, reviewer, retention
  policy, target propagation objective, and legal-hold decision.
- [ ] Create a uniquely identifiable synthetic record with source, parsed, chunk,
  vector, index, cache, export, replica, backup, and restore lineage documented.
- [ ] Prove the authorized test principal can retrieve it and denied tenant,
  principal, and group paths cannot.
- [ ] Capture relevant immutable versions, timestamps, queries, jobs, and backup
  points without recording content in telemetry labels.

## Exercise

1. Submit the deletion through the approved control plane; record request ID and
   UTC acceptance time.
2. Verify state progresses through the contract's deletion states without losing
   tenant, ACL, region, classification, or lineage fields.
3. At the approved interval, test source/current storage, parsed/chunk stores,
   active index and replicas, caches, serving retrieval, analytics/export copies,
   and queued/retry paths.
4. Run positive non-deleted control queries and negative deleted-record queries
   through the deployed application. A deleted record must not be retrievable or
   cited.
5. Verify backup/versioning handling: deletion tombstone or suppression state is
   retained for every recoverable copy according to approved policy.
6. In an isolated authorized restore target, restore a pre-deletion backup and
   prove the deletion ledger is re-applied before any serving or export access.
7. Confirm diagnostic/audit records retain only approved metadata and follow their
   separate retention/legal-hold policy.

Stop and declare an incident if deleted content remains available beyond the
approved objective, a restore can bypass the deletion ledger, or another tenant's
record is affected.

## Evidence record

| Field | Value |
|---|---|
| Exercise ID/environment/region | `<values>` |
| Synthetic record and lineage references | `<content-free IDs>` |
| Deletion accepted UTC | `<timestamp>` |
| Objective and observed propagation | `<duration>` |
| Store/index/cache/export checks | `<evidence references>` |
| Positive and negative serving checks | `<evidence references>` |
| Backup selected and isolated restore result | `<evidence references>` |
| Legal hold/retention decision | `<reference>` |
| Defects, incident, and corrective owners | `<references>` |
| Operator/reviewer and closure decision | `<values>` |
