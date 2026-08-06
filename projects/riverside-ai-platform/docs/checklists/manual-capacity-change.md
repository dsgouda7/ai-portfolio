# Manual Capacity Change Record

> **Status:** Procedure and blank evidence record only. No capacity change has
> been executed or approved by this document. Use until representative scale-out,
> cold-start, drain, dependency-limit, and recovery evidence supports an approved
> autoscale policy.

## Preconditions

- [ ] Incident/change ID, environment, UTC window, operator, and independent
  reviewer are recorded.
- [ ] Current release, index, APIM policy, Container Apps revision, Azure ML slot,
  instance/replica counts, quota, and regional service health are captured.
- [ ] Offered and achieved load, concurrency, queueing, latency, errors,
  rejections, retries, successful tokens/second, and telemetry freshness are
  captured for the same window.
- [ ] The bottleneck is identified. Scaling one tier will not exceed downstream
  Databricks, Azure ML, APIM, network, subnet, or telemetry limits.
- [ ] Incremental hourly/window cost and quota headroom are approved.
- [ ] Known-good values, rollback owner, observation window, and abort thresholds
  are recorded.

## Procedure

1. Freeze rollout, index promotion, and unrelated capacity/config changes.
2. Change one capacity control at a time through reviewed source and the approved
   deployment path. Do not make an unrecorded portal edit.
3. Keep the configured bounds in view: production source currently models
   Container Apps minimum `2`, maximum `10`, and HTTP concurrency target `20` per
   replica. These are review bounds, not proven capacity.
4. Confirm desired and actual state, readiness, dependency connection pressure,
   and absence of retry amplification.
5. Observe through the approved window. Compare the same workload and signals
   captured in preconditions; do not use request rate alone.
6. Roll back if safety, authorization, deadline, error, quality, cost, quota, or
   dependency abort criteria are crossed.
7. Close or extend the temporary change explicitly. Update the capacity model only
   from retained representative evidence.

## Evidence record

| Field | Value |
|---|---|
| Change/incident ID | `<id>` |
| Environment/region | `<environment>/<region>` |
| Workload version and window | `<version>; <UTC start/end>` |
| Bottleneck hypothesis and evidence | `<content-free references>` |
| Before values | `<tier and values>` |
| Approved target values | `<tier and values>` |
| Quota and incremental cost approval | `<reference>` |
| Abort thresholds | `<approved thresholds>` |
| After state and observation | `<evidence reference>` |
| Rollback result, if invoked | `<evidence reference or not invoked>` |
| Operator/reviewer | `<names or approved identities>` |
| Decision and expiry | `<retain/revert/extend>; <UTC>` |
