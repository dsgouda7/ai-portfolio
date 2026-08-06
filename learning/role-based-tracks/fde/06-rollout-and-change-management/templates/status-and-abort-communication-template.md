# Status, Pause, Abort, and Rollback Communications

## Routine cohort status

**Subject:** `[cohort]` rollout status: `[GO / HOLD / PAUSED]`

- **Window:** `[start/end UTC]`
- **Current exposure:** `[users, tenants, regions, workflows, tools]`
- **Known:** `[measured facts with release and population]`
- **Unknown:** `[open evidence and owner]`
- **Customer impact:** `[observed impact or "none observed in the stated window"]`
- **Decision:** `[continue, hold, reduce, or advance]`
- **Next update / decision:** `[UTC]`
- **Owner:** `[business and technical owners]`

## Abort / rollback notice

**Subject:** Riverside Editorial Copilot cohort paused; workflow writes disabled

At `[UTC]`, `[trigger]` crossed the approved abort condition for `[cohort/release]`.
We stopped candidate expansion and routed new work to `[known-good release/manual
workflow]`. `[read-only/degraded capability]` remains available where approved.

**Known:**

- `[fact, source, and scope]`
- `[containment completed]`

**Unknown:**

- `[question and evidence owner]`

**Committed actions:** `[none identified / count and reconciliation status]`.
Traffic rollback does not reverse actions already committed in PageTurn. `[owner]`
is reconciling those records before any correction is approved.

The next evidence-backed update is due at `[UTC]`. Do not promise a recovery time
or root cause before the incident commander approves it.

## Re-enablement notice

State containment retained, root/contributing cause status, regression evidence,
observation window, residual risk, restored exposure, re-enablement authority,
temporary-control expiry, and the next health review. Re-enablement requires the
same or stronger evidence gate as the original release.
