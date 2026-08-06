# DSC-02 Current Workflow and DSC-03 Baseline

## Workflow boundary

| Field | Value |
|---|---|
| Workflow ID | `<WF-...>` |
| Trigger | `<event that starts the workflow>` |
| End state | `<observable completion>` |
| Population | `<users/tasks/titles>` |
| Observation window | `<date range>` |
| Workflow owner | `<authorized owner>` |
| Evidence class | `<class>` |

## Current-state steps

| Step ID | Actor | Action/decision | System/source IDs | Input | Output | Time | Failure modes | Exception owner | Evidence source |
|---|---|---|---|---|---|---:|---|---|---|
| `<WFSTEP-...>` | `<role>` | `<current action>` | `<IDs>` | `<input>` | `<output>` | `<value/unit>` | `<observed failures>` | `<owner>` | `<trace/interview/sample>` |

## Baseline metrics

| Metric ID | Definition | Value/unit | Population | Window | Method/source | Slice coverage | Evidence class | Limitations | Revalidation trigger |
|---|---|---|---|---|---|---|---|---|---|
| `<MET-...>` | `<numerator/denominator or timer boundary>` | `<value>` | `<N and population>` | `<window>` | `<method>` | `<represented/missing slices>` | `<class>` | `<bias/exclusion>` | `<change/event>` |

## Baseline gaps

| Gap ID | Missing evidence | Collection method | Owner | Needed by | Decision blocked |
|---|---|---|---|---|---|
| `<UNK-...>` | `<missing baseline>` | `<sample/query/observation>` | `<owner>` | `<gate>` | `<criterion/architecture choice>` |

## Health check

- [ ] The baseline describes the current workflow, not a proposed system.
- [ ] Every number retains population, window, method, and limitations.
- [ ] Median, tail, error, and exception behavior are separated where material.
- [ ] Missing slices are visible rather than averaged away.
