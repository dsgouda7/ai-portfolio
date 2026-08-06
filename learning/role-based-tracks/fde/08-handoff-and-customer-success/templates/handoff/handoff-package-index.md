# Handoff Package Index Template

## The document-dump failure

The following is not a handoff package:

| File | Present |
|---|---|
| `dashboard.pdf` | yes |
| `runbook.docx` | yes |
| `support.xlsx` | yes |
| `training-slides.pptx` | yes |
| `acceptance.pdf` | yes |

It cannot answer who owns the next alert, which action is safe, what evidence proves the procedure, what remains unsupported, or when the decision expires. Artifact count is not capability transfer.

## Package control

| Field | Value |
|---|---|
| Artifact ID | `HOF-PKG-01` |
| Package version | `<version>` |
| Engagement/release scope | `<workflow, users, tenants, regions, releases, indexes, policies>` |
| Package owner | `<owner>` |
| Receiving operations owner | `<owner>` |
| Acceptance authority | `<authorized role>` |
| Status | `draft / review / accepted with conditions / accepted / superseded` |
| Evidence freeze date | `<UTC date>` |
| Revalidate on | Material workflow, data, identity, policy, model, architecture, support, or ownership change |

## Artifact manifest

| Artifact ID | Capability transferred | Version/status | Accountable owner | Receiving owner | Decision/action enabled | Evidence reference and class | Drill/reference | Limitations/exclusions | Revalidation trigger |
|---|---|---|---|---|---|---|---|---|---|
| `HOF-01` | Readiness decision | `<version/status>` | `<owner>` | `<owner>` | Accept, condition, or block exposure | `<reference; class>` | `<review/drill>` | `<limits>` | `<trigger>` |
| `OPS-DASH-01` | Interpret health and choose boundary | `<version/status>` | `<owner>` | `<owner>` | Continue, contain, stop ramp, or investigate | `<reference; class>` | `<dashboard drill>` | `<limits>` | `<trigger>` |
| `OPS-01` | Route and respond to alerts | `<version/status>` | `<owner>` | `<owner>` | First safe action and escalation | `<reference; class>` | `<delivery drill>` | `<limits>` | `<trigger>` |
| `HOF-02` | Execute runbooks | `<version/status>` | `<owner>` | `<owner>` | Contain, roll back, compensate, re-enable | `<reference; class>` | `<timed drill>` | `<limits>` | `<trigger>` |
| `HOF-03` | Apply support boundary | `<version/status>` | `<owner>` | `<owner>` | Severity, response, communication, vendor escalation | `<reference; class>` | `<tabletop>` | `<limits>` | `<trigger>` |
| `HOF-04` | Demonstrate operator competence | `<version/status>` | `<owner>` | `<owner>` | Operate independently | `<Measured drill records>` | `<drills>` | `<limits>` | `<trigger>` |
| `HOF-05` | Accept scope and limitations | `<version/status>` | `<owner>` | `<owner>` | Accept, condition, or reject ownership | `<Customer-validated record>` | `<gate refs>` | `<limits>` | `<trigger>` |

## Capability coverage

| Capability | Named owner after hypercare | Artifact | Evidence current | Drill passed | FDE-only access removed | Status/blocker |
|---|---|---|---|---|---|---|
| Shift start/handoff and routine evidence | `<owner>` | `<ID>` | `yes/no` | `yes/no` | `yes/no` | `<status>` |
| Service/release/index/policy identification | `<owner>` | `<ID>` | `yes/no` | `yes/no` | `yes/no` | `<status>` |
| Alert triage and escalation | `<owner>` | `<ID>` | `yes/no` | `yes/no` | `yes/no` | `<status>` |
| Model/index/policy rollback | `<owner>` | `<ID>` | `yes/no` | `yes/no` | `yes/no` | `<status>` |
| Action reconciliation/compensation | `<owner>` | `<ID>` | `yes/no` | `yes/no` | `yes/no` | `<status>` |
| Data correction/reindex/deletion | `<owner>` | `<ID>` | `yes/no` | `yes/no` | `yes/no` | `<status>` |
| Identity containment/re-enablement | `<owner>` | `<ID>` | `yes/no` | `yes/no` | `yes/no` | `<status>` |
| Policy and evaluation threshold change | `<owner>` | `<ID>` | `yes/no` | `yes/no` | `yes/no` | `<status>` |
| Cost/capacity exception | `<owner>` | `<ID>` | `yes/no` | `yes/no` | `yes/no` | `<status>` |
| Recurring health and retirement review | `<owner>` | `<ID>` | `yes/no` | `yes/no` | `yes/no` | `<status>` |

## Package checks

- [ ] Every artifact enables a named decision or action.
- [ ] Every critical capability has a receiving owner and backup.
- [ ] Evidence class is preserved when copied into the package.
- [ ] Procedures link to measured drills; planned drills are not marked passed.
- [ ] Open limitations and unknowns appear in the manifest and acceptance record.
- [ ] No credential, customer content, or sensitive raw trace is embedded.
- [ ] Superseded artifacts remain traceable and are not silently overwritten.
- [ ] The package has no FDE-only access, owner, or undocumented step after exit.
