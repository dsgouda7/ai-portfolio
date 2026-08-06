# Data and Integration Inventory

| Source ID | System | Purpose/use case | Owner | Type/format | Classification | Tenant/region | ACL model | Update/freshness | Version/lineage | Retention/deletion | Read/write scope | Known failures | Evidence class | Approval/status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `<SRC-...>` | `<SYS-...>` | `<bounded purpose>` | `<owner>` | `<type>` | `<class>` | `<scope>` | `<authorization>` | `<mode/target>` | `<identifiers>` | `<rule/unknown>` | `<read/propose/write>` | `<failure list>` | `<class>` | `<decision>` |

## Integration questions

| Question ID | Source/system | Question | Validation method | Owner | Needed by | Blocked scope |
|---|---|---|---|---|---|---|
| `<UNK-...>` | `<ID>` | `<schema/access/rate/delete/idempotency question>` | `<sample/test/vendor confirmation>` | `<owner>` | `<gate>` | `<use case>` |

## Health check

- [ ] Every source has a bounded purpose and owner.
- [ ] Read, proposal, and write scopes are separate.
- [ ] ACL, tenant, region, retention, deletion, freshness, and lineage are visible.
- [ ] Estimated record counts and vendor statements remain claims until corroborated.
- [ ] A missing identifier or lifecycle field blocks use rather than defaulting silently.
