# Riverside Identity Flow

**Status:** `[Local-static]` design artifact. Not a deployed identity proof.

```mermaid
flowchart LR
    A["Synthetic actor\nAPI-USER-EDITOR-017"] --> B["Customer IdP\nidentity + tenant + groups"]
    B --> C["Gateway\nvalidate token; derive actor and tenant"]
    C --> D["Request-context policy\nroles + region + purpose + titles"]
    D --> E["Retrieval policy\nserver-derived filters"]
    D --> F["Tool policy\nconcrete action + payload + approval"]
    E --> G["Post-retrieval check\nfail closed"]
    F --> H["Bounded tool call\nservice identity"]
    G --> I["Response assembly\nauthorized evidence only"]
    H --> I
    D --> J["Restricted audit\nallow and deny decisions"]
    E --> J
    F --> J
    I --> K["Public v1 response\nno role/filter echo"]
    style A fill:#1e3a8a,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style B fill:#1d4ed8,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style C fill:#1d4ed8,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style D fill:#b45309,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style E fill:#1d4ed8,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style F fill:#1d4ed8,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style G fill:#b91c1c,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style H fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style I fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style J fill:#b45309,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style K fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
```

## Context propagation

| Boundary | Trusted input | Decision | Forwarded context | Must not cross |
|---|---|---|---|---|
| Actor to IdP | Authentication ceremony | Establish identity | Actor, tenant memberships, groups, status | Passwords, authenticators |
| IdP to gateway | Signed token | Validate issuer, audience, signature, expiry, scope | Actor, tenant, trusted groups | Unvalidated client tenant/role headers |
| Gateway to policy | Validated identity plus approved request intent | Derive effective roles; validate region, purpose, titles | All seven required context fields | Caller-added authority |
| Policy to retrieval | Normalized context | Build mandatory filters | Tenant, region, purpose, title scope, effective roles, trace | Model-generated filters as authority |
| Policy to tool | Normalized context plus concrete action | Allow, deny, or require exact-payload approval | Minimum action scope and service identity | Caller token, broad credentials |
| Components to audit | Decision facts | Preserve allow and deny evidence | Synthetic actor/tenant, trace, policy/rule, outcome, reason | Prompt, completion, manuscript, token |
| Response assembly to client | Authorized evidence and decision | Minimize public envelope | Answer/refusal, citations, trace, deployment metadata | Internal roles, filters, token claims, backend errors |

## Invariants

1. Authentication does not grant a purpose or a tool action.
2. Requested roles may narrow but never expand trusted entitlements.
3. The region field is policy metadata until deployed routing/storage is externally proven.
4. Retrieval enforces filters before search and authorization again after search.
5. The model proposes intent; deterministic policy controls authority.
6. Audit records identity decisions; metrics use only bounded, non-identifying dimensions.

## Open external validations

- Token issuance, nested-group behavior, revocation, and reconciliation freshness.
- Managed/workload identity permissions and exact Azure scopes.
- Private network and DNS behavior for every hop.
- Audit retention, legal hold, support access, and deletion policy.
- Regional storage, processing, backup, diagnostics, and subprocessor paths.
