# Skills: Discovery, Composition, and Versioning - Theory Notes

## 1. The Capability-Layer Mental Model

Keep five primitives separate because they solve different problems.

| Primitive | Mental model | Primary responsibility |
|---|---|---|
| Tool | Executable action | Validate inputs and perform one operation |
| Prompt | Reusable text template | Supply wording, structure, and placeholders |
| Skill | Versioned know-how bundle | Explain how to complete a task with instructions, examples, checks, and pinned capabilities |
| Plugin | Host package or adapter | Install or connect behavior inside a framework |
| MCP server | Protocol endpoint | Publish capabilities across a process or ownership boundary |

A skill can explain how to triage suppliers with inventory and quote tools. It does not become either service or the approval policy. Removing the skill removes know-how; the tools and their controls remain.

## 2. The Manifest-as-Release Mental Model

A loose instruction file is reusable but hard to govern. A manifest turns know-how into a release unit by naming its version, summary, instructions, examples, evaluation cases, pinned tools, risk, access, and status.

The summary supports cheap discovery. Instructions and examples describe intent; evaluation cases test it. Pins identify exact executable dependencies. Risk, access, and status let the runtime reject unsuitable drafts before sensitive content loads.

The manifest does not prove a release is good. It makes the release identifiable enough to evaluate, approve, persist, and roll back.

## 3. The Progressive-Disclosure Mental Model

Discovery should narrow context. A registry first returns identity, version, summary, capabilities, risk, access, and status. The runtime requests the task's least capabilities, then filters by allow-list, tenant, role, risk ceiling, promotion status, and dependency health.

Only then should full instructions, examples, and checks load. Irrelevant content consumes no context, and unauthorized or unhealthy bundles cannot influence reasoning merely by appearing in a catalog.

Semantic similarity may help propose candidates, but similarity is not authorization. Deterministic policy still owns every gate.

## 4. The Least-Capability Mental Model

Discovery is a constrained selection problem: find a promoted, healthy, authorized skill covering the required capabilities with the least surplus.

Supplier triage needs inventory and quote reads. A quote-only skill is insufficient; a purchase-commitment skill is excessive and outside the buyer's risk ceiling. The narrow skill covers the task without importing commitment authority or unrelated instructions.

Least capability is the skill equivalent of least privilege. It limits both cognitive surface area and accidental power.

## 5. The Composition-Without-Authority Mental Model

Composition eases reuse without merging security boundaries. Supplier triage can sequence inventory and quote tools, but every invocation still passes registry, argument, health, and authority checks.

A skill cannot grant finance authority, validate a negative quantity, or turn supplier instructions into policy. The runtime supplies identity and scopes; the tool gateway enforces them. Skill text recommends, while deterministic software authorizes and validates.

The practical test is simple: if changing skill instructions can weaken a tool's validation or mint a new scope, the architecture has placed authority in the wrong layer.

## 6. The Pinned-Graph Mental Model

A skill version is reproducible only when its dependency graph is reproducible. Suppose supplier quote version 1 selects the lowest fresh trusted price, while version 2 selects the fastest fresh trusted response. Both implementations may be valid. A floating dependency still makes the same skill version choose different suppliers over time.

Pinning supplier quote version 1 keeps supplier triage version 1 stable even after version 2 enters the registry. Adopting version 2 requires a skill version bump because the skill's behavior, instructions, and expected outcomes changed together.

Pinning is not resistance to improvement. It is the mechanism that makes improvement explicit, reviewable, and attributable.

## 7. The Evaluation-Gate Mental Model

Schema validation asks whether a candidate manifest is well formed. An evaluation gate asks whether the candidate behaves acceptably. The difference matters because a perfectly valid manifest can still encode a regression.

OrderFlow evaluates candidates on committed deterministic purchase requests. A candidate that pins the latency-first tool but retains the old price-first expectation must fail. A later candidate can pass only after its instructions and checks explicitly accept the new supplier under a reviewed price-premium limit.

Promotion changes the active release only after all required checks pass. Rejection leaves the current version untouched. Rollback restores a known prior release rather than asking operators to reconstruct old instructions and dependencies during an incident. Release state should be recorded so every transition is auditable.

## 8. The Deterministic-Conflict Mental Model

Overlapping skills are inevitable. A global triage skill may coexist with a tenant-specific one; several versions may coexist during rollout. Selection therefore needs visible ordering rules.

A practical order is: least surplus capability, most specific tenant scope, lower risk, then newest promoted version. These rules should be encoded in the resolver and tested. Catalog insertion order, filename order, or similarity-score noise must never become hidden policy.

If two distinct skills remain tied after every declared rule, discovery should fail closed. Ambiguity is information the system must surface, not uncertainty the agent should conceal with a guess.

## 9. Durable Workflows and Specialists

Chapter 03's durable workflows need to persist the selected skill identity, skill version, and pinned tool versions beside workflow state. A resumed purchase order must continue with the reviewed dependency graph, not rediscover whatever is newest after a crash.

Chapter 08's specialist agents need the same discipline at assignment time. A supervisor should request the capabilities a task requires, discover one allowed skill, and load it only for the specialist whose role and tenant match. The skill guides that specialist but does not expand its authority scope. Inventory and supplier specialists remain unable to approve purchases even when they share a triage bundle.

## 10. Practical Failure Modes

1. **Full-catalog loading:** every instruction bundle enters context before relevance or authorization checks, wasting tokens and exposing sensitive guidance.
2. **Floating dependencies:** a tool upgrade changes a decision without a skill release or review trail.
3. **Self-declared authority:** skill text claims permissions that the runtime never granted.
4. **Schema-only promotion:** a candidate parses correctly but fails the task behavior it was meant to preserve.
5. **Uncommitted evaluations:** changing fixtures let a candidate pass today and fail tomorrow with no attributable code change.
6. **Catalog-order conflict resolution:** equally suitable skills produce different choices after an unrelated registration reorder.
7. **Rediscovery on resume:** a durable workflow restarts with a newer skill or tool graph than the one that began the transaction.
8. **Rollback by reconstruction:** operators edit prompts during an incident because no prior release remains addressable.

**Durable closing line:** a skill is trustworthy when its know-how is discoverable, its authority stays elsewhere, its dependencies are pinned, and every behavioral change earns a named release.
