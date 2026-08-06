# Incident Communication: `<INCIDENT-ID>`

> Draft status: `INTERNAL REVIEW ONLY | APPROVED FOR CUSTOMER | SENT`
> Classification: `<approved handling label>`
> Broad channels receive only the approved customer text. Restricted evidence remains in the approved evidence system.

## Approval and audience

| Field | Record |
|---|---|
| Audience and channel | `<customer roles/channel>` |
| Prepared at (UTC) | `<timestamp>` |
| Incident commander approval | `<role, decision reference, time>` |
| Communications approval | `<role, decision reference, time>` |
| Security/privacy/legal review | `<role/reference or N/A with authorized reason>` |
| Contract/regulatory process checked by | `<authorized owner/reference>` |
| Next update due (UTC) | `<timestamp>` |

## Customer-ready update

**Status at `<UTC timestamp>`**

Answer all five questions. Use one bounded sentence per answer unless the approved
audience needs more detail. The example text is illustrative, not incident evidence.

### 1. Known

`<One sentence containing verified facts only; no raw content, IDs, or root-cause claim.>`

Example: "At 14:30 UTC on 2026-07-20, a negative authorization test showed that
a disabled contractor-class identity could still reach the EU content route."

### 2. Unknown

`<One sentence naming the material scope, impact, or cause still being established.>`

Example: "We are determining how many production requests used that identity class
and whether content was returned to customers."

### 3. Affected

`<Evidence-backed population, workflow, region, and time bound, or state that scope is still being determined.>`

Example: "The current review is bounded to EU-route requests using contractor-class
tokens that were active before 14:30 UTC."

### 4. Contained

`<What is disabled, pinned, revoked, paused, or degraded; what approved path remains available.>`

Example: "The EU route is disabled for contractor-class identities; approved employee
access remains available, and the manual workflow is available at `<approved URL>`."

Customer action: `<none, or one approved bounded action>`

### 5. Next

"We will provide another update by `<exact UTC timestamp>` with `<next decision or
scope finding>`, even if the investigation is still in progress."

## Internal review notes: do not send

### Redaction hardening

Reject the draft when any item is present without explicit approval, a demonstrated
need, and channel controls:

- [ ] No raw customer, user, tenant, document, or internal resource IDs.
- [ ] No credentials, tokens, secrets, or exploitable endpoints/configuration.
- [ ] No manuscript text, prompts, retrieved passages, responses, or request bodies.
- [ ] No unapproved personal data or entitlement details.
- [ ] No unverified root cause, blame, legal conclusion, or breach determination.
- [ ] No unsupported recovery time, compensation, or notification promise.

| Candidate statement | Class | Evidence reference | Safe for audience? | Reviewer decision |
|---|---|---|---|---|
| `<statement>` | `FACT | HYPOTHESIS | UNKNOWN | DECISION` | `<approved reference>` | `YES | NO` | `APPROVE | REDACT | REMOVE: <reason>` |

Restricted technical context: `<stable evidence references only>`

Unapproved recovery estimate: `<NONE or remove before approval>`

Legal/contract questions routed to: `<authorized owner>`

## Redaction gate

Reject the draft if it contains any of the following without explicit approved
need and channel controls:

- manuscript, prompt, retrieved passage, model output, or request/response body;
- raw customer, user, tenant, document, token, credential, endpoint, or internal
  resource identifier;
- personal data, secrets, access tokens, entitlement details, or exploitable
  control configuration;
- an unverified root cause, unsupported blast radius, blame, legal conclusion,
  breach determination, or regulatory interpretation;
- a recovery promise, service credit, contractual commitment, or notification
  statement not approved by the authorized owner.

## Send record

| Field | Record |
|---|---|
| Final approved version/reference | `<immutable reference>` |
| Sent at (UTC) | `<timestamp>` |
| Sent by | `<role>` |
| Audience | `<bounded list/role group>` |
| Delivery confirmation | `<reference>` |
| Supersedes | `<prior update reference or NONE>` |

## Health check

- [ ] The update states known, unknown, affected, contained, and next update.
- [ ] Scope is evidence-backed and bounded.
- [ ] Root cause is absent unless verified and approved.
- [ ] The text contains no raw content, credentials, personal data, or exploitable detail.
- [ ] Customer action is explicit, even when it is “none.”
- [ ] Incident command and communications approvals are recorded.
- [ ] Legal, privacy, contract, and regulator decisions remain with authorized owners.
