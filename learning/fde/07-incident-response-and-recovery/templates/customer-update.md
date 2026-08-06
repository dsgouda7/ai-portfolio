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

We are investigating `<bounded service or workflow symptom>` affecting
`<confirmed scope, or “a subset still being determined”>`.

**Known:** `<verified facts only; no customer content, raw IDs, or root-cause claim>`

**Unknown:** `<material scope or cause still being established>`

**Containment:** `<what path is disabled, pinned, revoked, paused, or degraded>`

**Customer action:** `<none, or one approved bounded action>`

**Next update:** We will provide another update by `<UTC timestamp>`, even if
the investigation is still in progress.

## Internal review notes: do not send

| Candidate statement | Class | Evidence reference | Safe for audience? | Reviewer decision |
|---|---|---|---|---|
| `<statement>` | `FACT | HYPOTHESIS | UNKNOWN | DECISION` | `<reference>` | `YES | NO` | `<reason>` |

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
