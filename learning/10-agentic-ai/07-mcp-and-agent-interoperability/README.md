# Chapter 07 — MCP and Agent Interoperability

**Contents:** [one-line map of this directory](CONTENTS.md)

**Failure being repaired:** N-by-M adapters hide capability and schema drift, so an old client shape can reach an incompatible provider.

**Prerequisites:** [Chapter 06](../06-safety-human-control-and-governance/README.md), Chapter 00 tool contracts, and client/server lifecycle basics.

**Before the protocol vocabulary:** the client asks a capability server what actions it offers, checks the returned contract, and only then sends a call. **JSON-RPC** is a structured request/reply envelope with a method name, arguments, and an identifier tying the reply to the request. **MCP** standardizes that discovery and call lifecycle; it does not decide whether OrderFlow is authorized to buy.

## Local Learning Contract

- **Predict:** Decide whether an old-shape call should reach a changed handler or stop at the client's schema guard.
- **Implement:** Add capability discovery, cached schema validation, cancellation boundaries, drift refusal, and configuration-only provider switching.
- **Inspect:** Read initialization/discovery order, advertised schema/version, client cache, validation result, cancellation signal, and provider configuration.
- **Prove:** Reject schema drift before handler execution and switch providers without changing agent logic.
- **Required evidence:** The notebook's lifecycle-order, drift-rejection, no-handler-call, cancellation, and provider-switch assertions.
- **Bounded transfer:** Copy the schema-drift case and add the newly required field; prove validation now passes and the handler runs exactly once.
- **Relevant system-design reference:** [03 — Tool, MCP & Skill Registry](../system-design/03-tool-mcp-and-skill-registry.md), **reference-only depth**.

**Previous:** [Chapter 06 — Safety, Human Control, and Governance](../06-safety-human-control-and-governance/README.md) · **Next:** [Chapter 08 — Multi-Agent Coordination](../08-multi-agent-communication-and-coordination/README.md)

[Open the notebook](07-mcp-and-agent-interoperability.ipynb)
