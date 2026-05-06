# Ch.2 — Model Context Protocol (MCP)

> **The story.** **Anthropic** announced the **Model Context Protocol** on **25 November 2024** as an open standard built on JSON-RPC 2.0. The motivating problem was the **N×M integration explosion** — every agent had to ship custom adapter code for every data source. MCP defined three reusable primitives — *Resources*, *Tools*, *Prompts* — plus standard transports (stdio for local, SSE/HTTP for remote). Adoption was unusually fast for a protocol: by mid-2025 OpenAI, Microsoft, and Google had all shipped MCP support; Claude Desktop, Cursor, Zed, and VS Code Copilot all spoke MCP natively; and the public registry had passed several thousand servers. **MCP is now the protocol for tool/data integration in multi-agent systems**, the same way HTTP is the protocol for hypertext.
>
> **Where you are in the curriculum.** [Ch.1](../ch01_message_formats) gave you the message envelope. This chapter answers: **what problem does MCP solve that plain function calling does not, and how does the JSON-RPC 2.0 protocol turn any data source or executable function into something any compliant agent can discover and use without bespoke adapter code?** Master this and the [A2A](../ch03_a2a) chapter — agent-to-agent delegation — will compose cleanly with it.
>
> **Notation.** `MCP` = Model Context Protocol (Anthropic, November 2024). `JSON-RPC 2.0` = the wire transport format (`method`, `params`, `id`, `result` / `error`). `Resource` = read-only data exposed by an MCP server. `Tool` = executable function exposed by an MCP server. `Prompt` = reusable prompt template registered with an MCP server. `SSE` = Server-Sent Events (HTTP streaming transport for remote MCP servers). `stdio` = standard-input/output transport for local in-process MCP servers.
<!-- notation: key variables defined here -->

---

## § 0 · The Challenge — Where We Are

> 🎯 **The mission**: Build **OrderFlow** — AI-native B2B purchase order automation satisfying 8 constraints:
> 1. **THROUGHPUT**: 1,000 POs/day — 2. **LATENCY**: <4hr SLA — 3. **ACCURACY**: <2% error — 4. **SCALABILITY**: 10 agents/PO — 5. **RELIABILITY**: >99.9% uptime — 6. **AUDITABILITY**: Full traceability — 7. **OBSERVABILITY**: Real-time monitoring — 8. **DEPLOYABILITY**: Zero-downtime updates

**What we know so far**:
- ✅ **Ch.1 Message Formats**: Decomposed single agent into 8 specialized agents (Intake, Pricing, Negotiation, Legal, Finance, Drafting, Sending, Reconciliation)
- ✅ **Context overflow eliminated**: Each agent stays under 4k token budget (50% of 8k limit)
- ✅ **Error rate improved**: 5% → 3.8% (structured message schemas prevent parsing failures)
- ⚡ **Current metrics**: 10 POs/day throughput, 36 hours median latency, 3.8% error rate
- ❌ **But we still can't ground agents in real-time data!** Each agent needs access to ~20 data sources (ERP, pricing APIs, supplier APIs, email, legal templates). Without a standard protocol, that's **8 agents × 20 integrations = 160 bespoke implementations**.

**What's blocking us**:

🚨 **The N×M Integration Explosion**

You're the Lead Architect at OrderFlow. Your 8 agents are working, but they're blind. The Pricing agent needs live supplier quotes — someone hardcoded an HTTP client for TechFurnish's API. Then OfficeDepot. Then 18 other suppliers. The Negotiation agent needs the same data — a different engineer wrote different wrappers. The Finance agent needs ERP access — a third team wrote a third set of adapters.

**Current situation**: Three teams, nine custom wrappers (3 agents × 3 systems), zero reusability. Each wrapper has subtly different error handling. One silently swallows 404s. Another retries infinitely on timeout. The third crashes the agent.

```
Problems:
1. ❌ **Integration explosion**: N agents × M tools = N×M bespoke adapters (8 × 20 = 160 for OrderFlow) → **Blocks #4 SCALABILITY**
2. ❌ **No schema discovery**: Agent code hardcodes API schemas; when supplier API changes, agents break silently → **Blocks #3 ACCURACY**
3. ❌ **Zero observability**: Custom wrappers don't log tool calls consistently; cannot debug which agent called which supplier → **Blocks #7 OBSERVABILITY**
```

**Business impact**: You hired 2 engineers for 6 months just to write integration adapters ($180k labor cost). When TechFurnish changed their pricing API, 3 agents broke in production. OrderFlow processed zero POs for 4 hours. The CTO is demanding: **"Why can't we add a new supplier without rewriting half the codebase?"**

**What this chapter unlocks**:

🚀 **Model Context Protocol (MCP) — collapse N×M to N+M**:
1. **Standard protocol for tool access**: JSON-RPC 2.0 transport → any agent connects to any data source without custom code
2. **Self-describing servers**: Tools declare their JSON Schema at runtime → agents discover capabilities dynamically, no hardcoded schemas
3. **Integration collapse**: 160 bespoke integrations → **8 MCP clients + 20 MCP servers = 28 components** (94% reduction)

⚡ **Expected improvements**:
- **Throughput**: 10 → 10 POs/day (no change yet — still sequential architecture)
- **Latency**: 36 hours → 36 hours (no change yet — still synchronous)
- **Error rate**: 3.8% → **3.2%** (agents grounded in real-time ERP data, no hallucinated pricing)
- **Scalability**: 160 integrations → **28 components** (94% reduction) → **#4 SCALABILITY foundation ✅**
- **Auditability**: Basic logging → **MCP tool call logging** (partial observability) → **#6 AUDITABILITY improved ⚡**
- **Observability**: Message structure only → **MCP request/response logged** (standardized format) → **#7 OBSERVABILITY improved ⚡**

### Progress on the 8 Constraints

| Constraint | Status | Evidence |
|------------|--------|----------|
| #1 THROUGHPUT | ❌ **BLOCKED** | Still 10 POs/day (integration bottleneck) |
| #2 LATENCY | ❌ **BLOCKED** | 36 hours median (manual baseline) |
| #3 ACCURACY | ⚡ **IMPROVED** | 3.8% → **3.2% error** (agents grounded in real ERP data, no hallucinated pricing) |
| #4 SCALABILITY | ✅ **VALIDATED** | 8 agents share 20 MCP servers (no integration duplication) |
| #5 RELIABILITY | ❌ **BLOCKED** | No graceful degradation |
| #6 AUDITABILITY | ⚡ **IMPROVED** | MCP servers log all tool calls (partial observability) |
| #7 OBSERVABILITY | ⚡ **IMPROVED** | MCP tool calls logged (but no distributed tracing) |
| #8 DEPLOYABILITY | ❌ **BLOCKED** | No deployment automation |

**What's still blocking**: Agents on different servers can't delegate tasks to each other (e.g., Intake agent can't call Negotiation agent across Kubernetes pods). *(Ch.3 — A2A solves this.)*

---

## § 1 · The Core Idea

**Model Context Protocol (MCP)** collapses the N×M integration problem to N+M by defining a standard JSON-RPC 2.0 protocol for agent-tool communication. You write each tool integration once as an MCP server; any compliant agent becomes an MCP client and discovers available tools at runtime without hardcoded schemas. **Integration count scales linearly with agents plus tools, not multiplicatively.**

### Agent vs MCP Server — Role Clarity

One common point of confusion: **what is an agent, and what is an MCP server?**

| Aspect | Agent (MCP Client) | MCP Server |
|--------|-------------------|------------|
| **Role** | **Consumes** tools and resources | **Exposes** tools and resources |
| **Example** | Pricing Agent, Negotiation Agent | ERP Server, Supplier Quote Server |
| **Protocol Side** | Sends `tools/call` requests | Responds with `result` or `error` |
| **Implementation** | LangChain, AutoGen, Semantic Kernel, Claude Desktop | Python MCP SDK, Node.js MCP SDK, custom JSON-RPC server |
| **Cardinality** | N agents (8 in OrderFlow) | M servers (20 in OrderFlow) |
| **Reusability** | Agent-specific logic (orchestration, decision-making) | Cross-agent reusable (any client can call any server) |
| **LLM Required?** | Yes (agent makes decisions) | No (server is deterministic code) |

**Key insight**: One agent is an MCP client. It can call multiple MCP servers. One MCP server can be called by multiple agents. This is the **N+M collapse** — instead of writing N×M custom integrations, you write N clients + M servers.

---

## § 1.5 · The Practitioner Workflow — Your 4-Phase MCP Integration

> ⚠️ **Two ways to read this chapter:**
> - **Theory-first (recommended for learning):** Read §0→§3 sequentially to understand MCP concepts, then use this workflow as your integration reference
> - **Workflow-first (practitioners building production systems):** Use this diagram as a jump-to guide when integrating MCP into existing agent infrastructure
>
> **Note:** Section numbers don't follow phase order because the chapter teaches concepts pedagogically (protocol specification before implementation). The workflow below shows how to APPLY those concepts when building your MCP integration.

**Before diving into protocol details, understand the workflow you'll follow when integrating any agent with any MCP server:**

> 📊 **What you'll build by the end:** A production-grade MCP client that discovers tools at runtime, validates parameters against server-provided schemas, handles transport failures gracefully, and logs all tool calls for auditability.

```
Phase 1: INITIALIZE        Phase 2: DISCOVER          Phase 3: CALL             Phase 4: HANDLE
──────────────────────────────────────────────────────────────────────────────────────────────────
Connect to server:         List available tools:      Execute tool:             Handle failures:

• Send initialize          • Send tools/list          • Send tools/call         • Detect error codes
• Negotiate protocol       • Receive JSON Schemas     • Validate params         • Implement retry logic
• Confirm capabilities     • Cache tool registry      • Parse result/error      • Log all interactions

→ DECISION:                → DECISION:                → DECISION:               → DECISION:
  Which transport?           Which tool to expose?      Validate before call?     Retry strategy?
  • Local tool: stdio        • Agent needs: Tool        • Always validate         • -32603 (internal):
  • Remote service:          • Read-only: Resource        against schema            exponential backoff
    HTTP + SSE               • Template: Prompt         • Fail fast on bad        • -32601 (not found):
  • Concurrent clients:      • List all on startup        input                     no retry, alert user
    HTTP (stdio is 1:1)                                                           • Transport timeout:
                                                                                    reconnect + replay
```

> 💡 **Usage note:** Phases 1-2 happen once per agent startup or server connection. Phase 3 executes on every tool call. Phase 4 is continuous monitoring — every phase can fail and must be handled.

**Real-world integration timeline:**
- **Day 1-2:** Implement Phase 1 (handshake) + Phase 2 (discovery) → agent can connect to server and list available tools
- **Day 3-4:** Implement Phase 3 (tool invocation) → agent can call one tool successfully
- **Day 5-7:** Implement Phase 4 (error handling, logging, retry logic) → production-ready integration
- **Week 2+:** Add new servers (just implement server-side logic; client code unchanged)

### Phase Dependencies and Execution Flow

```mermaid
stateDiagram-v2
    [*] --> Phase1_Initialize
    Phase1_Initialize --> Phase2_Discover: Handshake successful
    Phase1_Initialize --> Phase4_Handle: Connection failed
    Phase2_Discover --> Phase3_Call: Tools discovered
    Phase2_Discover --> Phase4_Handle: Discovery failed
    Phase3_Call --> Phase3_Call: Next tool call
    Phase3_Call --> Phase4_Handle: Call failed
    Phase4_Handle --> Phase1_Initialize: Reconnect
    Phase4_Handle --> Phase3_Call: Retry succeeded
    Phase4_Handle --> [*]: Unrecoverable error
```

**Critical invariants:**
1. **Never skip Phase 1:** Always handshake before calling tools (protocol version mismatch causes silent failures)
2. **Cache Phase 2 results:** Don't re-discover tools on every call (adds 10-50ms latency per call)
3. **Validate in Phase 3:** Always validate tool arguments against schema before sending (server-side validation is last resort)
4. **Log everything in Phase 4:** Every MCP interaction must be logged for compliance audits

**Common pitfall:** Hardcoding tool schemas in agent code. **This defeats MCP's purpose.** The entire point is runtime discovery — your agent should work with any compliant server without code changes. If you find yourself writing `if tool_name == "get_supplier_quote":`, you're doing it wrong.

---

## § 2 · Running Example: PO #2024-1847 Pricing Lookup

Your Pricing agent needs to quote 10 standing desks from TechFurnish and OfficeDepot. Before MCP: you wrote `techfurnish_client.py` with hardcoded HTTP endpoints and response parsing. When TechFurnish changed their API schema, your agent crashed. When you added OfficeDepot, you wrote `officedepot_client.py` — same pattern, different bugs.

With MCP: you write `pricing-mcp-server` once, exposing `get_supplier_quote(supplier_name, item_id, quantity)` as a Tool. Both TechFurnish and OfficeDepot are wrapped behind this single server. Your Pricing agent calls `tools/call` with `{"name": "get_supplier_quote", "arguments": {"supplier_name": "TechFurnish", "item_id": "DESK-001", "quantity": 10}}`. The server returns `{"price": 789, "delivery_days": 14}`. When TechFurnish's API changes, you update the server implementation — zero agent code changes.

**Result**: PO #2024-1847 pricing lookup succeeded in 847ms (real-time supplier API call). Before MCP, this same lookup would have failed silently when TechFurnish changed their schema 3 weeks ago.

---

## § 3 · The Protocol Specification **[Phase 1: INITIALIZE]**

MCP is built on **JSON-RPC 2.0** — a lightweight remote procedure call protocol using JSON serialization. Every MCP interaction is a request-response pair over stdio (subprocess pipes) or HTTP+SSE (Server-Sent Events) transports.

> 🏭 **Industry Standard — JSON-RPC 2.0 Foundation**
>
> MCP builds on JSON-RPC 2.0 (published 2010), the same protocol powering Ethereum clients (Geth, Nethermind), Language Server Protocol (VSCode, IntelliJ), and Jupyter kernels. **Why this matters for production:** Every major programming language has battle-tested JSON-RPC libraries (Python: `jsonrpcserver`, Node: `jayson`, Go: `gorilla/rpc`). You're not adopting a bleeding-edge protocol — you're leveraging 15 years of tooling maturity. Debugging tools like [JSON-RPC Tester](https://www.jsonrpc.org/specification) work out-of-the-box with MCP traffic.
>
> **Key decision point:** JSON-RPC 2.0 is **stateless** — every request includes full context. This means MCP servers can scale horizontally behind a load balancer without session affinity. The protocol-level cost is slightly larger payloads (~50 bytes overhead per request) compared to binary protocols like gRPC, but the operational simplicity (no connection pooling, no sticky sessions) makes this the right tradeoff for agent-tool communication.

### The N×M Integration Problem

Without MCP, every integration between an agent and a tool is a custom adapter: the agent team writes a Python wrapper for the ERP, another team writes a different wrapper for the pricing API, and none of them are reusable across agents or reusable by agents built in different frameworks.

With `N` agents and `M` tools, that is `N × M` bespoke integrations. MCP collapses it to `N + M`: each tool becomes an MCP server (written once), and each agent becomes an MCP client (written once per framework). Any client connects to any server through a shared protocol.

```
Without MCP:                        With MCP:
Agent A → ERP adapter               Agent A ─────┐
Agent A → Pricing adapter           Agent B ─────┼──▶ MCP Protocol ──▶ ERP Server
Agent B → ERP adapter               Agent C ─────┘                ──▶ Pricing Server
Agent B → Pricing adapter                                          ──▶ Email Server
   = N × M adapters                    = N clients + M servers
```

### Protocol Mechanics

MCP is an open standard published by Anthropic (November 2024). It is built on **JSON-RPC 2.0** — a lightweight remote procedure call protocol that uses JSON as its serialisation format and requires no HTTP (though HTTP is supported).

### Handshake — Protocol Version Negotiation

**Phase 1 in action:** Every MCP integration begins with the `initialize` handshake. This is where client and server agree on protocol version and declare their capabilities.

```json
// Client → Server: initialise the connection and negotiate capabilities
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "tools": {},           // Client can invoke tools
      "sampling": {}         // Client supports LLM sampling (optional)
    },
    "clientInfo": {
      "name": "OrderFlow-PricingAgent",
      "version": "1.2.0"
    }
  }
}

// Server → Client: confirm capabilities
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "tools": {
        "listChanged": true    // Server will notify on tool changes
      },
      "resources": {
        "subscribe": true      // Server supports resource subscriptions
      }
    },
    "serverInfo": {
      "name": "pricing-mcp-server",
      "version": "2.1.3"
    }
  }
}
```

**Critical validation checkpoint:**

```python
# validate protocol version compatibility — reject incompatible servers early
if server_version != client_version:
    if is_backward_compatible(server_version, client_version):
        log.warning(f"Protocol version mismatch: {client_version} → {server_version} (compatible)")
    else:
        raise ProtocolVersionError(
            f"Incompatible protocol versions: client={client_version}, server={server_version}"
        )
```

> ⚡ **Why this matters:** In production, your OrderFlow Pricing agent (client v1.2.0) might connect to an old ERP server (protocol 2024-09-15) and a new supplier API server (protocol 2024-11-05). The handshake is your compatibility firewall — reject incompatible servers before any tool call happens.

The server is now self-described. The client does not need prior knowledge of what the server can do — it discovers it through the protocol.

---

## The Three Primitives **[Phase 2: DISCOVER]**

MCP defines exactly three types of thing a server can expose. Understanding the semantic difference between them is the most common interview test.

**Phase 2 in action:** After successful handshake, the client calls `tools/list`, `resources/list`, and `prompts/list` to discover everything the server offers. This happens once per connection — the results are cached.

> 🏭 **Industry Practice — MCP SDK Implementations**
>
> Don't build MCP clients from scratch. Use official SDKs: **Python** ([`mcp` package](https://pypi.org/project/mcp/)), **TypeScript** ([`@modelcontextprotocol/sdk`](https://www.npmjs.com/package/@modelcontextprotocol/sdk)), **Rust** ([`mcp-rs`](https://crates.io/crates/mcp)). These handle JSON-RPC serialization, transport abstraction (stdio/HTTP), capability negotiation, and tool schema validation.
>
> **Real-world integration times** (from production deployments):
> - **Using SDK:** 2-3 days for full Phase 1-4 implementation (handshake → discovery → invocation → error handling)
> - **From scratch:** 2-3 weeks + high bug surface (stdio buffer deadlocks, SSE reconnection logic, schema validation edge cases)
>
> The SDK also future-proofs you: when Anthropic adds new capabilities (e.g., bidirectional streaming in future MCP versions), you get them via `pip install --upgrade mcp`. Custom implementations require manual protocol updates.

### Resources — Read-only data the agent can inspect

Resources are URI-addressable content. The agent requests a resource, the server returns its content. The agent does not modify it (that would be a Tool).

```python
# Server exposes order records as resources
@mcp_server.resource("order://{order_id}")
def get_order(order_id: str) -> str:
    record = db.fetch_order(order_id)
    return json.dumps(record)

# Client reads the resource — same pattern as fetching a URL
content = await mcp_client.read_resource(f"order://PO-4812")
```

**Examples:** database records, file contents, API schema documentation, product catalogues.

### Tools — Callable functions with side effects

Tools are functions the agent can invoke. Unlike Resources, Tools can mutate state.

```python
# Server exposes a tool
@mcp_server.tool()
def send_purchase_order(po_document: str, supplier_email: str) -> dict:
    """Send a purchase order to a supplier via email."""
    result = email_client.send(to=supplier_email, body=po_document)
    return {"message_id": result.id, "status": "sent"}
```

The critical detail: the server's `tools/list` response includes the full JSON Schema for each tool's input parameters. The agent never has to guess or hardcode the schema — the server declares it.

**Discovery request (Phase 2):**

```json
// Client → Server: Discover available tools
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/list"
}

// Server → Client: Tool registry with full schemas
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "tools": [
      {
        "name": "get_supplier_quote",
        "description": "Fetch real-time pricing from supplier API",
        "inputSchema": {
          "type": "object",
          "properties": {
            "supplier_name": {"type": "string", "enum": ["TechFurnish", "OfficeDepot"]},
            "item_id": {"type": "string", "pattern": "^[A-Z]+-[0-9]+$"},
            "quantity": {"type": "integer", "minimum": 1, "maximum": 10000}
          },
          "required": ["supplier_name", "item_id", "quantity"]
        }
      },
      {
        "name": "send_purchase_order",
        "description": "Send a purchase order to a supplier via email.",
        "inputSchema": {
          "type": "object",
          "properties": {
            "po_document": {"type": "string"},
            "supplier_email": {"type": "string", "format": "email"}
          },
          "required": ["po_document", "supplier_email"]
        }
      }
    ]
  }
}
```

**Client-side caching (Phase 2 best practice):**

```python
# cache tool schemas on discovery — avoid re-discovering on every call
class MCPClient:
    def __init__(self):
        self._tool_registry: dict[str, dict] = {}  # Cache schemas here

    async def discover_tools(self):
        """Phase 2: Discover and cache all tools (call once per connection)"""
        response = await self.call("tools/list")
        for tool in response["result"]["tools"]:
            self._tool_registry[tool["name"]] = tool["inputSchema"]
        logger.info(f"Discovered {len(self._tool_registry)} tools: {list(self._tool_registry.keys())}")

    def get_tool_schema(self, tool_name: str) -> dict:
        """Retrieve cached schema (no network call)"""
        if tool_name not in self._tool_registry:
            raise ToolNotFoundError(f"Tool '{tool_name}' not in registry. Did you call discover_tools()?")
        return self._tool_registry[tool_name]
```

> ⚡ **Performance insight:** Discovery adds 10-50ms latency (1 round-trip). Cache the results. In the OrderFlow Pricing agent, we call `discover_tools()` once at startup, then handle 1,000 POs/day with zero re-discovery overhead.

### Prompts — Reusable, parameterised instruction templates

Prompts are pre-built instruction templates stored server-side. The client can request a prompt by name, pass parameters, and receive a fully-formed message list. This is where domain-specific instruction engineering lives, separated from agent code.

```python
# Server exposes a prompt template
@mcp_server.prompt()
def negotiate_price_prompt(supplier_name: str, target_price: float) -> list[dict]:
    return [
        {"role": "system", "content": f"You are a procurement specialist negotiating with {supplier_name}."},
        {"role": "user", "content": f"Your target price is ${target_price:.2f} per unit. Begin negotiation."}
    ]

# Client retrieves it
messages = await mcp_client.get_prompt("negotiate_price_prompt",
                                        arguments={"supplier_name": "Acme Corp", "target_price": 13.50})
```

**Why Prompts belong in MCP:** The prompt to negotiate with a specific supplier changes when the business changes. Storing it server-side means the agent binary does not need to be redeployed when the instruction changes.

---

## Transport Options — stdio vs HTTP+SSE

> 🏭 **Industry Standard — Transport Selection Matrix**
>
> MCP supports two transports, and the choice is **not arbitrary** — it's driven by deployment topology and concurrency requirements. Here's the decision matrix used in production:

| Transport | Mechanism | Latency | Concurrency | Typical use case | When NOT to use |
|-----------|-----------|---------|-------------|------------------|------------------|
| `stdio` | stdin/stdout pipes to a subprocess | <1ms | 1 client per server | Local tools: code executor, file system scanner, Git client | Multi-agent systems where N agents need the same tool (would spawn N server processes) |
| `HTTP + SSE` | HTTP POST for requests, Server-Sent Events for streaming responses | ~10ms | Unlimited (stateless) | Remote services: ERP API, pricing API, email gateway | Local-only tools with no network access (e.g., Docker socket) |

**Choosing transport — Decision tree:**

```python
# choose transport: stdio for local single-client, http+sse for remote/multi-client
def choose_transport(tool_properties: dict) -> str:
    if tool_properties["requires_local_filesystem"]:
        return "stdio"  # e.g., code execution sandbox, Git operations

    if tool_properties["concurrent_agents"] > 1:
        return "http+sse"  # e.g., shared ERP, pricing APIs

    if tool_properties["network_latency_acceptable"]:
        return "http+sse"  # e.g., external APIs, microservices

    return "stdio"  # default for local, single-agent tools
```

**Real-world example (OrderFlow):**
- **stdio transport:** Code execution tool (runs untrusted supplier scripts in sandbox), Git commit tool (local repository)
- **HTTP+SSE transport:** ERP server (8 agents query concurrently), pricing server (shared supplier API wrapper), email gateway (stateless, scales horizontally)

> ⚠️ **Common trap:** Using stdio for shared tools. If 8 agents each spawn their own stdio pricing-server subprocess, you have 8 redundant HTTP connection pools to TechFurnish API → rate limit violations. **Solution:** Deploy pricing-server as HTTP+SSE service once, all agents connect to `http://pricing-server:8080`.

---

## § 4 · How It Works — Step by Step **[Phase 3: CALL]**

**OrderFlow MCP interaction flow** (Pricing agent queries TechFurnish supplier):

```
PricingAgent (MCP Client)          pricing-mcp-server          TechFurnish API
    |                                     |                          |
    |──1. initialize──────────────────▶  |                          |
    |◀─────capabilities: {tools}─────────|                          |
    |                                     |                          |
    |──2. tools/list──────────────────▶  |                          |
    |◀─────[get_supplier_quote]──────────|                          |
    |                                     |                          |
    |──3. tools/call────────────────────▶|                          |
    |   {name: "get_supplier_quote",     |──HTTP POST──────────────▶|
    |    arguments: {supplier: "TechF",  |                          |
    |                item: "DESK-001",   |◀─────{price: 789}────────|
    |                qty: 10}}            |                          |
    |◀─────result: {price: 789}──────────|                          |
```

**Step-by-step**:
1. **Handshake** (Phase 1): Agent sends `initialize` with protocol version, server confirms capabilities
2. **Discovery** (Phase 2): Agent calls `tools/list`, server returns JSON Schema for each tool
3. **Invocation** (Phase 3): Agent calls `tools/call` with tool name + arguments, server validates against schema and executes
4. **Response** (Phase 3): Server returns result or error in standard JSON-RPC format

💡 **Key insight**: The agent never hardcoded TechFurnish's API. The server wrapped it. When you add OfficeDepot tomorrow, the Pricing agent's code doesn't change — you just add OfficeDepot logic inside the MCP server.

### Phase 3 Implementation — Tool Invocation with Validation

```python
# Phase 3: Tool invocation with client-side validation
async def call_tool(client: MCPClient, tool_name: str, arguments: dict) -> dict:
    """
    Phase 3 implementation: Validate arguments against schema, invoke tool, handle response.

    Args:
        client: MCP client with cached tool registry (from Phase 2)
        tool_name: Name of the tool to call
        arguments: Tool parameters (must match inputSchema)

    Returns:
        Tool result dict

    Raises:
        ValidationError: Arguments don't match schema
        ToolExecutionError: Server returned error response
    """
    # validate arguments against cached schema before sending to server
    schema = client.get_tool_schema(tool_name)  # Cached from Phase 2
    try:
        jsonschema.validate(instance=arguments, schema=schema)
    except jsonschema.ValidationError as e:
        raise ValidationError(
            f"Invalid arguments for tool '{tool_name}': {e.message}\n"
            f"Expected schema: {json.dumps(schema, indent=2)}"
        )

    # Send tools/call request
    request = {
        "jsonrpc": "2.0",
        "id": client.next_request_id(),
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        }
    }

    response = await client.send_request(request)

    # Handle response
    if "result" in response:
        return response["result"]
    elif "error" in response:
        # Phase 4 will handle retry logic for specific error codes
        raise ToolExecutionError(
            code=response["error"]["code"],
            message=response["error"]["message"]
        )
```

**Complete Phase 3 example (PO #2024-1847):**

```python
# Real tool call from OrderFlow Pricing agent
try:
    result = await call_tool(
        client=pricing_mcp_client,
        tool_name="get_supplier_quote",
        arguments={
            "supplier_name": "TechFurnish",
            "item_id": "DESK-001",
            "quantity": 10
        }
    )
    print(f"✅ Quote received: ${result['price']} per unit, {result['delivery_days']} days")
    # Output: ✅ Quote received: $789 per unit, 14 days
except ValidationError as e:
    print(f"❌ Invalid arguments: {e}")
except ToolExecutionError as e:
    print(f"❌ Tool execution failed: {e.code} - {e.message}")
    # Phase 4 retry logic would trigger here
```

> ⚡ **Why client-side validation matters:** In OrderFlow, we caught 23% of tool call errors during local validation (wrong argument types, missing required fields) before sending the request. This saved 847ms × 0.23 = **195ms per call** (no round-trip to server).

---

## § 5 · Key Diagrams

*(See illustration at end of chapter)*

---

## § 6 · Production Considerations **[Phase 4: HANDLE]**

**Phase 4 in action:** Error handling, retry strategies, and observability. This phase is continuous — every request can fail and must be handled gracefully.

> 🏭 **Industry Standard — JSON-RPC 2.0 Error Code Conventions**
>
> MCP inherits JSON-RPC 2.0's standardized error codes. **These are not arbitrary numbers** — they're from the JSON-RPC spec (2010), battle-tested across thousands of production systems:

| Code | Meaning | Retry Strategy | Example |
|------|---------|----------------|----------|
| **-32700** | Parse error (malformed JSON) | ❌ **No retry** (client bug) | Sent `{"jsonrpc": "2.0", "method: "tools/call"}` (missing closing quote) |
| **-32600** | Invalid request (missing required field) | ❌ **No retry** (client bug) | Sent `tools/call` without `params.name` |
| **-32601** | Method not found | ❌ **No retry** (tool doesn't exist) | Called `get_supplier_price` (typo: should be `get_supplier_quote`) |
| **-32602** | Invalid params (schema validation failed) | ❌ **No retry** (fix arguments first) | Sent `quantity: -5` (violates `minimum: 1`) |
| **-32603** | Internal error (server-side crash) | ✅ **Retry with exponential backoff** | Supplier API timed out |
| **-32000 to -32099** | Server-defined errors | ⚠️ **Depends on error message** | `-32050: Rate limit exceeded` (wait 60s, retry) |

**Phase 4 implementation — Retry logic:**

```python
import asyncio
from typing import Optional

class MCPClient:
    async def call_tool_with_retry(
        self,
        tool_name: str,
        arguments: dict,
        max_retries: int = 3,
        base_delay: float = 1.0
    ) -> dict:
        """
        Phase 4: Call tool with exponential backoff retry for transient errors.

        Retry logic:
        - -32603 (internal error): Retry with exponential backoff
        - -32700, -32600, -32601, -32602: No retry (client/schema errors)
        - Transport timeout: Reconnect + retry
        - All other errors: No retry, propagate to caller
        """
        last_error: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                # validate then execute tool call
                result = await self.call_tool(tool_name, arguments)

                # Success: log and return
                logger.info(
                    f"Tool '{tool_name}' succeeded",
                    extra={
                        "attempt": attempt + 1,
                        "tool_name": tool_name,
                        "latency_ms": result.get("_latency_ms", 0)
                    }
                )
                return result

            except ToolExecutionError as e:
                # Check if error is retryable
                if e.code == -32603:  # Internal error
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"Tool '{tool_name}' failed with -32603 (internal error), "
                        f"retrying in {delay}s (attempt {attempt + 1}/{max_retries})",
                        extra={"error_message": e.message}
                    )
                    await asyncio.sleep(delay)
                    last_error = e
                    continue

                elif e.code in {-32700, -32600, -32601, -32602}:
                    # Client error: no retry, log and re-raise immediately
                    logger.error(
                        f"Tool '{tool_name}' failed with non-retryable error {e.code}",
                        extra={
                            "error_code": e.code,
                            "error_message": e.message,
                            "arguments": arguments
                        }
                    )
                    raise

                else:
                    # Unknown error: log and re-raise
                    logger.error(
                        f"Tool '{tool_name}' failed with unknown error {e.code}",
                        extra={"error_message": e.message}
                    )
                    raise

            except TransportTimeout as e:
                # Transport failure: reconnect and retry
                logger.warning(
                    f"Transport timeout on attempt {attempt + 1}/{max_retries}, reconnecting..."
                )
                await self.reconnect()  # Phase 1: re-handshake
                await self.discover_tools()  # Phase 2: re-discover
                last_error = e
                continue

        # All retries exhausted
        logger.error(
            f"Tool '{tool_name}' failed after {max_retries} attempts",
            extra={"last_error": str(last_error)}
        )
        raise ToolExecutionError(
            code=-1,
            message=f"Tool '{tool_name}' failed after {max_retries} retries: {last_error}"
        )
```

**Observability — Logging every Phase:**

```python
# Production logging: every MCP interaction is an audit event
import structlog

logger = structlog.get_logger()

# Phase 1: Handshake
logger.info(
    "mcp.handshake",
    phase="initialize",
    protocol_version="2024-11-05",
    server_name="pricing-mcp-server",
    client_name="OrderFlow-PricingAgent"
)

# Phase 2: Discovery
logger.info(
    "mcp.discovery",
    phase="tools_list",
    tools_discovered=["get_supplier_quote", "send_purchase_order"],
    latency_ms=42
)

# Phase 3: Tool invocation
logger.info(
    "mcp.tool_call",
    phase="call",
    tool_name="get_supplier_quote",
    arguments={"supplier_name": "TechFurnish", "item_id": "DESK-001", "quantity": 10},
    correlation_id="PO-2024-1847",
    latency_ms=847
)

# Phase 4: Error handling
logger.error(
    "mcp.tool_error",
    phase="handle",
    tool_name="get_supplier_quote",
    error_code=-32603,
    error_message="Supplier API timeout",
    retry_attempt=1,
    correlation_id="PO-2024-1847"
)
```

> ⚡ **Why structured logging matters:** In OrderFlow, when CFO asked "Which agent queried TechFurnish for PO #2024-1847?", we ran: `grep 'correlation_id="PO-2024-1847"' logs/*.jsonl | grep tool_name="get_supplier_quote"` → found the full decision chain in 30 seconds. Before structured MCP logging, this forensics took 2+ hours.

| Concern | MCP Solution | OrderFlow Implementation |
|---------|-------------|-------------------------|
| **Latency** | stdio: <1ms; HTTP+SSE: ~10ms | stdio for code exec (local), HTTP+SSE for ERP/pricing (remote) |
| **Error handling** | JSON-RPC 2.0 error codes (see table above) | Retry -32603 with exponential backoff; fail fast on -32600/-32601/-32602 |
| **Authentication** | OAuth 2.0 (HTTP); process-level (stdio) | OAuth 2.0 bearer tokens for all HTTP servers; stdio servers run in agent's security context |
| **Rate limiting** | Server-side token bucket | pricing-server: 100 req/min per agent (prevents TechFurnish API ban) |
| **Monitoring** | JSON-RPC requests → middleware logs | Every tool call logged to Elasticsearch with correlation_id |
| **Deployment** | stdio: binaries; HTTP: containers | stdio servers: Docker images; HTTP servers: Kubernetes Deployment with HPA |

⚠️ **Common trap**: Exposing raw database access as MCP Resources. Instead, expose semantic operations as Tools (e.g., `get_inventory_level(item_id)` not `SELECT * FROM inventory`). This prevents agents from issuing arbitrary SQL.

---

## MCP vs Plain Function Calling

| Dimension | Plain function calling | MCP |
|-----------|----------------------|-----|
| Schema discovery | Schema is hardcoded in the client or passed in the system prompt | Schema is declared by the server at runtime via `tools/list` |
| Cross-agent reuse | The wrapper is coupled to the agent framework | Any MCP client can call any MCP server regardless of framework |
| Versioning | Client breaks silently if the tool's signature changes | Server negotiates protocol version during handshake; schema is always current |
| Observability | Tool calls are opaque unless the agent framework provides hooks | Every MCP call is a JSON-RPC request; standard middleware can intercept and log |
| Authentication | Ad hoc per integration | Standardised — OAuth 2.0 bearer tokens for HTTP transport; process-level trust for stdio |

---

## § 7 · What Can Go Wrong

**1. Tool schema drift**: Server updates tool signature without versioning → clients send old argument shape → validation fails.
- **Fix**: Version your tools (e.g., `get_supplier_quote_v2`) or use semantic versioning in MCP server manifest. Deprecate old versions gracefully.

**2. stdio deadlock**: Agent writes to server stdin, blocks waiting for response, but server is blocked writing to stdout (buffer full) → both processes hang.
- **Fix**: Use non-blocking I/O or larger OS pipe buffers. Production: prefer HTTP+SSE transport for concurrent clients.

**3. Exposing raw database access as Resources**: Agent requests `resource://db/inventory/all` and gets 500 MB of JSON → context window explosion.
- **Fix**: Resources should be semantic, not raw. Expose `resource://inventory/item/{id}` not `resource://db/*`. Rate-limit resource size to <10 KB.

**4. No authentication**: HTTP MCP server has no auth → any network client can call `tools/delete_all_orders`.
- **Fix**: Require OAuth 2.0 bearer tokens for HTTP transport. Validate tokens before processing `tools/call`.

**5. Logging proxy breaks observability**: Custom MCP proxy strips `correlation_id` from requests → distributed tracing fails.
- **Fix**: Proxies must preserve all JSON-RPC fields. Add new fields (e.g., `audit_timestamp`) but never remove existing ones.

---

## § 8 · Progress Check — What We Can Solve Now

```mermaid
graph LR
    Ch1["Ch.1\nMessage Formats"]:::done
    Ch2["Ch.2\nMCP"]:::current
    Ch3["Ch.3\nA2A"]:::upcoming
    Ch4["Ch.4\nEvent-Driven"]:::upcoming
    Ch5["Ch.5\nShared Memory"]:::upcoming
    Ch6["Ch.6\nTrust & Sandboxing"]:::upcoming
    Ch7["Ch.7\nAgent Frameworks"]:::upcoming
    Ch1 --> Ch2 --> Ch3 --> Ch4 --> Ch5 --> Ch6 --> Ch7
    classDef done fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    classDef current fill:#1d4ed8,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    classDef upcoming fill:#1e3a8a,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
```

✅ **Unlocked capabilities**:
- ✅ **Standard tool protocol**: Any agent can call any tool via JSON-RPC 2.0 without custom adapter code
- ✅ **Runtime schema discovery**: Agents discover tool schemas dynamically at connection time (no hardcoded schemas)
- ✅ **Integration collapse**: 160 bespoke integrations reduced to 28 components (8 MCP clients + 20 MCP servers)
- ✅ **Real-time data grounding**: Agents now query live ERP inventory, supplier pricing APIs, legal templates (no hallucinated data)

### Constraint Status After Ch.2

| Constraint | Before | After Ch.2 | Change |
|------------|--------|------------|--------|
| #1 THROUGHPUT | 10 POs/day | Still 10 POs/day | ❌ No change |
| #2 LATENCY | 36 hours median | 36 hours median | ❌ No change |
| #3 ACCURACY | 3.8% error | **3.2% error** | ⚡ **16% better** (grounded in real ERP data) |
| #4 SCALABILITY | 8 agents, but 160 integrations | **8 clients + 20 servers = 28 components** | ✅ **94% reduction** |
| #5 RELIABILITY | No retry logic | No retry logic | ❌ No change |
| #6 AUDITABILITY | Basic logging | MCP tool call logging | ⚡ **Improved** |
| #7 OBSERVABILITY | Message structure only | MCP tool calls logged | ⚡ **Improved** |
| #8 DEPLOYABILITY | No automation | No automation | ❌ No change |

### What We Can Solve Now

✅ **Scenario 1: Add new supplier without rewriting agents**
```
Before Ch.2:
New supplier API → write custom wrapper → update 3 agents → deploy → test → 2 weeks

After Ch.2:
New supplier API → add to existing pricing-mcp-server → test server → deploy server → 2 days
Agents unchanged, automatically discover new supplier via tools/list

Result: ✅ 7× faster supplier onboarding ($14k engineering cost → $2k)
```

✅ **Scenario 2: Recover from supplier API schema change**
```
Before Ch.2:
TechFurnish changes API → 3 agents crash → emergency hotfix → 4-hour production outage

After Ch.2:
TechFurnish changes API → update pricing-mcp-server (single file) → agents reconnect → discover new schema → 15-minute fix

Result: ✅ 16× faster recovery (4 hours → 15 minutes), zero agent code changes
```

✅ **Scenario 3: Compliance audit of tool usage**
```
Before Ch.2:
CFO asks "Which agent approved PO #2024-1847?" → manually grep 8 agent logs → inconsistent formats → 2-hour forensics

After Ch.2:
Every MCP tools/call logged in standard JSON-RPC format → query audit DB:
  SELECT * FROM mcp_calls WHERE correlation_id = 'PO-2024-1847' AND tool_name = 'approve_purchase_order'

Result: ✅ Full decision chain reconstructed in 30 seconds
```

### MCP Servers Deployed

- `erp-server`: Inventory levels, PO history, approval workflows (Resources)
- `pricing-server`: `get_supplier_quote(item_id, quantity)` (Tool)
- `email-server`: `send_email()`, `fetch_inbox()` (Tools)
- `legal-server`: Contract templates, approval policies (Resources)
- `audit-proxy-server`: Logging proxy that records all tool calls to compliance database

### What's Still Blocking

❌ **Cross-service agent delegation**: Your Intake agent (Kubernetes Pod 1) receives PO #2024-1847. It needs to delegate pricing lookup to the Pricing agent (Pod 3) and negotiation to the Negotiation agent (Pod 5). MCP solves agent-to-tool communication, but you have no protocol for agent-to-agent task handoff. You're still hardcoding HTTP endpoints: `POST http://pricing-agent:8080/lookup`. When you add a 9th agent, you update 8 agent configs. **Constraint #4 SCALABILITY blocked: can't add agents without N² coordination.**

❌ **Synchronous bottleneck**: Intake agent calls Pricing agent (847ms supplier API call), waits for response, then calls Negotiation agent (12-second LLM call), waits, then calls Approval agent. Total: 2 min + 5 min + 8 min + 15 min + 2 min + 3 min = **35 minutes best-case, 36 hours with queue time**. **Constraint #1 THROUGHPUT blocked at 10 POs/day** (need 1,000 POs/day).

❌ **No shared state management**: Pricing agent and Negotiation agent both need to update the PO line items. Race condition: both agents read `quantity: 10`, Pricing agent updates to `quantity: 10, price: 789`, Negotiation agent updates to `quantity: 10, price: 749`, last write wins → Pricing agent's work silently lost. **Constraint #3 ACCURACY risk: data corruption in concurrent workflows**.

**Real-world status**: You can now add tools (suppliers, APIs, services) without rewriting agents. Error rate improved 3.8% → 3.2% (agents grounded in real data). But you're still at 10 POs/day throughput with 36-hour latency — agents can't delegate to each other across service boundaries.

**Next up:** [Ch.3 — Agent-to-Agent Protocol (A2A)](../ch03_a2a) gives us **agent delegation** — Intake agent discovers and calls Pricing agent via Agent Cards, task lifecycle (submitted → working → completed → failed), and SSE streaming for long-running tasks. Unlocks distributed multi-agent orchestration across Kubernetes cluster.

---

## § 9 · Bridge to Chapter 3

MCP collapsed agent-to-tool integration from N×M to N+M, but agent-to-agent coordination is still point-to-point. Ch.3 (A2A) defines the delegation protocol — Agent Cards advertise capabilities, task lifecycle tracks request→response, SSE streams long-running results → **enables hierarchical orchestration without hardcoded agent endpoints**.

---

## § 10 · The Math

### The N×M Integration Problem

Without a standard protocol, $N$ agents connecting to $M$ data sources require $N \times M$ bespoke integrations. MCP changes this to $N + M$ (each agent and server implements the standard once):

$$\text{Without MCP:} \quad \text{integrations} = N \times M$$
$$\text{With MCP:} \quad \text{integrations} = N + M$$

For OrderFlow: 8 agents × 20 tools = **160 custom integrations** without MCP → **28 implementations** with MCP (8 client adapters + 20 server adapters).

### Tool Schema as a Type Contract

Each MCP tool has a JSON Schema input specification. The agent sends:

$$\text{call} = \bigl\{\text{name}: t, \ \text{arguments}: \mathbf{a}\bigr\}$$

where $t$ is the tool name and $\mathbf{a}$ is a JSON object validated against the tool's schema $\mathcal{S}_t$. Validation passes iff $\mathbf{a} \models \mathcal{S}_t$.

The server returns a result object $r_t(\mathbf{a})$. The MCP protocol guarantees that the shape of $r_t$ is declared in the server's capability manifest, making the contract machine-verifiable.

| Symbol | Meaning |
|--------|---------|
| $N$ | Number of MCP clients (agents) |
| $M$ | Number of MCP servers (tool providers) |
| $t$ | Tool name |
| $\mathbf{a}$ | Tool arguments (JSON object) |
| $\mathcal{S}_t$ | Tool's JSON Schema contract |
| $r_t(\mathbf{a})$ | Tool result for arguments $\mathbf{a}$ |



---

## § 11 · Where This Reappears

| Chapter | How MCP concepts appear |
|---------|------------------------|
| **Ch.1 — Message Formats** | MCP tool calls use the same `tool_calls`/`tool` role message format from Ch.1; MCP is the standardised envelope around function calling |
| **Ch.3 — A2A** | A2A agents can advertise their tools as MCP Resources; the two protocols are complementary — MCP for tool access, A2A for agent-to-agent task delegation |
| **Ch.7 — Agent Frameworks** | LangChain's `load_mcp_tools`, LangGraph's node-level tool access, and Semantic Kernel's plugin model all support MCP as the underlying tool discovery mechanism |
| **AI track — ReAct** | The ReAct Thought→Action→Observation loop calls MCP tools in the Action step; MCP is the production protocol that replaces hardcoded function schemas |
| **AI Infrastructure — Inference Optimization** | MCP servers can be deployed behind vLLM inference endpoints; the MCP client routes tool calls to the serving layer |

---

## Interview Questions

**Q: What are the three MCP primitive types and how do you decide which to use?**
**Resource** when the agent needs to read data without modifying it (a product catalogue, a database record). **Tool** when the agent needs to take an action or mutate state (send an email, write a record). **Prompt** when you have reusable, parameterised instruction templates that should live server-side rather than in agent code.

**Q: What problem does MCP solve that plain function calling does not?**
Discovery and reuse. With plain function calling, the agent must already know the tool's schema (hardcoded or injected via system prompt). With MCP, the server self-describes its capabilities at connection time. Any MCP-compliant agent can discover and use any MCP-compliant server without prior configuration — the `N × M` integration problem becomes `N + M`.

**Q: What is the difference between stdio and HTTP+SSE transports, and when would you choose each?**
`stdio` starts the MCP server as a subprocess and communicates via stdin/stdout — lowest latency, suitable for local tools (code execution, local file access), but limited to one client at a time. `HTTP + SSE` exposes the server over a network — supports multiple concurrent clients, can be scaled independently, required for remote services. Use stdio in development and for trusted local tools; use HTTP+SSE for production services.

**Q: Does MCP replace RAG?**
No. MCP exposes data as Resources that an agent can address by URI — it handles the access and delivery layer. The retrieval strategy (what to retrieve, how to chunk, how to rank), the vector database, and the embedding model are still entirely your responsibility. MCP and RAG are complementary: the RAG retrieval logic can be packaged as an MCP Tool.

---

## Notebook

`notebook.ipynb_solution.ipynb` (reference) or `notebook.ipynb_exercise.ipynb` (practice) implements:
1. A minimal MCP server exposing one Resource, one Tool, and one Prompt (using the `mcp` Python SDK)
2. An MCP client that discovers and calls them
3. The OrderFlow scenario: three MCP servers (ERP mock, pricing mock, email mock) called by a single orchestrator agent
4. The logging proxy: an MCP server that wraps another MCP server and records all calls

---

## Prerequisites

- [Ch.1 — Message Formats & Shared Context](../ch01_message_formats) — tool calls in the OpenAI message schema
- [AI / ReActAndSemanticKernel](../.03-ai/ch06_react_and_semantic_kernel/react-and-semantic-kernel.md) — the ReAct tool-use loop that MCP standardises

## Next

→ [Ch.3 — Agent-to-Agent Protocol (A2A)](../a2a) — how agents delegate tasks to other agents across service boundaries

## Illustrations

![MCP - N x M to N + M, handshake, primitives, transports](img/MCP.png)
