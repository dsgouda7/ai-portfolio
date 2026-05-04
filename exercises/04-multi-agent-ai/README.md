# Exercise 04: Multi-Agent AI System (TODO Version)

> **Learning Challenge:** Build a collaborative multi-agent system with task decomposition, message passing, and coordination

**Scaffolding Level:** 🟡 Medium (guided TODOs with time estimates)

See [main.py](main.py), [src/models.py](src/models.py), and [src/features.py](src/features.py) for implementation.

## Quick Start

```bash
# Install dependencies
pip install rich

# Run demonstration (after implementing TODOs)
python main.py
```

> **Note:** Infrastructure files (Dockerfile, docker-compose.yml, Makefile) have been removed for simplicity. Use setup.ps1 (Windows) or setup.sh (Unix) for environment setup.

## 0. Grand Challenge: OrderFlow

> 🎯 **The mission**: Build **OrderFlow** — AI-native B2B purchase order automation satisfying 8 constraints:
> 1. **THROUGHPUT**: 1,000 POs/day (manual baseline: 50/day)
> 2. **LATENCY**: <4hr SLA (manual baseline: 24-48hr)
> 3. **ACCURACY**: <2% error (manual baseline: 5%)
> 4. **SCALABILITY**: 10 agents/PO without context overflow
> 5. **RELIABILITY**: >99.9% uptime + graceful degradation
> 6. **AUDITABILITY**: Full traceability (every agent decision logged)
> 7. **OBSERVABILITY**: Real-time monitoring (latency, error rates, throughput)
> 8. **DEPLOYABILITY**: Zero-downtime updates + <5min rollback

**What OrderFlow is:**  
An AI-native operations platform that automates the end-to-end lifecycle of B2B purchase orders — receiving freeform email requests, checking inventory and pricing, negotiating with suppliers, drafting and sending POs, and reconciling confirmations. Currently, your company processes 50 POs/day with 3 procurement staff ($420k/year labor cost) at a 5% error rate. CFO Elena Vasquez needs 1,000 POs/day capacity to support business growth, but scaling the manual approach would require 60 staff ($8.4M/year) — economically non-viable.

**Current constraint status:**
- ✅ **Manual baseline established**: 50 POs/day throughput measured, 5% error rate documented, $420k/year cost benchmarked
- ✅ **Single-agent prototype validated** (from AI track): ReAct agent handles simple 2-step PO flows (<5 supplier emails)
- ❌ **Context overflow after 3 supplier negotiations**: Single agent hits 8k token limit (proven in notes Ch.1)
- ❌ **Synchronous bottleneck**: 3 blocking threads × 8hr wait time = 24 POs/day max throughput (Ch.4 analysis)
- ❌ **No audit trail**: CFO Elena rejects deployment due to $120k fraud incident from previous unaudited procurement bot (Ch.6 requirement)

**What's blocking us:**
1. **Context overflow**: Single-agent system accumulates 24k tokens after 3 supplier email threads (8k context limit exceeded) → agent loses conversation history → 5% error rate spikes to 12%
2. **Synchronous coordination**: Orchestrator waits synchronously for each worker response (pricing check: 2hr, supplier negotiation: 6hr, approval: 30min) → 3 sequential threads × 8hr = 24 POs/day max → cannot reach 1,000 POs/day target
3. **Silent approval violations**: Previous procurement bot auto-approved POs without logging `agent_id` or `rule_applied` → $120k overbilling fraud undetected for 8 months → CFO requires explicit audit trail for every financial commitment

**What this exercise unlocks:**
- **Constraint #4 (Scalability)**: Message format decomposition → 8 specialist agents × 3k tokens each (vs. 1 generalist × 24k tokens) → context budget under 80% throughout full PO lifecycle → error rate drops from 12% → <2%
- **Foundation for Ch.4 (Event-Driven)**: Async message passing infrastructure built here enables pub/sub migration → 1,000 concurrent POs without blocking threads → <4hr SLA achievable
- **Foundation for Ch.6 (Trust & Audit)**: Structured message envelopes with `agent_id`, `role`, `content` fields → every agent action logged → HMAC signing layer (Ch.6) builds on this foundation → zero un-audited commitments requirement satisfied

**After completing this exercise:**
- You'll understand *why* message formats matter: structured payloads preserve context across agent handoffs without token overflow
- You'll see *where* coordination bottlenecks appear: synchronous request/response patterns vs. async pub/sub event-driven alternatives
- You'll know *what* to measure: context token usage per agent, message round-trip latency, error rate under concurrent load — the 3 key metrics that validate OrderFlow's scalability constraint

## What You'll Build

A multi-agent system with:
- **CoordinatorAgent**: Decomposes tasks and orchestrates workers
- **WorkerAgent**: Executes atomic tasks in parallel  
- **ResearchAgent**: Gathers information with caching
- **Message passing**: Request/response/broadcast patterns
- **Shared state**: Conflict detection and versioning
- **Metrics**: Real-time coordination analytics

## Implementation Tasks

### Phase 1: Agents (src/models.py)

1. **CoordinatorAgent.process_task()** - Task decomposition and worker assignment
2. **CoordinatorAgent.respond_to_message()** - Handle worker responses
3. **WorkerAgent.process_task()** - Execute tasks independently
4. **WorkerAgent.respond_to_message()** - Process task requests
5. **ResearchAgent.process_task()** - Research with caching
6. **ExperimentRunner.run_experiment()** - Orchestrate multi-agent collaboration
7. **ExperimentRunner.print_metrics()** - Display coordination metrics

### Phase 2: Infrastructure (src/features.py)

8. **MessageParser.parse_message()** - Validate and parse messages
9. **MessageParser.extract_task_from_message()** - Extract task details
10. **MessageParser.validate_response()** - Match requests and responses
11. **SharedStateManager.update()** - State updates with conflict detection
12. **SharedStateManager.get()** - Retrieve state values
13. **SharedStateManager.lock/unlock()** - Exclusive state access
14. **ConversationHistory.add_message()** - Track conversation history
15. **ConversationHistory.get_conversation()** - Retrieve conversation thread
16. **MessageRouter.route()** - Route messages with priority handling

**Total: 17 TODOs** (see inline code comments for details)

## Core Concepts

### Agent Autonomy
Each agent has inbox, outbox, state, and metrics. They operate independently.

### Task Decomposition
- **Low complexity**: 1 subtask
- **Medium complexity**: 3 subtasks (research → execute → validate)
- **High complexity**: 5 subtasks (research → design → execute → test → validate)

### Message Passing
```python
Message(sender, recipient, content, timestamp, message_type)
```
Types: request, response, broadcast

### Coordination Pattern
```
Coordinator receives task → Decomposes → Assigns to workers (round-robin) 
→ Workers execute in parallel → Coordinator aggregates results
```

### Shared State
- Locks prevent conflicts
- Versioning enables rollback
- Timestamps track ordering

### Emergent Behavior
Simple agent rules → complex system behavior (load balancing, caching, specialization)

## Success Criteria

- [ ] All 17 TODOs implemented
- [ ] `python main.py` runs without errors
- [ ] 3 demo tasks complete with visible output
- [ ] Coordination metrics display properly
- [ ] Message routing works between agents

## Extension Ideas

**Level 1:** CriticAgent, priority queues, retry logic, task dependencies

**Level 2:** LLM integration (OpenAI), threading, agent learning, visualization

**Level 3:** Persistent state (Redis), distributed deployment, AutoGen, Prometheus

## Reference

Full documentation with detailed explanations available in files:
- [src/models.py](src/models.py) - Agent implementations with TODOs
- [src/features.py](src/features.py) - Infrastructure with TODOs
- [main.py](main.py) - Demonstration script

Each TODO is a 1-line description. See _REFERENCE/ directory for complete implementations.

---

*For questions, see [CONTRIBUTING.md](../../CONTRIBUTING.md)*
