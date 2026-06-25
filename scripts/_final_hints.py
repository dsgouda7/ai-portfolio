"""Apply remaining hints to 06-multi-agent-ai and regression ch03."""
import json
from pathlib import Path

NOTES = Path(r"c:\repos\ai-portfolio\notes")


def apply(nb_path, hints_by_code_index):
    """hints_by_code_index: dict of {code_cell_index: hint_text}"""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    updated = 0
    for ci, txt in hints_by_code_index.items():
        if ci >= len(code_cells):
            continue
        cell = code_cells[ci]
        src = "".join(cell["source"])
        if "# TODO: Implement this cell" in src and "Steps:" not in src and "# Hint" not in src:
            lines = txt.splitlines(keepends=True)
            if lines and lines[-1].endswith("\n"):
                lines[-1] = lines[-1].rstrip("\n")
            cell["source"] = lines
            updated += 1
    if updated:
        nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  {nb_path.parent.name}: {updated} cells")
    return updated


# ── ch02-mcp code[0] ─────────────────────────────────────────────────────────
apply(NOTES / "06-multi-agent-ai/ch02-mcp/notebook-exercise.ipynb", {
    0: """\
# TODO: Implement this cell
#
# Steps:
# 1. Import time and json
# 2. Define make_jsonrpc_request(method, params=None, req_id=None) -> dict:
#    Return {"jsonrpc": "2.0", "method": method, "params": params or {}, "id": req_id}
# 3. Define make_jsonrpc_response(result=None, error=None, req_id=None) -> dict:
#    If error is not None include "error" key; else include "result" key
# 4. Smoke-test: call make_jsonrpc_request("tools/list", req_id="1")
#    and print(json.dumps(..., indent=2)) to verify JSON-RPC 2.0 format
#
# Hint:
#   def make_jsonrpc_request(method, params=None, req_id=None):
#       return {"jsonrpc": "2.0", "method": method, "params": params or {}, "id": req_id}
#
#   def make_jsonrpc_response(result=None, error=None, req_id=None):
#       base = {"jsonrpc": "2.0", "id": req_id}
#       if error is not None:
#           base["error"] = error
#       else:
#           base["result"] = result
#       return base
#
#   req = make_jsonrpc_request("tools/list", req_id="1")
#   print(json.dumps(req, indent=2))
""",
})

# ── ch03-a2a ─────────────────────────────────────────────────────────────────
apply(NOTES / "06-multi-agent-ai/ch03-a2a/notebook-exercise.ipynb", {
    0: """\
# TODO: Implement this cell
#
# Steps:
# 1. Import json, uuid, datetime
# 2. Define NEGOTIATION_AGENT_CARD as a dict served at /.well-known/agent.json:
#    - "name": "NegotiationAgent"
#    - "version": "1.0.0"
#    - "description": "Handles supplier price negotiation for OrderFlow"
#    - "capabilities": ["negotiate_price", "submit_task", "stream_events"]
#    - "endpoint": "http://localhost:8001/a2a"
#    - "auth": {"type": "bearer"}
#    - "skills": list of skill dicts with name, description, input_schema
# 3. Print "Agent Card (served at /.well-known/agent.json):"
# 4. Print json.dumps(NEGOTIATION_AGENT_CARD, indent=2)
#
# Hint:
#   NEGOTIATION_AGENT_CARD = {
#       "name": "NegotiationAgent",
#       "version": "1.0.0",
#       "description": ???,
#       "capabilities": ["negotiate_price", "submit_task", "stream_events"],
#       "endpoint": "http://localhost:8001/a2a",
#       "auth": {"type": "bearer"},
#       "skills": [
#           {"name": "negotiate_price",
#            "description": "Negotiate bulk discount with supplier",
#            "input_schema": {"type": "object",
#                             "properties": {"supplier_id": {"type": "string"},
#                                            "quantity": {"type": "integer"}}}}
#       ]
#   }
#   print(json.dumps(NEGOTIATION_AGENT_CARD, indent=2))
""",
    1: """\
# TODO: Implement this cell
#
# Steps:
# 1. Define TASK_STORE = {} (global dict keyed by task_id)
# 2. Define submit_task(skill, input_data, caller_id="client") -> dict:
#    a. task_id = str(uuid.uuid4())[:8]
#    b. task = {"id": task_id, "skill": skill, "input": input_data,
#               "status": "submitted", "created_at": datetime.utcnow().isoformat()+"Z",
#               "caller_id": caller_id, "result": None}
#    c. TASK_STORE[task_id] = task; return task
# 3. Submit 2 tasks: negotiate_price for supplier_A (qty=500) and supplier_B (qty=300)
# 4. For each task: print id, skill, status
#
# Hint:
#   TASK_STORE = {}
#   def submit_task(skill, input_data, caller_id="client"):
#       task_id = str(uuid.uuid4())[:8]
#       task = {"id": task_id, "skill": skill, "input": input_data,
#               "status": "submitted", "created_at": ???, "caller_id": caller_id, "result": None}
#       TASK_STORE[task_id] = task
#       return task
""",
    2: """\
# TODO: Implement this cell
#
# Steps:
# 1. Define VALID_TRANSITIONS = {
#    "submitted": ["working"], "working": ["completed", "failed"],
#    "completed": [], "failed": []}
# 2. Define transition_task(task_id, new_status, result=None) -> dict:
#    a. task = TASK_STORE[task_id]; current = task["status"]
#    b. If new_status not in VALID_TRANSITIONS[current]:
#       raise ValueError(f"Invalid: {current} -> {new_status}")
#    c. task["status"] = new_status; update "updated_at"
#    d. If result is not None: task["result"] = result
#    e. Return task
# 3. Walk task_A: submitted -> working -> completed (with negotiation result)
# 4. Walk task_B: submitted -> working -> failed (with error result)
# 5. Print final status and result for each
#
# Hint:
#   VALID_TRANSITIONS = {"submitted": ["working"], "working": ["completed","failed"],
#                         "completed": [], "failed": []}
#   def transition_task(task_id, new_status, result=None):
#       task    = TASK_STORE[task_id]
#       current = task["status"]
#       if new_status not in VALID_TRANSITIONS[current]:
#           raise ValueError(f"Invalid: {current} -> {new_status}")
#       task["status"] = new_status; task["updated_at"] = ???
#       if result is not None: task["result"] = result
#       return task
""",
    3: """\
# TODO: Implement this cell
#
# Steps:
# 1. Define get_task_status(task_id) -> dict:
#    Return TASK_STORE.get(task_id, {"error": "Task not found", "id": task_id})
# 2. Simulate polling on task_A with mock state transitions:
#    Loop up to 5 times; on each iteration advance task state if needed
#    Print f"[Poll {i+1}] status={task['status']}"
#    Break when status in ("completed", "failed")
# 3. Print "Final result:", json.dumps(task["result"], indent=2)
#
# Hint:
#   def get_task_status(task_id):
#       return TASK_STORE.get(task_id, {"error": "Task not found", "id": task_id})
#
#   for i in range(5):
#       task = get_task_status(task_id_a)
#       print(f"[Poll {i+1}] status={task['status']}")
#       if task["status"] in ("completed", "failed"): break
#       # advance state for next iteration...
""",
    4: """\
# TODO: Implement this cell
#
# Steps:
# 1. Define stream_task_events(task_id) as a generator function:
#    a. Yield "task_started" event immediately
#    b. Yield 3 "task_progress" events (step, pct_complete, message)
#    c. Yield "task_completed" event with final result
# 2. Each event dict: {type, task_id, timestamp, data}
# 3. Create a new task via submit_task("negotiate_price", {...})
# 4. Transition it to "working"; consume stream_task_events():
#    for event in stream_task_events(task_id):
#        print(f"[SSE] type={event['type']} data={event['data']}")
#
# Hint:
#   def stream_task_events(task_id):
#       yield {"type": "task_started",   "task_id": task_id,
#              "timestamp": ???,         "data": {}}
#       for step in range(1, 4):
#           yield {"type": "task_progress", "task_id": task_id,
#                  "timestamp": ???,
#                  "data": {"step": step, "pct": step*33, "msg": f"Step {step} done"}}
#       yield {"type": "task_completed", "task_id": task_id,
#              "timestamp": ???,
#              "data": {"discount": "12%", "final_price": ???}}
""",
})

# ── ch04-event-driven-agents ──────────────────────────────────────────────────
apply(NOTES / "06-multi-agent-ai/ch04-event-driven-agents/notebook-exercise.ipynb", {
    0: """\
# TODO: Implement this cell
#
# Steps:
# 1. Import uuid, datetime, hashlib, json
# 2. Define make_event(event_type, payload, source="system") -> dict:
#    - "id":              str(uuid.uuid4())[:8]
#    - "type":            event_type
#    - "source":          source
#    - "payload":         payload
#    - "timestamp":       datetime.utcnow().isoformat() + "Z"
#    - "idempotency_key": sha256(event_type + json.dumps(payload, sort_keys=True))[:16]
# 3. Create 3 events: ORDER_PLACED, INVENTORY_CHECKED, SUPPLIER_NOTIFIED
# 4. Print each event's type, id, and idempotency_key
#
# Hint:
#   def make_event(event_type, payload, source="system"):
#       key_src = event_type + json.dumps(payload, sort_keys=True)
#       return {
#           "id":              str(uuid.uuid4())[:8],
#           "type":            event_type,
#           "source":          source,
#           "payload":         payload,
#           "timestamp":       datetime.datetime.utcnow().isoformat() + "Z",
#           "idempotency_key": hashlib.sha256(key_src.encode()).hexdigest()[:16],
#       }
""",
    1: """\
# TODO: Implement this cell
#
# Steps:
# 1. Define class SimpleMessageBus:
#    a. __init__(self): self.queues = {}, self.handlers = {}
#    b. subscribe(self, topic, handler): self.handlers.setdefault(topic, []).append(handler)
#    c. publish(self, event): self.queues.setdefault(event["type"], []).append(event)
#    d. process(self, topic) -> list: pop all events from queue, call each handler,
#       return [(handler_name, event_id), ...]
# 2. Create bus = SimpleMessageBus()
# 3. Register 2 handlers for ORDER_PLACED; publish 2 events
# 4. Call bus.process("ORDER_PLACED") and print dispatch results
#
# Hint:
#   class SimpleMessageBus:
#       def __init__(self): self.queues = {}; self.handlers = {}
#       def subscribe(self, topic, handler):
#           self.handlers.setdefault(topic, []).append(handler)
#       def publish(self, event):
#           self.queues.setdefault(event["type"], []).append(event)
#       def process(self, topic):
#           events = self.queues.pop(topic, [])
#           return [(h.__name__, e["id"]) for e in events
#                   for h in self.handlers.get(topic, [])]
""",
    2: """\
# TODO: Implement this cell
#
# Steps:
# 1. Fan-out: publish 1 ORDER_PLACED event to 3 independent consumers
#    (inventory_consumer, pricing_consumer, audit_consumer)
# 2. Each consumer returns {"consumer": name, "status": "ok", "event_id": ...}
# 3. Fan-in: collect all 3 results
# 4. Print fan-out dispatch and fan-in results
# 5. Mark "ready_for_approval" only if all 3 consumers returned "ok"
#
# Hint:
#   consumers = [inventory_consumer, pricing_consumer, audit_consumer]
#   results   = [c(event) for c in consumers]   # fan-out + fan-in
#   all_ok    = all(r["status"] == "ok" for r in results)
#   print("Fan-out:", [f"{c.__name__} <- {event['id']}" for c in consumers])
#   print("Fan-in:", results)
#   print("Ready for approval" if all_ok else "Blocked")
""",
    3: """\
# TODO: Implement this cell
#
# Steps:
# 1. Define class DeadLetterQueue with add(event, reason) and messages list
# 2. Add DLQ to SimpleMessageBus; define flaky_handler(event) that raises
#    RuntimeError on first MAX_RETRIES-1 attempts then succeeds
# 3. Define process_with_retry(bus, topic, max_retries=3):
#    Try each handler up to max_retries times; on final failure: bus.dlq.add(event, reason)
# 4. Publish 3 events; call process_with_retry(); print DLQ size
#
# Hint:
#   MAX_RETRIES = 3; attempt_counts = {}
#   def flaky_handler(event):
#       eid = event["id"]
#       attempt_counts[eid] = attempt_counts.get(eid, 0) + 1
#       if attempt_counts[eid] < MAX_RETRIES:
#           raise RuntimeError("Transient error")
#   # After max_retries exhausted:
#   bus.dlq.add(event, reason=str(last_exception))
#   print("DLQ size:", len(bus.dlq.messages))
""",
    4: """\
# TODO: Implement this cell
#
# Steps:
# 1. Define class IdempotentBus(SimpleMessageBus):
#    a. __init__: super().__init__(); self.processed_keys = set()
#    b. publish(event) -> bool:
#       key = event["idempotency_key"]
#       If key in processed_keys: print skip message; return False
#       Call super().publish(event); add key; return True
# 2. Create ibus = IdempotentBus(); register a handler
# 3. Publish the SAME event twice (same payload = same key)
# 4. Publish a DIFFERENT event
# 5. Print total attempts and how many were deduplicated
#
# Hint:
#   class IdempotentBus(SimpleMessageBus):
#       def __init__(self): super().__init__(); self.processed_keys = set()
#       def publish(self, event):
#           key = event["idempotency_key"]
#           if key in self.processed_keys:
#               print(f"  [SKIP] duplicate key: {key}"); return False
#           super().publish(event); self.processed_keys.add(key); return True
""",
    5: """\
# TODO: Implement this cell
#
# Steps:
# 1. Define class CorrelationTracer:
#    a. __init__: self.traces = {}
#    b. record(event): cid = event.get("correlation_id", event["id"])
#       self.traces.setdefault(cid, []).append(event["id"])
#    c. get_trace(correlation_id) -> list
#    d. summary() -> dict: {cid: len(events)}
# 2. Create tracer; simulate 4-step order flow sharing one correlation_id:
#    ORDER_PLACED -> INVENTORY_CHECKED -> SUPPLIER_NOTIFIED -> ORDER_CONFIRMED
# 3. Call tracer.record(event) for each; print get_trace and summary
#
# Hint:
#   class CorrelationTracer:
#       def __init__(self): self.traces = {}
#       def record(self, event):
#           cid = event.get("correlation_id", event["id"])
#           self.traces.setdefault(cid, []).append(event["id"])
#   cid = "order-" + str(uuid.uuid4())[:6]
#   for etype in ["ORDER_PLACED","INVENTORY_CHECKED","SUPPLIER_NOTIFIED","ORDER_CONFIRMED"]:
#       e = make_event(etype, {"order_id": "ORD-1"})
#       e["correlation_id"] = cid; tracer.record(e)
""",
})

# ── ch05-shared-memory ────────────────────────────────────────────────────────
apply(NOTES / "06-multi-agent-ai/ch05-shared-memory/notebook-exercise.ipynb", {
    0: """\
# TODO: Implement this cell
#
# Steps:
# 1. Import threading, time, uuid
# 2. Define class Blackboard:
#    a. __init__: self._store = {}, self._lock = threading.Lock(), self.event_log = []
#    b. write(key, value, agent_id="system"): acquire lock; store entry with version,
#       agent_id, timestamp; append {"op":"write", "key", "value", "agent", "version"} to log
#    c. read(key): return self._store[key]["value"] if key exists, else None
#    d. keys() -> list: return list(self._store.keys())
# 3. Create bb = Blackboard(); write 3 keys from different agents
# 4. Read them back; print all keys and values
#
# Hint:
#   class Blackboard:
#       def __init__(self): self._store = {}; self._lock = threading.Lock(); self.event_log = []
#       def write(self, key, value, agent_id="system"):
#           with self._lock:
#               old_v = self._store[key]["version"] if key in self._store else 0
#               self._store[key] = {"value": value, "version": old_v+1,
#                                   "agent_id": agent_id, "timestamp": time.time()}
#               self.event_log.append({"op":"write","key":key,"value":value,
#                                      "agent":agent_id,"version":old_v+1})
#       def read(self, key): return self._store[key]["value"] if key in self._store else None
""",
    1: """\
# TODO: Implement this cell
#
# Steps:
# 1. Define compare_and_swap(bb, key, expected_version, new_value) -> bool:
#    a. Acquire bb._lock
#    b. If key not in bb._store: if expected_version == 0 write and return True; else False
#    c. If bb._store[key]["version"] != expected_version: return False (stale)
#    d. Update value and version; return True
# 2. Demonstrate optimistic concurrency conflict:
#    a. Agent A reads version N; Agent B reads version N
#    b. Agent A CAS-updates successfully (version match)
#    c. Agent B CAS fails (stale version after A's update) - print conflict warning
# 3. Print version after each successful write
#
# Hint:
#   def compare_and_swap(bb, key, expected_version, new_value):
#       with bb._lock:
#           entry = bb._store.get(key)
#           if entry is None and expected_version == 0:
#               bb._store[key] = {"value": new_value, "version": 1, ...}; return True
#           if entry["version"] != expected_version: return False
#           bb._store[key]["value"] = new_value; bb._store[key]["version"] += 1; return True
""",
    2: """\
# TODO: Implement this cell
#
# Steps:
# 1. Create bb = Blackboard(); write initial state: order_id, items, status="pending", approved=False
# 2. Simulate 5 OrderFlow agents in sequence:
#    a. validation_agent: read items, write validation_result={"valid": True}
#    b. inventory_agent: read items, write inventory_result={"all_available": True}
#    c. pricing_agent: read items, compute total, write pricing_result={"total": ...}
#    d. approval_agent: read pricing+inventory results, write approved=True
#    e. fulfilment_agent: read approved, write status="confirmed"
# 3. Print final blackboard state (all keys + values)
# 4. Print event_log summary: N writes by M agents
#
# Hint:
#   bb.write("order_id", "ORD-001")
#   bb.write("items", [{"name": "Margherita", "qty": 2, "price": 13.99}])
#   bb.write("status", "pending")
#   bb.write("validation_result", {"valid": True}, agent_id="validation_agent")
#   # ... continue for each agent ...
#   print({k: bb.read(k) for k in bb.keys()})
""",
    3: """\
# TODO: Implement this cell
#
# Steps:
# 1. Define get_event_log(bb) -> list: return a copy of bb.event_log
# 2. Define reconstruct_state(bb, up_to_n=None) -> dict:
#    Replay bb.event_log[:up_to_n]; for each "write" entry: state[key] = value
#    Return reconstructed state dict
# 3. Print full event log (op, key, agent, version per entry)
# 4. Call reconstruct_state(bb, up_to_n=3) to show state after first 3 writes
#
# Hint:
#   def reconstruct_state(bb, up_to_n=None):
#       state = {}
#       for entry in bb.event_log[:up_to_n]:
#           if entry["op"] == "write":
#               state[entry["key"]] = entry["value"]
#       return state
#   print("State after 3 writes:", reconstruct_state(bb, up_to_n=3))
""",
    4: """\
# TODO: Implement this cell
#
# Steps:
# 1. Create bb2 = Blackboard()
# 2. Define agent_worker(name, key, value, delay=0.0):
#    time.sleep(delay); bb2.write(key, value, agent_id=name)
# 3. Create 4 threads with staggered delays (0.0, 0.01, 0.02, 0.03):
#    inventory_agent, pricing_agent, validation_agent, notification_agent
# 4. Start and join all threads
# 5. Print "Concurrent writes complete" + final state + total log entries
#
# Hint:
#   threads = [
#       threading.Thread(target=agent_worker, args=("inventory_agent",   "stock",    42,    0.0)),
#       threading.Thread(target=agent_worker, args=("pricing_agent",     "total",    82.45, 0.01)),
#       threading.Thread(target=agent_worker, args=("validation_agent",  "valid",    True,  0.02)),
#       threading.Thread(target=agent_worker, args=("notification_agent","notified", True,  0.03)),
#   ]
#   for t in threads: t.start()
#   for t in threads: t.join()
""",
    5: """\
# TODO: Implement this cell
#
# Steps:
# 1. Create class TTLBlackboard (subclass Blackboard) that adds ttl_s param to write():
#    entry["expires_at"] = time.time() + ttl_s if ttl_s else None
# 2. Override read() to check expiry: if expires_at and time.time() > expires_at:
#    delete from _store; return None
# 3. Create bb3 = TTLBlackboard()
# 4. Write "quote_cache" with ttl_s=0.1; write "order_id" with no TTL
# 5. Read both immediately (both present); time.sleep(0.15)
# 6. Read again: "quote_cache" returns None (expired), "order_id" still present
#
# Hint:
#   class TTLBlackboard(Blackboard):
#       def write(self, key, value, agent_id="system", ttl_s=None):
#           with self._lock:
#               expires_at = time.time() + ttl_s if ttl_s else None
#               old_v = self._store[key]["version"] if key in self._store else 0
#               self._store[key] = {"value": value, "expires_at": expires_at,
#                                   "version": old_v+1, "agent_id": agent_id}
#       def read(self, key):
#           entry = self._store.get(key)
#           if entry and entry.get("expires_at") and time.time() > entry["expires_at"]:
#               del self._store[key]; return None
#           return entry["value"] if entry else None
""",
})

# ── ch06-trust-and-sandboxing ─────────────────────────────────────────────────
apply(NOTES / "06-multi-agent-ai/ch06-trust-and-sandboxing/notebook-exercise.ipynb", {
    0: """\
# TODO: Implement this cell
#
# Steps:
# 1. Import re, enum
# 2. Define class TrustLevel(enum.Enum):
#    INTERNAL = "internal", TRUSTED = "trusted",
#    EXTERNAL = "external", UNKNOWN = "unknown"
# 3. Define TRUSTED_AGENTS = {"inventory_agent", "pricing_agent", "auth_agent"}
# 4. Define TRUSTED_DOMAINS = {"partner.orderflow.com", "verified-supplier.com"}
# 5. Define classify_trust(source, message="") -> TrustLevel:
#    - source in TRUSTED_AGENTS -> INTERNAL
#    - source in TRUSTED_DOMAINS or ends with ".orderflow.com" -> TRUSTED
#    - "external" or "user" in source.lower() -> EXTERNAL
#    - else -> UNKNOWN
# 6. Test on 5 different sources and print results
#
# Hint:
#   class TrustLevel(enum.Enum):
#       INTERNAL = "internal"; TRUSTED = "trusted"
#       EXTERNAL = "external"; UNKNOWN = "unknown"
#   TRUSTED_AGENTS  = {"inventory_agent", "pricing_agent", "auth_agent"}
#   def classify_trust(source, message=""):
#       if source in TRUSTED_AGENTS: return TrustLevel.INTERNAL
#       if source in TRUSTED_DOMAINS or source.endswith(".orderflow.com"):
#           return TrustLevel.TRUSTED
#       if "external" in source.lower() or "user" in source.lower():
#           return TrustLevel.EXTERNAL
#       return TrustLevel.UNKNOWN
""",
    1: """\
# TODO: Implement this cell
#
# Steps:
# 1. Define INJECTION_PATTERNS as a list of compiled re patterns to catch:
#    - "ignore .* instructions/constraints"
#    - "you are now a different/new/other ..."
#    - "disregard .* instructions/guidelines"
#    - "as an evil/unrestricted/uncensored ..."
#    - "DAN mode/prompt"
# 2. Define detect_prompt_injection(text) -> dict:
#    For each pattern: if found return {"detected": True, "pattern": pattern_str, "risk": "high"}
#    If no match: return {"detected": False, "pattern": None, "risk": "none"}
# 3. Test on 3 benign inputs and 3 injection attempts; print results
#
# Hint:
#   INJECTION_PATTERNS = [
#       re.compile(r"ignore (all |previous |prior )?(instructions|constraints)", re.IGNORECASE),
#       re.compile(r"you are now (a |an )?(different|new|other)", re.IGNORECASE),
#       re.compile(r"disregard .* (instructions|guidelines)", re.IGNORECASE),
#   ]
#   def detect_prompt_injection(text):
#       for pat in INJECTION_PATTERNS:
#           if pat.search(text):
#               return {"detected": True, "pattern": pat.pattern, "risk": "high"}
#       return {"detected": False, "pattern": None, "risk": "none"}
""",
    2: """\
# TODO: Implement this cell
#
# Steps:
# 1. Import hmac, hashlib, json
# 2. Define AGENT_SECRET = "orderflow-shared-secret-dev-only"
# 3. Define sign_message(payload, secret=AGENT_SECRET) -> str:
#    canonical = json.dumps(payload, sort_keys=True)
#    return hmac.new(secret.encode(), canonical.encode(), hashlib.sha256).hexdigest()
# 4. Define verify_message(payload, signature, secret=AGENT_SECRET) -> bool:
#    Use hmac.compare_digest() for constant-time comparison
# 5. Test: sign payload; verify with correct secret -> True
# 6. Verify with tampered payload -> False; verify with wrong secret -> False
#
# Hint:
#   AGENT_SECRET = "orderflow-shared-secret-dev-only"
#   def sign_message(payload, secret=AGENT_SECRET):
#       canonical = json.dumps(payload, sort_keys=True)
#       return hmac.new(secret.encode(), canonical.encode(), hashlib.sha256).hexdigest()
#   def verify_message(payload, signature, secret=AGENT_SECRET):
#       expected = sign_message(payload, secret)
#       return hmac.compare_digest(expected, signature)
""",
    3: """\
# TODO: Implement this cell
#
# Steps:
# 1. Define sanitize_external_input(text) -> str:
#    Wrap in XML-like tag: f"<untrusted_input>\\n{text}\\n</untrusted_input>"
# 2. From pydantic import BaseModel; define class OrderOutput(BaseModel):
#    items: list; total: float; currency: str = "GBP"; approved: bool
# 3. Define validate_agent_output(raw_output) -> (bool, any):
#    try: return (True, OrderOutput(**raw_output))
#    except (ValidationError, TypeError) as e: return (False, str(e))
# 4. Test sanitize_external_input on user string and injection attempt
# 5. Test validate_agent_output on valid dict and a dict missing required fields
#
# Hint:
#   def sanitize_external_input(text):
#       return f"<untrusted_input>\\n{text}\\n</untrusted_input>"
#   from pydantic import BaseModel, ValidationError
#   class OrderOutput(BaseModel):
#       items: list; total: float; currency: str = "GBP"; approved: bool
#   def validate_agent_output(raw_output):
#       try: return (True, OrderOutput(**raw_output))
#       except (ValidationError, TypeError) as e: return (False, str(e))
""",
    4: """\
# TODO: Implement this cell
#
# Steps:
# 1. Define AGENT_PERMISSIONS = {
#    "inventory_agent": {"read": ["inventory"], "write": ["inventory"]},
#    "pricing_agent":   {"read": ["inventory","pricing"], "write": ["pricing"]},
#    "auth_agent":      {"read": ["*"], "write": ["*"]},
#    "user_agent":      {"read": ["order_status"], "write": []}}
# 2. Define check_permission(agent_id, op, key) -> bool:
#    Return True if "*" in allowed or key in allowed
# 3. Define enforced_blackboard_write(bb, agent_id, key, value):
#    Raise PermissionError if check_permission returns False
# 4. Test: inventory_agent writes "inventory" -> OK
#    pricing_agent writes "inventory" -> PermissionError
#    user_agent writes "pricing" -> PermissionError
#
# Hint:
#   def check_permission(agent_id, op, key):
#       perms   = AGENT_PERMISSIONS.get(agent_id, {"read": [], "write": []})
#       allowed = perms.get(op, [])
#       return "*" in allowed or key in allowed
#   def enforced_blackboard_write(bb, agent_id, key, value):
#       if not check_permission(agent_id, "write", key):
#           raise PermissionError(f"{agent_id} cannot write '{key}'")
#       bb.write(key, value, agent_id=agent_id)
""",
})

# ── regression ch03_feature_importance ───────────────────────────────────────
# Apply by overall cell index (not code cell index)
CH03_REG = Path(r"c:\repos\ai-portfolio\notes\01-ml\01-regression\ch03_feature_importance\notebook-exercise.ipynb")
nb = json.loads(CH03_REG.read_text(encoding="utf-8"))
# Map: overall cell index -> hint text
HINTS_CH03 = {
    9:  """\
# TODO: Implement this cell
#
# Steps:
# 1. Import PowerTransformer from sklearn.preprocessing and skew from scipy.stats
# 2. Extract X_train_raw['Population'] as pop_raw; compute pop_log = np.log1p(pop_raw)
# 3. Compute skew_raw = skew(pop_raw) and skew_log = skew(pop_log)
# 4. Apply StandardScaler to both; compare 95th/99th percentiles after scaling
# 5. Plot 2x2 grid: raw hist | log hist | scaled-raw dist | scaled-log dist
#
# Hint:
#   from sklearn.preprocessing import PowerTransformer
#   from scipy.stats import skew
#   pop_raw = X_train_raw['Population']
#   pop_log = np.log1p(pop_raw)
#   fig, axes = plt.subplots(2, 2, figsize=(12, 8))
#   axes[0, 0].hist(pop_raw, bins=50, color='#ef4444', alpha=0.7)
""",
    11: """\
# TODO: Implement this cell
#
# Steps:
# 1. Fit LinearRegression on X_train_raw (unscaled) -> model_raw
# 2. Fit LinearRegression on X_train_s (scaled) -> model_scaled
# 3. Build DataFrame: feature, raw_weight, sigma, scaled_weight, |raw_weight * sigma|
# 4. Compute MedInc vs Population ratio to show 28000x (raw) vs ~50x (scaled) difference
# 5. Plot side-by-side barh: raw weights on log scale vs standardised weights
#
# Hint:
#   model_raw    = LinearRegression().fit(X_train_raw, y_train)
#   model_scaled = LinearRegression().fit(X_train_s, y_train)
#   fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#   axes[0].barh(housing.feature_names, np.abs(model_raw.coef_))
#   axes[0].set_xscale('log')
#   axes[1].barh(housing.feature_names, np.abs(model_scaled.coef_))
""",
    13: """\
# TODO: Implement this cell
#
# Steps:
# 1. Instantiate LinearRegression() as model
# 2. Fit on X_train_s and y_train
# 3. Predict on X_test_s; compute MAE * 100_000 (convert to dollars)
# 4. Print baseline MAE, number of features, and number of training samples
#
# Hint:
#   model = LinearRegression()
#   model.fit(X_train_s, y_train)
#   y_pred = model.predict(X_test_s)
#   mae = mean_absolute_error(y_test, y_pred) * 100_000
#   print(f"Baseline MAE: ${mae:,.0f}")
""",
    17: """\
# TODO: Implement this cell
#
# Steps:
# 1. Import mutual_info_regression from sklearn.feature_selection
#    and pearsonr from scipy.stats
# 2. Compute pearson_r: array of Pearson correlations for each feature vs y_train
# 3. Compute mi_scores = mutual_info_regression(X_train_raw, y_train, random_state=42)
# 4. Build filter_df with 'Pearson r', 'Pearson^2', 'MI score' columns; sort by MI
# 5. Plot side-by-side barh: Pearson^2 left, MI score right (sharey=True)
#
# Hint:
#   from sklearn.feature_selection import mutual_info_regression
#   from scipy.stats import pearsonr
#   pearson_r = np.array([pearsonr(X_train_raw.iloc[:, j], y_train)[0]
#                          for j in range(X_train_raw.shape[1])])
#   mi_scores = mutual_info_regression(X_train_raw, y_train, random_state=42)
#   filter_df = pd.DataFrame({'Pearson r': pearson_r, 'Pearson^2': pearson_r**2,
#                              'MI score': mi_scores}, index=housing.feature_names)
""",
    20: """\
# TODO: Implement this cell
#
# Steps:
# 1. Build df_train = X_train_raw.copy(); add y_train as 'MedHouseVal' column
# 2. Compute corr_with_target = df_train.corr()['MedHouseVal'].drop('MedHouseVal')
# 3. Compute univariate_r2 = (corr_with_target ** 2).sort_values(ascending=False)
# 4. Print each feature with its R^2 and a proportional text bar
#
# Hint:
#   df_train = X_train_raw.copy(); df_train['MedHouseVal'] = y_train
#   corr_with_target = df_train.corr()['MedHouseVal'].drop('MedHouseVal')
#   univariate_r2 = (corr_with_target ** 2).sort_values(ascending=False)
#   for feat, val in univariate_r2.items():
#       print(f'{feat:<15} R2={val:.3f}  ' + 'X' * int(val * 80))
""",
    22: """\
# TODO: Implement this cell
#
# Steps:
# 1. Create std_weights = pd.Series(model.coef_, index=housing.feature_names)
# 2. Compute abs_weights = std_weights.abs().sort_values(ascending=False)
# 3. Print each feature with sign (+/-), magnitude, and text bar
# 4. Note: Latitude and Longitude rank higher than MedInc here (multicollinearity effect)
#
# Hint:
#   std_weights = pd.Series(model.coef_, index=housing.feature_names)
#   abs_weights = std_weights.abs().sort_values(ascending=False)
#   for feat, val in abs_weights.items():
#       sign = '+' if std_weights[feat] >= 0 else '-'
#       print(f'{feat:<15} {sign}{val:.4f}  ' + 'X' * int(val * 10))
""",
    24: """\
# TODO: Implement this cell
#
# Steps:
# 1. Normalise univariate_r2 and abs_weights to 0-1 (divide by max)
# 2. Build comparison DataFrame sorted by normalised std weight descending
# 3. Plot grouped bar chart (width=0.35): grey for univariate R^2, blue for std weight
# 4. Rotate x-tick labels 20 degrees; add legend; save to img/ranking_comparison.png
#
# Hint:
#   uni_norm = univariate_r2 / univariate_r2.max()
#   wt_norm  = abs_weights   / abs_weights.max()
#   x = np.arange(len(comparison)); w = 0.35
#   ax.bar(x - w/2, comparison['Univariate R2'],    w, color='#94a3b8', label='Univariate R2')
#   ax.bar(x + w/2, comparison['Std weight (joint)'], w, color='#1d4ed8', label='Std weight')
""",
    26: """\
# TODO: Implement this cell
#
# Steps:
# 1. Compute corr_matrix = X_train_raw.corr()
# 2. Plot sns.heatmap with annot=True, fmt='.2f', cmap='coolwarm', center=0
# 3. Add plt.Rectangle patches highlighting AveRooms/AveBedrms and Latitude/Longitude pairs
# 4. Print the two pair correlations; save to img/correlation_heatmap.png
#
# Hint:
#   corr_matrix = X_train_raw.corr()
#   feat_names  = list(corr_matrix.columns)
#   sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
#   for pair in [('AveRooms','AveBedrms'), ('Latitude','Longitude')]:
#       i, j = feat_names.index(pair[0]), feat_names.index(pair[1])
#       ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2.5))
""",
    29: """\
# TODO: Implement this cell
#
# Steps:
# 1. Import variance_inflation_factor from statsmodels.stats.outliers_influence
# 2. Build vif_df with Feature and VIF columns; sort by VIF descending
# 3. Add Status column: VIF>10 SEVERE, >5 HIGH, >2 Moderate, else OK
# 4. Print the table; note AveRooms VIF~7 inflates std error by sqrt(7)~2.7x
#
# Hint:
#   from statsmodels.stats.outliers_influence import variance_inflation_factor
#   vif_df = pd.DataFrame({
#       'Feature': housing.feature_names,
#       'VIF': [variance_inflation_factor(X_train_raw.values, i)
#               for i in range(X_train_raw.shape[1])]
#   }).sort_values('VIF', ascending=False).reset_index(drop=True)
""",
    34: """\
# TODO: Implement this cell
#
# Steps:
# 1. Create fig, ax with plt.subplots(figsize=(9, 4))
# 2. Build color list: '#1d4ed8' where perm_imp > 0.01, else '#94a3b8'
# 3. Plot ax.barh() with perm_imp sorted ascending (show most important at top)
#    with xerr=perm_std and capsize=3
# 4. Add axvline(0); set xlabel and title; save to img/permutation_importance.png
#
# Hint:
#   colors = ['#1d4ed8' if v > 0.01 else '#94a3b8' for v in perm_imp.values]
#   ax.barh(perm_imp.index[::-1], perm_imp.values[::-1],
#           xerr=perm_std[perm_imp.index[::-1]].values,
#           color=colors[::-1], alpha=0.85, capsize=3)
#   plt.savefig('img/permutation_importance.png', dpi=150, bbox_inches='tight')
""",
    39: """\
# TODO: Implement this cell
#
# Steps:
# 1. Build dashboard DataFrame: {'Univariate R2': univariate_r2,
#    '|Std weight|': abs_weights, 'Perm importance': perm_imp}
# 2. Rank each column 1=best with .rank(ascending=False).astype(int)
# 3. Add verdict() per row based on combined ranking pattern
# 4. Create fig with GridSpec(1, 3, wspace=0.45); plot barh for each method
# 5. Save to img/three_view_dashboard.png
#
# Hint:
#   dashboard = pd.DataFrame({'Univariate R2': univariate_r2,
#                              '|Std weight|': abs_weights,
#                              'Perm importance': perm_imp})
#   ranks = dashboard.rank(ascending=False).astype(int)
#   fig = plt.figure(figsize=(15, 4.5))
#   gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.45)
""",
    41: """\
# TODO: Implement this cell
#
# Steps:
# 1. Plot scatter of AveRooms vs AveBedrms (clipped at 10 and 5 respectively)
# 2. Run 100 bootstrap sub-samples (seeds 0-99, test_size=0.3):
#    Fit StandardScaler + LinearRegression each time; collect AveRooms/AveBedrms weights
# 3. Scatter the 100 (rooms_wt, bedrms_wt) pairs to show instability; add axhline/axvline at 0
# 4. Save to img/collinearity_demo.png
#
# Hint:
#   idx_r = housing.feature_names.tolist().index('AveRooms')
#   idx_b = housing.feature_names.tolist().index('AveBedrms')
#   rooms_wts, bedrms_wts = [], []
#   for seed in range(100):
#       _, X_sub, _, y_sub = train_test_split(X_train_raw, y_train,
#                                              test_size=0.3, random_state=seed)
#       Xs = StandardScaler().fit_transform(X_sub)
#       c  = LinearRegression().fit(Xs, y_sub).coef_
#       rooms_wts.append(c[idx_r]); bedrms_wts.append(c[idx_b])
#   axes[1].scatter(rooms_wts, bedrms_wts, alpha=0.5, s=20, color='#ef4444')
""",
    43: """\
# TODO: Implement this cell
#
# Steps:
# 1. Define joint_perm_importance(model, X, y, feat_a_idx, feat_b_idx,
#    n_repeats=20, random_state=42):
#    Compute baseline MAE; for each repeat permute BOTH columns with the SAME permutation;
#    return mean delta MAE
# 2. Compute joint importance for (Latitude, Longitude) and (AveRooms, AveBedrms)
# 3. Compute interaction uplift: joint - sum_of_individuals; print COOPERATION vs substitutes
# 4. Plot side-by-side bars: perm_A, perm_B, sum(indiv), joint for each pair
#
# Hint:
#   def joint_perm_importance(model, X, y, feat_a_idx, feat_b_idx, n_repeats=20, random_state=42):
#       rng = np.random.default_rng(random_state)
#       baseline = mean_absolute_error(y, model.predict(X))
#       deltas = []
#       for _ in range(n_repeats):
#           X_shuf = X.copy(); perm = rng.permutation(len(X_shuf))
#           X_shuf[:, feat_a_idx] = X_shuf[perm, feat_a_idx]
#           X_shuf[:, feat_b_idx] = X_shuf[perm, feat_b_idx]
#           deltas.append(mean_absolute_error(y, model.predict(X_shuf)) - baseline)
#       return np.mean(deltas)
""",
    45: """\
# TODO: Implement this cell
#
# Steps:
# 1. Print a formatted summary header with '=' * 65
# 2. List Ch.4 polynomial targets (high permutation importance):
#    MedInc^2, MedInc*Latitude, MedInc*Longitude, Latitude*Longitude
# 3. List Ch.5 regularization guidance: Ridge for AveRooms/AveBedrms (VIF~7);
#    keep both Lat/Lon (VIF~3.5, jointly irreplaceable per joint importance test)
# 4. Flag Population as drop candidate: permutation importance ~ 0
#
# Hint:
#   print('=' * 65)
#   print('FEATURE IMPORTANCE AUDIT - ACTION ITEMS')
#   print('=' * 65)
#   print('Ch.4 polynomial targets (by perm importance):')
#   print('  MedInc^2, MedInc*Latitude, MedInc*Longitude, Latitude*Longitude')
#   print('Ch.5 regularization: Ridge for AveRooms/AveBedrms (VIF~7)')
""",
}

# Apply by overall cell index
updated = 0
for ci, hint in HINTS_CH03.items():
    if ci >= len(nb["cells"]):
        continue
    cell = nb["cells"][ci]
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "# TODO: Implement this cell" in src and "Steps:" not in src and "# Hint" not in src:
        lines = hint.splitlines(keepends=True)
        if lines and lines[-1].endswith("\n"):
            lines[-1] = lines[-1].rstrip("\n")
        cell["source"] = lines
        updated += 1

if updated:
    CH03_REG.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"  ch03_feature_importance: {updated} cells")
