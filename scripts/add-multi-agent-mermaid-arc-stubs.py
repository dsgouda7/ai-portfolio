"""
add_multi_agent_mermaid_arc_stubs.py
Append a 7-chapter Mermaid graph LR stub to each ## 11 · Progress Check section
(or ## § 11.5 · Progress Check if not yet normalized) in Multi-Agent AI README files.
"""
import re
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MA_ROOT = os.path.join(REPO_ROOT, "notes", "04-multi_agent_ai")

CHAPTERS = [
    ("ch01_message_formats",      "Ch.1 — Message Formats"),
    ("ch02_mcp",                  "Ch.2 — MCP"),
    ("ch03_a2a",                  "Ch.3 — A2A"),
    ("ch04_event_driven_agents",  "Ch.4 — Event-Driven"),
    ("ch05_shared_memory",        "Ch.5 — Shared Memory"),
    ("ch06_trust_and_sandboxing", "Ch.6 — Trust & Sandboxing"),
    ("ch07_agent_frameworks",     "Ch.7 — Agent Frameworks"),
]

MERMAID_TEMPLATE = """
```mermaid
graph LR
    Ch1["Ch.1\\nMessage Formats"]:::done
    Ch2["Ch.2\\nMCP"]:::done
    Ch3["Ch.3\\nA2A"]:::done
    Ch4["Ch.4\\nEvent-Driven"]:::done
    Ch5["Ch.5\\nShared Memory"]:::done
    Ch6["Ch.6\\nTrust & Sandboxing"]:::done
    Ch7["Ch.7\\nAgent Frameworks"]:::done
    Ch1 --> Ch2 --> Ch3 --> Ch4 --> Ch5 --> Ch6 --> Ch7
    classDef done fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    classDef current fill:#1d4ed8,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    classDef upcoming fill:#1e3a8a,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
```
"""

# Pattern matches both normalized and un-normalized progress check headings
PROGRESS_HEADING = re.compile(
    r"(^## (?:§ 11\.5|11) · Progress Check[^\n]*\n)", re.MULTILINE
)

for chapter_dir, chapter_label in CHAPTERS:
    filepath = os.path.join(MA_ROOT, chapter_dir, "README.md")
    if not os.path.exists(filepath):
        print(f"SKIP (not found): {filepath}")
        continue

    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    if "```mermaid" in content and "graph LR" in content:
        print(f"Mermaid arc already present in {chapter_dir}/README.md")
        continue

    match = PROGRESS_HEADING.search(content)
    if not match:
        print(f"No Progress Check heading found in {chapter_dir}/README.md")
        continue

    # Insert the mermaid stub right after the heading line
    insert_pos = match.end()
    new_content = content[:insert_pos] + MERMAID_TEMPLATE + content[insert_pos:]

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"Inserted Mermaid arc stub in {chapter_dir}/README.md")
