# context-optimizer demo

Interactive portal showing hierarchical RAG tree traversal in real time.

## Quick start — straight from the repo, no PyPI needed

```bash
# 1 — install direct dependencies only (FastAPI + HuggingFace)
pip install fastapi "uvicorn[standard]" transformers torch sentence-transformers chromadb pydantic python-dotenv

# 2 — build the index (downloads flan-t5-small ~250 MB on first run)
cd projects/context-optimizer/demo
python setup_demo.py

# 3 — start the portal
python run_demo.py
# → open http://localhost:8000
```

The scripts automatically register `../src/` as the `context_optimizer` package — no `pip install context-optimizer` required.

---

## Editable install (alternative)

If you'd rather install the package in dev mode:

```bash
cd projects/context-optimizer
pip install -e ".[hf]"
cd demo
python setup_demo.py
python run_demo.py
```

---

## With a reasoning model (Ollama)

```bash
# Pull a small model first
ollama pull qwen2.5:1.5b

# Start with reasoning enabled
python run_demo.py --model qwen2.5:1.5b
```

---

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | 8000 | HTTP port |
| `--host` | 127.0.0.1 | Bind address |
| `--index` | `./.index` | Path to built index |
| `--model` | *(none)* | Ollama model for answer synthesis |

---

## What you'll see

| Panel | Contents |
|-------|----------|
| **Left** | File tree of every indexed document. Click to read raw content on the right. |
| **Center** | Query input + animated D3 force graph of the trie. Submit a query to watch the agent traverse clusters → blocks. |
| **Right** | Raw file viewer. Populated by clicking a file or a block node in the graph. |

The animation replays the actual server-side traversal steps returned by the API, so you can see exactly which cluster nodes were searched, which blocks were expanded, and which raw chunks were fetched to produce the answer.
