"""
run_demo.py — Start the context-optimizer demo web portal.

    cd projects/context-optimizer/demo
    python run_demo.py [--port 8000] [--index ./.index] [--model qwen2.5:1.5b]

No pip install required — runs directly from the repo source.
Optional editable install (slightly faster import):  pip install -e '../[hf]'

Reasoning model (--model) is optional.  Without it the portal shows tree
traversal and retrieval results without a synthesized answer.
With Ollama running, pass any small model for full end-to-end demos:
    python run_demo.py --model qwen2.5:1.5b
"""
from __future__ import annotations

# ── Dev-mode bootstrap ─────────────────────────────────────────────────────────
# Wire src/ as context_optimizer if running from repo without a pip install.
import importlib.util as _ilu
from pathlib import Path
_bs = Path(__file__).parent / "_bootstrap.py"
_bs_spec = _ilu.spec_from_file_location("_bootstrap", _bs)
_bs_mod  = _ilu.module_from_spec(_bs_spec)
_bs_spec.loader.exec_module(_bs_mod)
del _ilu, _bs, _bs_spec, _bs_mod
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start the context-optimizer demo web portal"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="HTTP port (default 8000)"
    )
    parser.add_argument(
        "--index",
        default=str(Path(__file__).parent / ".index"),
        metavar="DIR",
        help="Path to the built index (default: ./.index)",
    )
    parser.add_argument(
        "--model",
        default="",
        metavar="MODEL",
        help="Ollama model for reasoning (e.g. qwen2.5:1.5b). Leave empty for retrieval-only.",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Bind host (default 127.0.0.1)"
    )
    args = parser.parse_args()

    index_path = Path(args.index).expanduser().resolve()
    if not index_path.exists():
        print(f"[demo] Index not found: {index_path}")
        print("[demo] Run `python setup_demo.py` first.")
        sys.exit(1)

    try:
        import uvicorn
    except ImportError:
        print("[demo] uvicorn not installed. Run: pip install 'context-optimizer[mcp]'")
        sys.exit(1)

    # Pass config via env so server.py picks it up
    import os
    os.environ["DEMO_INDEX_DIR"] = str(index_path)
    os.environ["DEMO_REASONING_MODEL"] = args.model

    print(f"[demo] Index   : {index_path}")
    print(f"[demo] Reasoner: {args.model or '(none — retrieval-only mode)'}")
    print(f"[demo] URL     : http://{args.host}:{args.port}")
    print()

    uvicorn.run(
        "app.server:app",
        host=args.host,
        port=args.port,
        reload=False,
        app_dir=str(Path(__file__).parent),
        log_level="info",
    )


if __name__ == "__main__":
    main()
