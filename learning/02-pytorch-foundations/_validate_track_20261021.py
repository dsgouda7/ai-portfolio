"""Task-specific validation; remove after evidence is retained."""
import ast
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import unquote

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
SESSION = Path(r"..\..\Users\deepaksha\.copilot\session-state\78a59e14-e8bd-471f-9577-f35a620524b8\files\pytorch-track-20261021")


def validate_sources():
    totals = []
    for path in sorted(ROOT.rglob("*.ipynb")):
        book = nbformat.read(path, as_version=4)
        nbformat.validate(book)
        ids = [c.id for c in book.cells]
        assert len(ids) == len(set(ids))
        code = [c for c in book.cells if c.cell_type == "code"]
        for i, cell in enumerate(book.cells):
            if cell.cell_type == "code":
                ast.parse(cell.source)
                assert not cell.outputs and cell.execution_count is None
                assert i > 0 and book.cells[i - 1].cell_type == "markdown"
                assert len(book.cells[i - 1].source.split()) >= 85, (path, i)
            else:
                assert "```mermaid" not in cell.source
        baseline = json.loads((SESSION / f"{path.stem}.backup").read_bytes())
        original_code = {c["id"]: c for c in baseline["cells"] if c["cell_type"] == "code"}
        for cell in code:
            assert cell.id in original_code
            assert cell.metadata == original_code[cell.id]["metadata"]
            assert cell.outputs == original_code[cell.id]["outputs"]
        assert set(ids).issuperset(c["id"] for c in baseline["cells"])
        totals.append({"notebook": str(path.relative_to(ROOT)), "code_experiments": len(code)})
    for path in [*ROOT.rglob("*.md"), *ROOT.rglob("*.ipynb")]:
        if path.suffix == ".ipynb":
            text = "\n".join(c.source for c in nbformat.read(path, as_version=4).cells if c.cell_type == "markdown")
        else:
            text = path.read_text(encoding="utf-8")
        assert "```mermaid" not in text
        for target in re.findall(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+[^)]*)?\)", text):
            if target.startswith(("https:", "http:", "mailto:", "#", "data:")):
                continue
            target = unquote(target.split("#", 1)[0])
            assert (path.parent / target).exists(), (path.relative_to(ROOT), target)
    for path in ROOT.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"))
    (SESSION / "source-validation.json").write_text(json.dumps(totals, indent=2), encoding="utf-8")
    return totals


def execute():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    os.environ["PATH"] = str(REPO / ".venv" / "Scripts") + os.pathsep + os.environ["PATH"]
    summary = []
    for path in sorted(ROOT.rglob("*.ipynb")):
        if path.stem == "01-deep-learning-map":
            continue
        source = nbformat.read(path, as_version=4)
        started = time.monotonic()
        manager = KernelManager(kernel_name="python3")
        manager.kernel_spec.argv[0] = sys.executable
        client = NotebookClient(
            source,
            km=manager,
            timeout=600,
            allow_errors=False,
            resources={"metadata": {"path": str(REPO)}},
        )
        executed_path = SESSION / f"{path.stem}-executed.ipynb"
        print(f"Executing {path.relative_to(ROOT)} with {sys.executable}", flush=True)
        try:
            client.execute()
        finally:
            nbformat.write(source, executed_path)
        executed = nbformat.read(executed_path, as_version=4)
        clean = nbformat.read(path, as_version=4)
        assert [(c.id, c.source) for c in executed.cells] == [(c.id, c.source) for c in clean.cells]
        code = [c for c in executed.cells if c.cell_type == "code"]
        assert all(c.execution_count is not None for c in code)
        assert not any(o.output_type == "error" for c in code for o in c.outputs)
        output = "\n".join(
            o.text for c in code for o in c.outputs if o.output_type == "stream"
        )
        assert f"Kernel interpreter: {sys.executable}" in output
        (SESSION / f"{path.stem}-execution.log").write_text(output, encoding="utf-8")
        result = {
            "notebook": str(path.relative_to(ROOT)),
            "executed": str(executed_path.resolve()),
            "code_experiments": len(code),
            "kernel": sys.executable,
            "seconds": round(time.monotonic() - started, 2),
            "complete_source_id_equality": True,
        }
        summary.append(result)
        (SESSION / "execution-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Passed {len(code)} code experiments in {result['seconds']}s", flush=True)
    return summary


if __name__ == "__main__":
    if "--execute" in sys.argv:
        execute()
    else:
        print(json.dumps(validate_sources(), indent=2))
