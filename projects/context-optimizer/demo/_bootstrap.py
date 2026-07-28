"""
_bootstrap.py — Dev-mode path shim.

Imported by setup_demo.py and run_demo.py.  If ``context_optimizer`` is
already installed (``pip install -e .`` or from PyPI) this is a no-op.
Otherwise it wires ``projects/context-optimizer/src/`` as the
``context_optimizer`` package so the demo scripts work straight from the repo
with no install step at all.

Usage in a script
-----------------
    # At the TOP of setup_demo.py / run_demo.py, before any other imports:
    import importlib.util, sys
    from pathlib import Path
    _bs_path = Path(__file__).parent / "_bootstrap.py"
    _bs_spec = importlib.util.spec_from_file_location("_bootstrap", _bs_path)
    _bs_mod  = importlib.util.module_from_spec(_bs_spec)
    _bs_spec.loader.exec_module(_bs_mod)
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def ensure_importable() -> None:
    """Register src/ as *context_optimizer* if the package is not installed."""
    try:
        import context_optimizer  # noqa: F401
        return  # already available — nothing to do
    except ImportError:
        pass

    # demo/_bootstrap.py lives one directory below projects/context-optimizer/
    src = (Path(__file__).parent.parent / "src").resolve()
    if not src.exists():
        raise ImportError(
            f"context_optimizer source not found at {src}.\n"
            "Either run:  pip install -e '.[hf]'  from the context-optimizer "
            "directory, or ensure you're executing this script from inside the "
            "repo."
        )

    # Register src/ as the top-level package in sys.modules.  Setting __path__
    # to [src] means Python will resolve sub-imports (providers/, extractors/,
    # adapters/, …) as sub-packages under src/ automatically.
    pkg = types.ModuleType("context_optimizer")
    pkg.__path__ = [str(src)]
    pkg.__package__ = "context_optimizer"
    pkg.__file__ = str(src / "__init__.py")
    sys.modules["context_optimizer"] = pkg

    init_py = src / "__init__.py"
    if init_py.exists():
        spec = importlib.util.spec_from_file_location(
            "context_optimizer",
            init_py,
            submodule_search_locations=[str(src)],
        )
        spec.loader.exec_module(pkg)


ensure_importable()
