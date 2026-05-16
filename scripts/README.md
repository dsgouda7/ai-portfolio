# scripts/

Utility scripts for environment setup, notebook hygiene, rendering helpers, and git hook installation.

## Quick start

Use exactly one setup script for your OS:

```powershell
.\scripts\setup.ps1
```

```bash
bash scripts/setup.sh
```

Both scripts are designed to be re-runnable and will skip components that are already installed.

## Script inventory

### Setup & Environment

| Script | Purpose |
|--------|---------|
| `setup.ps1` / `setup.sh` | Full environment setup: creates `.venv`, installs deps, registers kernels. Add `--enable-slm-assistant` for Ollama/Kilo, `--enable-mkdocs-server` for docs, `--enable-gpu-notebook-stack` for CUDA |
| `teardown.ps1` / `teardown.sh` | Clean environment teardown |
| `install-hooks.ps1` / `install-hooks.sh` | Install git hooks from `hooks/` directory |

### Validation & Maintenance

| Script | Purpose |
|--------|---------|
| `check-md-links.py` / `check-md-links.ps1` | Scan all markdown files for broken links |
| `check-notebooks.py` | Validate notebook JSON/syntax |
| `set-default-kernel.py` | Set correct kernel metadata for notebooks by track |
| `scan-hardcoded-paths.py` | Find hardcoded absolute paths |
| `audit-ml-images.py` | Find and clean unreferenced images in ML track |
| `audit-workflow-pattern.py` | Validate chapter README against workflow patterns |
| `update-notebook-links.py` | Update notebook cross-references |
| `update-shared-imports.py` | Update shared code imports |

### Generation & Rendering

| Script | Purpose |
|--------|---------|
| `render-html-to-png.py` | Convert HTML/SVG to PNG using headless browser |
| `generate-exercise-notebooks.py` | Generate exercise/solution notebook pairs |
| `mathjax.js` | MathJax config for MkDocs |

### Watchers & Hooks

| Script | Purpose |
|--------|---------|
| `cce-watcher.ps1` / `cce-watcher.sh` | File watcher for CCE indexing |
| `hooks/pre-commit` | Git pre-commit hook for secret scanning |

## Generated files

The setup scripts generate or update these runtime files:

| Generated file | Source | Purpose |
|---|---|---|
| `scripts/ollama-watcher.ps1` | `setup.ps1` | Starts/stops Ollama with VS Code workspace lifecycle when `--enable-slm-assistant` is used |
| `scripts/ollama-watcher.sh` | `setup.sh` | Starts/stops Ollama with VS Code workspace lifecycle when `--enable-slm-assistant` is used |
| `.vscode/tasks.json` | setup scripts | Adds `ollama-start`, `ollama-stop`, and Kilo launch task when `--enable-slm-assistant` is used |
| `.vscode/settings.json` | setup scripts | Applies notebook read-only and default-kernel settings |

Do not hand-edit generated watcher scripts unless you also update setup logic.

## Common usage

Run notebook integrity scan:

```bash
python scripts/check_notebooks.py
```

Re-apply default kernels after creating new notebooks:

```bash
python scripts/set_default_kernel.py
```

Install commit hooks:

```powershell
.\scripts\install-hooks.ps1
```

```bash
bash scripts/install-hooks.sh
```

Render an HTML file to PNG (Windows):

```powershell
python scripts/render_html_to_png.py input.html output.png 1400 920
```

## Notes

- Setup configures the Kilo Code + Ollama assistant bundle only when `--enable-slm-assistant` is supplied; default setup keeps the footprint smaller. MkDocs is also opt-in via `--enable-mkdocs-server`, and the CUDA notebook stack is opt-in via `--enable-gpu-notebook-stack`.
- On Windows, standard venv activation script name is `Activate.ps1`.
- If activation scripts are unavailable, setup now falls back to direct venv interpreter/path wiring.
- **Ad-hoc scripts removed:** One-time setup/stub-generation scripts (e.g., `create-*-img-dirs.py`, `add-*-stubs.py`, `remove-emojis.py`) have been deleted after completing their tasks. Only reusable maintenance scripts remain.
