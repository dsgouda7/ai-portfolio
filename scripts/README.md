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

| Script | Needed | When to use it | Notes |
|---|---|---|---|
| `scripts/setup.ps1` | Yes (Windows) | First-time full local setup on Windows; also safe for repair/re-run | Creates/uses `.venv`, installs Python deps, registers kernels, starts Jupyter; add `--enable-slm-assistant` for VS Code + Kilo Code, Ollama, and workspace wiring, `--enable-mkdocs-server` for the local docs server, and `--enable-gpu-notebook-stack` for CUDA PyTorch + fine-tuning deps |
| `scripts/setup.sh` | Yes (macOS/Linux) | First-time full local setup on macOS/Linux; also safe for repair/re-run | Linux/macOS counterpart of `setup.ps1`; add `--enable-slm-assistant` for the optional SLM assistant bundle, `--enable-mkdocs-server` for the local docs server, and `--enable-gpu-notebook-stack` for CUDA PyTorch + fine-tuning deps |
| `scripts/install-hooks.ps1` | Optional | Install repository git hooks on Windows | Copies files from `scripts/hooks/` into `.git/hooks/` |
| `scripts/install-hooks.sh` | Optional | Install repository git hooks on macOS/Linux | Same behavior as PowerShell variant |
| `scripts/check_notebooks.py` | Optional | Validate all notebooks under `notes/` for JSON/syntax issues | Best-effort static checker; useful before large commits |
| `scripts/set_default_kernel.py` | Yes (indirect) | Usually run by setup scripts; run manually after adding/moving notebooks | Sets `metadata.kernelspec` by top-level track folder |
| `scripts/render_html_to_png.py` | Optional | Convert HTML/SVG diagram files to PNG snapshots on Windows | Uses headless Edge/Chrome; primarily for docs asset generation |
| `scripts/mathjax.js` | Optional | MkDocs/MathJax rendering support in docs site | Referenced by MkDocs config/assets, not executed directly |
| `scripts/hooks/pre-commit` | Optional but recommended | Secret scanning for staged changes before commit | Installed via `install-hooks.*`; blocks likely credential leaks |

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
