#!/usr/bin/env bash
# Create the shared FDE environment without starting Docker or cloud services.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"
VENV_DIR="$SCRIPT_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
KERNEL_NAME="fde"
KERNEL_DISPLAY_NAME="Python (FDE .venv)"
KERNEL_SETTER="$REPO_ROOT/scripts/set-notebook-kernel.py"
AZURE_TUTORIAL="$REPO_ROOT/learning/ai-infrastructure/09-azure-operational-llm-serving"
RIVERSIDE_PROJECT="$REPO_ROOT/projects/riverside-ai-platform"
RAG_ROOT="$REPO_ROOT/projects/rag-knowledge-pipeline"
SKIP_KERNEL=0
SKIP_PROJECTS=0
SKIP_RAG=0
INCLUDE_AZUREML=0

for argument in "$@"; do
    case "$argument" in
        --skip-kernel) SKIP_KERNEL=1 ;;
        --skip-projects) SKIP_PROJECTS=1 ;;
        --skip-rag) SKIP_RAG=1 ;;
        --include-azureml) INCLUDE_AZUREML=1 ;;
        *) echo "Unknown option: $argument" >&2; exit 1 ;;
    esac
done

if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "Python was not found on PATH. Install Python 3.11-3.13 and rerun this script." >&2
    exit 1
fi

PYTHON_VERSION="$($PYTHON -c "import sys; print('.'.join(map(str, sys.version_info[:3])))")"
"$PYTHON" -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 14) else 1)" || {
    echo "Python $PYTHON_VERSION is unsupported. The shared project environment requires Python 3.11-3.13." >&2
    exit 1
}

for required_path in "$REQUIREMENTS" "$KERNEL_SETTER" "$AZURE_TUTORIAL" "$RIVERSIDE_PROJECT" "$RAG_ROOT"; do
    [ -e "$required_path" ] || { echo "Required setup input was not found: $required_path" >&2; exit 1; }
done

if [ -x "$VENV_PYTHON" ]; then
    echo "Reusing virtual environment at $VENV_DIR"
else
    echo "Creating virtual environment with Python $PYTHON_VERSION at $VENV_DIR"
    "$PYTHON" -m venv "$VENV_DIR"
fi

echo "Upgrading pip, setuptools, and wheel..."
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel --quiet

echo "Installing FDE notebook and local Azure lab dependencies..."
"$VENV_PYTHON" -m pip install -r "$REQUIREMENTS"

if [ "$SKIP_PROJECTS" -eq 0 ]; then
    RIVERSIDE_EXTRAS="test,telemetry"
    if [ "$INCLUDE_AZUREML" -eq 1 ]; then
        RIVERSIDE_EXTRAS="test,telemetry,azureml"
    fi
    RIVERSIDE_REQUIREMENT="$RIVERSIDE_PROJECT[$RIVERSIDE_EXTRAS]"
    echo "Installing Riverside AI Platform in editable mode with $RIVERSIDE_EXTRAS extras..."
    "$VENV_PYTHON" -m pip install --editable "$RIVERSIDE_REQUIREMENT"
fi

if [ "$SKIP_RAG" -eq 0 ]; then
    echo "Installing local RAG phase 1, 2, and 3 validation dependencies..."
    "$VENV_PYTHON" -m pip install \
        -r "$RAG_ROOT/phase1-ingest/requirements.txt" \
        -r "$RAG_ROOT/phase2-vectorize/requirements.txt" \
        -r "$RAG_ROOT/phase3-serve/requirements.txt" \
        "pytest>=8,<9" \
        "langchain>=0.1,<1" \
        "langchain-community>=0.0.10,<1" \
        "langchain-core>=0.1,<1"
    RAG_INGEST_REQUIREMENT="$RAG_ROOT/phase1-ingest[test]"
    "$VENV_PYTHON" -m pip install \
        --editable "$RAG_ROOT/shared" \
        --editable "$RAG_INGEST_REQUIREMENT"
fi

echo "Verifying installed imports..."
"$VENV_PYTHON" -c "import jsonschema, pandas, numpy, matplotlib; print('FDE notebook imports: OK')"
if [ "$SKIP_PROJECTS" -eq 0 ]; then
    "$VENV_PYTHON" -c "import fastapi, httpx, pytest; from azure.identity import DefaultAzureCredential; import app, endpoint_client, rag_orchestrator, release_gates, telemetry; print('Riverside project and test imports: OK')"
fi
if [ "$SKIP_RAG" -eq 0 ]; then
    "$VENV_PYTHON" -c "import bs4, pypdf, docx, deltalake, datasets, pyarrow, langchain, langchain_community, langchain_core, chromadb, torch, pytest, shared; from databricks.ai_search.client import AISearchClient; from databricks.sdk import WorkspaceClient; print('RAG dependency imports: OK')"
    "$VENV_PYTHON" -c "import sys; sys.path.insert(0, r'$RAG_ROOT/phase1-ingest/src'); import remote.config, remote.contracts; print('RAG phase 1 imports: OK')"
    "$VENV_PYTHON" -c "import sys; sys.path.insert(0, r'$RAG_ROOT/phase2-vectorize/src'); import remote.contracts, remote.databricks_adapters; print('RAG phase 2 imports: OK')"
    "$VENV_PYTHON" -c "import sys; sys.path.insert(0, r'$RAG_ROOT/phase3-serve'); import src.rag_pipeline, src.retrieval, src.server; print('RAG phase 3 imports: OK')"
fi

if [ "$SKIP_KERNEL" -eq 0 ]; then
    echo "Registering Jupyter kernel '$KERNEL_NAME'..."
    "$VENV_PYTHON" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY_NAME"
    echo "Assigning '$KERNEL_NAME' to FDE and Azure operational notebooks..."
    "$VENV_PYTHON" "$KERNEL_SETTER" --directory "$SCRIPT_DIR" --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY_NAME"
    "$VENV_PYTHON" "$KERNEL_SETTER" --directory "$AZURE_TUTORIAL" --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY_NAME"
fi

echo ""
echo "FDE setup complete."
echo "Virtual environment: $VENV_DIR"
echo "Jupyter kernel: $KERNEL_DISPLAY_NAME"
echo "No Docker or Azure service was installed, started, or contacted."
