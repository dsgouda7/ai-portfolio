#!/usr/bin/env bash
# setup.sh — AI/ML Dev Environment Setup (macOS / Linux)
# Provisions Python, a full AI/ML library stack, VS Code, and optionally a local SLM assistant bundle.
# Run from anywhere:
#   bash scripts/setup.sh
#
# Steps implemented so far:
#   1. Python + AI/ML libraries  ✔
#   2. VS Code install            ✔
#   3. Kilo Code (agentic AI) extension  ✔
#   4. Ollama server install & first launch  ✔
#   5. Lifecycle wiring (Ollama runs with VS Code)  ✔
#   6. Pull DeepSeek-R1 reasoning SLM for Kilo Code  ✔

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# scripts/ → repo root
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PATH="$REPO_ROOT/.venv"
VSCODE_DIR="$REPO_ROOT/.vscode"

# ─── Helpers ──────────────────────────────────────────────────────────────────

step()  { echo; echo "▶ $*"; }
ok()    { echo "  ✓ $*"; }
warn()  { echo "  ! $*"; }
fail()  { echo "  ✗ $*" >&2; exit 1; }
group() { echo; echo "  ── $*"; }

ENABLE_SLM_ASSISTANT=false
ENABLE_MKDOCS_SERVER=false
ENABLE_GPU_NOTEBOOK_STACK=false
for arg in "$@"; do
    case "$arg" in
        --enable-slm-assistant)
            ENABLE_SLM_ASSISTANT=true
            ;;
        --enable-mkdocs-server)
            ENABLE_MKDOCS_SERVER=true
            ;;
        --enable-gpu-notebook-stack)
            ENABLE_GPU_NOTEBOOK_STACK=true
            ;;
    esac
done

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ -f /etc/os-release ]]; then
    # shellcheck source=/dev/null
    source /etc/os-release
    case "${ID:-}" in
        ubuntu|debian|linuxmint|pop)  OS="debian" ;;
        fedora|rhel|centos|rocky|alma) OS="fedora" ;;
        arch|manjaro)                  OS="arch"   ;;
        *)                             OS="linux"  ;;
    esac
fi

# ─── STEP 1: Python + AI/ML Libraries ────────────────────────────────────────

echo ""
echo "══════════════════════════════════════════════"
echo "  AI/ML Dev Environment Setup — Step 1/7"
echo "  Python + AI/ML Library Stack"
echo "══════════════════════════════════════════════"

# ─── 1a. Python ───────────────────────────────────────────────────────────────

step "Checking Python 3.11+"

PYTHON=""
for candidate in python3.11 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" --version 2>&1)
        major=$(echo "$ver" | sed 's/Python \([0-9]*\)\..*/\1/')
        minor=$(echo "$ver" | sed 's/Python [0-9]*\.\([0-9]*\)\..*/\1/')
        if [ "$major" -ge 3 ] && [ "$minor" -ge 11 ]; then
            PYTHON="$candidate"
            ok "$ver (meets 3.11+ requirement)"
            break
        elif [ "$major" -ge 3 ] && [ "$minor" -ge 9 ]; then
            warn "$ver found — Python 3.11+ recommended; continuing with $ver"
            PYTHON="$candidate"
            break
        else
            warn "$ver found but Python 3.9+ is required"
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    warn "Python not found — attempting install ..."
    case "$OS" in
        macos)
            if command -v brew &>/dev/null; then
                brew install python@3.11
                PYTHON="python3.11"
            else
                fail "Homebrew not found. Install it from https://brew.sh/ then re-run, or install Python manually from https://www.python.org/downloads/"
            fi
            ;;
        debian)
            sudo apt-get update -qq
            sudo apt-get install -y python3.11 python3.11-venv python3-pip
            PYTHON="python3.11"
            ;;
        fedora)
            sudo dnf install -y python3.11
            PYTHON="python3.11"
            ;;
        arch)
            sudo pacman -Sy --noconfirm python
            PYTHON="python"
            ;;
        *)
            fail "Cannot auto-install Python on this system. Install Python 3.11+ manually from https://www.python.org/downloads/ and re-run."
            ;;
    esac
    ok "Python installed: $("$PYTHON" --version 2>&1)"
fi

# ─── 1b. pip ──────────────────────────────────────────────────────────────────

step "Checking pip"
if "$PYTHON" -m pip --version &>/dev/null; then
    ok "$("$PYTHON" -m pip --version)"
else
    warn "pip not available — bootstrapping ..."
    "$PYTHON" -m ensurepip --upgrade 2>/dev/null || {
        case "$OS" in
            debian) sudo apt-get install -y python3-pip ;;
            fedora) sudo dnf install -y python3-pip ;;
            macos)  brew install python@3.11 ;;
            *)      fail "Cannot bootstrap pip. Install it manually and re-run." ;;
        esac
    }
    ok "$("$PYTHON" -m pip --version)"
fi

# ─── 1c. Virtual environment ──────────────────────────────────────────────────

step "Virtual environment (.venv)"

if [ -d "$VENV_PATH" ]; then
    ok "Existing .venv found — reusing"
else
    warn "No .venv found — creating ..."
    "$PYTHON" -m venv "$VENV_PATH"
    ok "Created .venv at $VENV_PATH"
fi

# Activate if available; otherwise wire PATH directly to venv binaries.
VENV_BIN="$VENV_PATH/bin"
VENV_PYTHON="$VENV_BIN/python"

if [ ! -x "$VENV_PYTHON" ]; then
    fail "Cannot find venv Python interpreter at $VENV_PYTHON"
fi

if [ -f "$VENV_BIN/activate" ]; then
    # shellcheck source=/dev/null
    source "$VENV_BIN/activate"
    ok "Activated .venv via $VENV_BIN/activate"
else
    warn "activate script not found under .venv/bin; continuing with direct venv PATH wiring"
    export VIRTUAL_ENV="$VENV_PATH"
    export PATH="$VENV_BIN:$PATH"
    ok "Using venv interpreter directly: $VENV_PYTHON"
fi

# Upgrade build tools quietly
python -m pip install --upgrade pip setuptools wheel --quiet
ok "pip / setuptools / wheel up to date"

# ─── 1d. Package installation ─────────────────────────────────────────────────

step "Installing AI/ML package stack"

normalize_pkg_key() {
    echo "$1" | sed 's/\[.*\]//' | sed 's/[>=<!].*//' | tr '[:upper:]' '[:lower:]' | xargs
}

INSTALLED_PIP_KEYS=$(python -m pip list --format=columns 2>/dev/null | tail -n +3 | awk '{print tolower($1)}')

pip_has_key() {
    local key="$1"
    echo "$INSTALLED_PIP_KEYS" | grep -qx "$key"
}

mark_installed_key() {
    local key="$1"
    if ! pip_has_key "$key"; then
        INSTALLED_PIP_KEYS="${INSTALLED_PIP_KEYS}"$'\n'"$key"
    fi
}

# Helper: install a group of packages
install_group() {
    local group_name="$1"
    shift
    local extra_args=()
    # Collect packages until "--extra-args" sentinel
    local packages=()
    local in_extra=0
    for arg in "$@"; do
        if [ "$arg" = "--extra-args" ]; then
            in_extra=1
        elif [ "$in_extra" -eq 1 ]; then
            extra_args+=("$arg")
        else
            packages+=("$arg")
        fi
    done

    group "$group_name"
    for pkg in "${packages[@]}"; do
        # Normalise: strip extras/version specifiers for lookup
        local key
        key=$(normalize_pkg_key "$pkg")
        if pip_has_key "$key"; then
            ok "$pkg already installed"
        else
            warn "$pkg missing — installing ..."
            if [ ${#extra_args[@]} -gt 0 ]; then
                if ! python -m pip install "$pkg" "${extra_args[@]}" --quiet; then
                    warn "Initial install failed for $pkg — retrying with --no-cache-dir"
                    python -m pip install "$pkg" "${extra_args[@]}" --quiet --no-cache-dir
                fi
            else
                if ! python -m pip install "$pkg" --quiet; then
                    warn "Initial install failed for $pkg — retrying with --no-cache-dir"
                    python -m pip install "$pkg" --quiet --no-cache-dir
                fi
            fi
            mark_installed_key "$key"
            ok "$pkg installed"
        fi
    done
}

CORE_SCIENTIFIC=(numpy pandas scipy matplotlib seaborn)
MACHINE_LEARNING=(scikit-learn xgboost lightgbm)
DEEP_LEARNING_TF=(tensorflow tensorboard keras)
PYTORCH_CPU=(torch torchvision torchaudio)
NOTEBOOK_TOOLING=(notebook ipykernel ipywidgets jupyterlab)
GENERATIVE_AI=(
    transformers diffusers accelerate datasets tokenizers
    huggingface-hub openai langchain langchain-community
    sentence-transformers faiss-cpu chromadb
)
UTILITIES=(python-dotenv tqdm pillow requests httpx pydantic)
DOCS_SITE=(mkdocs-material pymdown-extensions mkdocs-jupyter)
NOTEBOOK_EXTRAS=(
    mlflow tiktoken mcp fastapi "uvicorn[standard]" anyio redis
    langgraph langchain-core langchain-openai
    autogen-agentchat semantic-kernel ollama
)
CODE_INTELLIGENCE=(
    code-context-engine   # local MCP server — AST-aware codebase indexing + cross-session memory
)

ALL_REQUIRED_PACKAGES=(
    "${CORE_SCIENTIFIC[@]}"
    "${MACHINE_LEARNING[@]}"
    "${DEEP_LEARNING_TF[@]}"
    "${PYTORCH_CPU[@]}"
    "${NOTEBOOK_TOOLING[@]}"
    "${GENERATIVE_AI[@]}"
    "${UTILITIES[@]}"
    "${DOCS_SITE[@]}"
    "${NOTEBOOK_EXTRAS[@]}"
    "${CODE_INTELLIGENCE[@]}"
)

ALL_DEPENDENCIES_MET=true
for pkg in "${ALL_REQUIRED_PACKAGES[@]}"; do
    key=$(normalize_pkg_key "$pkg")
    if ! pip_has_key "$key"; then
        ALL_DEPENDENCIES_MET=false
        break
    fi
done

if [ "$ALL_DEPENDENCIES_MET" = true ]; then
    ok "All Python package dependencies already satisfied — skipping package installation step"
else
    # Core scientific stack
    install_group "Core scientific stack" "${CORE_SCIENTIFIC[@]}"

    # Machine learning
    install_group "Machine learning" "${MACHINE_LEARNING[@]}"

    # Deep learning — TensorFlow
    install_group "Deep learning / TensorFlow" "${DEEP_LEARNING_TF[@]}"

    # PyTorch — CPU-safe build (no CUDA required on stock machines)
    install_group "PyTorch (CPU build)" "${PYTORCH_CPU[@]}" \
        --extra-args --index-url https://download.pytorch.org/whl/cpu

    # Notebook tooling
    install_group "Notebook tooling" "${NOTEBOOK_TOOLING[@]}"

    # Generative AI / LLM utilities
    install_group "Generative AI / LLM utilities" "${GENERATIVE_AI[@]}"

    # General utilities
    install_group "Utilities" "${UTILITIES[@]}"

    # Docs / study site (MkDocs Material — browse notes/ in a web browser)
    # mkdocs-jupyter renders every notebook.ipynb as a page alongside the .md files.
    install_group "Docs site (MkDocs Material)" "${DOCS_SITE[@]}"

    # Notebook extras — dependencies pulled in by per-notes setup scripts
    install_group "Notebook extras (AIInfrastructure + MultiAgentAI)" "${NOTEBOOK_EXTRAS[@]}"

    # Code intelligence tooling
    install_group "Code intelligence (CCE)" "${CODE_INTELLIGENCE[@]}"

    # AI Infrastructure (ML Experiment Tracking) dependencies
    echo ""
    echo "📦 Installing AI Infrastructure dependencies..."
    install_group "AI Infrastructure (ML Experiment Tracking)" mlflow evidently dvc tensorboard wandb
    echo "✅ AI Infrastructure setup complete"
    echo "   Verify: mlflow --version && dvc --version"

    # DevOps Fundamentals dependencies
    echo ""
    echo "📦 Installing DevOps Fundamentals dependencies..."

    # Docker
    echo ""
    echo "  • Checking Docker..."
    if command -v docker &>/dev/null; then
        ok "Docker already installed: $(docker --version)"
    else
        warn "Docker not found — attempting install..."
        case "$OS" in
            debian)
                # Install Docker on Ubuntu/Debian
                sudo apt-get update -qq
                sudo apt-get install -y ca-certificates curl gnupg
                sudo install -m 0755 -d /etc/apt/keyrings
                curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
                sudo chmod a+r /etc/apt/keyrings/docker.gpg
                echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
                sudo apt-get update -qq
                sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
                sudo usermod -aG docker "$USER"
                ok "Docker installed — you may need to log out and back in for group permissions"
                ;;
            macos)
                warn "On macOS, install Docker Desktop manually from: https://www.docker.com/products/docker-desktop"
                ;;
            *)
                warn "Cannot auto-install Docker on this system. Install manually from: https://docs.docker.com/engine/install/"
                ;;
        esac
    fi

    # Kind
    echo ""
    echo "  • Checking Kind (Kubernetes in Docker)..."
    if command -v kind &>/dev/null; then
        ok "Kind already installed: $(kind --version)"
    else
        warn "Kind not found — installing..."
        case "$OS" in
            macos)
                if command -v brew &>/dev/null; then
                    brew install kind
                    ok "Kind installed via Homebrew"
                else
                    # Download binary for macOS
                    KIND_VERSION="v0.20.0"
                    curl -Lo /tmp/kind https://kind.sigs.k8s.io/dl/${KIND_VERSION}/kind-darwin-amd64
                    chmod +x /tmp/kind
                    sudo mv /tmp/kind /usr/local/bin/kind
                    ok "Kind installed to /usr/local/bin/kind"
                fi
                ;;
            *)
                # Download binary for Linux
                KIND_VERSION="v0.20.0"
                curl -Lo /tmp/kind https://kind.sigs.k8s.io/dl/${KIND_VERSION}/kind-linux-amd64
                chmod +x /tmp/kind
                sudo mv /tmp/kind /usr/local/bin/kind
                ok "Kind installed to /usr/local/bin/kind"
                ;;
        esac
    fi

    # kubectl
    echo ""
    echo "  • Checking kubectl..."
    if command -v kubectl &>/dev/null; then
        ok "kubectl already installed: $(kubectl version --client --short 2>/dev/null || kubectl version --client)"
    else
        warn "kubectl not found — installing..."
        case "$OS" in
            macos)
                if command -v brew &>/dev/null; then
                    brew install kubectl
                    ok "kubectl installed via Homebrew"
                else
                    # Download binary for macOS
                    KUBECTL_VERSION="$(curl -L -s https://dl.k8s.io/release/stable.txt)"
                    curl -LO "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/darwin/amd64/kubectl"
                    chmod +x kubectl
                    sudo mv kubectl /usr/local/bin/kubectl
                    ok "kubectl installed to /usr/local/bin/kubectl"
                fi
                ;;
            debian)
                # Install kubectl via apt repository
                sudo apt-get update -qq
                sudo apt-get install -y apt-transport-https ca-certificates curl
                curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.28/deb/Release.key | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
                echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.28/deb/ /' | sudo tee /etc/apt/sources.list.d/kubernetes.list
                sudo apt-get update -qq
                sudo apt-get install -y kubectl
                ok "kubectl installed via apt"
                ;;
            *)
                # Download binary for generic Linux
                KUBECTL_VERSION="$(curl -L -s https://dl.k8s.io/release/stable.txt)"
                curl -LO "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl"
                chmod +x kubectl
                sudo mv kubectl /usr/local/bin/kubectl
                ok "kubectl installed to /usr/local/bin/kubectl"
                ;;
        esac
    fi

    # Terraform
    echo ""
    echo "  • Checking Terraform..."
    if command -v terraform &>/dev/null; then
        ok "Terraform already installed: $(terraform --version | head -n1)"
    else
        warn "Terraform not found — installing..."
        case "$OS" in
            macos)
                if command -v brew &>/dev/null; then
                    brew tap hashicorp/tap
                    brew install hashicorp/tap/terraform
                    ok "Terraform installed via Homebrew"
                else
                    warn "Homebrew not found. Install Terraform manually from: https://developer.hashicorp.com/terraform/install"
                fi
                ;;
            debian)
                # Install Terraform via HashiCorp repository
                wget -O- https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
                echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
                sudo apt-get update -qq
                sudo apt-get install -y terraform
                ok "Terraform installed via HashiCorp repository"
                ;;
            *)
                warn "Cannot auto-install Terraform on this system. Install manually from: https://developer.hashicorp.com/terraform/install"
                ;;
        esac
    fi

    # K9s (optional)
    echo ""
    echo "  • Checking K9s (optional Kubernetes TUI)..."
    if command -v k9s &>/dev/null; then
        ok "K9s already installed: $(k9s version --short 2>/dev/null || echo 'installed')"
    else
        warn "K9s not found — installing (optional)..."
        case "$OS" in
            macos)
                if command -v brew &>/dev/null; then
                    brew install derailed/k9s/k9s
                    ok "K9s installed via Homebrew"
                else
                    warn "Homebrew not found. Install K9s manually from: https://k9scli.io/topics/install/"
                fi
                ;;
            debian)
                # Install via webinstall.dev
                curl -sS https://webi.sh/k9s | sh
                export PATH="$HOME/.local/bin:$PATH"
                ok "K9s installed via webinstall.dev"
                ;;
            *)
                warn "Cannot auto-install K9s on this system. Install manually from: https://k9scli.io/topics/install/"
                ;;
        esac
    fi

    echo ""
    echo "✅ DevOps Fundamentals setup complete"
    echo "   Verify installations:"
    echo "   • docker --version"
    echo "   • kind --version"
    echo "   • kubectl version --client"
    echo "   • terraform --version"
    echo "   • k9s version"
fi

if [ "$ENABLE_GPU_NOTEBOOK_STACK" = true ]; then
    step "Installing GPU notebook stack"
    group "GPU notebook packages"

    # Replace the default CPU-only torch stack with a CUDA-enabled wheel.
    if ! python -m pip install --upgrade --force-reinstall \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121 --quiet; then
        fail "Failed to install CUDA-enabled PyTorch wheels"
    fi
    ok "PyTorch CUDA wheel installed"

    install_group "Fine-tuning / GPU notebook deps" accelerate datasets peft trl
fi

# ─── 1d'. Dependency health check / compatibility repairs ───────────────────

step "Checking dependency health (pip check)"

if python -m pip check >/dev/null 2>&1; then
    ok "pip check passed"
else
    warn "pip check reported conflicts — applying compatibility pin(s)"
    # TensorFlow 2.21 requires protobuf >=6.31.1,<8.0.0.
    python -m pip install "protobuf>=6.31.1,<8.0.0" --quiet --no-cache-dir || true

    if python -m pip check >/dev/null 2>&1; then
        ok "Dependency conflicts resolved"
    else
        warn "Dependency conflicts remain after automatic repair:"
        python -m pip check || true
        warn "Setup will continue, but you may need to pin package versions manually."
    fi
fi

# ─── 1f. Code Context Engine (CCE) init ──────────────────────────────────────
#
# cce init:
#   • downloads the embedding model (one-time, ~67 MB)
#   • creates .mcp.json + .vscode/mcp.json  (MCP server registration)
#   • installs 3 git hooks (auto-reindex on commit)
#   • builds the local vector + BM25 index for context_search
# Re-indexing is NOT needed periodically — git hooks keep the index current.
# To force a full re-index manually: cce index

step "Initialising Code Context Engine (cce init)"

if [ -f "$REPO_ROOT/.mcp.json" ]; then
    ok "CCE already initialised (.mcp.json exists) — skipping"
else
    CCE_BIN="$VENV_BIN/cce"
    if [ -x "$CCE_BIN" ]; then
        warn "Running cce init (indexes codebase — may take a few minutes on first run) ..."
        if (cd "$REPO_ROOT" && "$CCE_BIN" init); then
            ok "cce init complete — reload VS Code to activate the MCP server"
        else
            warn "cce init exited non-zero — CCE may not be fully set up"
        fi
    else
        warn "cce binary not found at $CCE_BIN — skipping CCE init"
    fi
fi

# ─── 1e. Register Jupyter kernels ─────────────────────────────────────────────

step "Registering Jupyter kernels"

kernel_exists() {
    local kernel_name="$1"
    python - "$kernel_name" <<'PYEOF'
import json, subprocess, sys
name = sys.argv[1]
try:
    out = subprocess.check_output([sys.executable, "-m", "jupyter", "kernelspec", "list", "--json"], text=True)
    data = json.loads(out)
    sys.exit(0 if name in (data.get("kernelspecs") or {}) else 1)
except Exception:
    sys.exit(2)
PYEOF
    return $?
}

ALL_KERNELS=(ai-ml-dev ml-notes ai-infrastructure multi-agent-ai)
ALL_KERNELS_PRESENT=true
for k in "${ALL_KERNELS[@]}"; do
    if ! kernel_exists "$k"; then
        ALL_KERNELS_PRESENT=false
        break
    fi
done

if [ "$ALL_KERNELS_PRESENT" = true ]; then
    ok "All required Jupyter kernels already registered — skipping kernel registration"
else
    if kernel_exists "ai-ml-dev"; then
        ok "Kernel 'ai-ml-dev' already registered"
    else
        python -m ipykernel install --user --name "ai-ml-dev" --display-name "AI/ML Dev (venv)" &>/dev/null
        ok "Kernel 'ai-ml-dev' registered"
    fi

    if kernel_exists "ml-notes"; then
        ok "Kernel 'ml-notes' already registered"
    else
        python -m ipykernel install --user --name "ml-notes" --display-name "ML Notes (venv)" &>/dev/null
        ok "Kernel 'ml-notes' registered"
    fi

    if kernel_exists "ai-infrastructure"; then
        ok "Kernel 'ai-infrastructure' already registered"
    else
        python -m ipykernel install --user --name "ai-infrastructure" --display-name "Python (AI Infrastructure)" &>/dev/null
        ok "Kernel 'ai-infrastructure' registered"
    fi

    if kernel_exists "multi-agent-ai"; then
        ok "Kernel 'multi-agent-ai' already registered"
    else
        python -m ipykernel install --user --name "multi-agent-ai" --display-name "Multi-Agent AI" &>/dev/null
        ok "Kernel 'multi-agent-ai' registered"
    fi
fi

step "Setting default kernel on every notebook under notes/"
python "$SCRIPT_DIR/set_default_kernel.py" || warn "set_default_kernel.py exited non-zero"

step "Setting notebook permissions (read-only for solutions, writable for exercises)"

# Read-only for solution notebooks (chmod 444)
solution_count=$(find "$REPO_ROOT/notes" -name "*_solution.ipynb" 2>/dev/null | wc -l)
if [ "$solution_count" -gt 0 ]; then
    find "$REPO_ROOT/notes" -name "*_solution.ipynb" -exec chmod 444 {} \; 2>/dev/null || true
fi

# Writable for exercise notebooks (chmod 644)
exercise_count=$(find "$REPO_ROOT/notes" -name "*_exercise.ipynb" 2>/dev/null | wc -l)
if [ "$exercise_count" -gt 0 ]; then
    find "$REPO_ROOT/notes" -name "*_exercise.ipynb" -exec chmod 644 {} \; 2>/dev/null || true
fi

if [ "$solution_count" -gt 0 ] || [ "$exercise_count" -gt 0 ]; then
    ok "Permissions set: $solution_count solution notebooks (read-only), $exercise_count exercise notebooks (writable)"
else
    warn "No exercise/solution notebooks found — skipping permission setup"
fi

# ─── Done ─────────────────────────────────────────────────────────────────────

# ─── STEP 2: Visual Studio Code ─────────────────────────────────────────────

echo ""
echo "══════════════════════════════════════════════"
echo "  AI/ML Dev Environment Setup — Step 2/7"
echo "  Visual Studio Code"
echo "══════════════════════════════════════════════"

step "Checking for Visual Studio Code"

CODE_CMD=""
for candidate in code code-insiders; do
    if command -v "$candidate" &>/dev/null; then
        code_ver=$("$candidate" --version 2>&1 | head -1)
        if echo "$code_ver" | grep -qE '[0-9]+\.[0-9]+'; then
            CODE_CMD="$candidate"
            ok "VS Code $code_ver already installed ($candidate)"
            break
        fi
    fi
done

if [ -z "$CODE_CMD" ]; then
    warn "VS Code not found — installing ..."
    case "$OS" in
        macos)
            if command -v brew &>/dev/null; then
                brew install --cask visual-studio-code
            else
                fail "Homebrew not found. Install VS Code manually from https://code.visualstudio.com/ and re-run."
            fi
            ;;
        debian)
            # Add Microsoft apt repository
            sudo apt-get install -y wget gpg
            wget -qO- https://packages.microsoft.com/keys/microsoft.asc \
                | gpg --dearmor \
                | sudo tee /usr/share/keyrings/microsoft-archive-keyring.gpg > /dev/null
            echo "deb [arch=amd64,arm64,armhf signed-by=/usr/share/keyrings/microsoft-archive-keyring.gpg] \
https://packages.microsoft.com/repos/code stable main" \
                | sudo tee /etc/apt/sources.list.d/vscode.list > /dev/null
            sudo apt-get update -qq
            sudo apt-get install -y code
            ;;
        fedora)
            sudo rpm --import https://packages.microsoft.com/keys/microsoft.asc
            sudo tee /etc/yum.repos.d/vscode.repo > /dev/null <<'EOF'
[code]
name=Visual Studio Code
baseurl=https://packages.microsoft.com/yumrepos/vscode
enabled=1
gpgcheck=1
gpgkey=https://packages.microsoft.com/keys/microsoft.asc
EOF
            sudo dnf install -y code
            ;;
        arch)
            # Use AUR helper if available, otherwise advise manual install
            if command -v yay &>/dev/null; then
                yay -S --noconfirm visual-studio-code-bin
            elif command -v paru &>/dev/null; then
                paru -S --noconfirm visual-studio-code-bin
            else
                warn "AUR helper not found. Install VS Code manually: https://code.visualstudio.com/"
                warn "Continuing without VS Code — Steps 3–5 may fail."
            fi
            ;;
        *)
            warn "Cannot auto-install VS Code on this OS. Download from https://code.visualstudio.com/"
            warn "Continuing — Steps 3–5 may fail if 'code' is not on PATH."
            ;;
    esac

    # Refresh PATH and verify
    export PATH="$PATH:/usr/bin:/usr/local/bin"
    if command -v code &>/dev/null; then
        CODE_CMD="code"
        code_ver=$(code --version 2>&1 | head -1)
        ok "VS Code $code_ver installed successfully"
    else
        warn "'code' not on PATH yet. Restart your terminal after this script finishes, then re-run for remaining steps."
        CODE_CMD="code"   # optimistically continue
    fi
else
    ok "Skipping install — VS Code already present"
fi

# ─── STEP 2.5: Core VS Code Extensions (Python/Jupyter Stack) ───────────────

echo ""
echo "══════════════════════════════════════════════"
echo "  AI/ML Dev Environment Setup — Step 2.5/7"
echo "  Core Python/Jupyter Extensions"
echo "══════════════════════════════════════════════"

CORE_EXTENSIONS=(
    "ms-python.python"              # Python language support
    "ms-python.vscode-pylance"      # Fast Python language server
    "ms-toolsai.jupyter"            # Jupyter notebook support
    "ms-python.black-formatter"     # Black code formatter
    "ms-python.isort"               # Import organization
    "yzhang.markdown-all-in-one"    # Markdown editing
    "esbenp.prettier-vscode"        # Prettier formatter (Markdown/JSON)
)

step "Installing core extensions for Python/Jupyter development"

installed_count=0
skipped_count=0
failed_count=0

for ext_id in "${CORE_EXTENSIONS[@]}"; do
    ext_name="${ext_id##*.}"  # Extract short name after last dot

    # Check if already installed
    already_installed=false
    if command -v "${CODE_CMD}" &>/dev/null; then
        if "${CODE_CMD}" --list-extensions 2>/dev/null | grep -qi "^${ext_id}$"; then
            echo "  ✓ $ext_name (already installed)"
            skipped_count=$((skipped_count + 1))
            already_installed=true
        fi
    fi

    if [ "$already_installed" = false ]; then
        if command -v "${CODE_CMD}" &>/dev/null; then
            "${CODE_CMD}" --install-extension "$ext_id" --force &>/dev/null || true
            # Verify installation
            if "${CODE_CMD}" --list-extensions 2>/dev/null | grep -qi "^${ext_id}$"; then
                ok "$ext_name installed"
                installed_count=$((installed_count + 1))
            else
                warn "$ext_name install ran but not yet detected"
                installed_count=$((installed_count + 1))  # Count as success; may appear after restart
            fi
        else
            warn "Cannot install $ext_name: 'code' not on PATH"
            failed_count=$((failed_count + 1))
        fi
    fi
done

echo ""
echo "  Extension Summary:"
echo "    • Installed: $installed_count"
echo "    • Already present: $skipped_count"
if [ "$failed_count" -gt 0 ]; then
    echo "    • Failed: $failed_count (install manually via Extensions panel)"
fi
echo ""

# ─── STEP 3: Kilo Code (Agentic AI) Extension ───────────────────────────────
#
# Kilo Code is an open-source agentic coding assistant (fork of Roo/Cline) that
# can plan, edit files, and run commands. We point it at a locally-hosted
# DeepSeek-R1 reasoning SLM via Ollama (configured in Step 6d).

if [ "$ENABLE_SLM_ASSISTANT" = true ]; then

echo ""
echo "══════════════════════════════════════════════"
echo "  AI/ML Dev Environment Setup — Step 3/7"
echo "  Kilo Code — Agentic AI Extension"
echo "══════════════════════════════════════════════"

KILO_EXT_ID="kilocode.kilo-code"

step "Checking Kilo Code extension ($KILO_EXT_ID)"

EXTENSION_INSTALLED=false
if command -v "${CODE_CMD}" &>/dev/null; then
    if "${CODE_CMD}" --list-extensions 2>/dev/null | grep -qi "$KILO_EXT_ID"; then
        ok "Kilo Code already installed"
        EXTENSION_INSTALLED=true
    fi
else
    warn "'${CODE_CMD}' not on PATH — skipping extension check"
fi

if [ "$EXTENSION_INSTALLED" = false ]; then
    warn "Kilo Code not found — installing ..."
    if command -v "${CODE_CMD}" &>/dev/null; then
        "${CODE_CMD}" --install-extension "$KILO_EXT_ID" --force &>/dev/null || true
        # Verify
        if "${CODE_CMD}" --list-extensions 2>/dev/null | grep -qi "$KILO_EXT_ID"; then
            ok "Kilo Code installed successfully"
        else
            warn "Install ran but extension not detected yet — it may appear after VS Code restarts"
        fi
    else
        warn "Cannot install Kilo Code: 'code' not on PATH."
        warn "Install manually: open VS Code → Extensions → search 'Kilo Code' → Install"
    fi
fi

step "Kilo Code post-install configuration note"
echo ""
echo "  After launching VS Code:"
echo "    1. Open the Kilo Code sidebar (kangaroo icon on the Activity Bar)"
echo "    2. Click 'Settings' → API Provider: Ollama"
echo "    3. Base URL: http://localhost:11434   (the auto-discover button works too)"
echo "    4. Model: deepseek-r1:8b  (or deepseek-r1:1.5b on low-RAM machines)"
echo "    5. Save — Kilo Code will now drive agentic edits with DeepSeek-R1 reasoning"
echo ""

# ─── STEP 4: Ollama Server Install & First Launch ────────────────────────────

echo ""
echo "══════════════════════════════════════════════"
echo "  AI/ML Dev Environment Setup — Step 4/7"
echo "  Ollama Local Inference Server"
echo "══════════════════════════════════════════════"

OLLAMA_PORT=11434
OLLAMA_BASE_URL="http://localhost:${OLLAMA_PORT}"
PID_FILE="$REPO_ROOT/.ollama.pid"

# ── 4a. Install Ollama ────────────────────────────────────────────────────────

step "Checking Ollama binary"

OLLAMA_INSTALLED=false
if command -v ollama &>/dev/null; then
    ollama_ver=$(ollama --version 2>&1 || true)
    ok "Ollama already installed: $ollama_ver"
    OLLAMA_INSTALLED=true
fi

if [ "$OLLAMA_INSTALLED" = false ]; then
    warn "Ollama not found — installing ..."
    case "$OS" in
        macos)
            if command -v brew &>/dev/null; then
                brew install ollama
            else
                fail "Homebrew not found. Install Ollama manually from https://ollama.com/download and re-run."
            fi
            ;;
        debian|fedora|arch|linux)
            # Official install script (works on most Linux distros)
            curl -fsSL https://ollama.com/install.sh | sh
            ;;
        *)
            fail "Cannot auto-install Ollama on this OS. Download from https://ollama.com/download and re-run."
            ;;
    esac
    export PATH="$PATH:/usr/local/bin:/usr/bin"
    if command -v ollama &>/dev/null; then
        ok "Ollama installed: $(ollama --version 2>&1 || true)"
    else
        fail "Ollama installation completed but 'ollama' not found on PATH. Restart terminal and re-run."
    fi
fi

# ── 4b. Start the Ollama server ───────────────────────────────────────────────

step "Starting Ollama server"

# Check if already listening
SERVER_RUNNING=false
if curl -sf --max-time 3 "$OLLAMA_BASE_URL" &>/dev/null; then
    ok "Ollama server already running at $OLLAMA_BASE_URL"
    SERVER_RUNNING=true
fi

if [ "$SERVER_RUNNING" = false ]; then
    warn "Ollama server not running — starting in background ..."

    # Pin Ollama to a single loaded model with no parallelism, so the 8B
    # reasoning SLM owns the GPU/RAM exclusively while Kilo Code is working.
    export OLLAMA_MAX_LOADED_MODELS=1
    export OLLAMA_NUM_PARALLEL=1
    export OLLAMA_CONTEXT_LENGTH=4096

    # Start server detached, redirect output to a log file
    nohup env OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NUM_PARALLEL=1 OLLAMA_CONTEXT_LENGTH=4096 \
        ollama serve > "$REPO_ROOT/.ollama.log" 2>&1 &
    OLLAMA_BG_PID=$!
    echo "$OLLAMA_BG_PID" > "$PID_FILE"
    ok "Ollama server started (PID $OLLAMA_BG_PID, log: .ollama.log)"
    ok "Single-model mode: OLLAMA_MAX_LOADED_MODELS=1, OLLAMA_NUM_PARALLEL=1, OLLAMA_CONTEXT_LENGTH=4096"
    ok "PID saved to .ollama.pid"

    # Health-check with retries
    MAX_RETRIES=12
    RETRIES=0
    HEALTHY=false
    while [ "$RETRIES" -lt "$MAX_RETRIES" ] && [ "$HEALTHY" = false ]; do
        sleep 1
        if curl -sf --max-time 2 "$OLLAMA_BASE_URL" &>/dev/null; then
            HEALTHY=true
        fi
        RETRIES=$((RETRIES + 1))
    done

    if [ "$HEALTHY" = true ]; then
        ok "Ollama server is healthy at $OLLAMA_BASE_URL"
    else
        warn "Ollama server did not respond within ${MAX_RETRIES}s — it may still be starting up"
        warn "Check manually: curl $OLLAMA_BASE_URL  or  cat .ollama.log"
    fi
fi

# ─── STEP 5: Ollama Lifecycle Wiring ──────────────────────────────────────────
#
# Strategy: write .vscode/tasks.json with a folderOpen task that starts
# ollama serve, and a companion stop task.  VS Code has no native onClose
# hook, so we also write a small watcher script that monitors the 'code'
# process and stops Ollama when it exits.

echo ""
echo "══════════════════════════════════════════════"
echo "  AI/ML Dev Environment Setup — Step 5/7"
echo "  Ollama Lifecycle Wiring"
echo "══════════════════════════════════════════════"

VSCODE_DIR="$REPO_ROOT/.vscode"
TASKS_JSON="$VSCODE_DIR/tasks.json"
WATCHER_SCRIPT="$REPO_ROOT/scripts/ollama-watcher.sh"

# ── 5a. Write .vscode/tasks.json ──────────────────────────────────────────────

step "Configuring .vscode/tasks.json"

mkdir -p "$VSCODE_DIR"

WRITE_TASKS=true
if [ -f "$TASKS_JSON" ]; then
    if grep -q 'ollama-start' "$TASKS_JSON" 2>/dev/null && grep -q 'cce-watcher' "$TASKS_JSON" 2>/dev/null; then
        ok "tasks.json already contains ollama + cce-watcher tasks — skipping"
        WRITE_TASKS=false
    else
        warn "tasks.json exists but is missing tasks — backing up and rewriting"
        cp "$TASKS_JSON" "${TASKS_JSON}.bak"
    fi
fi

if [ "$WRITE_TASKS" = true ]; then
    cat > "$TASKS_JSON" << 'TASKSJSON'
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "ollama-start",
            "type": "shell",
            "command": "bash",
            "args": [
                "${workspaceFolder}/scripts/ollama-watcher.sh"
            ],
            "runOptions": {
                "runOn": "folderOpen"
            },
            "presentation": {
                "reveal": "never",
                "panel": "dedicated",
                "showReuseMessage": false
            },
            "problemMatcher": []
        },
        {
            "label": "ollama-stop",
            "type": "shell",
            "command": "bash",
            "args": [
                "-c",
                "pid=$(cat '${workspaceFolder}/.ollama.pid' 2>/dev/null); [ -n \"$pid\" ] && kill \"$pid\" 2>/dev/null; pkill -x ollama 2>/dev/null; true"
            ],
            "presentation": {
                "reveal": "never",
                "showReuseMessage": false
            },
            "problemMatcher": []
        },
        {
            "label": "cce-watcher",
            "detail": "File-system watcher — triggers cce index --changed-only on new files and notebook cell changes",
            "type": "shell",
            "command": "bash",
            "args": ["${workspaceFolder}/scripts/cce-watcher.sh"],
            "runOptions": {
                "runOn": "folderOpen"
            },
            "presentation": {
                "reveal": "never",
                "panel": "dedicated",
                "showReuseMessage": false,
                "close": false
            },
            "problemMatcher": []
        }
    ]
}
TASKSJSON
    ok "Written: .vscode/tasks.json"
    warn "ACTION REQUIRED: open VS Code → Terminal → Run Task → 'Allow Automatic Tasks'"
fi

# ── 5b. Write the watcher script ──────────────────────────────────────────────

step "Writing ollama-watcher.sh"

# Always rewrite: the watcher content evolves (env vars, model pins, ...) and
# we want re-running setup.sh to keep it in sync.
if [ -f "$WATCHER_SCRIPT" ]; then
    ok "Overwriting existing ollama-watcher.sh"
fi

cat > "$WATCHER_SCRIPT" << 'WATCHER'
#!/usr/bin/env bash
# ollama-watcher.sh
# Launched automatically when this VS Code workspace opens (via tasks.json folderOpen).
# - Starts ollama serve if not already running
# - Monitors the VS Code process
# - Stops ollama serve when VS Code exits

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$REPO_ROOT")"  # scripts/ -> repo root
PID_FILE="$REPO_ROOT/.ollama.pid"
OLLAMA_URL="http://localhost:11434"

is_ollama_running() {
    curl -sf --max-time 2 "$OLLAMA_URL" &>/dev/null
}

# Start ollama if not already up
if ! is_ollama_running; then
    # Pin to a single loaded model with no parallelism and a 4096-token ctx.
    export OLLAMA_MAX_LOADED_MODELS=1
    export OLLAMA_NUM_PARALLEL=1
    export OLLAMA_CONTEXT_LENGTH=4096
    nohup env OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NUM_PARALLEL=1 OLLAMA_CONTEXT_LENGTH=4096 \
        ollama serve >> "$REPO_ROOT/.ollama.log" 2>&1 &
    echo $! > "$PID_FILE"
    sleep 3
fi

# Wait until all VS Code windows are gone
while true; do
    sleep 5
    if ! pgrep -x code &>/dev/null && ! pgrep -x "Code" &>/dev/null; then
        # VS Code has exited — stop Ollama
        saved_pid=$(cat "$PID_FILE" 2>/dev/null || true)
        [ -n "$saved_pid" ] && kill "$saved_pid" 2>/dev/null || true
        pkill -x ollama 2>/dev/null || true
        rm -f "$PID_FILE"
        break
    fi
done
WATCHER
chmod +x "$WATCHER_SCRIPT"
ok "Written: scripts/ollama-watcher.sh"

# ─── STEP 6: Pull DeepSeek-R1 Reasoning SLM ──────────────────────────────────
#
# DeepSeek-R1 is the reasoning model that powers Kilo Code's agentic planning.
#   Primary:  deepseek-r1:8b-llama-distill-q4_K_M  (~5 GB, needs ~10 GB free RAM)
#   Fallback: deepseek-r1:1.5b-qwen-distill-q4_0   (~1.1 GB, needs ~3 GB free RAM)
# Selection is automatic based on detected system RAM.
#
# After pulling, we derive a companion model tagged '-ctx4k' with
# `PARAMETER num_ctx 4096` baked in, so every client (Kilo Code, curl, raw API)
# gets a 4096-token context window without having to pass num_ctx explicitly.

echo ""
echo "══════════════════════════════════════════════"
echo "  AI/ML Dev Environment Setup — Step 6/7"
echo "  Pull DeepSeek-R1 Reasoning SLM"
echo "══════════════════════════════════════════════"

PRIMARY_BASE="deepseek-r1:8b-llama-distill-q4_K_M"
FALLBACK_BASE="deepseek-r1:1.5b-qwen-distill-q4_0"
CTX_TOKENS=4096

# ── 6a. Detect system RAM ────────────────────────────────────────────────────

step "Detecting system RAM"

TOTAL_RAM_GB=0
case "$OS" in
    macos)
        total_bytes=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
        TOTAL_RAM_GB=$(( total_bytes / 1073741824 ))
        ;;
    debian|fedora|arch|linux|*)
        # /proc/meminfo reports kB
        mem_kb=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}' || echo 0)
        TOTAL_RAM_GB=$(( mem_kb / 1048576 ))
        ;;
esac

ok "Total RAM: ${TOTAL_RAM_GB} GB"

if [ "$TOTAL_RAM_GB" -ge 10 ]; then
    BASE_MODEL="$PRIMARY_BASE"
    ok "RAM ≥ 10 GB — selecting primary base: $BASE_MODEL"
else
    BASE_MODEL="$FALLBACK_BASE"
    warn "RAM < 10 GB — selecting fallback base: $BASE_MODEL"
fi

# Derived tag: same base with '-ctx4k' suffix. This is what Kilo Code targets.
CHOSEN_MODEL="$(echo "$BASE_MODEL" | cut -d: -f1):$(echo "$BASE_MODEL" | cut -d: -f2)-ctx4k"
ok "Derived model (num_ctx=${CTX_TOKENS}): $CHOSEN_MODEL"

# ── 6b. Check if base model already pulled ──────────────────────────────────

step "Checking if $BASE_MODEL is already available"

BASE_PRESENT=false
if ollama list 2>/dev/null | grep -qF "$BASE_MODEL"; then
    ok "$BASE_MODEL already present in Ollama"
    BASE_PRESENT=true
fi

# ── 6c. Pull the base model ──────────────────────────────────────────────────

if [ "$BASE_PRESENT" = false ]; then
    step "Pulling $BASE_MODEL (this may take a few minutes on first run)"
    echo "  Downloading model — progress shown below:"
    echo ""
    if ollama pull "$BASE_MODEL"; then
        ok "$BASE_MODEL pulled successfully"
    else
        warn "Pull failed — check your internet connection and retry: ollama pull $BASE_MODEL"
    fi
else
    step "Skipping pull — $BASE_MODEL already present"
fi

# ── 6c'. Create the derived -ctx4k model with num_ctx=4096 baked in ─────────

step "Creating derived model $CHOSEN_MODEL with num_ctx=${CTX_TOKENS}"

if ollama list 2>/dev/null | grep -qF "$CHOSEN_MODEL"; then
    ok "$CHOSEN_MODEL already present — skipping create"
else
    MODELFILE="$REPO_ROOT/.ollama.Modelfile"
    cat > "$MODELFILE" << MFEOF
FROM $BASE_MODEL
PARAMETER num_ctx $CTX_TOKENS
MFEOF
    if ollama create "$CHOSEN_MODEL" -f "$MODELFILE"; then
        ok "Created $CHOSEN_MODEL (num_ctx=${CTX_TOKENS})"
    else
        warn "ollama create failed — Kilo Code will have to pass num_ctx itself"
        CHOSEN_MODEL="$BASE_MODEL"
    fi
    rm -f "$MODELFILE"
fi

# ── 6d. Configure Kilo Code to use the DeepSeek-R1 model ─────────────────────
#
# The current Kilo Code extension (built on the Kilo CLI) reads its config from
#   ~/.config/kilo/kilo.jsonc        (global, used by both VS Code & CLI)
# and from a project-level kilo.jsonc / .kilo/kilo.jsonc.
#
# We write BOTH:
#   • the global file → makes our local Ollama model the default everywhere
#   • a project-level .kilo/kilo.jsonc → shared with anyone who opens this repo
#
# Schema reference: https://app.kilo.ai/config.json
# Docs: https://kilo.ai/docs/code-with-ai/agents/custom-models

step "Writing Kilo Code config (global + project) so DeepSeek-R1 is the default model"

KILO_MODEL_KEY="$CHOSEN_MODEL"
KILO_MODEL_REF="ollama/$KILO_MODEL_KEY"

KILO_CONFIG_JSON=$(cat << KILOCFG
{
  "\$schema": "https://app.kilo.ai/config.json",
  "model": "${KILO_MODEL_REF}",
  "provider": {
    "ollama": {
      "options": {
        "baseURL": "http://localhost:11434/v1",
        "timeout": 600000
      },
      "models": {
        "${KILO_MODEL_KEY}": {
          "name": "DeepSeek-R1 (local Ollama, 4k ctx)",
          "tool_call": true,
          "reasoning": true,
          "limit": {
            "context": ${CTX_TOKENS},
            "output": ${CTX_TOKENS}
          }
        }
      }
    }
  }
}
KILOCFG
)

# 1) Global config ── ~/.config/kilo/kilo.jsonc
KILO_GLOBAL_DIR="$HOME/.config/kilo"
KILO_GLOBAL_FILE="$KILO_GLOBAL_DIR/kilo.jsonc"
mkdir -p "$KILO_GLOBAL_DIR"
if [ -f "$KILO_GLOBAL_FILE" ]; then
    cp "$KILO_GLOBAL_FILE" "${KILO_GLOBAL_FILE}.bak"
    warn "Existing global Kilo config backed up to ${KILO_GLOBAL_FILE}.bak"
fi
echo "$KILO_CONFIG_JSON" > "$KILO_GLOBAL_FILE"
ok "Global Kilo config written: $KILO_GLOBAL_FILE"

# 2) Project config ── <repo>/.kilo/kilo.jsonc
KILO_PROJECT_DIR="$REPO_ROOT/.kilo"
KILO_PROJECT_FILE="$KILO_PROJECT_DIR/kilo.jsonc"
mkdir -p "$KILO_PROJECT_DIR"
echo "$KILO_CONFIG_JSON" > "$KILO_PROJECT_FILE"
ok "Project Kilo config written: .kilo/kilo.jsonc"

# 3) Auto-launch the Kilo Code sidebar when this workspace opens.
# Add a folderOpen command-task that focuses the Kilo Code view container.
step "Wiring Kilo Code sidebar to auto-open with this workspace"

if [ -f "$TASKS_JSON" ] && command -v python &>/dev/null; then
    python - "$TASKS_JSON" "$CODE_CMD" << 'PYEOF'
import json, sys
p = sys.argv[1]
code_cmd = sys.argv[2]
try:
    with open(p, 'r') as f:
        data = json.load(f)
except Exception:
    data = {"version": "2.0.0", "tasks": []}
data.setdefault("tasks", [])
if not any(t.get("label") == "kilo-code-launch" for t in data["tasks"]):
    data["tasks"].append({
        "label": "kilo-code-launch",
        "type": "shell",
        "command": "bash",
        "args": ["-c", f"'{code_cmd}' --command kilo-code.SidebarProvider.focus 2>/dev/null; exit 0"],
        "runOptions": {"runOn": "folderOpen"},
        "presentation": {"reveal": "never", "panel": "dedicated", "showReuseMessage": False},
        "problemMatcher": []
    })
    with open(p, 'w') as f:
        json.dump(data, f, indent=4)
    print("  ✓ Added 'kilo-code-launch' folderOpen task to .vscode/tasks.json")
else:
    print("  ✓ tasks.json already has 'kilo-code-launch' — skipping")
PYEOF
fi

# 4) Recommend the Kilo extension at the workspace level so VS Code surfaces it.
EXTENSIONS_JSON="$VSCODE_DIR/extensions.json"
if [ -f "$EXTENSIONS_JSON" ] && command -v python &>/dev/null; then
    python - "$EXTENSIONS_JSON" "$KILO_EXT_ID" << 'PYEOF'
import json, sys
p, ext = sys.argv[1], sys.argv[2]
try:
    with open(p, 'r') as f:
        data = json.load(f)
except Exception:
    data = {}
recs = data.setdefault("recommendations", [])
if ext not in recs:
    recs.append(ext)
    with open(p, 'w') as f:
        json.dump(data, f, indent=4)
print(f"  ✓ Kilo Code present in extensions.json recommendations")
PYEOF
else
    cat > "$EXTENSIONS_JSON" << EXTEOF
{
  "recommendations": ["${KILO_EXT_ID}"]
}
EXTEOF
    ok "Created .vscode/extensions.json with Kilo Code recommended"
fi


# ─── VS Code notebook settings ───────────────────────────────────────────────
#
# Keep notebooks read-only in VS Code so Jupyter Lab stays the only editor for
# .ipynb files in this workspace.

else
    warn "Skipping the SLM assistant bundle. Re-run with --enable-slm-assistant to install Kilo Code, Ollama, and the local model wiring."
fi

step "Writing .vscode/settings.json (notebooks read-only in VS Code)"

SETTINGS_JSON="$VSCODE_DIR/settings.json"

if [ -f "$SETTINGS_JSON" ]; then
    python - "$SETTINGS_JSON" << 'PYEOF'
import json, sys
p = sys.argv[1]
try:
    with open(p, "r") as f:
        data = json.load(f)
except Exception:
    data = {}
ro = data.setdefault("files.readonlyInclude", {})
ro["**/*.ipynb"] = True
data["notebook.defaultKernel"] = "ai-ml-dev"
with open(p, "w") as f:
    json.dump(data, f, indent=4)
print("  \u2713 Merged read-only rule into existing .vscode/settings.json")
PYEOF
else
    mkdir -p "$VSCODE_DIR"
    cat > "$SETTINGS_JSON" << 'SETTINGSJSON'
{
    "files.readonlyInclude": {
        "**/*.ipynb": true
    },
    "notebook.defaultKernel": "ai-ml-dev"
}
SETTINGSJSON
    ok "Written: .vscode/settings.json"
fi

# ─── STEP 7: Launch Study Servers (Jupyter Lab + MkDocs) ─────────────────────
#
# Fixed local ports so bookmarks stay stable:
#   • Jupyter Lab  → http://localhost:8888   (hands-on coding in notebooks)
#   • MkDocs site  → http://localhost:8000   (read notes/ in a web browser)
#
# Both run as detached background processes so this script can exit.
# PIDs saved to .jupyter.pid / .mkdocs.pid for a later stop command.

echo ""
echo "══════════════════════════════════════════════"
echo "  AI/ML Dev Environment Setup — Step 7/7"
echo "  Launch Study Servers (Jupyter + MkDocs)"
echo "══════════════════════════════════════════════"

JUPYTER_PORT=8888
MKDOCS_PORT=8000
JUPYTER_PID_FILE="$REPO_ROOT/.jupyter.pid"
MKDOCS_PID_FILE="$REPO_ROOT/.mkdocs.pid"
JUPYTER_LOG="$REPO_ROOT/.jupyter.log"
MKDOCS_LOG="$REPO_ROOT/.mkdocs.log"

port_in_use() {
    local port="$1"
    if command -v lsof &>/dev/null; then
        lsof -iTCP:"$port" -sTCP:LISTEN &>/dev/null
    elif command -v ss &>/dev/null; then
        ss -ltn "sport = :$port" 2>/dev/null | grep -q ":$port"
    else
        # Last resort: try to bind
        (echo > "/dev/tcp/127.0.0.1/$port") &>/dev/null
    fi
}

# ── 7a. Jupyter Lab ──────────────────────────────────────────────────────────

step "Starting Jupyter Lab on port $JUPYTER_PORT"

if port_in_use "$JUPYTER_PORT"; then
    ok "Port $JUPYTER_PORT already in use — assuming Jupyter Lab is running"
else
    nohup python -m jupyter lab \
        --no-browser \
        --ServerApp.ip=127.0.0.1 \
        --ServerApp.port="$JUPYTER_PORT" \
        --ServerApp.port_retries=0 \
        --ServerApp.root_dir="$REPO_ROOT" \
        --ServerApp.open_browser=False \
        > "$JUPYTER_LOG" 2>&1 &
    echo $! > "$JUPYTER_PID_FILE"
    ok "Jupyter Lab started (PID $(cat "$JUPYTER_PID_FILE")) — log: .jupyter.log"
    echo "    Check .jupyter.log for the one-time login token/URL."
fi

# ── 7b. MkDocs site ──────────────────────────────────────────────────────────

if [ "$ENABLE_MKDOCS_SERVER" = true ]; then
    step "Starting MkDocs site on port $MKDOCS_PORT"

    if port_in_use "$MKDOCS_PORT"; then
        ok "Port $MKDOCS_PORT already in use — assuming MkDocs is running"
    else
        nohup python -m mkdocs serve \
            -f "$REPO_ROOT/mkdocs.yml" \
            -a "127.0.0.1:$MKDOCS_PORT" \
            > "$MKDOCS_LOG" 2>&1 &
        echo $! > "$MKDOCS_PID_FILE"
        ok "MkDocs started (PID $(cat "$MKDOCS_PID_FILE")) — log: .mkdocs.log"
    fi
else
    warn "Skipping MkDocs site; pass --enable-mkdocs-server when you want a local docs server."
fi

# ─── ALL DONE ─────────────────────────────────────────────────────────────────────

echo ""
echo "══════════════════════════════════════════════"
if [ "$ENABLE_SLM_ASSISTANT" = true ]; then
    echo "  Setup complete (all 7 steps)"
else
    echo "  Setup complete (assistant bundle skipped)"
fi
echo ""
echo "  Python env  : $VENV_PATH"
echo "  Activate    : source .venv/bin/activate"
echo "  VS Code     : ${CODE_CMD}"
if [ "$ENABLE_SLM_ASSISTANT" = true ]; then
    echo "  Kilo Code   : $KILO_EXT_ID"
    echo "  Ollama      : $OLLAMA_BASE_URL  (single-model mode, ctx=${CTX_TOKENS})"
    echo "  Reasoning   : $CHOSEN_MODEL (DeepSeek-R1, ${CTX_TOKENS}-token ctx)"
else
    echo "  SLM assistant: disabled (pass --enable-slm-assistant to install Kilo Code, Ollama, and model wiring)"
fi
if [ "$ENABLE_MKDOCS_SERVER" = true ]; then
    echo "  MkDocs      : http://localhost:$MKDOCS_PORT"
else
    echo "  MkDocs      : disabled (pass --enable-mkdocs-server to launch the local docs server)"
fi
echo ""
echo "  Study servers (running in background):"
echo "    Hands-on notebooks  → http://localhost:$JUPYTER_PORT"
if [ "$ENABLE_MKDOCS_SERVER" = true ]; then
    echo "    Reading (MkDocs)    → http://localhost:$MKDOCS_PORT"
fi
echo ""
echo "  To stop them:"
if [ "$ENABLE_MKDOCS_SERVER" = true ]; then
    echo "    kill \$(cat .jupyter.pid .mkdocs.pid 2>/dev/null)"
else
    echo "    kill \$(cat .jupyter.pid 2>/dev/null)"
fi
echo ""
if [ "$ENABLE_SLM_ASSISTANT" = true ]; then
    echo "  Next: open VS Code in this folder — Ollama will start automatically."
    echo "  If prompted, click 'Allow Automatic Tasks' to enable the watcher."
fi
echo "══════════════════════════════════════════════"
echo ""
