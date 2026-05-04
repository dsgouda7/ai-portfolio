# setup.ps1 — AI/ML Dev Environment Setup (Windows / PowerShell)
# Provisions Python, a full AI/ML library stack, VS Code, and optionally a local SLM assistant bundle.
# Run from anywhere:
#   .\scripts\setup.ps1
#
# Steps implemented so far:
#   1. Python + AI/ML libraries  ✔
#   2. VS Code install            ✔
#   3. Kilo Code (agentic AI) extension  ✔
#   4. Ollama server install & first launch  ✔
#   5. Lifecycle wiring (Ollama runs with VS Code)  ✔
#   6. Pull DeepSeek-R1 reasoning SLM for Kilo Code  ✔

# Bootstrap: this script is designed for PowerShell 7+.
# If launched from Windows PowerShell 5.1, install pwsh (if needed) and re-run there.
$RunningInPwsh7 = ($PSVersionTable.PSEdition -eq "Core" -and $PSVersionTable.PSVersion.Major -ge 7)
if (-not $RunningInPwsh7) {
    Write-Host ""
    Write-Host "[setup] PowerShell 7 is required. Checking for pwsh..." -ForegroundColor Cyan

    $pwshCommand = Get-Command pwsh -ErrorAction SilentlyContinue
    if (-not $pwshCommand) {
        Write-Host "[setup] pwsh not found. Attempting install via winget..." -ForegroundColor Yellow
        if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
            Write-Host "[setup] winget is not available, cannot auto-install PowerShell 7." -ForegroundColor Red
            Write-Host "[setup] Install manually: https://aka.ms/powershell-release?tag=stable" -ForegroundColor Red
            exit 1
        }

        try {
            winget install --id Microsoft.PowerShell --source winget --silent --accept-package-agreements --accept-source-agreements
            # Refresh PATH in current process to pick up freshly installed pwsh.
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                        [System.Environment]::GetEnvironmentVariable("Path", "User")
            $pwshCommand = Get-Command pwsh -ErrorAction SilentlyContinue
        } catch {
            Write-Host "[setup] Failed to install PowerShell 7 automatically." -ForegroundColor Red
            Write-Host "[setup] Install manually: https://aka.ms/powershell-release?tag=stable" -ForegroundColor Red
            exit 1
        }
    }

    if (-not $pwshCommand) {
        Write-Host "[setup] pwsh is still unavailable after installation attempt." -ForegroundColor Red
        exit 1
    }

    Write-Host "[setup] Relaunching setup in PowerShell 7..." -ForegroundColor Green
    $forwardArgs = @("-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $PSCommandPath) + $args
    & $pwshCommand.Source @forwardArgs
    exit $LASTEXITCODE
}

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# scripts/ → repo root
$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $RepoRoot ".venv"
$VscodeDirPath = Join-Path $RepoRoot ".vscode"

$EnableSlmAssistant = $false
$EnableMkdocsServer = $false
$EnableGpuNotebookStack = $false
foreach ($arg in $args) {
    if ($arg -eq "--enable-slm-assistant" -or $arg -eq "-EnableSlmAssistant") {
        $EnableSlmAssistant = $true
    } elseif ($arg -eq "--enable-mkdocs-server" -or $arg -eq "-EnableMkdocsServer") {
        $EnableMkdocsServer = $true
    }
}

# ─── Helpers ──────────────────────────────────────────────────────────────────

function Write-Step  { param($msg) Write-Host "`n▶ $msg" -ForegroundColor Cyan }
function Write-Ok    { param($msg) Write-Host "  ✓ $msg" -ForegroundColor Green }
function Write-Warn  { param($msg) Write-Host "  ! $msg" -ForegroundColor Yellow }
function Write-Fail  { param($msg) Write-Host "  ✗ $msg" -ForegroundColor Red; exit 1 }
function Write-Group { param($msg) Write-Host "`n  ── $msg" -ForegroundColor DarkCyan }

# ─── STEP 1: Python + AI/ML Libraries ────────────────────────────────────────

Write-Host ""
Write-Host "══════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host "  AI/ML Dev Environment Setup — Step 1/7" -ForegroundColor White
Write-Host "  Python + AI/ML Library Stack" -ForegroundColor White
Write-Host "══════════════════════════════════════════════" -ForegroundColor DarkGray

# ─── 1a. Python ───────────────────────────────────────────────────────────────

Write-Step "Checking Python 3.11+"

$Python = $null
foreach ($candidate in @("python", "python3")) {
    try {
        $ver = & $candidate --version 2>&1
        if ($ver -match "Python (\d+)\.(\d+)") {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            if ($major -ge 3 -and $minor -ge 11) {
                $Python = $candidate
                Write-Ok "$ver"
                break
            } elseif ($major -ge 3 -and $minor -ge 9) {
                Write-Warn "$ver found — Python 3.11+ is recommended; continuing with $ver"
                $Python = $candidate
                break
            } else {
                Write-Warn "$ver found but Python 3.9+ is required"
            }
        }
    } catch { }
}

if (-not $Python) {
    Write-Warn "Python not found — attempting install via winget ..."
    try {
        winget install --id Python.Python.3.11 --source winget --silent --accept-package-agreements --accept-source-agreements
        # Refresh PATH for this session
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                    [System.Environment]::GetEnvironmentVariable("Path", "User")
        $Python = "python"
        $ver = & python --version 2>&1
        Write-Ok "Installed: $ver"
    } catch {
        Write-Fail "Could not install Python automatically. Install Python 3.11+ from https://www.python.org/downloads/ and re-run."
    }
}

# ─── 1b. pip ──────────────────────────────────────────────────────────────────

Write-Step "Checking pip"
try {
    $pipVer = & $Python -m pip --version 2>&1
    Write-Ok $pipVer
} catch {
    Write-Warn "pip not available — bootstrapping ..."
    & $Python -m ensurepip --upgrade
    $pipVer = & $Python -m pip --version 2>&1
    Write-Ok $pipVer
}

# ─── 1c. Virtual environment ──────────────────────────────────────────────────

Write-Step "Virtual environment (.venv)"

if (Test-Path $VenvPath) {
    Write-Ok "Existing .venv found — reusing"
} else {
    Write-Warn "No .venv found — creating ..."
    & $Python -m venv $VenvPath
    Write-Ok "Created .venv at $VenvPath"
}

$VenvScriptsPath = Join-Path $VenvPath "Scripts"
$VenvPython = Join-Path $VenvScriptsPath "python.exe"
$ActivateScript = Join-Path $VenvScriptsPath "Activate.ps1"

if (-not (Test-Path $VenvPython)) {
    Write-Fail "Cannot find venv Python interpreter at $VenvPython"
}

if (Test-Path $ActivateScript) {
    . $ActivateScript
    Write-Ok "Activated .venv via $ActivateScript"
} else {
    Write-Warn "Activate.ps1 not found under .venv/Scripts; continuing with direct venv PATH wiring"
    $env:VIRTUAL_ENV = $VenvPath
    $pathEntries = $env:Path -split ';'
    if ($pathEntries -notcontains $VenvScriptsPath) {
        $env:Path = "$VenvScriptsPath;$env:Path"
    }
    Write-Ok "Using venv interpreter directly: $VenvPython"
}

# Upgrade pip + build tools quietly
& python -m pip install --upgrade pip setuptools wheel --quiet
Write-Ok "pip / setuptools / wheel up to date"

# ─── 1d. Package installation ─────────────────────────────────────────────────

Write-Step "Installing AI/ML package stack"

function Normalize-PackageKey {
    param([string]$Package)
    return ($Package -replace '\[.*\]', '' -replace '[>=<!\s].*', '').ToLower().Trim()
}

$script:InstalledPkgSet = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
try {
    $installedNow = (& python -m pip list --format=columns 2>&1 | Select-Object -Skip 2) -replace '\s+.*', ''
    foreach ($name in $installedNow) {
        if ($name) { [void]$script:InstalledPkgSet.Add($name.ToLower()) }
    }
} catch {
    Write-Warn "Could not snapshot installed pip packages; installation checks will run per package"
}

function Install-Group {
    param(
        [string]   $GroupName,
        [string[]] $Packages,
        [string]   $ExtraArgs = ""
    )
    Write-Group $GroupName

    foreach ($pkg in $Packages) {
        # Normalise: strip extras/version for display key
        $key = Normalize-PackageKey $pkg
        if ($script:InstalledPkgSet.Contains($key)) {
            Write-Ok "$pkg already installed"
        } else {
            Write-Warn "$pkg missing — installing ..."
            $extraArgList = @()
            if ($ExtraArgs) {
                $extraArgList = $ExtraArgs.Split(" ") | Where-Object { $_ -and $_.Trim() }
            }

            # First attempt: normal pip install
            & python -m pip install $pkg @extraArgList --quiet

            if ($LASTEXITCODE -ne 0) {
                # Retry without cache to avoid local wheel cache permission/lock issues.
                Write-Warn "Initial install failed for $pkg — retrying with --no-cache-dir"
                & python -m pip install $pkg @extraArgList --quiet --no-cache-dir
            }

            if ($LASTEXITCODE -ne 0) { Write-Fail "Failed to install $pkg after retry" }
            [void]$script:InstalledPkgSet.Add($key)
            Write-Ok "$pkg installed"
        }
    }
}

# Package groups
$coreScientific = @(
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "seaborn"
)

$machineLearning = @(
    "scikit-learn",
    "xgboost",
    "lightgbm"
)

$deepLearningTensorflow = @(
    "tensorflow",
    "tensorboard",
    "keras"
)

$pytorchCpu = @(
    "torch",
    "torchvision",
    "torchaudio"
)

$notebookTooling = @(
    "notebook",
    "ipykernel",
    "ipywidgets",
    "jupyterlab"
)

$generativeAi = @(
    "transformers",
    "diffusers",
    "accelerate",
    "datasets",
    "tokenizers",
    "huggingface-hub",
    "openai",
    "langchain",
    "langchain-community",
    "sentence-transformers",
    "faiss-cpu",
    "chromadb"
)

$utilities = @(
    "python-dotenv",
    "tqdm",
    "pillow",
    "requests",
    "httpx",
    "pydantic"
)

$docsSite = @(
    "mkdocs-material",
    "pymdown-extensions",
    "mkdocs-jupyter"
)

# Notebook extras — dependencies pulled in by per-notes setup scripts
#   notes/AIInfrastructure : mlflow
#   notes/MultiAgentAI     : tiktoken, mcp, fastapi, uvicorn, anyio, redis,
#                            langgraph, langchain-core, langchain-openai,
#                            autogen-agentchat, semantic-kernel, ollama
$notebookExtras = @(
    "mlflow",
    "tiktoken",
    "mcp",
    "fastapi",
    "uvicorn[standard]",
    "anyio",
    "redis",
    "langgraph",
    "langchain-core",
    "langchain-openai",
    "autogen-agentchat",
    "semantic-kernel",
    "ollama"
)

$codeIntelligence = @(
    "code-context-engine"   # local MCP server — AST-aware codebase indexing + cross-session memory
)

$requiredPackageKeys = @(
    $coreScientific +
    $machineLearning +
    $deepLearningTensorflow +
    $pytorchCpu +
    $notebookTooling +
    $generativeAi +
    $utilities +
    $docsSite +
    $notebookExtras +
    $codeIntelligence
) | ForEach-Object { Normalize-PackageKey $_ } | Sort-Object -Unique

$missingPackageCount = ($requiredPackageKeys | Where-Object { -not $script:InstalledPkgSet.Contains($_) }).Count

if ($missingPackageCount -eq 0) {
    Write-Ok "All Python package dependencies already satisfied — skipping package installation step"
} else {
    # Core scientific stack
    Install-Group "Core scientific stack" $coreScientific

    # Machine learning
    Install-Group "Machine learning" $machineLearning

    # Deep learning — TensorFlow
    Install-Group "Deep learning / TensorFlow" $deepLearningTensorflow

    # PyTorch — CPU-safe build (no CUDA required)
    Install-Group "PyTorch (CPU build)" $pytorchCpu "--index-url https://download.pytorch.org/whl/cpu"

    # Notebook tooling
    Install-Group "Notebook tooling" $notebookTooling

    # Generative AI / LLM utilities
    Install-Group "Generative AI / LLM utilities" $generativeAi

    # General utilities
    Install-Group "Utilities" $utilities

    # Docs / study site (MkDocs Material — browse notes/ in a web browser)
    # mkdocs-jupyter renders every notebook.ipynb as a page alongside the .md files.
    Install-Group "Docs site (MkDocs Material)" $docsSite

    # Notebook extras
    Install-Group "Notebook extras (AIInfrastructure + MultiAgentAI)" $notebookExtras

    # Code intelligence tooling
    Install-Group "Code intelligence (CCE)" $codeIntelligence

    # AI Infrastructure (ML Experiment Tracking) dependencies
    Write-Host ""
    Write-Host "📦 Installing AI Infrastructure dependencies..." -ForegroundColor Cyan
    Install-Group "AI Infrastructure (ML Experiment Tracking)" @(
        "mlflow",
        "evidently",
        "dvc",
        "tensorboard",
        "wandb"
    )
    Write-Host "✅ AI Infrastructure setup complete" -ForegroundColor Green
    Write-Host "   Verify: mlflow --version && dvc --version" -ForegroundColor Cyan

    # DevOps Fundamentals dependencies
    Write-Host ""
    Write-Host "📦 Installing DevOps Fundamentals dependencies..." -ForegroundColor Cyan

    # Docker Desktop (manual installation required)
    Write-Host ""
    Write-Host "⚠️  Docker Desktop - Manual setup required:" -ForegroundColor Yellow
    Write-Host "   1. Download and install Docker Desktop from: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
    Write-Host "   2. After installation, verify with: docker --version" -ForegroundColor Yellow
    Write-Host "   3. Ensure Docker Desktop is running before using DevOps notebooks" -ForegroundColor Yellow

    # Check if Chocolatey is available
    $chocoAvailable = Get-Command choco -ErrorAction SilentlyContinue
    if (-not $chocoAvailable) {
        Write-Host ""
        Write-Host "⚠️  Chocolatey not found - attempting install..." -ForegroundColor Yellow
        try {
            Set-ExecutionPolicy Bypass -Scope Process -Force
            [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
            Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
            # Refresh PATH to pick up choco
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                        [System.Environment]::GetEnvironmentVariable("Path", "User")
            Write-Ok "Chocolatey installed successfully"
            $chocoAvailable = Get-Command choco -ErrorAction SilentlyContinue
        } catch {
            Write-Warn "Could not install Chocolatey automatically. Install manually from https://chocolatey.org/install"
        }
    }

    if ($chocoAvailable) {
        Write-Host ""
        Write-Host "Installing DevOps tools via Chocolatey..." -ForegroundColor Cyan

        # Kind
        Write-Host "  • Installing Kind (Kubernetes in Docker)..." -ForegroundColor Cyan
        try {
            choco install kind -y --no-progress 2>&1 | Out-Null
            Write-Ok "Kind installed"
        } catch {
            Write-Warn "Failed to install Kind via Chocolatey"
        }

        # kubectl
        Write-Host "  • Installing kubectl..." -ForegroundColor Cyan
        try {
            choco install kubernetes-cli -y --no-progress 2>&1 | Out-Null
            Write-Ok "kubectl installed"
        } catch {
            Write-Warn "Failed to install kubectl via Chocolatey"
        }

        # Terraform
        Write-Host "  • Installing Terraform..." -ForegroundColor Cyan
        try {
            choco install terraform -y --no-progress 2>&1 | Out-Null
            Write-Ok "Terraform installed"
        } catch {
            Write-Warn "Failed to install Terraform via Chocolatey"
        }

        # K9s (optional)
        Write-Host "  • Installing K9s (optional Kubernetes TUI)..." -ForegroundColor Cyan
        try {
            choco install k9s -y --no-progress 2>&1 | Out-Null
            Write-Ok "K9s installed"
        } catch {
            Write-Warn "Failed to install K9s via Chocolatey (optional tool)"
        }

        # Refresh PATH for this session
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                    [System.Environment]::GetEnvironmentVariable("Path", "User")
    } else {
        Write-Host ""
        Write-Host "⚠️  Chocolatey not available - skipping automated tool installation" -ForegroundColor Yellow
        Write-Host "   Install tools manually:" -ForegroundColor Yellow
        Write-Host "   • Kind: https://kind.sigs.k8s.io/docs/user/quick-start/#installation" -ForegroundColor Yellow
        Write-Host "   • kubectl: https://kubernetes.io/docs/tasks/tools/install-kubectl-windows/" -ForegroundColor Yellow
        Write-Host "   • Terraform: https://developer.hashicorp.com/terraform/install" -ForegroundColor Yellow
        Write-Host "   • K9s: https://k9scli.io/topics/install/" -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Host "✅ DevOps Fundamentals setup complete" -ForegroundColor Green
    Write-Host "   Verify installations:" -ForegroundColor Cyan
    Write-Host "   • docker --version" -ForegroundColor Cyan
    Write-Host "   • kind --version" -ForegroundColor Cyan
    Write-Host "   • kubectl version --client" -ForegroundColor Cyan
    Write-Host "   • terraform --version" -ForegroundColor Cyan
    Write-Host "   • k9s version" -ForegroundColor Cyan
}

if ($EnableGpuNotebookStack) {
    Write-Step "Installing GPU notebook stack"
    Write-Group "GPU notebook packages"

    # Replace the default CPU-only torch stack with a CUDA-enabled wheel.
    & python -m pip install --upgrade --force-reinstall `
        torch torchvision torchaudio `
        --index-url https://download.pytorch.org/whl/cu121 --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "Failed to install CUDA-enabled PyTorch wheels"
    }
    Write-Ok "PyTorch CUDA wheel installed"

    Install-Group "Fine-tuning / GPU notebook deps" @(
        "accelerate",
        "datasets",
        "peft",
        "trl"
    )
}
# ─── 1d'. Dependency health check / compatibility repairs ───────────────────

Write-Step "Checking dependency health (pip check)"

$pipCheckOutput = & python -m pip check 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Ok "pip check passed"
} else {
    Write-Warn "pip check reported conflicts — applying compatibility pin(s)"
    # TensorFlow 2.21 requires protobuf >=6.31.1,<8.0.0.
    & python -m pip install "protobuf>=6.31.1,<8.0.0" --quiet --no-cache-dir

    $pipCheckOutput = & python -m pip check 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Ok "Dependency conflicts resolved"
    } else {
        Write-Warn "Dependency conflicts remain after automatic repair:"
        $pipCheckOutput | ForEach-Object { Write-Warn "$_" }
        Write-Warn "Setup will continue, but you may need to pin package versions manually."
    }
}

# ─── 1f. Code Context Engine (CCE) init ──────────────────────────────────────
#
# cce init:
#   • downloads the embedding model (one-time, ~67 MB)
#   • creates .mcp.json + .vscode/mcp.json  (MCP server registration)
#   • installs 3 git hooks (auto-reindex on commit)
#   • builds the local vector + BM25 index for context_search
# Re-indexing is NOT needed periodically — git hooks keep the index current.
# To force a full re-index manually: cce index

Write-Step "Initialising Code Context Engine (cce init)"

$McpJsonPath = Join-Path $RepoRoot ".mcp.json"
if (Test-Path $McpJsonPath) {
    Write-Ok "CCE already initialised (.mcp.json exists) — skipping"
} else {
    $CcePath = Join-Path $VenvScriptsPath "cce.exe"
    if (Test-Path $CcePath) {
        Write-Warn "Running cce init (indexes codebase — may take a few minutes on first run) ..."
        Push-Location $RepoRoot
        try {
            & $CcePath init
            if ($LASTEXITCODE -eq 0) {
                Write-Ok "cce init complete — reload VS Code to activate the MCP server"
            } else {
                Write-Warn "cce init exited with code $LASTEXITCODE — CCE may not be fully set up"
            }
        } finally {
            Pop-Location
        }
    } else {
        Write-Warn "cce binary not found at $CcePath — skipping CCE init"
    }
}

# ─── 1e. Register Jupyter kernels ─────────────────────────────────────────────

Write-Step "Registering Jupyter kernels"

$kernelTargets = @(
    @{ Name = "ai-ml-dev";         Display = "AI/ML Dev (venv)" },
    @{ Name = "ml-notes";          Display = "ML Notes (venv)" },
    @{ Name = "ai-infrastructure"; Display = "Python (AI Infrastructure)" },
    @{ Name = "multi-agent-ai";    Display = "Multi-Agent AI" }
)

$existingKernelNames = @{}
try {
    $kernelJson = & python -m jupyter kernelspec list --json 2>$null
    if ($kernelJson) {
        $parsed = $kernelJson | ConvertFrom-Json
        if ($parsed.kernelspecs) {
            foreach ($prop in $parsed.kernelspecs.PSObject.Properties.Name) {
                $existingKernelNames[$prop] = $true
            }
        }
    }
} catch {
    Write-Warn "Could not enumerate existing kernels; kernel checks will run individually"
}

$allKernelsPresent = $true
foreach ($k in $kernelTargets) {
    if (-not $existingKernelNames.ContainsKey($k.Name)) { $allKernelsPresent = $false; break }
}

if ($allKernelsPresent) {
    Write-Ok "All required Jupyter kernels already registered — skipping kernel registration"
} else {
    foreach ($k in $kernelTargets) {
        if ($existingKernelNames.ContainsKey($k.Name)) {
            Write-Ok "Kernel '$($k.Name)' already registered"
            continue
        }
        & python -m ipykernel install --user --name $k.Name --display-name $k.Display 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "Failed to register kernel '$($k.Name)' — continuing"
        } else {
            Write-Ok "Kernel '$($k.Name)' registered"
        }
    }
}

Write-Step "Setting default kernel on every notebook under notes/"
& python (Join-Path $PSScriptRoot "set_default_kernel.py")
if ($LASTEXITCODE -ne 0) { Write-Warn "set_default_kernel.py exited with code $LASTEXITCODE" }

Write-Step "Setting notebook permissions (read-only for solutions, writable for exercises)"
try {
    # Read-only for solution notebooks
    $solutionCount = 0
    Get-ChildItem -Path (Join-Path $RepoRoot "notes") -Filter "*_solution.ipynb" -Recurse | ForEach-Object {
        Set-ItemProperty -Path $_.FullName -Name IsReadOnly -Value $true -ErrorAction SilentlyContinue
        $solutionCount++
    }

    # Writable for exercise notebooks (ensure not read-only)
    $exerciseCount = 0
    Get-ChildItem -Path (Join-Path $RepoRoot "notes") -Filter "*_exercise.ipynb" -Recurse | ForEach-Object {
        Set-ItemProperty -Path $_.FullName -Name IsReadOnly -Value $false -ErrorAction SilentlyContinue
        $exerciseCount++
    }

    Write-Ok "Permissions set: $solutionCount solution notebooks (read-only), $exerciseCount exercise notebooks (writable)"
} catch {
    Write-Warn "Failed to set some notebook permissions: $_"
    Write-Warn "Notebooks will still work, but file permissions may not be ideal"
}

# ─── STEP 2: Visual Studio Code ─────────────────────────────────────────────

Write-Host ""
Write-Host "══════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host "  AI/ML Dev Environment Setup — Step 2/7" -ForegroundColor White
Write-Host "  Visual Studio Code" -ForegroundColor White
Write-Host "══════════════════════════════════════════════" -ForegroundColor DarkGray

Write-Step "Checking for Visual Studio Code"

$CodeCmd = $null
foreach ($candidate in @("code", "code-insiders")) {
    try {
        $codeVer = & $candidate --version 2>&1 | Select-Object -First 1
        if ($codeVer -match '\d+\.\d+') {
            $CodeCmd = $candidate
            Write-Ok "VS Code $codeVer already installed ($candidate)"
            break
        }
    } catch { }
}

if (-not $CodeCmd) {
    Write-Warn "VS Code not found — installing via winget ..."
    try {
        winget install --id Microsoft.VisualStudioCode `
            --source winget `
            --silent `
            --accept-package-agreements `
            --accept-source-agreements

        # Refresh PATH so 'code' is immediately available
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                    [System.Environment]::GetEnvironmentVariable("Path", "User")

        # Confirm
        $codeVer = & code --version 2>&1 | Select-Object -First 1
        if ($codeVer -match '\d+\.\d+') {
            $CodeCmd = "code"
            Write-Ok "VS Code $codeVer installed successfully"
        } else {
            Write-Warn "winget completed but 'code' is not yet on PATH."
            Write-Warn "Restart your terminal after this script finishes, then re-run for remaining steps."
            $CodeCmd = "code"   # optimistically continue; extensions added in Step 3
        }
    } catch {
        Write-Fail "Could not install VS Code automatically. Download from https://code.visualstudio.com/ and re-run."
    }
} else {
    Write-Ok "Skipping install — VS Code already present"
}

# ─── STEP 2.5: Core VS Code Extensions (Python/Jupyter Stack) ───────────────

Write-Host ""
Write-Host "══════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host "  AI/ML Dev Environment Setup — Step 2.5/7" -ForegroundColor White
Write-Host "  Core Python/Jupyter Extensions" -ForegroundColor White
Write-Host "══════════════════════════════════════════════" -ForegroundColor DarkGray

$CoreExtensions = @(
    "ms-python.python",              # Python language support
    "ms-python.vscode-pylance",      # Fast Python language server
    "ms-toolsai.jupyter",            # Jupyter notebook support
    "ms-python.black-formatter",     # Black code formatter
    "ms-python.isort",               # Import organization
    "yzhang.markdown-all-in-one",    # Markdown editing
    "esbenp.prettier-vscode"         # Prettier formatter (Markdown/JSON)
)

Write-Step "Installing core extensions for Python/Jupyter development"

$installedCount = 0
$skippedCount = 0
$failedCount = 0

foreach ($extId in $CoreExtensions) {
    $extName = $extId -replace '.*\.', ''  # Extract short name

    # Check if already installed
    $alreadyInstalled = $false
    try {
        $extList = & $CodeCmd --list-extensions 2>&1
        if ($extList -match [regex]::Escape($extId)) {
            Write-Host "  ✓ $extName (already installed)" -ForegroundColor DarkGray
            $skippedCount++
            $alreadyInstalled = $true
        }
    } catch {
        Write-Warn "Could not query extensions — will attempt install"
    }

    if (-not $alreadyInstalled) {
        try {
            & $CodeCmd --install-extension $extId --force 2>&1 | Out-Null
            # Verify installation
            $extList = & $CodeCmd --list-extensions 2>&1
            if ($extList -match [regex]::Escape($extId)) {
                Write-Ok "$extName installed"
                $installedCount++
            } else {
                Write-Warn "$extName install command ran but extension not yet detected"
                $installedCount++  # Count as success; may appear after restart
            }
        } catch {
            Write-Warn "Failed to install $extName ($extId)"
            $failedCount++
        }
    }
}

Write-Host ""
Write-Host "  Extension Summary:" -ForegroundColor White
Write-Host "    • Installed: $installedCount" -ForegroundColor Green
Write-Host "    • Already present: $skippedCount" -ForegroundColor DarkGray
if ($failedCount -gt 0) {
    Write-Host "    • Failed: $failedCount (install manually via Extensions panel)" -ForegroundColor Yellow
}
Write-Host ""

# ─── STEP 3: Kilo Code (Agentic AI) Extension ───────────────────────────────
#
# Kilo Code is an open-source agentic coding assistant (fork of Roo/Cline) that
# can plan, edit files, and run commands. We point it at a locally-hosted
# DeepSeek-R1 reasoning SLM via Ollama (configured in Step 6d).

if ($EnableSlmAssistant) {

Write-Host ""
Write-Host "══════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host "  AI/ML Dev Environment Setup — Step 3/7" -ForegroundColor White
Write-Host "  Kilo Code — Agentic AI Extension" -ForegroundColor White
Write-Host "══════════════════════════════════════════════" -ForegroundColor DarkGray

$KiloExtId = "kilocode.kilo-code"

Write-Step "Checking Kilo Code extension ($KiloExtId)"

$extensionInstalled = $false
try {
    $extList = & $CodeCmd --list-extensions 2>&1
    if ($extList -match [regex]::Escape($KiloExtId)) {
        Write-Ok "Kilo Code already installed"
        $extensionInstalled = $true
    }
} catch {
    Write-Warn "Could not query VS Code extensions — will attempt install anyway"
}

if (-not $extensionInstalled) {
    Write-Warn "Kilo Code not found — installing ..."
    try {
        & $CodeCmd --install-extension $KiloExtId --force 2>&1 | Out-Null
        # Verify
        $extList = & $CodeCmd --list-extensions 2>&1
        if ($extList -match [regex]::Escape($KiloExtId)) {
            Write-Ok "Kilo Code installed successfully"
        } else {
            Write-Warn "Install command ran but extension not detected yet — it may appear after VS Code restarts"
        }
    } catch {
        Write-Warn "Could not install Kilo Code automatically."
        Write-Warn "Install manually: open VS Code → Extensions → search 'Kilo Code' → Install"
    }
}

Write-Step "Kilo Code post-install configuration note"
Write-Host ""
Write-Host "  After launching VS Code:" -ForegroundColor White
Write-Host "    1. Open the Kilo Code sidebar (kangaroo icon on the Activity Bar)" -ForegroundColor DarkGray
Write-Host "    2. Click 'Settings' → API Provider: Ollama" -ForegroundColor DarkGray
Write-Host "    3. Base URL: http://localhost:11434   (the auto-discover button works too)" -ForegroundColor DarkGray
Write-Host "    4. Model: deepseek-r1:8b  (or deepseek-r1:1.5b on low-RAM machines)" -ForegroundColor DarkGray
Write-Host "    5. Save — Kilo Code will now drive agentic edits with DeepSeek-R1 reasoning" -ForegroundColor DarkGray
Write-Host ""

# ─── STEP 4: Ollama Server Install & First Launch ────────────────────────────

Write-Host ""
Write-Host "══════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host "  AI/ML Dev Environment Setup — Step 4/7" -ForegroundColor White
Write-Host "  Ollama Local Inference Server" -ForegroundColor White
Write-Host "══════════════════════════════════════════════" -ForegroundColor DarkGray

$OllamaPort = 11434
$OllamaBaseUrl = "http://localhost:$OllamaPort"

# ── 4a. Install Ollama ────────────────────────────────────────────────────────

Write-Step "Checking Ollama binary"

$OllamaInstalled = $false
try {
    $ollamaVer = & ollama --version 2>&1
    if ($ollamaVer -match 'ollama') {
        Write-Ok "Ollama already installed: $ollamaVer"
        $OllamaInstalled = $true
    }
} catch { }

if (-not $OllamaInstalled) {
    Write-Warn "Ollama not found — installing via winget ..."
    try {
        winget install --id Ollama.Ollama `
            --source winget `
            --silent `
            --accept-package-agreements `
            --accept-source-agreements

        # Refresh PATH
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                    [System.Environment]::GetEnvironmentVariable("Path", "User")

        $ollamaVer = & ollama --version 2>&1
        Write-Ok "Ollama installed: $ollamaVer"
    } catch {
        Write-Fail "Could not install Ollama automatically. Download from https://ollama.com/download and re-run."
    }
}

# ── 4b. Start the Ollama server ───────────────────────────────────────────────

Write-Step "Starting Ollama server"

# Check if it is already listening
$serverRunning = $false
try {
    $resp = Invoke-WebRequest -Uri $OllamaBaseUrl -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
    if ($resp.StatusCode -eq 200) {
        Write-Ok "Ollama server already running at $OllamaBaseUrl"
        $serverRunning = $true
    }
} catch { }

if (-not $serverRunning) {
    Write-Warn "Ollama server not running — starting in background ..."

    # Pin Ollama to a single loaded model with no parallelism, so the 8B
    # reasoning SLM owns the GPU/RAM exclusively while Kilo Code is working.
    $env:OLLAMA_MAX_LOADED_MODELS = "1"
    $env:OLLAMA_NUM_PARALLEL      = "1"
    $env:OLLAMA_CONTEXT_LENGTH    = "4096"
    [System.Environment]::SetEnvironmentVariable("OLLAMA_MAX_LOADED_MODELS", "1", "User")
    [System.Environment]::SetEnvironmentVariable("OLLAMA_NUM_PARALLEL",      "1", "User")
    [System.Environment]::SetEnvironmentVariable("OLLAMA_CONTEXT_LENGTH",    "4096", "User")

    # Launch as a hidden background job so this script keeps running
    $OllamaJob = Start-Job -ScriptBlock {
        $env:OLLAMA_MAX_LOADED_MODELS = "1"
        $env:OLLAMA_NUM_PARALLEL      = "1"
        $env:OLLAMA_CONTEXT_LENGTH    = "4096"
        ollama serve
    } -Name "OllamaServe"
    Write-Ok "Ollama server started (background job id: $($OllamaJob.Id))"
    Write-Ok "Single-model mode: OLLAMA_MAX_LOADED_MODELS=1, OLLAMA_NUM_PARALLEL=1, OLLAMA_CONTEXT_LENGTH=4096"

    # Save PID file so the lifecycle step (Step 5) can stop it
    $pidFile = Join-Path $RepoRoot ".ollama.pid"
    # The job spawns a child process; wait briefly then capture the PID
    Start-Sleep -Seconds 3
    $ollamaProc = Get-Process -Name "ollama" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($ollamaProc) {
        $ollamaProc.Id | Set-Content $pidFile
        Write-Ok "Ollama PID $($ollamaProc.Id) saved to .ollama.pid"
    } else {
        Write-Warn "Could not locate ollama process to save PID — lifecycle management in Step 5 may need manual setup"
    }

    # Health-check with retries
    $maxRetries = 10
    $retries = 0
    $healthy = $false
    while ($retries -lt $maxRetries -and -not $healthy) {
        Start-Sleep -Seconds 1
        try {
            $resp = Invoke-WebRequest -Uri $OllamaBaseUrl -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
            if ($resp.StatusCode -eq 200) { $healthy = $true }
        } catch { }
        $retries++
    }

    if ($healthy) {
        Write-Ok "Ollama server is healthy at $OllamaBaseUrl"
    } else {
        Write-Warn "Ollama server did not respond within ${maxRetries}s — it may still be starting up"
        Write-Warn "Run 'ollama serve' manually and check http://localhost:$OllamaPort"
    }
}

# ─── STEP 5: Ollama Lifecycle Wiring ──────────────────────────────────────────
#
# Strategy: write .vscode/tasks.json with a folderOpen task that starts
# ollama serve, and a companion stop task.  VS Code has no native onClose
# hook, so we also write a small wrapper script that monitors the 'code'
# process and stops Ollama when it exits.

Write-Host ""
Write-Host "══════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host "  AI/ML Dev Environment Setup — Step 5/7" -ForegroundColor White
Write-Host "  Ollama Lifecycle Wiring" -ForegroundColor White
Write-Host "══════════════════════════════════════════════" -ForegroundColor DarkGray

$VscodeDirPath  = Join-Path $RepoRoot ".vscode"
$TasksJsonPath  = Join-Path $VscodeDirPath "tasks.json"
$WatcherScript  = Join-Path $RepoRoot "scripts\ollama-watcher.ps1"

# ── 5a. Write .vscode/tasks.json ──────────────────────────────────────────────

Write-Step "Configuring .vscode/tasks.json"

if (-not (Test-Path $VscodeDirPath)) {
    New-Item -ItemType Directory -Path $VscodeDirPath | Out-Null
}

# Only write if tasks.json doesn't already contain our tasks
$writeTasksJson = $true
if (Test-Path $TasksJsonPath) {
    $existing = Get-Content $TasksJsonPath -Raw
    if (($existing -match 'ollama-start') -and ($existing -match 'cce-watcher')) {
        Write-Ok "tasks.json already contains ollama + cce-watcher tasks — skipping"
        $writeTasksJson = $false
    } else {
        Write-Warn "tasks.json exists but is missing tasks — backing up and rewriting"
        Copy-Item $TasksJsonPath "$TasksJsonPath.bak" -Force
    }
}

if ($writeTasksJson) {
    $tasksJson = @'
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "ollama-start",
            "type": "shell",
            "command": "pwsh",
            "args": [
                "-NonInteractive",
                "-File",
                "${workspaceFolder}/scripts/ollama-watcher.ps1"
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
            "command": "pwsh",
            "args": [
                "-NonInteractive",
                "-Command",
                "$p = Get-Content '${workspaceFolder}/.ollama.pid' -ErrorAction SilentlyContinue; if ($p) { Stop-Process -Id ([int]$p) -Force -ErrorAction SilentlyContinue }"
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
            "windows": {
                "command": "pwsh",
                "args": [
                    "-NonInteractive",
                    "-ExecutionPolicy", "Bypass",
                    "-File", "${workspaceFolder}/scripts/cce-watcher.ps1"
                ]
            },
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
'@
    Set-Content -Path $TasksJsonPath -Value $tasksJson -Encoding UTF8
    Write-Ok "Written: .vscode/tasks.json"
    Write-Warn "ACTION REQUIRED: open VS Code → Terminal → Run Task → 'Allow Automatic Tasks' to enable folderOpen trigger"
}

# ── 5b. Write the watcher script ──────────────────────────────────────────────

Write-Step "Writing ollama-watcher.ps1"

# Always rewrite: the watcher content evolves (env vars, model pins, ...) and
# we want re-running setup.ps1 to keep it in sync.
if (Test-Path $WatcherScript) {
    Write-Ok "Overwriting existing ollama-watcher.ps1"
}

$watcherContent = @'
# ollama-watcher.ps1
# Launched automatically when this VS Code workspace opens (via tasks.json folderOpen).
# - Starts ollama serve if not already running
# - Monitors the VS Code process
# - Stops ollama serve when VS Code exits

$RepoRoot  = Split-Path -Parent $PSScriptRoot
$PidFile   = Join-Path $RepoRoot ".ollama.pid"
$OllamaUrl = "http://localhost:11434"

function Is-OllamaRunning {
    try { $null = Invoke-WebRequest $OllamaUrl -UseBasicParsing -TimeoutSec 2 -EA Stop; return $true } catch { return $false }
}

# Start ollama if not already up
if (-not (Is-OllamaRunning)) {
    # Pin to a single loaded model with no parallelism and a 4096-token ctx.
    $env:OLLAMA_MAX_LOADED_MODELS = "1"
    $env:OLLAMA_NUM_PARALLEL      = "1"
    $env:OLLAMA_CONTEXT_LENGTH    = "4096"
    $job = Start-Job -ScriptBlock {
        $env:OLLAMA_MAX_LOADED_MODELS = "1"
        $env:OLLAMA_NUM_PARALLEL      = "1"
        $env:OLLAMA_CONTEXT_LENGTH    = "4096"
        ollama serve
    }
    Start-Sleep -Seconds 3
    $proc = Get-Process -Name "ollama" -EA SilentlyContinue | Select-Object -First 1
    if ($proc) { $proc.Id | Set-Content $PidFile }
}

# Wait until all VS Code windows are closed
while ($true) {
    Start-Sleep -Seconds 5
    $codeRunning = Get-Process -Name "code" -ErrorAction SilentlyContinue
    if (-not $codeRunning) {
        # VS Code has exited — stop Ollama
        $savedPid = Get-Content $PidFile -ErrorAction SilentlyContinue
        if ($savedPid) {
            Stop-Process -Id ([int]$savedPid) -Force -ErrorAction SilentlyContinue
        }
        # Belt-and-suspenders: kill any remaining ollama processes
        Get-Process -Name "ollama" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
        Remove-Item $PidFile -ErrorAction SilentlyContinue
        break
    }
}
'@
Set-Content -Path $WatcherScript -Value $watcherContent -Encoding UTF8
Write-Ok "Written: scripts/ollama-watcher.ps1"

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

Write-Host ""
Write-Host "══════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host "  AI/ML Dev Environment Setup — Step 6/7" -ForegroundColor White
Write-Host "  Pull DeepSeek-R1 Reasoning SLM" -ForegroundColor White
Write-Host "══════════════════════════════════════════════" -ForegroundColor DarkGray

# ── 6a. Detect system RAM ────────────────────────────────────────────────────

Write-Step "Detecting system RAM"

$TotalRamGB = 0
try {
    $cs = Get-CimInstance -ClassName Win32_ComputerSystem
    $TotalRamGB = [math]::Round($cs.TotalPhysicalMemory / 1GB, 1)
    Write-Ok "Total RAM: ${TotalRamGB} GB"
} catch {
    Write-Warn "Could not detect RAM — defaulting to conservative model"
    $TotalRamGB = 0
}

# Choose model based on available RAM
$PrimaryBase   = "deepseek-r1:8b-llama-distill-q4_K_M"
$FallbackBase  = "deepseek-r1:1.5b-qwen-distill-q4_0"
$BaseModel     = if ($TotalRamGB -ge 10) { $PrimaryBase } else { $FallbackBase }
$CtxTokens     = 4096

# Derived tag: same base with '-ctx4k' suffix. This is what Kilo Code targets.
$ChosenModel   = "$($BaseModel.Split(':')[0]):$($BaseModel.Split(':')[1])-ctx4k"

if ($TotalRamGB -ge 10) {
    Write-Ok "RAM ≥ 10 GB — selecting primary base: $BaseModel"
} else {
    Write-Warn "RAM < 10 GB — selecting fallback base: $BaseModel"
}
Write-Ok "Derived model (num_ctx=$CtxTokens): $ChosenModel"

# ── 6b. Check if base model already pulled ───────────────────────────────────

Write-Step "Checking if $BaseModel is already available"

$basePresent = $false
try {
    $modelList = & ollama list 2>&1
    if ($modelList -match [regex]::Escape($BaseModel)) {
        Write-Ok "$BaseModel already present in Ollama"
        $basePresent = $true
    }
} catch {
    Write-Warn "Could not query ollama list — will attempt pull"
}

# ── 6c. Pull the base model ──────────────────────────────────────────────────

if (-not $basePresent) {
    Write-Step "Pulling $BaseModel (this may take a few minutes on first run)"
    Write-Host "  Downloading model — progress shown below:" -ForegroundColor DarkGray
    Write-Host ""
    & ollama pull $BaseModel
    if ($LASTEXITCODE -eq 0) {
        Write-Ok "$BaseModel pulled successfully"
    } else {
        Write-Warn "Pull exited with code $LASTEXITCODE — check your internet connection and retry: ollama pull $BaseModel"
    }
} else {
    Write-Step "Skipping pull — $BaseModel already present"
}

# ── 6c'. Create the derived -ctx4k model with num_ctx=4096 baked in ──────────

Write-Step "Creating derived model $ChosenModel with num_ctx=$CtxTokens"

$derivedPresent = $false
try {
    $modelList = & ollama list 2>&1
    if ($modelList -match [regex]::Escape($ChosenModel)) {
        Write-Ok "$ChosenModel already present — skipping create"
        $derivedPresent = $true
    }
} catch { }

if (-not $derivedPresent) {
    $ModelfilePath = Join-Path $RepoRoot ".ollama.Modelfile"
    @"
FROM $BaseModel
PARAMETER num_ctx $CtxTokens
"@ | Set-Content -Path $ModelfilePath -Encoding ASCII
    & ollama create $ChosenModel -f $ModelfilePath
    if ($LASTEXITCODE -eq 0) {
        Write-Ok "Created $ChosenModel (num_ctx=$CtxTokens)"
    } else {
        Write-Warn "ollama create exited with code $LASTEXITCODE — Kilo Code will have to pass num_ctx itself"
        $ChosenModel = $BaseModel  # fall back to base
    }
    Remove-Item $ModelfilePath -ErrorAction SilentlyContinue
}

# ── 6d. Configure Kilo Code to use the DeepSeek-R1 model ─────────────────────
#
# The current Kilo Code extension (built on the Kilo CLI) reads its config from
#   ~/.config/kilo/kilo.jsonc        (global, used by both VS Code & CLI)
# and from a project-level kilo.jsonc / .kilo/kilo.jsonc.
#
# We write BOTH:
#   • the global file → makes our local Ollama model the default everywhere
#   • a project-level .kilo/kilo.jsonc → checked into the repo so anyone who
#     opens this workspace gets the same default
#
# Schema reference: https://app.kilo.ai/config.json
# Docs: https://kilo.ai/docs/code-with-ai/agents/custom-models

Write-Step "Writing Kilo Code config (global + project) so DeepSeek-R1 is the default model"

$KiloModelKey = $ChosenModel  # e.g. deepseek-r1:8b-llama-distill-q4_K_M-ctx4k
$KiloModelRef = "ollama/$KiloModelKey"

$kiloConfig = [ordered]@{
    '$schema' = "https://app.kilo.ai/config.json"
    model     = $KiloModelRef
    provider  = [ordered]@{
        ollama = [ordered]@{
            options = [ordered]@{
                baseURL = "http://localhost:11434/v1"
                timeout = 600000
            }
            models = [ordered]@{
                $KiloModelKey = [ordered]@{
                    name      = "DeepSeek-R1 (local Ollama, 4k ctx)"
                    tool_call = $true
                    reasoning = $true
                    limit     = [ordered]@{
                        context = $CtxTokens
                        output  = $CtxTokens
                    }
                }
            }
        }
    }
}

$kiloConfigJson = $kiloConfig | ConvertTo-Json -Depth 10

# 1) Global config ── ~/.config/kilo/kilo.jsonc
$KiloGlobalDir  = Join-Path $env:USERPROFILE ".config\kilo"
$KiloGlobalPath = Join-Path $KiloGlobalDir "kilo.jsonc"
if (-not (Test-Path $KiloGlobalDir)) {
    New-Item -ItemType Directory -Path $KiloGlobalDir -Force | Out-Null
}
if (Test-Path $KiloGlobalPath) {
    Copy-Item $KiloGlobalPath "$KiloGlobalPath.bak" -Force
    Write-Warn "Existing global Kilo config backed up to $KiloGlobalPath.bak"
}
Set-Content -Path $KiloGlobalPath -Value $kiloConfigJson -Encoding UTF8
Write-Ok "Global Kilo config written: $KiloGlobalPath"

# 2) Project config ── <repo>/.kilo/kilo.jsonc (overrides global for this workspace)
$KiloProjectDir  = Join-Path $RepoRoot ".kilo"
$KiloProjectPath = Join-Path $KiloProjectDir "kilo.jsonc"
if (-not (Test-Path $KiloProjectDir)) {
    New-Item -ItemType Directory -Path $KiloProjectDir -Force | Out-Null
}
Set-Content -Path $KiloProjectPath -Value $kiloConfigJson -Encoding UTF8
Write-Ok "Project Kilo config written: .kilo/kilo.jsonc"

# 3) Auto-launch the Kilo Code sidebar when this workspace opens.
# Add a folderOpen command-task that focuses the Kilo Code view container.
# View ID: kilo-code.SidebarProvider (per the extension's package.json).
Write-Step "Wiring Kilo Code sidebar to auto-open with this workspace"

$KiloLaunchTask = [ordered]@{
    label   = "kilo-code-launch"
    type    = "shell"
    command = "pwsh"
    args    = @(
        "-NonInteractive",
        "-Command",
        "& '$CodeCmd' --command kilo-code.SidebarProvider.focus 2>$null; exit 0"
    )
    runOptions   = [ordered]@{ runOn = "folderOpen" }
    presentation = [ordered]@{
        reveal           = "never"
        panel            = "dedicated"
        showReuseMessage = $false
    }
    problemMatcher = @()
}

# Merge into existing tasks.json (already created in Step 5a)
if (Test-Path $TasksJsonPath) {
    try {
        if ($PSVersionTable.PSVersion.Major -ge 6) {
            $tasksObj = Get-Content $TasksJsonPath -Raw | ConvertFrom-Json -AsHashtable
        } else {
            $tj = Get-Content $TasksJsonPath -Raw | ConvertFrom-Json
            $tasksObj = @{}
            $tj.PSObject.Properties | ForEach-Object { $tasksObj[$_.Name] = $_.Value }
        }
        if (-not $tasksObj.ContainsKey("tasks")) { $tasksObj["tasks"] = @() }
        $hasKilo = $false
        foreach ($t in $tasksObj["tasks"]) {
            if ($t.label -eq "kilo-code-launch") { $hasKilo = $true; break }
        }
        if (-not $hasKilo) {
            $tasksObj["tasks"] = @($tasksObj["tasks"]) + $KiloLaunchTask
            $tasksObj | ConvertTo-Json -Depth 10 | Set-Content $TasksJsonPath -Encoding UTF8
            Write-Ok "Added 'kilo-code-launch' folderOpen task to .vscode/tasks.json"
        } else {
            Write-Ok "tasks.json already has 'kilo-code-launch' — skipping"
        }
    } catch {
        Write-Warn "Could not merge kilo-code-launch task into tasks.json: $($_.Exception.Message)"
    }
}

# 4) Make sure VS Code does NOT disable the Kilo extension on this workspace.
# Add it to recommended extensions and ensure no workspace-level disabled list.
$ExtensionsJsonPath = Join-Path $VscodeDirPath "extensions.json"
$extensionsObj = [ordered]@{ recommendations = @($KiloExtId) }
if (Test-Path $ExtensionsJsonPath) {
    try {
        if ($PSVersionTable.PSVersion.Major -ge 6) {
            $exObj = Get-Content $ExtensionsJsonPath -Raw | ConvertFrom-Json -AsHashtable
        } else {
            $ex = Get-Content $ExtensionsJsonPath -Raw | ConvertFrom-Json
            $exObj = @{}
            $ex.PSObject.Properties | ForEach-Object { $exObj[$_.Name] = $_.Value }
        }
        if (-not $exObj.ContainsKey("recommendations")) { $exObj["recommendations"] = @() }
        if ($exObj["recommendations"] -notcontains $KiloExtId) {
            $exObj["recommendations"] = @($exObj["recommendations"]) + $KiloExtId
        }
        $exObj | ConvertTo-Json -Depth 10 | Set-Content $ExtensionsJsonPath -Encoding UTF8
        Write-Ok "Kilo Code added to .vscode/extensions.json recommendations"
    } catch {
        Write-Warn "Could not update extensions.json: $($_.Exception.Message)"
    }
} else {
    $extensionsObj | ConvertTo-Json -Depth 10 | Set-Content $ExtensionsJsonPath -Encoding UTF8
    Write-Ok "Created .vscode/extensions.json with Kilo Code recommended"
}


# ─── VS Code notebook settings ───────────────────────────────────────────────
#
# Keep notebooks read-only in VS Code so Jupyter Lab stays the only editor for
# .ipynb files in this workspace.

} else {
    Write-Warn "Skipping the SLM assistant bundle. Re-run with --enable-slm-assistant to install Kilo Code, Ollama, and the local model wiring."
}

Write-Step "Writing .vscode/settings.json (notebooks read-only in VS Code)"

$SettingsJsonPath = Join-Path $VscodeDirPath "settings.json"
$readOnlyPatch = [ordered]@{
    "files.readonlyInclude" = [ordered]@{ "**/*.ipynb" = $true }
    "notebook.defaultKernel" = "ai-ml-dev"
}

if (Test-Path $SettingsJsonPath) {
    try {
        if ($PSVersionTable.PSVersion.Major -ge 6) {
            $existing = Get-Content $SettingsJsonPath -Raw | ConvertFrom-Json -AsHashtable
        } else {
            $jsonObj = Get-Content $SettingsJsonPath -Raw | ConvertFrom-Json
            $existing = @{}
            $jsonObj.PSObject.Properties | ForEach-Object { $existing[$_.Name] = $_.Value }
        }
    } catch {
        $existing = @{}
    }
    # Merge files.readonlyInclude map
    if (-not $existing.ContainsKey("files.readonlyInclude")) {
        $existing["files.readonlyInclude"] = @{}
    }
    $existing["files.readonlyInclude"]["**/*.ipynb"] = $true
    $existing["notebook.defaultKernel"] = "ai-ml-dev"
    $existing | ConvertTo-Json -Depth 10 | Set-Content $SettingsJsonPath -Encoding UTF8
    Write-Ok "Merged read-only rule into existing .vscode/settings.json"
} else {
    if (-not (Test-Path $VscodeDirPath)) {
        New-Item -ItemType Directory -Path $VscodeDirPath | Out-Null
    }
    $readOnlyPatch | ConvertTo-Json -Depth 10 | Set-Content $SettingsJsonPath -Encoding UTF8
    Write-Ok "Written: .vscode/settings.json"
}

# ─── STEP 7: Launch Study Servers (Jupyter Lab + MkDocs) ─────────────────────
#
# Fixed local ports so bookmarks stay stable:
#   • Jupyter Lab  → http://localhost:8888   (hands-on coding in notebooks)
#   • MkDocs site  → http://localhost:8000   (read notes/ in a web browser)
#
# Both run as detached background processes so this script can exit.
# PIDs are saved to .jupyter.pid / .mkdocs.pid for a later stop command.

Write-Host ""
Write-Host "══════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host "  AI/ML Dev Environment Setup — Step 7/7" -ForegroundColor White
Write-Host "  Launch Study Servers (Jupyter + MkDocs)" -ForegroundColor White
Write-Host "══════════════════════════════════════════════" -ForegroundColor DarkGray

$JupyterPort  = 8888
$MkdocsPort   = 8000
$JupyterPid   = Join-Path $RepoRoot ".jupyter.pid"
$MkdocsPid    = Join-Path $RepoRoot ".mkdocs.pid"
$JupyterLog   = Join-Path $RepoRoot ".jupyter.log"
$MkdocsLog    = Join-Path $RepoRoot ".mkdocs.log"
$VenvPython   = Join-Path $VenvPath "Scripts\python.exe"

function Test-PortInUse { param([int]$Port)
    try {
        $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop
        return [bool]$conn
    } catch { return $false }
}

# ── 7a. Jupyter Lab ──────────────────────────────────────────────────────────

Write-Step "Starting Jupyter Lab on port $JupyterPort"

if (Test-PortInUse -Port $JupyterPort) {
    Write-Ok "Port $JupyterPort already in use — assuming Jupyter Lab is running"
} else {
    $jupyterArgs = @(
        "-m", "jupyter", "lab",
        "--no-browser",
        "--ServerApp.ip=127.0.0.1",
        "--ServerApp.port=$JupyterPort",
        "--ServerApp.port_retries=0",
        "--ServerApp.root_dir=$RepoRoot",
        "--ServerApp.open_browser=False"
    )
    $jupyterProc = Start-Process -FilePath $VenvPython `
        -ArgumentList $jupyterArgs `
        -WorkingDirectory $RepoRoot `
        -RedirectStandardOutput $JupyterLog `
        -RedirectStandardError  "$JupyterLog.err" `
        -WindowStyle Hidden `
        -PassThru
    $jupyterProc.Id | Set-Content $JupyterPid
    Write-Ok "Jupyter Lab started (PID $($jupyterProc.Id)) — log: .jupyter.log"
    Write-Host "    Check .jupyter.log for the one-time login token/URL." -ForegroundColor DarkGray
}

# ── 7b. MkDocs site ──────────────────────────────────────────────────────────

if ($EnableMkdocsServer) {
    Write-Step "Starting MkDocs site on port $MkdocsPort"

    if (Test-PortInUse -Port $MkdocsPort) {
        Write-Ok "Port $MkdocsPort already in use — assuming MkDocs is running"
    } else {
        $mkdocsArgs = @(
            "-m", "mkdocs", "serve",
            "-f", (Join-Path $RepoRoot "mkdocs.yml"),
            "-a", "127.0.0.1:$MkdocsPort"
        )
        $mkdocsProc = Start-Process -FilePath $VenvPython `
            -ArgumentList $mkdocsArgs `
            -WorkingDirectory $RepoRoot `
            -RedirectStandardOutput $MkdocsLog `
            -RedirectStandardError  "$MkdocsLog.err" `
            -WindowStyle Hidden `
            -PassThru
        $mkdocsProc.Id | Set-Content $MkdocsPid
        Write-Ok "MkDocs started (PID $($mkdocsProc.Id)) — log: .mkdocs.log"
    }
} else {
    Write-Warn "Skipping MkDocs site; pass --enable-mkdocs-server when you want a local docs server."
}

# ─── ALL DONE ─────────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "══════════════════════════════════════════════" -ForegroundColor DarkGray
if ($EnableSlmAssistant) {
    Write-Host "  Setup complete (all 7 steps)" -ForegroundColor Green
} else {
    Write-Host "  Setup complete (assistant bundle skipped)" -ForegroundColor Green
}
Write-Host ""
Write-Host "  Python env  : $VenvPath" -ForegroundColor White
Write-Host "  Activate    : .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  VS Code     : $CodeCmd" -ForegroundColor White
if ($EnableSlmAssistant) {
    Write-Host "  Kilo Code   : $KiloExtId" -ForegroundColor White
    Write-Host "  Ollama      : $OllamaBaseUrl  (single-model mode, ctx=$CtxTokens)" -ForegroundColor White
    Write-Host "  Reasoning   : $ChosenModel (DeepSeek-R1, $CtxTokens-token ctx)" -ForegroundColor White
} else {
    Write-Host "  SLM assistant: disabled (pass --enable-slm-assistant to install Kilo Code, Ollama, and model wiring)" -ForegroundColor White
}
if ($EnableMkdocsServer) {
    Write-Host "  MkDocs      : http://localhost:$MkdocsPort" -ForegroundColor White
} else {
    Write-Host "  MkDocs      : disabled (pass --enable-mkdocs-server to launch the local docs server)" -ForegroundColor White
}
Write-Host ""
Write-Host "  Study servers (running in background):" -ForegroundColor Cyan
Write-Host "    Hands-on notebooks  → http://localhost:$JupyterPort" -ForegroundColor White
if ($EnableMkdocsServer) {
    Write-Host "    Reading (MkDocs)    → http://localhost:$MkdocsPort"  -ForegroundColor White
}
Write-Host ""
Write-Host "  To stop them:" -ForegroundColor DarkGray
if ($EnableMkdocsServer) {
    Write-Host "    Get-Content .jupyter.pid,.mkdocs.pid | % { Stop-Process -Id ([int]`$_) -Force }" -ForegroundColor DarkGray
} else {
    Write-Host "    Get-Content .jupyter.pid | % { Stop-Process -Id ([int]`$_) -Force }" -ForegroundColor DarkGray
}
Write-Host ""
if ($EnableSlmAssistant) {
    Write-Host "  Next: open VS Code in this folder — Ollama will start automatically." -ForegroundColor Cyan
    Write-Host "  If prompted, click 'Allow Automatic Tasks' to enable the watcher." -ForegroundColor Cyan
}
Write-Host "══════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host ""
