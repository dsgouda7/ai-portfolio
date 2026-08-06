<#
.SYNOPSIS
    Creates the shared local environment for the complete FDE route.

.DESCRIPTION
    Creates or reuses learning/role-based-tracks/fde/.venv, installs all FDE notebook and local
    Azure operational lab dependencies, and optionally installs the Riverside
    platform plus local RAG validation dependencies. By default it registers one
    `fde` kernel and assigns it to all FDE notebooks and the Azure operational
    tutorial notebook. It does not start Docker or contact Azure services.
#>
param(
    [switch]$SkipKernel,
    [switch]$SkipProjects,
    [switch]$SkipRag,
    [switch]$IncludeAzureML
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..\..\..")).Path
$Requirements = Join-Path $ScriptDir "requirements.txt"
$VenvDir = Join-Path $ScriptDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$KernelName = "fde"
$KernelDisplayName = "Python (FDE .venv)"
$KernelSetter = Join-Path $RepoRoot "scripts\set-notebook-kernel.py"
$AzureTutorial = Join-Path $RepoRoot "learning\ai-infrastructure\09-azure-operational-llm-serving"
$RiversideProject = Join-Path $RepoRoot "projects\riverside-ai-platform"
$RagRoot = Join-Path $RepoRoot "projects\rag-knowledge-pipeline"

$PythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (-not $PythonCommand) {
    $PythonCommand = Get-Command python3 -ErrorAction SilentlyContinue
}
if (-not $PythonCommand) {
    throw "Python was not found on PATH. Install Python 3.11-3.13 and rerun this script."
}

$PythonVersionText = (& $PythonCommand.Source -c "import sys; print('.'.join(map(str, sys.version_info[:3])))").Trim()
$PythonVersion = [version]$PythonVersionText
if ($PythonVersion -lt [version]"3.11" -or $PythonVersion -ge [version]"3.14") {
    throw "Python $PythonVersionText is unsupported. The shared project environment requires Python 3.11-3.13."
}

foreach ($RequiredPath in @($Requirements, $KernelSetter, $AzureTutorial, $RiversideProject, $RagRoot)) {
    if (-not (Test-Path $RequiredPath)) {
        throw "Required setup input was not found: $RequiredPath"
    }
}

if (Test-Path $VenvPython) {
    Write-Host "Reusing virtual environment at $VenvDir"
} else {
    Write-Host "Creating virtual environment with Python $PythonVersionText at $VenvDir"
    & $PythonCommand.Source -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed." }
}

Write-Host "Upgrading pip, setuptools, and wheel..."
& $VenvPython -m pip install --upgrade pip setuptools wheel --quiet
if ($LASTEXITCODE -ne 0) { throw "Build-tool installation failed." }

Write-Host "Installing FDE notebook and local Azure lab dependencies..."
& $VenvPython -m pip install -r $Requirements
if ($LASTEXITCODE -ne 0) { throw "FDE dependency installation failed." }

if (-not $SkipProjects) {
    $RiversideExtras = if ($IncludeAzureML) { "test,telemetry,azureml" } else { "test,telemetry" }
    $RiversideRequirement = "${RiversideProject}[$RiversideExtras]"
    Write-Host "Installing Riverside AI Platform in editable mode with $RiversideExtras extras..."
    & $VenvPython -m pip install --editable $RiversideRequirement
    if ($LASTEXITCODE -ne 0) { throw "Riverside AI Platform installation failed." }
}

if (-not $SkipRag) {
    Write-Host "Installing local RAG phase 1, 2, and 3 validation dependencies..."
    & $VenvPython -m pip install `
        -r (Join-Path $RagRoot "phase1-ingest\requirements.txt") `
        -r (Join-Path $RagRoot "phase2-vectorize\requirements.txt") `
        -r (Join-Path $RagRoot "phase3-serve\requirements.txt") `
        "pytest>=8,<9" `
        "langchain>=0.1,<1" `
        "langchain-community>=0.0.10,<1" `
        "langchain-core>=0.1,<1"
    if ($LASTEXITCODE -ne 0) { throw "RAG dependency installation failed." }

    $RagSharedRequirement = Join-Path $RagRoot "shared"
    $RagIngestRequirement = "$(Join-Path $RagRoot 'phase1-ingest')[test]"
    & $VenvPython -m pip install `
        --editable $RagSharedRequirement `
        --editable $RagIngestRequirement
    if ($LASTEXITCODE -ne 0) { throw "Editable RAG package installation failed." }
}

Write-Host "Verifying installed imports..."
& $VenvPython -c "import jsonschema, pandas, numpy, matplotlib; print('FDE notebook imports: OK')"
if ($LASTEXITCODE -ne 0) { throw "FDE notebook import verification failed." }

if (-not $SkipProjects) {
    & $VenvPython -c "import fastapi, httpx, pytest; from azure.identity import DefaultAzureCredential; import app, endpoint_client, rag_orchestrator, release_gates, telemetry; print('Riverside project and test imports: OK')"
    if ($LASTEXITCODE -ne 0) { throw "Riverside project import verification failed." }
}

if (-not $SkipRag) {
    $RagIngestSource = Join-Path $RagRoot "phase1-ingest\src"
    $RagVectorizeSource = Join-Path $RagRoot "phase2-vectorize\src"
    $RagServeSource = Join-Path $RagRoot "phase3-serve"
    & $VenvPython -c "import bs4, pypdf, docx, deltalake, datasets, pyarrow, langchain, langchain_community, langchain_core, chromadb, torch, pytest, shared; from databricks.ai_search.client import AISearchClient; from databricks.sdk import WorkspaceClient; print('RAG dependency imports: OK')"
    if ($LASTEXITCODE -ne 0) { throw "RAG import verification failed." }
    & $VenvPython -c "import sys; sys.path.insert(0, r'$RagIngestSource'); import remote.config, remote.contracts; print('RAG phase 1 imports: OK')"
    if ($LASTEXITCODE -ne 0) { throw "RAG phase 1 import verification failed." }
    & $VenvPython -c "import sys; sys.path.insert(0, r'$RagVectorizeSource'); import remote.contracts, remote.databricks_adapters; print('RAG phase 2 imports: OK')"
    if ($LASTEXITCODE -ne 0) { throw "RAG phase 2 import verification failed." }
    & $VenvPython -c "import sys; sys.path.insert(0, r'$RagServeSource'); import src.rag_pipeline, src.retrieval, src.server; print('RAG phase 3 imports: OK')"
    if ($LASTEXITCODE -ne 0) { throw "RAG phase 3 import verification failed." }
}

if (-not $SkipKernel) {
    Write-Host "Registering Jupyter kernel '$KernelName'..."
    & $VenvPython -m ipykernel install --user --name $KernelName --display-name $KernelDisplayName
    if ($LASTEXITCODE -ne 0) { throw "Jupyter kernel registration failed." }

    Write-Host "Assigning '$KernelName' to FDE and Azure operational notebooks..."
    & $VenvPython $KernelSetter --directory $ScriptDir --name $KernelName --display-name $KernelDisplayName
    if ($LASTEXITCODE -ne 0) { throw "FDE notebook kernel assignment failed." }
    & $VenvPython $KernelSetter --directory $AzureTutorial --name $KernelName --display-name $KernelDisplayName
    if ($LASTEXITCODE -ne 0) { throw "Azure tutorial kernel assignment failed." }
}

Write-Host ""
Write-Host "FDE setup complete." -ForegroundColor Green
Write-Host "Virtual environment: $VenvDir"
Write-Host "Jupyter kernel: $KernelDisplayName"
Write-Host "No Docker or Azure service was installed, started, or contacted."
