<#
.SYNOPSIS
    Creates the chapter-local environment for GenAI 04 LLM Evaluation.

.DESCRIPTION
    Creates or reuses `.venv` next to this script, installs every dependency
    from the adjacent requirements.txt, registers the `genai-04-llm-evaluation` Jupyter
    kernel, and assigns that kernel to every notebook in this chapter.

    Pass -SkipKernel to install dependencies without registering or assigning
    the Jupyter kernel.
#>
param(
    [switch]$SkipKernel
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Requirements = Join-Path $ScriptDir "requirements.txt"
$VenvDir = Join-Path $ScriptDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$KernelName = "genai-04-llm-evaluation"
$KernelDisplayName = "Python (GenAI 04 LLM Evaluation .venv)"
$KernelSetter = Join-Path $ScriptDir "..\..\..\scripts\set-notebook-kernel.py"

$PythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (-not $PythonCommand) {
    $PythonCommand = Get-Command python3 -ErrorAction SilentlyContinue
}
if (-not $PythonCommand) {
    throw "Python was not found on PATH. Install Python 3.10+ and rerun this script."
}
if (-not (Test-Path $Requirements)) {
    throw "requirements.txt was not found at $Requirements"
}
if (-not (Test-Path $KernelSetter)) {
    throw "Kernel metadata helper was not found at $KernelSetter"
}

if (Test-Path $VenvPython) {
    Write-Host "Reusing virtual environment at $VenvDir"
} else {
    $PythonVersion = (& $PythonCommand.Source --version 2>&1).Trim()
    Write-Host "Creating virtual environment with $PythonVersion at $VenvDir"
    & $PythonCommand.Source -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed." }
}

Write-Host "Upgrading pip, setuptools, and wheel..."
& $VenvPython -m pip install --upgrade pip setuptools wheel --quiet
if ($LASTEXITCODE -ne 0) { throw "Build-tool installation failed." }

Write-Host "Installing dependencies from $Requirements..."
& $VenvPython -m pip install -r $Requirements
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }

Write-Host "Downloading NLTK data used by the evaluation notebooks..."
$NltkDataScript = @'
import nltk

resources = [
    "punkt",
    "punkt_tab",
    "wordnet",
    "stopwords",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "maxent_ne_chunker",
    "maxent_ne_chunker_tab",
    "words",
]
for resource in resources:
    nltk.download(resource, quiet=True)
'@
& $VenvPython -c $NltkDataScript
if ($LASTEXITCODE -ne 0) { throw "NLTK data download failed." }

if (-not $SkipKernel) {
    Write-Host "Registering Jupyter kernel '$KernelName'..."
    & $VenvPython -m ipykernel install --user --name $KernelName --display-name $KernelDisplayName
    if ($LASTEXITCODE -ne 0) { throw "Jupyter kernel registration failed." }

    Write-Host "Assigning '$KernelName' to chapter notebooks..."
    & $VenvPython $KernelSetter --directory $ScriptDir --name $KernelName --display-name $KernelDisplayName
    if ($LASTEXITCODE -ne 0) { throw "Notebook kernel metadata update failed." }
}

Write-Host ""
Write-Host "Setup complete for GenAI 04 LLM Evaluation." -ForegroundColor Green
Write-Host "Virtual environment: $VenvDir"
Write-Host "Jupyter kernel: $KernelDisplayName"
