<#
.SYNOPSIS
    Creates the chapter-local environment for Training Data Quality and Lineage.

.DESCRIPTION
    Creates or reuses `.venv`, installs requirements, registers the
    `ai-engineer-data-quality` Jupyter kernel, and assigns it to this chapter's
    notebook. The script does not execute notebook cells.
#>
param(
    [switch]$SkipKernel
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Requirements = Join-Path $ScriptDir "requirements.txt"
$VenvDir = Join-Path $ScriptDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$KernelName = "ai-engineer-data-quality"
$KernelDisplayName = "Python (AI Engineer Data Quality .venv)"
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

if (-not $SkipKernel) {
    Write-Host "Registering Jupyter kernel '$KernelName'..."
    & $VenvPython -m ipykernel install --user --name $KernelName --display-name $KernelDisplayName
    if ($LASTEXITCODE -ne 0) { throw "Jupyter kernel registration failed." }

    Write-Host "Assigning '$KernelName' to chapter notebooks..."
    & $VenvPython $KernelSetter --directory $ScriptDir --name $KernelName --display-name $KernelDisplayName
    if ($LASTEXITCODE -ne 0) { throw "Notebook kernel metadata update failed." }
}

Write-Host ""
Write-Host "Setup complete for AI Engineer Training Data Quality and Lineage." -ForegroundColor Green
Write-Host "Virtual environment: $VenvDir"
Write-Host "Jupyter kernel: $KernelDisplayName"
Write-Host "Notebook cells were not executed."
