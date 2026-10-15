<#
.SYNOPSIS
    Creates the chapter-local environment for AI Engineer 05.

.DESCRIPTION
    Creates or reuses `.venv`, installs requirements.txt, registers the chapter
    kernel, and assigns it to notebooks in this directory. It does not execute
    any notebook or contact a model, provider, or cloud service.
#>
param(
    [switch]$SkipKernel
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Requirements = Join-Path $ScriptDir "requirements.txt"
$VenvDir = Join-Path $ScriptDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$KernelName = "ai-engineer-05-feedback-drift"
$KernelDisplayName = "Python (AI Engineer 05 Feedback and Drift .venv)"
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
    Write-Host "Creating virtual environment at $VenvDir"
    & $PythonCommand.Source -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed." }
}

& $VenvPython -m pip install --upgrade pip setuptools wheel --quiet
if ($LASTEXITCODE -ne 0) { throw "Build-tool installation failed." }
& $VenvPython -m pip install -r $Requirements
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }

if (-not $SkipKernel) {
    & $VenvPython -m ipykernel install --user --name $KernelName --display-name $KernelDisplayName
    if ($LASTEXITCODE -ne 0) { throw "Jupyter kernel registration failed." }
    & $VenvPython $KernelSetter --directory $ScriptDir --name $KernelName --display-name $KernelDisplayName
    if ($LASTEXITCODE -ne 0) { throw "Notebook kernel metadata update failed." }
}

Write-Host ""
Write-Host "Setup complete for AI Engineer 05 Feedback and Drift." -ForegroundColor Green
Write-Host "No notebook was executed and no external service was contacted."
