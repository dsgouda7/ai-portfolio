<#
.SYNOPSIS
    Creates a local virtual environment and installs everything needed to run
    gpu-hardware-foundations.ipynb.

.DESCRIPTION
    Creates a `.venv` next to this script (if it does not already exist),
    installs the dependencies from requirements.txt into it, and registers a
    Jupyter kernel named "gpu-hardware" pointing at that venv. Select the
    "Python (gpu-hardware .venv)" kernel in VS Code to use it.

    Pass -SkipKernel to install into the venv without registering the kernel.

.EXAMPLE
    .\setup.ps1
    .\setup.ps1 -SkipKernel
#>
param(
    [switch]$SkipKernel
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Requirements = Join-Path $ScriptDir "requirements.txt"
$VenvDir = Join-Path $ScriptDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

$pythonCmd = if (Get-Command python -ErrorAction SilentlyContinue) { "python" }
             elseif (Get-Command python3 -ErrorAction SilentlyContinue) { "python3" }
             else { $null }

if (-not $pythonCmd) {
    Write-Host "Error: Python was not found on PATH. Install Python 3.9+ and re-run." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $Requirements)) {
    Write-Host "Error: requirements.txt not found next to this script." -ForegroundColor Red
    exit 1
}

if (Test-Path $VenvPython) {
    Write-Host "Reusing existing virtual environment at $VenvDir"
} else {
    $pyVersion = (& $pythonCmd --version 2>&1).Trim()
    Write-Host "Creating virtual environment with $pyVersion at $VenvDir"
    & $pythonCmd -m venv $VenvDir
}

Write-Host "Upgrading pip..."
& $VenvPython -m pip install --upgrade pip -q

Write-Host "Installing notebook dependencies from requirements.txt..."
& $VenvPython -m pip install -r $Requirements -q

Write-Host "Installing Jupyter kernel support (ipykernel)..."
& $VenvPython -m pip install ipykernel -q

Write-Host "Installing nbconvert for headless execution..."
& $VenvPython -m pip install nbconvert -q

if (-not $SkipKernel) {
    Write-Host "Registering Jupyter kernel 'gpu-hardware'..."
    & $VenvPython -m ipykernel install --user --name gpu-hardware --display-name "Python (gpu-hardware .venv)"
}

Write-Host "`nSetup complete." -ForegroundColor Green
Write-Host "Open gpu-hardware-foundations.ipynb and pick the 'Python (gpu-hardware .venv)' kernel."
Write-Host "The venv lives at: $VenvDir"
