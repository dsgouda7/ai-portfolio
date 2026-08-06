<#
.SYNOPSIS
    Creates the local environment for azure-operational-llm-serving.ipynb.

.DESCRIPTION
    Creates a .venv beside this script, installs requirements.txt, and
    optionally registers an azure-operational-serving Jupyter kernel. It does
    not install Azure tooling, start Docker, or contact Azure.

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

$PythonCmd = if (Get-Command python -ErrorAction SilentlyContinue) { "python" }
             elseif (Get-Command python3 -ErrorAction SilentlyContinue) { "python3" }
             else { $null }

if (-not $PythonCmd) {
    Write-Host "Error: Python 3.10+ was not found on PATH." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $Requirements)) {
    Write-Host "Error: requirements.txt was not found next to this script." -ForegroundColor Red
    exit 1
}

if (Test-Path $VenvPython) {
    Write-Host "Reusing virtual environment at $VenvDir"
} else {
    Write-Host "Creating virtual environment at $VenvDir"
    & $PythonCmd -m venv $VenvDir
}

& $VenvPython -m pip install --upgrade pip -q
& $VenvPython -m pip install -r $Requirements -q
& $VenvPython -m pip install ipykernel -q

if (-not $SkipKernel) {
    & $VenvPython -m ipykernel install --user `
        --name azure-operational-serving `
        --display-name "Python (azure-operational-serving .venv)"
}

Write-Host "`nSetup complete. No Azure or container service was contacted." -ForegroundColor Green
Write-Host "Open azure-operational-llm-serving.ipynb and select the chapter kernel."
