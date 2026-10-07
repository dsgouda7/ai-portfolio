param([switch]$SkipKernel)
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $ScriptDir ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"
$Kernel = "pytorch-llms-02"
$Display = "Python (PyTorch for LLMs 02)"
$Setter = Join-Path $ScriptDir "..\..\..\scripts\set-notebook-kernel.py"
if (-not (Test-Path $Python)) {
    python -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed." }
}
& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
& $Python -m pip install -r (Join-Path $ScriptDir "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
if (-not $SkipKernel) {
    & $Python -m ipykernel install --user --name $Kernel --display-name $Display
    if ($LASTEXITCODE -ne 0) { throw "Kernel registration failed." }
    & $Python $Setter --directory $ScriptDir --name $Kernel --display-name $Display
    if ($LASTEXITCODE -ne 0) { throw "Notebook kernel assignment failed." }
}
Write-Host "Ready: $Display" -ForegroundColor Green
