# install-hooks.ps1 — copy hooks from scripts/hooks/ into .git/hooks/
#
# Usage (from repo root):
#   .\scripts\install-hooks.ps1

$ErrorActionPreference = 'Stop'

$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$HooksSource = Join-Path $ScriptDir 'hooks'
$GitRoot     = git rev-parse --show-toplevel
$HooksDest   = Join-Path $GitRoot '.git' 'hooks'

if (-not (Test-Path $HooksSource)) {
    Write-Error "Hooks directory not found: $HooksSource"
    exit 1
}

if (-not (Test-Path $HooksDest)) {
    New-Item -ItemType Directory -Path $HooksDest | Out-Null
}

Write-Host "`nInstalling Git hooks..." -ForegroundColor Cyan

$installed = 0

# Install pre-commit hook
$preCommitSource = Join-Path $HooksSource 'pre-commit'
if (Test-Path $preCommitSource) {
    $dest = Join-Path $HooksDest 'pre-commit'
    Copy-Item -Path $preCommitSource -Destination $dest -Force
    Write-Host "  ✓ " -ForegroundColor Green -NoNewline
    Write-Host "Installed: pre-commit"
    $installed++
}

# Install pre-push hook (secret removal)
$prePushSource = Join-Path $HooksSource 'pre-push-remove-secrets.ps1'
if (Test-Path $prePushSource) {
    $dest = Join-Path $HooksDest 'pre-push'
    Copy-Item -Path $prePushSource -Destination $dest -Force
    Write-Host "  ✓ " -ForegroundColor Green -NoNewline
    Write-Host "Installed: pre-push (secret removal)"
    $installed++
}

Write-Host ""
Write-Host "$installed hook(s) installed into .git\hooks\" -ForegroundColor Green
Write-Host "Test with: " -NoNewline
Write-Host "git add . ; git commit -m 'test'" -ForegroundColor Yellow -NoNewline
Write-Host " (on a branch)"
