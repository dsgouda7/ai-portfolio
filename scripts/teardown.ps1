# teardown.ps1 — Gracefully stop all background processes started by setup.ps1
# and deactivate the .venv.
#
# Usage:
#   .\scripts\teardown.ps1
#
# Stops (in order):
#   1. Jupyter Lab       (.jupyter.pid  → port 8888)
#   2. MkDocs site       (.mkdocs.pid   → port 8000)
#   3. Ollama server     (.ollama.pid   → port 11434)
#   4. Deactivates the virtual environment if active

Set-StrictMode -Version Latest
$ErrorActionPreference = "SilentlyContinue"

$RepoRoot = Split-Path -Parent $PSScriptRoot

# ─── Helpers ──────────────────────────────────────────────────────────────────

function Write-Step { param($msg) Write-Host "`n▶ $msg" -ForegroundColor Cyan }
function Write-Ok   { param($msg) Write-Host "  ✓ $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "  ! $msg" -ForegroundColor Yellow }
function Write-Skip { param($msg) Write-Host "  – $msg" -ForegroundColor DarkGray }

function Stop-ServiceByPidFile {
    param(
        [string]$Label,
        [string]$PidFile,
        [string]$FallbackProcessName
    )

    Write-Step "Stopping $Label"

    $stopped = $false

    # Try PID file first
    if (Test-Path $PidFile) {
        $savedPid = (Get-Content $PidFile -Raw -ErrorAction SilentlyContinue).Trim()
        if ($savedPid -match '^\d+$') {
            $proc = Get-Process -Id ([int]$savedPid) -ErrorAction SilentlyContinue
            if ($proc) {
                # Send CTRL_C_EVENT equivalent: close gracefully first, then force
                $proc.CloseMainWindow() | Out-Null
                Start-Sleep -Milliseconds 800
                if (-not $proc.HasExited) {
                    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
                }
                Write-Ok "$Label (PID $savedPid) stopped"
                $stopped = $true
            } else {
                Write-Warn "PID $savedPid from $([System.IO.Path]::GetFileName($PidFile)) is no longer running"
            }
        }
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    }

    # Fallback: find by process name
    if (-not $stopped -and $FallbackProcessName) {
        $procs = Get-Process -Name $FallbackProcessName -ErrorAction SilentlyContinue
        if ($procs) {
            foreach ($p in $procs) {
                $p.CloseMainWindow() | Out-Null
                Start-Sleep -Milliseconds 400
                if (-not $p.HasExited) {
                    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
                }
            }
            Write-Ok "$Label stopped via process name '$FallbackProcessName' ($($procs.Count) process(es))"
            $stopped = $true
        }
    }

    if (-not $stopped) {
        Write-Skip "$Label was not running"
    }
}

# ─── Banner ───────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "══════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host "  AI/ML Dev Environment Teardown" -ForegroundColor White
Write-Host "  Stopping background processes + venv" -ForegroundColor White
Write-Host "══════════════════════════════════════════════" -ForegroundColor DarkGray

# ─── 1. Jupyter Lab ───────────────────────────────────────────────────────────

Stop-ServiceByPidFile `
    -Label        "Jupyter Lab" `
    -PidFile      (Join-Path $RepoRoot ".jupyter.pid") `
    -FallbackProcessName "jupyter"

# ─── 2. MkDocs ────────────────────────────────────────────────────────────────

Stop-ServiceByPidFile `
    -Label        "MkDocs site" `
    -PidFile      (Join-Path $RepoRoot ".mkdocs.pid") `
    -FallbackProcessName "mkdocs"

# ─── 3. Ollama ────────────────────────────────────────────────────────────────

Stop-ServiceByPidFile `
    -Label        "Ollama server" `
    -PidFile      (Join-Path $RepoRoot ".ollama.pid") `
    -FallbackProcessName "ollama"

# ─── 4. Deactivate virtual environment ───────────────────────────────────────

Write-Step "Virtual environment"

if ($env:VIRTUAL_ENV) {
    $venvName = Split-Path -Leaf $env:VIRTUAL_ENV
    if (Get-Command "deactivate" -ErrorAction SilentlyContinue) {
        deactivate
        Write-Ok "Deactivated venv: $venvName"
    } else {
        # Manual cleanup when deactivate function is not in scope
        $env:PATH = ($env:PATH -split ';' |
            Where-Object { $_ -notlike "$($env:VIRTUAL_ENV)*" }) -join ';'
        Remove-Item Env:VIRTUAL_ENV -ErrorAction SilentlyContinue
        Write-Ok "Removed venv from PATH: $venvName"
    }
} else {
    Write-Skip "No active virtual environment detected"
}

# ─── Done ─────────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "══════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host "  Teardown complete" -ForegroundColor Green
Write-Host "══════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host ""
