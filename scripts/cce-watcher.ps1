# cce-watcher.ps1 — CCE file-system watcher (Windows)
#
# Keeps the Code Context Engine index up to date between git commits:
#   • Any new file created in the repo  →  cce index --changed-only  (immediate)
#   • Any file deleted from the repo    →  cce index --changed-only  (immediate)
#   • A .ipynb file accumulates ≥ 3 cell additions/changes  →  cce index --changed-only
#
# Usage:  .\scripts\cce-watcher.ps1
# The VS Code workspace task "cce-watcher" starts this automatically on folder open.

param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot)
)

$CcePath = Join-Path $RepoRoot ".venv\Scripts\cce.exe"
if (-not (Test-Path $CcePath)) {
    Write-Error "cce binary not found at $CcePath — run .\scripts\setup.ps1 first."
    exit 1
}

# Directories to ignore
$SkipPatterns = @('[\\/]\.venv[\\/]', '[\\/]\.git[\\/]', '[\\/]__pycache__[\\/]',
                  '[\\/]\.cce[\\/]', '[\\/]node_modules[\\/]')

# Shared state: synchronized hashtable so event-action blocks can read/write it
$script:State = [hashtable]::Synchronized(@{
    CcePath       = $CcePath
    RepoRoot      = $RepoRoot
    SkipPatterns  = $SkipPatterns
    CellCounts    = [System.Collections.Generic.Dictionary[string, int]]::new()
    PendingDeltas = [System.Collections.Generic.Dictionary[string, int]]::new()
    LastTriggered = [datetime]::MinValue
    DebounceSecs  = 2
})

# ── FileSystemWatcher ────────────────────────────────────────────────────────
$watcher = New-Object System.IO.FileSystemWatcher($RepoRoot)
$watcher.IncludeSubdirectories = $true
$watcher.NotifyFilter = [System.IO.NotifyFilters]::FileName -bor
                        [System.IO.NotifyFilters]::LastWrite -bor
                        [System.IO.NotifyFilters]::DirectoryName

# ── Created: any new non-notebook file → immediate index ────────────────────
$null = Register-ObjectEvent -InputObject $watcher -EventName Created `
    -SourceIdentifier "CCEWatcher.Created" `
    -MessageData $script:State `
    -Action {
        $path  = $Event.SourceEventArgs.FullPath
        $state = $Event.MessageData

        foreach ($pat in $state.SkipPatterns) {
            if ($path -match $pat) { return }
        }

        # Debounce
        if (([datetime]::Now - $state.LastTriggered).TotalSeconds -lt $state.DebounceSecs) { return }
        $state.LastTriggered = [datetime]::Now

        Start-Process -FilePath $state.CcePath `
            -ArgumentList "index", "--changed-only" `
            -WorkingDirectory $state.RepoRoot `
            -WindowStyle Hidden `
            -ErrorAction SilentlyContinue
    }

# ── Deleted: any removed file → immediate index ────────────────────────────
$null = Register-ObjectEvent -InputObject $watcher -EventName Deleted `
    -SourceIdentifier "CCEWatcher.Deleted" `
    -MessageData $script:State `
    -Action {
        $path  = $Event.SourceEventArgs.FullPath
        $state = $Event.MessageData

        foreach ($pat in $state.SkipPatterns) {
            if ($path -match $pat) { return }
        }

        # Debounce
        if (([datetime]::Now - $state.LastTriggered).TotalSeconds -lt $state.DebounceSecs) { return }
        $state.LastTriggered = [datetime]::Now

        Start-Process -FilePath $state.CcePath `
            -ArgumentList "index", "--changed-only" `
            -WorkingDirectory $state.RepoRoot `
            -WindowStyle Hidden `
            -ErrorAction SilentlyContinue
    }

# ── Changed: .ipynb with ≥ 3 accumulated cell changes → index ───────────────
$null = Register-ObjectEvent -InputObject $watcher -EventName Changed `
    -SourceIdentifier "CCEWatcher.Changed" `
    -MessageData $script:State `
    -Action {
        $path  = $Event.SourceEventArgs.FullPath
        $state = $Event.MessageData

        foreach ($pat in $state.SkipPatterns) {
            if ($path -match $pat) { return }
        }

        if ($path -notlike '*.ipynb') { return }

        # Parse notebook to count cells
        try {
            $nb           = Get-Content $path -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop
            $currentCount = ($nb.cells | Measure-Object).Count
        } catch { return }

        $prev    = if ($state.CellCounts.ContainsKey($path))    { $state.CellCounts[$path] }    else { 0 }
        $pending = if ($state.PendingDeltas.ContainsKey($path)) { $state.PendingDeltas[$path] } else { 0 }

        $delta   = [Math]::Abs($currentCount - $prev)
        $state.CellCounts[$path] = $currentCount
        $pending += $delta

        if ($pending -ge 3) {
            $state.PendingDeltas[$path] = 0

            # Debounce
            if (([datetime]::Now - $state.LastTriggered).TotalSeconds -lt $state.DebounceSecs) { return }
            $state.LastTriggered = [datetime]::Now

            Start-Process -FilePath $state.CcePath `
                -ArgumentList "index", "--changed-only" `
                -WorkingDirectory $state.RepoRoot `
                -WindowStyle Hidden `
                -ErrorAction SilentlyContinue
        } else {
            $state.PendingDeltas[$path] = $pending
        }
    }

$watcher.EnableRaisingEvents = $true

Write-Host "CCE watcher active — $RepoRoot"
Write-Host "  • New file created          → cce index --changed-only"
Write-Host "  • File deleted              → cce index --changed-only"
Write-Host "  • .ipynb ≥ 3 cells changed  → cce index --changed-only"
Write-Host "Press Ctrl+C to stop."

try {
    while ($true) {
        # Wait-Event processes queued events and yields CPU
        $null = Wait-Event -Timeout 1
    }
} finally {
    Unregister-Event -SourceIdentifier "CCEWatcher.Created" -ErrorAction SilentlyContinue
    Unregister-Event -SourceIdentifier "CCEWatcher.Deleted" -ErrorAction SilentlyContinue
    Unregister-Event -SourceIdentifier "CCEWatcher.Changed" -ErrorAction SilentlyContinue
    $watcher.EnableRaisingEvents = $false
    $watcher.Dispose()
    Write-Host "CCE watcher stopped."
}
