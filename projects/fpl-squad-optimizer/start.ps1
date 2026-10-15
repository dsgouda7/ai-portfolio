$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$webApp = Join-Path $PSScriptRoot 'fpl-generator\web.py'

if (-not (Test-Path $python)) {
    throw "Python environment not found. Run .\setup.ps1 first."
}

$previousOpenBrowser = $env:FPL_OPEN_BROWSER
try {
    $env:FPL_OPEN_BROWSER = '1'
    & $python $webApp
    $exitCode = $LASTEXITCODE
} finally {
    $env:FPL_OPEN_BROWSER = $previousOpenBrowser
}

exit $exitCode
