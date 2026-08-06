Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$result = Invoke-Pester -Script (Join-Path $PSScriptRoot 'Preflight.Tests.ps1') -PassThru
Write-Output "Pester passed=$($result.PassedCount) failed=$($result.FailedCount)"
if ($result.FailedCount -gt 0) {
    exit 1
}
