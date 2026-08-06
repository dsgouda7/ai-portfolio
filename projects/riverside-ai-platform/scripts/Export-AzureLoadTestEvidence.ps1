[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string]$ConfigPath,
    [Parameter(Mandatory)] [string]$MaterializedDirectory,
    [Parameter(Mandatory)] [string]$TestRunId,
    [Parameter(Mandatory)] [string]$OutputDirectory,
    [switch]$Apply
)

. "$PSScriptRoot/Common.ps1"

if ($TestRunId -cnotmatch '^[a-z0-9][a-z0-9_-]{0,63}$') { throw 'TestRunId must be an explicit lowercase identifier of at most 64 characters.' }
$config = Read-JsonObject $ConfigPath
$manifestPath = Join-Path $MaterializedDirectory 'materialization-manifest.json'
$manifest = Read-JsonObject $manifestPath
$runRecordPath = Join-Path $MaterializedDirectory "run-$TestRunId.json"
$runRecord = Read-JsonObject $runRecordPath
$subscriptionId = Assert-NonEmptyString (Get-RequiredProperty $config subscription_id) 'subscription_id'
$tenantId = Assert-NonEmptyString (Get-RequiredProperty $config tenant_id) 'tenant_id'
$resourceGroup = Assert-NonEmptyString (Get-RequiredProperty $config resource_group) 'resource_group'
$loadTestResource = Assert-NonEmptyString (Get-RequiredProperty $config load_test_resource) 'load_test_resource'
if ($runRecord.test_run_id -cne $TestRunId -or $runRecord.test_id -cne $manifest.test_id -or $runRecord.load_test_resource -cne $loadTestResource) { throw 'Run record does not match the explicit evidence target.' }
Assert-Digest $manifestPath $runRecord.materialization_manifest_sha256 'run.materialization_manifest' | Out-Null
if (-not $Apply) {
    Write-Host 'DRY RUN: run binding and local digests are valid. No Azure result was read or downloaded.'
    Write-Host "Would require terminal status, download result/report/log archives, export LoadTestRunMetrics and EngineHealthMetrics, and write an evidence digest manifest for $TestRunId."
    return
}

Assert-AzureContext $subscriptionId $tenantId
$outputRoot = [IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $outputRoot) { throw 'OutputDirectory must not already exist; evidence export never overwrites prior evidence.' }
New-Item -ItemType Directory -Path $outputRoot | Out-Null
$run = Invoke-AzChecked @('load', 'test-run', 'show', '--load-test-resource', $loadTestResource, '--resource-group', $resourceGroup, '--test-run-id', $TestRunId, '--output', 'json') -CaptureJson
$terminalStatuses = @('Passed', 'Failed', 'Completed', 'Error', 'Stopped')
if ($run.status -notin $terminalStatuses) { throw "Test run is not terminal; observed status '$($run.status)'." }
Write-JsonEvidence $run (Join-Path $outputRoot 'run.json')
Invoke-AzChecked @('load', 'test-run', 'download-files', '--load-test-resource', $loadTestResource, '--resource-group', $resourceGroup, '--test-run-id', $TestRunId, '--path', $outputRoot, '--result', '--report', '--log', '--force', '--only-show-errors') | Out-Null
$loadMetrics = Invoke-AzChecked @('load', 'test-run', 'metrics', 'list', '--load-test-resource', $loadTestResource, '--resource-group', $resourceGroup, '--test-run-id', $TestRunId, '--metric-namespace', 'LoadTestRunMetrics', '--output', 'json') -CaptureJson
$engineMetrics = Invoke-AzChecked @('load', 'test-run', 'metrics', 'list', '--load-test-resource', $loadTestResource, '--resource-group', $resourceGroup, '--test-run-id', $TestRunId, '--metric-namespace', 'EngineHealthMetrics', '--output', 'json') -CaptureJson
if ($null -eq $loadMetrics -or $null -eq $engineMetrics) { throw 'Azure Load Testing returned an empty metric payload.' }
Write-JsonEvidence $loadMetrics (Join-Path $outputRoot 'load-test-run-metrics.json')
Write-JsonEvidence $engineMetrics (Join-Path $outputRoot 'engine-health-metrics.json')
$evidenceFiles = @(Get-ChildItem -LiteralPath $outputRoot -File -Recurse | Sort-Object FullName)
if ($evidenceFiles.Count -lt 5) { throw 'Evidence export did not produce the required run, metric, and downloaded result files.' }
$digests = [ordered]@{}
foreach ($file in $evidenceFiles) {
    $relative = [IO.Path]::GetRelativePath($outputRoot, $file.FullName).Replace('\', '/')
    $digests[$relative] = Get-FileSha256 $file.FullName
}
Write-JsonEvidence ([ordered]@{
    schema_version = '1.0.0'
    test_run_id = $TestRunId
    test_id = $manifest.test_id
    status = $run.status
    source_config_sha256 = $manifest.source_config_sha256
    materialization_manifest_sha256 = Get-FileSha256 $manifestPath
    files = $digests
    evidence_valid_for_release_gate = ($run.status -eq 'Passed')
    note = 'Raw Azure output retained; result_parser normalization and reviewed acceptance remain separate required gates.'
}) (Join-Path $outputRoot 'evidence-manifest.json')
if ($run.status -ne 'Passed') { throw "Evidence was retained, but the Azure Load Testing run status is '$($run.status)', not Passed." }
Write-Host "Azure Load Testing evidence was exported to $outputRoot. Capacity and production readiness remain unproven until reviewed and normalized."
