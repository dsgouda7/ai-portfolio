[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string]$ConfigPath,
    [Parameter(Mandatory)] [string]$MaterializedDirectory,
    [Parameter(Mandatory)] [string]$TestRunId,
    [ValidateSet('create', 'update')] [string]$TestOperation = 'update',
    [switch]$Apply
)

. "$PSScriptRoot/Common.ps1"

if ($TestRunId -cnotmatch '^[a-z0-9][a-z0-9_-]{0,63}$') { throw 'TestRunId must be an explicit lowercase identifier of at most 64 characters.' }
$config = Read-JsonObject $ConfigPath
$manifest = Read-JsonObject (Join-Path $MaterializedDirectory 'materialization-manifest.json')
$subscriptionId = Assert-NonEmptyString (Get-RequiredProperty $config subscription_id) 'subscription_id'
$tenantId = Assert-NonEmptyString (Get-RequiredProperty $config tenant_id) 'tenant_id'
$resourceGroup = Assert-NonEmptyString (Get-RequiredProperty $config resource_group) 'resource_group'
$loadTestResource = Assert-NonEmptyString (Get-RequiredProperty $config load_test_resource) 'load_test_resource'
foreach ($property in $manifest.files.PSObject.Properties) {
    Assert-Digest (Join-Path $MaterializedDirectory $property.Name) ([string]$property.Value) "materialized.$($property.Name)" | Out-Null
}
$testFile = Join-Path $MaterializedDirectory 'azure-load-test.yaml'
$identityArguments = @('--engine-ref-id-type', $manifest.engine_identity_type)
if ($manifest.engine_identity_type -eq 'UserAssigned') {
    $identityArguments += @('--engine-ref-ids', (Assert-NonEmptyString $manifest.engine_identity_resource_id 'engine_identity_resource_id'))
}
$testArguments = @('load', 'test', $TestOperation, '--load-test-resource', $loadTestResource, '--resource-group', $resourceGroup, '--test-id', $manifest.test_id, '--load-test-config-file', $testFile) + $identityArguments + @('--only-show-errors')
$runArguments = @('load', 'test-run', 'create', '--load-test-resource', $loadTestResource, '--resource-group', $resourceGroup, '--test-id', $manifest.test_id, '--test-run-id', $TestRunId, '--display-name', $TestRunId, '--description', "Riverside evidence run for materialization $($manifest.source_config_sha256)", '--only-show-errors')
if (-not $Apply) {
    Write-Host 'DRY RUN: load-test assets and digests are valid. No test was published or started.'
    Write-Host ('az ' + ($testArguments -join ' '))
    Write-Host ('az ' + ($runArguments -join ' '))
    return
}

Assert-AzureContext $subscriptionId $tenantId
Invoke-AzChecked $testArguments | Out-Null
$run = Invoke-AzChecked ($runArguments + @('--output', 'json')) -CaptureJson
if ($run.testRunId -cne $TestRunId) { throw 'Azure Load Testing returned a different test run ID.' }
$recordPath = Join-Path $MaterializedDirectory "run-$TestRunId.json"
Write-JsonEvidence @{ test_run_id = $TestRunId; test_id = $manifest.test_id; load_test_resource = $loadTestResource; materialization_manifest_sha256 = Get-FileSha256 (Join-Path $MaterializedDirectory 'materialization-manifest.json'); create_response = $run } $recordPath
Write-Host "Azure Load Testing run '$TestRunId' was started. Invoke Export-AzureLoadTestEvidence.ps1 only after it reaches a terminal state."
