[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string]$ConfigPath,
    [Parameter(Mandatory)] [string]$MaterializedDirectory,
    [ValidateSet('blue', 'green', 'both')] [string]$Deployment = 'both',
    [switch]$Apply
)

. "$PSScriptRoot/Common.ps1"

$config = Read-JsonObject $ConfigPath
$manifest = Read-JsonObject (Join-Path $MaterializedDirectory 'materialization-manifest.json')
$subscriptionId = Assert-NonEmptyString (Get-RequiredProperty $config subscription_id) 'subscription_id'
$tenantId = Assert-NonEmptyString (Get-RequiredProperty $config tenant_id) 'tenant_id'
$resourceGroup = Assert-NonEmptyString (Get-RequiredProperty $config resource_group) 'resource_group'
$workspaceName = Assert-NonEmptyString (Get-RequiredProperty $config workspace_name) 'workspace_name'
foreach ($property in $manifest.files.PSObject.Properties) {
    Assert-Digest (Join-Path $MaterializedDirectory $property.Name) ([string]$property.Value) "materialized.$($property.Name)" | Out-Null
}
$endpointFile = Join-Path $MaterializedDirectory 'endpoint.yml'
$deploymentFiles = if ($Deployment -eq 'both') { @('blue', 'green') } else { @($Deployment) }
if (-not $Apply) {
    Write-Host 'DRY RUN: materialized digests are valid. Azure ML publication is not applied because -Apply was not supplied.'
    Write-Host "az ml online-endpoint create|update --file $endpointFile --resource-group $resourceGroup --workspace-name $workspaceName"
    foreach ($slot in $deploymentFiles) {
        Write-Host "az ml online-deployment create|update --file $(Join-Path $MaterializedDirectory "deployments/$slot.yml") --resource-group $resourceGroup --workspace-name $workspaceName"
    }
    return
}

Assert-AzureContext $subscriptionId $tenantId
$endpoints = @(Invoke-AzChecked @('ml', 'online-endpoint', 'list', '--resource-group', $resourceGroup, '--workspace-name', $workspaceName, '--output', 'json') -CaptureJson)
$endpointExisting = @($endpoints | Where-Object { $_.name -ceq $manifest.endpoint_name }) | Select-Object -First 1
$endpointVerb = if ($null -eq $endpointExisting) { 'create' } else { 'update' }
Invoke-AzChecked @('ml', 'online-endpoint', $endpointVerb, '--file', $endpointFile, '--resource-group', $resourceGroup, '--workspace-name', $workspaceName, '--only-show-errors') | Out-Null

foreach ($slot in $deploymentFiles) {
    $deploymentName = Assert-NonEmptyString (Get-RequiredProperty $config "${slot}_slot_name") "${slot}_slot_name"
    $deploymentFile = Join-Path $MaterializedDirectory "deployments/$slot.yml"
    $deployments = @(Invoke-AzChecked @('ml', 'online-deployment', 'list', '--endpoint-name', $manifest.endpoint_name, '--resource-group', $resourceGroup, '--workspace-name', $workspaceName, '--output', 'json') -CaptureJson)
    $existing = @($deployments | Where-Object { $_.name -ceq $deploymentName }) | Select-Object -First 1
    $verb = if ($null -eq $existing) { 'create' } else { 'update' }
    Invoke-AzChecked @('ml', 'online-deployment', $verb, '--file', $deploymentFile, '--resource-group', $resourceGroup, '--workspace-name', $workspaceName, '--only-show-errors') | Out-Null
}
Write-Host 'Azure ML endpoint/deployment publication completed with no traffic change. Smoke, network, identity, quota, and runtime validation are still required.'
