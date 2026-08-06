[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string]$ConfigPath,
    [Parameter(Mandatory)] [ValidateRange(0, 100)] [int]$BluePercent,
    [Parameter(Mandatory)] [ValidateRange(0, 100)] [int]$GreenPercent,
    [switch]$Apply
)

. "$PSScriptRoot/Common.ps1"

if ($BluePercent + $GreenPercent -ne 100) { throw 'BluePercent and GreenPercent must sum to 100.' }
$config = Read-JsonObject $ConfigPath
$subscriptionId = Assert-NonEmptyString (Get-RequiredProperty $config subscription_id) 'subscription_id'
$tenantId = Assert-NonEmptyString (Get-RequiredProperty $config tenant_id) 'tenant_id'
$resourceGroup = Assert-NonEmptyString (Get-RequiredProperty $config resource_group) 'resource_group'
$workspaceName = Assert-NonEmptyString (Get-RequiredProperty $config workspace_name) 'workspace_name'
$endpointName = Assert-NonEmptyString (Get-RequiredProperty $config endpoint_name) 'endpoint_name'
$blueName = Assert-NonEmptyString (Get-RequiredProperty $config blue_slot_name) 'blue_slot_name'
$greenName = Assert-NonEmptyString (Get-RequiredProperty $config green_slot_name) 'green_slot_name'
$traffic = "$blueName=$BluePercent $greenName=$GreenPercent"
if (-not $Apply) {
    Write-Host "DRY RUN: az ml online-endpoint update --name $endpointName --traffic '$traffic' --resource-group $resourceGroup --workspace-name $workspaceName"
    return
}

Assert-AzureContext $subscriptionId $tenantId
foreach ($slot in @($blueName, $greenName)) {
    Invoke-AzChecked @('ml', 'online-deployment', 'show', '--name', $slot, '--endpoint-name', $endpointName, '--resource-group', $resourceGroup, '--workspace-name', $workspaceName, '--output', 'none') | Out-Null
}
Invoke-AzChecked @('ml', 'online-endpoint', 'update', '--name', $endpointName, '--traffic', $traffic, '--resource-group', $resourceGroup, '--workspace-name', $workspaceName, '--only-show-errors') | Out-Null
$observed = Invoke-AzChecked @('ml', 'online-endpoint', 'show', '--name', $endpointName, '--resource-group', $resourceGroup, '--workspace-name', $workspaceName, '--query', 'traffic', '--output', 'json') -CaptureJson
if ([int]$observed.$blueName -ne $BluePercent -or [int]$observed.$greenName -ne $GreenPercent) {
    throw 'Azure ML traffic read-back does not match the requested allocation.'
}
Write-Host "Azure ML traffic is $traffic. End-to-end health and observation-window gates remain required."
