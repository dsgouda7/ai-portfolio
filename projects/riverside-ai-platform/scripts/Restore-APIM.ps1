[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string]$ConfigPath,
    [Parameter(Mandatory)] [string]$SnapshotPath,
    [Parameter(Mandatory)] [string]$ExpectedSnapshotSha256,
    [switch]$Apply
)

. "$PSScriptRoot/Common.ps1"

$config = Read-JsonObject $ConfigPath
$snapshot = Read-JsonObject $SnapshotPath
Assert-Digest $SnapshotPath $ExpectedSnapshotSha256 'snapshot' | Out-Null
$subscriptionId = Assert-NonEmptyString (Get-RequiredProperty $config subscription_id) 'subscription_id'
$tenantId = Assert-NonEmptyString (Get-RequiredProperty $config tenant_id) 'tenant_id'
$resourceGroup = Assert-NonEmptyString (Get-RequiredProperty $config resource_group) 'resource_group'
$serviceName = Assert-NonEmptyString (Get-RequiredProperty $config apim_service_name) 'apim_service_name'
$apiId = Assert-NonEmptyString (Get-RequiredProperty $config api_id) 'api_id'
$backendInputs = Get-RequiredProperty $config backends
$blueBackendName = Assert-NonEmptyString (Get-RequiredProperty $backendInputs blue_name 'backends') 'backends.blue_name'
$greenBackendName = Assert-NonEmptyString (Get-RequiredProperty $backendInputs green_name 'backends') 'backends.green_name'
$poolBackendName = Assert-NonEmptyString (Get-RequiredProperty $backendInputs pool_name 'backends') 'backends.pool_name'
if ($snapshot.subscription_id -cne $subscriptionId -or $snapshot.resource_group -cne $resourceGroup -or $snapshot.apim_service_name -cne $serviceName -or $snapshot.api_id -cne $apiId) {
    throw 'Snapshot target does not match the explicit APIM target inputs.'
}
$apiVersion = '2024-05-01'
$serviceBase = "/subscriptions/$subscriptionId/resourceGroups/$resourceGroup/providers/Microsoft.ApiManagement/service/$serviceName"

function Restore-Body([object]$Resource) {
    if ($null -eq $Resource -or $Resource.PSObject.Properties.Name -notcontains 'properties') { throw 'Snapshot resource has no properties.' }
    return @{ properties = $Resource.properties }
}
function Restore-Entry([object]$Entry) {
    if ($null -eq $Entry -or $Entry.PSObject.Properties.Name -notcontains 'exists') { throw 'Snapshot entry has no existence marker.' }
    if (-not [bool]$Entry.exists) { throw 'Cannot build a restore body for an absent snapshot entry.' }
    return Restore-Body $Entry.resource
}
function Put-ApimResource([string]$Uri, [object]$Body) {
    $bodyPath = Join-Path $env:TEMP ("riverside-apim-restore-" + [guid]::NewGuid().ToString('N') + '.json')
    try {
        Write-JsonEvidence $Body $bodyPath
        Invoke-AzChecked @('rest', '--method', 'put', '--uri', $Uri, '--body', "@$bodyPath", '--headers', 'Content-Type=application/json', '--output', 'none') | Out-Null
    } finally {
        Remove-Item -LiteralPath $bodyPath -Force -ErrorAction SilentlyContinue
    }
}
function Remove-ApimResource([string]$Uri) {
    Invoke-AzChecked @('rest', '--method', 'delete', '--uri', $Uri, '--output', 'none') | Out-Null
}

foreach ($property in $snapshot.resources.named_values.PSObject.Properties) {
    if ([bool]$property.Value.exists -and [bool]$property.Value.resource.properties.secret) { throw "Refusing to restore secret named value '$($property.Name)' from a file." }
}
if (-not $Apply) {
    Write-Host 'DRY RUN: snapshot digest and target binding are valid. No APIM resource was changed.'
    Write-Host 'Restore order: API, named values, single backends, pool backend, fragments, API policy.'
    return
}

Assert-AzureContext $subscriptionId $tenantId
if ([bool]$snapshot.resources.api.exists) { Put-ApimResource "$serviceBase/apis/$apiId`?api-version=$apiVersion" (Restore-Entry $snapshot.resources.api) }
foreach ($property in $snapshot.resources.named_values.PSObject.Properties | Where-Object { [bool]$_.Value.exists }) { Put-ApimResource "$serviceBase/namedValues/$($property.Name)`?api-version=$apiVersion" (Restore-Entry $property.Value) }
$backendProperties = @($snapshot.resources.backends.PSObject.Properties)
foreach ($property in $backendProperties | Where-Object { [bool]$_.Value.exists -and $_.Value.resource.properties.type -ne 'Pool' }) { Put-ApimResource "$serviceBase/backends/$($property.Name)`?api-version=$apiVersion" (Restore-Entry $property.Value) }
foreach ($property in $backendProperties | Where-Object { [bool]$_.Value.exists -and $_.Value.resource.properties.type -eq 'Pool' }) { Put-ApimResource "$serviceBase/backends/$($property.Name)`?api-version=$apiVersion" (Restore-Entry $property.Value) }
foreach ($property in $snapshot.resources.fragments.PSObject.Properties | Where-Object { [bool]$_.Value.exists }) { Put-ApimResource "$serviceBase/policyFragments/$($property.Name)`?api-version=$apiVersion" (Restore-Entry $property.Value) }
if ([bool]$snapshot.resources.policy.exists) { Put-ApimResource "$serviceBase/apis/$apiId/policies/policy?api-version=$apiVersion" (Restore-Entry $snapshot.resources.policy) }

if (-not [bool]$snapshot.resources.api.exists) {
    Remove-ApimResource "$serviceBase/apis/$apiId`?api-version=$apiVersion"
} elseif (-not [bool]$snapshot.resources.policy.exists) {
    Remove-ApimResource "$serviceBase/apis/$apiId/policies/policy?api-version=$apiVersion"
}
foreach ($property in $snapshot.resources.fragments.PSObject.Properties | Where-Object { -not [bool]$_.Value.exists }) { Remove-ApimResource "$serviceBase/policyFragments/$($property.Name)`?api-version=$apiVersion" }
foreach ($name in @($poolBackendName, $greenBackendName, $blueBackendName)) {
    $entry = $snapshot.resources.backends.$name
    if (-not [bool]$entry.exists) { Remove-ApimResource "$serviceBase/backends/$name`?api-version=$apiVersion" }
}
foreach ($property in $snapshot.resources.named_values.PSObject.Properties | Where-Object { -not [bool]$_.Value.exists }) { Remove-ApimResource "$serviceBase/namedValues/$($property.Name)`?api-version=$apiVersion" }
Write-Host 'APIM snapshot restore completed. Authentication, routing, retries, deadlines, and negative cases must be revalidated live.'
