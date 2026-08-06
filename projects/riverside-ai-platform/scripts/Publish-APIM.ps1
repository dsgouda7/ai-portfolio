[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string]$ConfigPath,
    [Parameter(Mandatory)] [string]$BackupDirectory,
    [switch]$Apply
)

. "$PSScriptRoot/Common.ps1"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$config = Read-JsonObject $ConfigPath
$subscriptionId = Assert-NonEmptyString (Get-RequiredProperty $config subscription_id) 'subscription_id'
$tenantId = Assert-NonEmptyString (Get-RequiredProperty $config tenant_id) 'tenant_id'
$resourceGroup = Assert-NonEmptyString (Get-RequiredProperty $config resource_group) 'resource_group'
$serviceName = Assert-NonEmptyString (Get-RequiredProperty $config apim_service_name) 'apim_service_name'
$apiId = Assert-NonEmptyString (Get-RequiredProperty $config api_id) 'api_id'
$apiPath = Assert-NonEmptyString (Get-RequiredProperty $config api_path) 'api_path'
$namedValueInputs = Get-RequiredProperty $config named_values
$backendInputs = Get-RequiredProperty $config backends
$apiVersion = '2024-05-01'
$serviceBase = "/subscriptions/$subscriptionId/resourceGroups/$resourceGroup/providers/Microsoft.ApiManagement/service/$serviceName"
$openApiPath = Join-Path $projectRoot 'apim/api/openapi.json'
$policyPath = Join-Path $projectRoot 'apim/policies/api-policy.xml'
$fragmentDirectory = Join-Path $projectRoot 'apim/policies/fragments'
$fragmentManifestPath = Join-Path $fragmentDirectory 'manifest.json'
$namedValueContractPath = Join-Path $projectRoot 'apim/parameters/named-values.json'
$backendTemplatePath = Join-Path $projectRoot 'apim/backends/backends.bicep'
$fragmentManifest = Read-JsonObject $fragmentManifestPath
$namedValueContract = Read-JsonObject $namedValueContractPath
$expected = Get-RequiredProperty $config expected_sha256
$sourceDigests = [ordered]@{
    openapi = Assert-Digest $openApiPath (Get-RequiredProperty $expected openapi) 'expected_sha256.openapi'
    api_policy = Assert-Digest $policyPath (Get-RequiredProperty $expected api_policy) 'expected_sha256.api_policy'
    fragments = Assert-Digest $fragmentDirectory (Get-RequiredProperty $expected fragments) 'expected_sha256.fragments'
    named_values_contract = Assert-Digest $namedValueContractPath (Get-RequiredProperty $expected named_values_contract) 'expected_sha256.named_values_contract'
    backend_template = Assert-Digest $backendTemplatePath (Get-RequiredProperty $expected backend_template) 'expected_sha256.backend_template'
}

$resolvedNamedValues = [ordered]@{}
foreach ($entry in $namedValueContract.named_values) {
    if ([bool]$entry.secret) { throw "Secret named value '$($entry.name)' is not supported by this identity-only workflow." }
    if ($namedValueInputs.PSObject.Properties.Name -notcontains $entry.name) { throw "named_values.$($entry.name) is required." }
    $value = Assert-NonEmptyString $namedValueInputs.($entry.name) "named_values.$($entry.name)"
    if ($entry.PSObject.Properties.Name -contains 'allowed_values' -and $value -notin $entry.allowed_values) {
        throw "named_values.$($entry.name) is outside the allowed values."
    }
    $resolvedNamedValues[$entry.name] = $value
}
if ($resolvedNamedValues['riverside-backend-pool-id'] -cne $backendInputs.pool_name) {
    throw 'riverside-backend-pool-id must exactly match backends.pool_name.'
}

$snapshot = [ordered]@{
    schema_version = '1.0.0'
    subscription_id = $subscriptionId
    resource_group = $resourceGroup
    apim_service_name = $serviceName
    api_id = $apiId
    resources = [ordered]@{ api = $null; named_values = [ordered]@{}; backends = [ordered]@{}; fragments = [ordered]@{}; policy = $null }
}

function Get-ApimResource([string]$Uri) {
    $result = Invoke-AzChecked @('rest', '--method', 'get', '--uri', $Uri, '--output', 'json') -CaptureJson
    return $result
}

function Get-SnapshotEntry([bool]$Exists, [string]$Uri) {
    if (-not $Exists) { return @{ exists = $false; resource = $null } }
    return @{ exists = $true; resource = Get-ApimResource $Uri }
}

function Put-ApimResource([string]$Uri, [object]$Body) {
    $bodyPath = Join-Path $env:TEMP ("riverside-apim-" + [guid]::NewGuid().ToString('N') + '.json')
    try {
        Write-JsonEvidence $Body $bodyPath
        Invoke-AzChecked @('rest', '--method', 'put', '--uri', $Uri, '--body', "@$bodyPath", '--headers', 'Content-Type=application/json', '--output', 'none') | Out-Null
    } finally {
        Remove-Item -LiteralPath $bodyPath -Force -ErrorAction SilentlyContinue
    }
}

$apiBody = @{ properties = @{ displayName = 'Riverside AI Platform Chat API'; description = 'Riverside application-facing chat API.'; path = $apiPath; protocols = @('https'); subscriptionRequired = $false; format = 'openapi+json'; value = Get-Content -LiteralPath $openApiPath -Raw -Encoding utf8 } }
$fragmentBodies = [ordered]@{}
foreach ($fragment in $fragmentManifest.fragments) {
    $fragmentPath = Join-Path $projectRoot "apim/policies/fragments/$($fragment.file)"
    $fragmentBodies[$fragment.id] = @{ properties = @{ description = "Riverside $($fragment.section) policy fragment, order $($fragment.order)."; format = 'rawxml'; value = Get-Content -LiteralPath $fragmentPath -Raw -Encoding utf8 } }
}
$namedValueBodies = [ordered]@{}
foreach ($entry in $resolvedNamedValues.GetEnumerator()) {
    $namedValueBodies[$entry.Key] = @{ properties = @{ displayName = $entry.Key; value = $entry.Value; secret = $false; tags = @('riverside-managed') } }
}
$policyBody = @{ properties = @{ format = 'rawxml'; value = Get-Content -LiteralPath $policyPath -Raw -Encoding utf8 } }
$backendParameters = @{
    apimServiceName = @{ value = $serviceName }
    blueBackendName = @{ value = (Assert-NonEmptyString $backendInputs.blue_name 'backends.blue_name') }
    greenBackendName = @{ value = (Assert-NonEmptyString $backendInputs.green_name 'backends.green_name') }
    poolBackendName = @{ value = (Assert-NonEmptyString $backendInputs.pool_name 'backends.pool_name') }
    blueBackendUrl = @{ value = (Assert-NonEmptyString $backendInputs.blue_url 'backends.blue_url') }
    greenBackendUrl = @{ value = (Assert-NonEmptyString $backendInputs.green_url 'backends.green_url') }
    blueWeight = @{ value = [int]$backendInputs.blue_weight }
    greenWeight = @{ value = [int]$backendInputs.green_weight }
    bluePriority = @{ value = [int]$backendInputs.blue_priority }
    greenPriority = @{ value = [int]$backendInputs.green_priority }
}
foreach ($urlName in @('blueBackendUrl', 'greenBackendUrl')) {
    if ($backendParameters[$urlName].value -notmatch '^https://[^/]+(?:/.*)?$') { throw "$urlName must be an explicit HTTPS URL." }
}
$backendParametersPath = Join-Path $BackupDirectory 'backend-parameters.json'
Write-JsonEvidence @{ '$schema' = 'https://schema.management.azure.com/schemas/2019-04-01/deploymentParameters.json#'; contentVersion = '1.0.0.0'; parameters = $backendParameters } $backendParametersPath

if (-not $Apply) {
    Write-Host 'DRY RUN: APIM source and explicit values are complete. No APIM resource was read or changed.'
    Write-Host "Backend preview: az deployment group what-if --resource-group $resourceGroup --template-file apim/backends/backends.bicep --parameters $backendParametersPath --validation-level Provider"
    Write-Host 'Apply order: API, named values, backend deployment, fragments, API policy.'
    return
}

Assert-AzureContext $subscriptionId $tenantId
New-Item -ItemType Directory -Path $BackupDirectory -Force | Out-Null
$apiUri = "$serviceBase/apis/$apiId`?api-version=$apiVersion"
$apis = Get-ApimResource "$serviceBase/apis?api-version=$apiVersion&`$top=1000"
$namedValues = Get-ApimResource "$serviceBase/namedValues?api-version=$apiVersion&`$top=1000"
$backends = Get-ApimResource "$serviceBase/backends?api-version=$apiVersion&`$top=1000"
$fragments = Get-ApimResource "$serviceBase/policyFragments?api-version=$apiVersion&`$top=1000"
$apiExists = @($apis.value | Where-Object { $_.name -ceq $apiId }).Count -eq 1
$snapshot.resources.api = Get-SnapshotEntry $apiExists $apiUri
foreach ($name in $resolvedNamedValues.Keys) {
    $exists = @($namedValues.value | Where-Object { $_.name -ceq $name }).Count -eq 1
    $entry = Get-SnapshotEntry $exists "$serviceBase/namedValues/$name`?api-version=$apiVersion"
    if ($entry.exists -and [bool]$entry.resource.properties.secret) { throw "Refusing to snapshot secret APIM named value '$name'." }
    $snapshot.resources.named_values[$name] = $entry
}
foreach ($name in @($backendInputs.blue_name, $backendInputs.green_name, $backendInputs.pool_name)) {
    $exists = @($backends.value | Where-Object { $_.name -ceq $name }).Count -eq 1
    $snapshot.resources.backends[$name] = Get-SnapshotEntry $exists "$serviceBase/backends/$name`?api-version=$apiVersion"
}
foreach ($fragment in $fragmentManifest.fragments) {
    $exists = @($fragments.value | Where-Object { $_.name -ceq $fragment.id }).Count -eq 1
    $snapshot.resources.fragments[$fragment.id] = Get-SnapshotEntry $exists "$serviceBase/policyFragments/$($fragment.id)`?api-version=$apiVersion"
}
$policyExists = $false
if ($apiExists) {
    $policies = Get-ApimResource "$serviceBase/apis/$apiId/policies?api-version=$apiVersion"
    $policyExists = @($policies.value | Where-Object { $_.name -ceq 'policy' }).Count -eq 1
}
$snapshot.resources.policy = Get-SnapshotEntry $policyExists "$serviceBase/apis/$apiId/policies/policy?api-version=$apiVersion"
$snapshotPath = Join-Path $BackupDirectory 'apim-snapshot.json'
Write-JsonEvidence $snapshot $snapshotPath
Write-JsonEvidence @{ source = $sourceDigests; snapshot_sha256 = Get-FileSha256 $snapshotPath } (Join-Path $BackupDirectory 'apim-evidence.json')

Put-ApimResource $apiUri $apiBody
foreach ($name in $namedValueBodies.Keys) { Put-ApimResource "$serviceBase/namedValues/$name`?api-version=$apiVersion" $namedValueBodies[$name] }
Invoke-AzChecked @('deployment', 'group', 'what-if', '--resource-group', $resourceGroup, '--template-file', $backendTemplatePath, '--parameters', $backendParametersPath, '--validation-level', 'Provider', '--only-show-errors') | Out-Null
Invoke-AzChecked @('deployment', 'group', 'create', '--name', "riverside-apim-$apiId", '--resource-group', $resourceGroup, '--template-file', $backendTemplatePath, '--parameters', $backendParametersPath, '--only-show-errors') | Out-Null
foreach ($id in $fragmentBodies.Keys) { Put-ApimResource "$serviceBase/policyFragments/$id`?api-version=$apiVersion" $fragmentBodies[$id] }
Put-ApimResource "$serviceBase/apis/$apiId/policies/policy?api-version=$apiVersion" $policyBody
Write-Host "APIM source was published. Snapshot retained at $snapshotPath; live policy behavior remains unvalidated."
