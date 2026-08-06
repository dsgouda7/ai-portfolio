[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string]$ConfigPath,
    [Parameter(Mandatory)] [string]$MaterializedDirectory,
    [switch]$Apply
)

. "$PSScriptRoot/Common.ps1"

$config = Read-JsonObject $ConfigPath
$manifestPath = Join-Path $MaterializedDirectory 'materialization-manifest.json'
$manifest = Read-JsonObject $manifestPath
$subscriptionId = Assert-NonEmptyString (Get-RequiredProperty $config subscription_id) 'subscription_id'
$tenantId = Assert-NonEmptyString (Get-RequiredProperty $config tenant_id) 'tenant_id'
$resourceGroup = Assert-NonEmptyString (Get-RequiredProperty $config resource_group) 'resource_group'
$workspaceName = Assert-NonEmptyString (Get-RequiredProperty $config workspace_name) 'workspace_name'

function Assert-RegisteredAssetId {
    param(
        [object]$Asset,
        [ValidateSet('environments', 'models')] [string]$AssetKind,
        [string]$Name,
        [string]$Version
    )
    if ($null -eq $Asset) { throw "Azure ML did not return $AssetKind asset ${Name}:$Version." }
    $actualId = Assert-NonEmptyString (Get-RequiredProperty $Asset id "$AssetKind.$Name.$Version") "$AssetKind.$Name.$Version.id"
    $expectedId = "/subscriptions/$subscriptionId/resourceGroups/$resourceGroup/providers/Microsoft.MachineLearningServices/workspaces/$workspaceName/$AssetKind/$Name/versions/$Version"
    if ($actualId -ine $expectedId) {
        throw "Azure ML returned an unexpected asset ID for ${Name}:$Version. Expected '$expectedId', observed '$actualId'."
    }
    if ([string](Get-RequiredProperty $Asset name "$AssetKind.$Name.$Version") -cne $Name -or
        [string](Get-RequiredProperty $Asset version "$AssetKind.$Name.$Version") -cne $Version) {
        throw "Azure ML returned mismatched name/version fields for ${Name}:$Version."
    }
    return $actualId
}

function Assert-AssetTags {
    param([object]$Asset, [hashtable]$ExpectedTags, [string]$Label)
    $tags = Get-RequiredProperty $Asset tags $Label
    foreach ($entry in $ExpectedTags.GetEnumerator()) {
        if ($tags.PSObject.Properties.Name -notcontains $entry.Key -or [string]$tags.($entry.Key) -cne [string]$entry.Value) {
            throw "$Label has an unexpected '$($entry.Key)' digest tag."
        }
    }
}

foreach ($property in $manifest.files.PSObject.Properties) {
    Assert-Digest (Join-Path $MaterializedDirectory $property.Name) ([string]$property.Value) "materialized.$($property.Name)" | Out-Null
}
foreach ($model in @($manifest.packages.blue_model, $manifest.packages.green_model)) {
    Assert-Digest $model.path $model.sha256 "model_package.$($model.name).$($model.version)" | Out-Null
}

$environmentFile = Join-Path $MaterializedDirectory 'environment/environment.yml'
$environmentAsset = $manifest.environment_asset
$planned = @(
    "az ml environment create --file $environmentFile --resource-group $resourceGroup --workspace-name $workspaceName",
    "az ml model create --name $($manifest.packages.blue_model.name) --version $($manifest.packages.blue_model.version) --path $($manifest.packages.blue_model.path) --type custom_model --resource-group $resourceGroup --workspace-name $workspaceName",
    "az ml model create --name $($manifest.packages.green_model.name) --version $($manifest.packages.green_model.version) --path $($manifest.packages.green_model.path) --type custom_model --resource-group $resourceGroup --workspace-name $workspaceName"
)
if (-not $Apply) {
    Write-Host 'DRY RUN: local digests are valid. Registration is not applied because -Apply was not supplied.'
    $planned | ForEach-Object { Write-Host $_ }
    return
}

Assert-AzureContext $subscriptionId $tenantId
$registeredModels = @()
$expectedEnvironmentTags = @{
    environment_template_sha256 = [string]$environmentAsset.template_sha256
    conda_sha256 = [string]$environmentAsset.conda_sha256
    image_sha256 = [string]$environmentAsset.environment_image_sha256
}
$environments = @(Invoke-AzChecked @('ml', 'environment', 'list', '--name', $environmentAsset.name, '--resource-group', $resourceGroup, '--workspace-name', $workspaceName, '--output', 'json') -CaptureJson)
$environmentExisting = @($environments | Where-Object { [string]$_.version -ceq [string]$environmentAsset.version }) | Select-Object -First 1
if ($null -ne $environmentExisting) {
    Assert-AssetTags $environmentExisting $expectedEnvironmentTags 'environment_asset'
    $registeredEnvironment = $environmentExisting
} else {
    $registeredEnvironment = Invoke-AzChecked @('ml', 'environment', 'create', '--file', $environmentFile, '--resource-group', $resourceGroup, '--workspace-name', $workspaceName, '--only-show-errors', '--output', 'json') -CaptureJson
    Assert-AssetTags $registeredEnvironment $expectedEnvironmentTags 'environment_asset'
}
$registeredEnvironmentEvidence = [ordered]@{
    id = Assert-RegisteredAssetId $registeredEnvironment 'environments' ([string]$environmentAsset.name) ([string]$environmentAsset.version)
    name = [string]$environmentAsset.name
    version = [string]$environmentAsset.version
    environment_image_sha256 = [string]$environmentAsset.environment_image_sha256
    conda_sha256 = [string]$environmentAsset.conda_sha256
    environment_template_sha256 = [string]$environmentAsset.template_sha256
}

foreach ($model in @($manifest.packages.blue_model, $manifest.packages.green_model)) {
    $expectedModelTags = @{
        package_sha256 = [string]$model.sha256
        release_manifest_sha256 = [string]$model.release_manifest_sha256
        release_id = [string]$model.release_id
    }
    $models = @(Invoke-AzChecked @('ml', 'model', 'list', '--name', $model.name, '--resource-group', $resourceGroup, '--workspace-name', $workspaceName, '--output', 'json') -CaptureJson)
    $existing = @($models | Where-Object { [string]$_.version -ceq [string]$model.version }) | Select-Object -First 1
    if ($null -ne $existing) {
        Assert-AssetTags $existing $expectedModelTags "model_asset.$($model.name).$($model.version)"
        $registeredModel = $existing
    } else {
        $registeredModel = Invoke-AzChecked @('ml', 'model', 'create', '--name', $model.name, '--version', $model.version, '--path', $model.path, '--type', 'custom_model', '--tags', "package_sha256=$($model.sha256)", "release_manifest_sha256=$($model.release_manifest_sha256)", "release_id=$($model.release_id)", '--resource-group', $resourceGroup, '--workspace-name', $workspaceName, '--only-show-errors', '--output', 'json') -CaptureJson
        Assert-AssetTags $registeredModel $expectedModelTags "model_asset.$($model.name).$($model.version)"
    }
    $registeredModels += [ordered]@{
        id = Assert-RegisteredAssetId $registeredModel 'models' ([string]$model.name) ([string]$model.version)
        name = [string]$model.name
        version = [string]$model.version
        release_id = [string]$model.release_id
        release_manifest_sha256 = [string]$model.release_manifest_sha256
        package_sha256 = [string]$model.sha256
    }
}
$registrationManifest = [ordered]@{
    schema_version = '1.0.0'
    source_materialization_manifest_sha256 = Get-FileSha256 $manifestPath
    subscription_id = $subscriptionId
    resource_group = $resourceGroup
    workspace_name = $workspaceName
    assets = [ordered]@{
        environment = $registeredEnvironmentEvidence
        models = $registeredModels
    }
}
Write-JsonEvidence $registrationManifest (Join-Path $MaterializedDirectory 'registration-manifest.json')
Write-Host 'Azure ML model and environment registrations completed. Live serving behavior remains unvalidated.'
