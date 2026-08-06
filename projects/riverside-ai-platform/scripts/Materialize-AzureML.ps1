[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string]$ConfigPath,
    [Parameter(Mandatory)] [string]$OutputDirectory
)

. "$PSScriptRoot/Common.ps1"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$config = Read-JsonObject $ConfigPath
$environment = Assert-NonEmptyString (Get-RequiredProperty $config environment) 'environment'
if ($environment -notin @('dev', 'staging', 'production')) { throw 'environment must be dev, staging, or production.' }
$applicationDeadline = [int](Get-RequiredProperty $config application_deadline_seconds)
if ($applicationDeadline -lt 1 -or $applicationDeadline -gt 120) {
    throw 'application_deadline_seconds must be between 1 and 120 and remain below the 180-second Azure ML outer timeout.'
}

function Assert-AzureMlAssetName([object]$Value, [string]$Name) {
    $text = ConvertTo-TemplateScalar $Value $Name
    if ($text -cnotmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$') {
        throw "$Name must be a valid Azure ML asset name."
    }
    return $text
}

function Assert-AzureMlAssetVersion([object]$Value, [string]$Name) {
    $text = ConvertTo-TemplateScalar $Value $Name
    if ($text -cnotmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$') {
        throw "$Name must be an immutable Azure ML asset version."
    }
    return $text
}

$paths = Get-RequiredProperty $config paths
$codePath = (Resolve-Path (Get-RequiredProperty $paths code_package)).Path
$blueModelPath = (Resolve-Path (Get-RequiredProperty $paths blue_model_package)).Path
$greenModelPath = (Resolve-Path (Get-RequiredProperty $paths green_model_package)).Path
$expected = Get-RequiredProperty $config expected_sha256
$codeDigest = Assert-Digest $codePath (Get-RequiredProperty $expected code_package) 'expected_sha256.code_package'
$blueDigest = Assert-Digest $blueModelPath (Get-RequiredProperty $expected blue_model_package) 'expected_sha256.blue_model_package'
$greenDigest = Assert-Digest $greenModelPath (Get-RequiredProperty $expected green_model_package) 'expected_sha256.green_model_package'
if (-not (Test-Path -LiteralPath (Join-Path $codePath 'score.py') -PathType Leaf)) {
    throw 'paths.code_package must contain score.py at its root.'
}
function Assert-ReleaseArtifacts([string]$PackagePath, [string]$Label) {
    $releaseManifestPath = Join-Path $PackagePath 'model-release-manifest.json'
    $release = Read-JsonObject $releaseManifestPath
    $releaseId = Assert-NonEmptyString (Get-RequiredProperty $release release_id $Label) "$Label.release_id"
    foreach ($entry in @(
        @{ value = $release.adapter; uri = 'uri'; digest = 'digest' },
        @{ value = $release.tokenizer; uri = 'uri'; digest = 'digest' },
        @{ value = $release.training_provenance; uri = 'manifest_uri'; digest = 'manifest_digest' },
        @{ value = $release.evaluation; uri = 'report_uri'; digest = 'report_digest' }
    )) {
        $uri = Assert-NonEmptyString (Get-RequiredProperty $entry.value $entry.uri $Label) "$Label.$($entry.uri)"
        if ($uri -notmatch '^repo://(.+)$') { throw "$Label contains a non-package artifact URI: $uri" }
        $relative = $Matches[1].Replace('/', [IO.Path]::DirectorySeparatorChar)
        if ($relative.Split([IO.Path]::DirectorySeparatorChar) -contains '..') { throw "$Label contains a parent-traversal artifact URI." }
        $digestObject = Get-RequiredProperty $entry.value $entry.digest $Label
        if ((Get-RequiredProperty $digestObject algorithm $Label) -cne 'sha256') { throw "$Label requires sha256 artifact digests." }
        Assert-Digest (Join-Path $PackagePath $relative) (Get-RequiredProperty $digestObject value $Label) "$Label.$relative" | Out-Null
    }
    if (-not (Test-Path -LiteralPath (Join-Path $PackagePath 'base-model') -PathType Container)) { throw "$Label is missing base-model/." }
    if (-not (Test-Path -LiteralPath (Join-Path $PackagePath 'adapter_config.json') -PathType Leaf)) { throw "$Label is missing adapter_config.json." }
    return [ordered]@{
        release_id = $releaseId
        release_manifest_sha256 = Get-FileSha256 $releaseManifestPath
    }
}
$blueRelease = Assert-ReleaseArtifacts $blueModelPath 'blue_model_package'
$greenRelease = Assert-ReleaseArtifacts $greenModelPath 'green_model_package'
$condaPath = Join-Path $projectRoot 'azureml/environment/conda.yml'
$environmentTemplatePath = Join-Path $projectRoot 'azureml/environment/environment.yml'
$condaDigest = Assert-Digest $condaPath (Get-RequiredProperty $expected conda_file) 'expected_sha256.conda_file'
$environmentTemplateDigest = Assert-Digest $environmentTemplatePath (Get-RequiredProperty $expected environment_template) 'expected_sha256.environment_template'

$endpointName = ConvertTo-TemplateScalar (Get-RequiredProperty $config endpoint_name) 'endpoint_name'
$region = ConvertTo-TemplateScalar (Get-RequiredProperty $config region) 'region'
$blueSlot = ConvertTo-TemplateScalar (Get-RequiredProperty $config blue_slot_name) 'blue_slot_name'
$greenSlot = ConvertTo-TemplateScalar (Get-RequiredProperty $config green_slot_name) 'green_slot_name'
$environmentAssetName = Assert-AzureMlAssetName (Get-RequiredProperty $config environment_asset_name) 'environment_asset_name'
$environmentAssetVersion = Assert-AzureMlAssetVersion (Get-RequiredProperty $config environment_asset_version) 'environment_asset_version'
$baseImage = ConvertTo-TemplateScalar (Get-RequiredProperty $config base_image_by_digest) 'base_image_by_digest'
if ($baseImage -cnotmatch '^[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:([a-f0-9]{64})$') { throw 'base_image_by_digest must be pinned by sha256 digest.' }
$environmentImageDigest = $Matches[1]
$deployedAtValue = Get-RequiredProperty $config deployed_at
$deployedAt = if ($deployedAtValue -is [DateTime]) {
    $deployedAtValue.ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
} elseif ($deployedAtValue -is [DateTimeOffset]) {
    $deployedAtValue.ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
} else {
    Assert-NonEmptyString $deployedAtValue 'deployed_at'
}
try { [DateTimeOffset]::ParseExact($deployedAt, 'yyyy-MM-ddTHH:mm:ssZ', $null) | Out-Null } catch { throw 'deployed_at must be an explicit UTC timestamp in yyyy-MM-ddTHH:mm:ssZ format.' }

$endpointValues = @{
    RIVERSIDE_ENVIRONMENT = ConvertTo-TemplateScalar $environment 'environment'
    RIVERSIDE_REGION = $region
}
$rolloutValues = @{
    RIVERSIDE_BLUE_SLOT_NAME = $blueSlot
    RIVERSIDE_GREEN_SLOT_NAME = $greenSlot
}
$environmentValues = @{
    RIVERSIDE_ENVIRONMENT_ASSET_NAME = $environmentAssetName
    RIVERSIDE_ENVIRONMENT_ASSET_VERSION = $environmentAssetVersion
    RIVERSIDE_BASE_IMAGE_BY_DIGEST = $baseImage
    RIVERSIDE_ENVIRONMENT_IMAGE_SHA256 = $environmentImageDigest
    RIVERSIDE_CONDA_SHA256 = $condaDigest
    RIVERSIDE_ENVIRONMENT_TEMPLATE_SHA256 = $environmentTemplateDigest
}
$deploymentCommon = @{
    RIVERSIDE_ENDPOINT_NAME = $endpointName
    RIVERSIDE_ENVIRONMENT_ASSET_NAME = $environmentAssetName
    RIVERSIDE_ENVIRONMENT_ASSET_VERSION = $environmentAssetVersion
    RIVERSIDE_CODE_PACKAGE_PATH = ConvertTo-TemplateScalar $codePath 'paths.code_package'
    RIVERSIDE_CODE_PACKAGE_SHA256 = $codeDigest
    RIVERSIDE_ENVIRONMENT_IMAGE_SHA256 = $environmentImageDigest
    RIVERSIDE_ENVIRONMENT_CONDA_SHA256 = $condaDigest
    RIVERSIDE_ENVIRONMENT_TEMPLATE_SHA256 = $environmentTemplateDigest
    RIVERSIDE_ENVIRONMENT = ConvertTo-TemplateScalar $environment 'environment'
    RIVERSIDE_REGION = $region
    RIVERSIDE_INDEX_VERSION = ConvertTo-TemplateScalar (Get-RequiredProperty $config index_version) 'index_version'
    RIVERSIDE_DEPLOYED_AT = $deployedAt
    RIVERSIDE_APPLICATION_DEADLINE_SECONDS = [string]$applicationDeadline
}
$blueValues = $deploymentCommon.Clone()
$blueValues.RIVERSIDE_BLUE_SLOT_NAME = $blueSlot
$blueValues.RIVERSIDE_BLUE_DEPLOYMENT_NAME = ConvertTo-TemplateScalar (Get-RequiredProperty $config blue_deployment_name) 'blue_deployment_name'
$blueValues.RIVERSIDE_BLUE_MODEL_NAME = Assert-AzureMlAssetName (Get-RequiredProperty $config blue_model_name) 'blue_model_name'
$blueValues.RIVERSIDE_BLUE_MODEL_VERSION = Assert-AzureMlAssetVersion (Get-RequiredProperty $config blue_model_version) 'blue_model_version'
$blueValues.RIVERSIDE_BLUE_MODEL_PACKAGE_SHA256 = $blueDigest
$blueValues.RIVERSIDE_BLUE_RELEASE_MANIFEST_SHA256 = $blueRelease.release_manifest_sha256
$greenValues = $deploymentCommon.Clone()
$greenValues.RIVERSIDE_GREEN_SLOT_NAME = $greenSlot
$greenValues.RIVERSIDE_GREEN_DEPLOYMENT_NAME = ConvertTo-TemplateScalar (Get-RequiredProperty $config green_deployment_name) 'green_deployment_name'
$greenValues.RIVERSIDE_GREEN_MODEL_NAME = Assert-AzureMlAssetName (Get-RequiredProperty $config green_model_name) 'green_model_name'
$greenValues.RIVERSIDE_GREEN_MODEL_VERSION = Assert-AzureMlAssetVersion (Get-RequiredProperty $config green_model_version) 'green_model_version'
$greenValues.RIVERSIDE_GREEN_MODEL_PACKAGE_SHA256 = $greenDigest
$greenValues.RIVERSIDE_GREEN_RELEASE_MANIFEST_SHA256 = $greenRelease.release_manifest_sha256

$outputRoot = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$files = [ordered]@{}
function Expand-AzureMlTemplateFile {
    param(
        [string]$TemplatePath,
        [string]$OutputPath,
        [hashtable]$Values,
        [hashtable]$LiteralValues = @{}
    )
    $content = Get-Content -LiteralPath $TemplatePath -Raw -Encoding utf8
    foreach ($key in ($Values.Keys | Sort-Object)) {
        $token = "__$key`__"
        if (-not $content.Contains($token)) { throw "Template token '$token' was not found in $TemplatePath." }
        $content = $content.Replace($token, [string]$Values[$key])
    }
    foreach ($sentinel in ($LiteralValues.Keys | Sort-Object)) {
        if (-not $content.Contains($sentinel)) { throw "Template sentinel '$sentinel' was not found in $TemplatePath." }
        $content = $content.Replace($sentinel, [string]$LiteralValues[$sentinel])
    }
    $unresolved = @([regex]::Matches($content, '__RIVERSIDE_[A-Z0-9_]+__') | ForEach-Object Value | Sort-Object -Unique)
    $remainingSentinels = @($LiteralValues.Keys | Where-Object { $content.Contains($_) })
    if ($unresolved.Count -gt 0 -or $remainingSentinels.Count -gt 0) {
        throw "Unresolved template values in ${TemplatePath}: $(@($unresolved) + $remainingSentinels -join ', ')"
    }
    $parent = Split-Path -Parent $OutputPath
    if ($parent) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    [IO.File]::WriteAllText($OutputPath, $content, [Text.UTF8Encoding]::new($false))
}

function Materialize([string]$RelativePath, [hashtable]$Values, [hashtable]$LiteralValues = @{}) {
    $source = Join-Path $projectRoot "azureml/$RelativePath"
    $destination = Join-Path $outputRoot $RelativePath
    Expand-AzureMlTemplateFile $source $destination $Values $LiteralValues
    $files[$RelativePath.Replace('\', '/')] = Get-FileSha256 $destination
}

Materialize 'endpoint.yml' $endpointValues @{ 'RIVERSIDE-ENDPOINT-NAME-REQUIRED' = $endpointName }
Materialize 'environment/environment.yml' $environmentValues
Copy-Item -LiteralPath $condaPath -Destination (Join-Path $outputRoot 'environment/conda.yml') -Force
$files['environment/conda.yml'] = Get-FileSha256 (Join-Path $outputRoot 'environment/conda.yml')
Materialize 'deployments/blue.yml' $blueValues
Materialize 'deployments/green.yml' $greenValues
Materialize 'rollout/blue-100.yml' $rolloutValues @{ 'RIVERSIDE-ENDPOINT-NAME-REQUIRED' = $endpointName }
Materialize 'rollout/green-canary-10.yml' $rolloutValues @{ 'RIVERSIDE-ENDPOINT-NAME-REQUIRED' = $endpointName }
Materialize 'rollout/green-100.yml' $rolloutValues @{ 'RIVERSIDE-ENDPOINT-NAME-REQUIRED' = $endpointName }

$manifest = [ordered]@{
    schema_version = '1.0.0'
    environment = $environment
    region = $region
    endpoint_name = $endpointName
    application_deadline_seconds = $applicationDeadline
    azureml_outer_container_timeout_seconds = 180
    source_config_sha256 = Get-FileSha256 $ConfigPath
    packages = [ordered]@{
        code = @{ path = $codePath; sha256 = $codeDigest }
        blue_model = @{ path = $blueModelPath; sha256 = $blueDigest; release_id = $blueRelease.release_id; release_manifest_sha256 = $blueRelease.release_manifest_sha256; name = $blueValues.RIVERSIDE_BLUE_MODEL_NAME; version = $blueValues.RIVERSIDE_BLUE_MODEL_VERSION }
        green_model = @{ path = $greenModelPath; sha256 = $greenDigest; release_id = $greenRelease.release_id; release_manifest_sha256 = $greenRelease.release_manifest_sha256; name = $greenValues.RIVERSIDE_GREEN_MODEL_NAME; version = $greenValues.RIVERSIDE_GREEN_MODEL_VERSION }
    }
    environment_asset = @{ name = $environmentAssetName; version = $environmentAssetVersion; template_sha256 = $environmentTemplateDigest; conda_sha256 = $condaDigest; base_image = $baseImage; environment_image_sha256 = $environmentImageDigest }
    files = $files
}
Write-JsonEvidence $manifest (Join-Path $outputRoot 'materialization-manifest.json')
Write-Host "Materialized Azure ML assets at $outputRoot. No Azure command was run."
