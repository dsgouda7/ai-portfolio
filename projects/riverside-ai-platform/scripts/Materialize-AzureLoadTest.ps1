[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string]$ConfigPath,
    [Parameter(Mandatory)] [string]$OutputDirectory
)

. "$PSScriptRoot/Common.ps1"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$config = Read-JsonObject $ConfigPath
$testId = ConvertTo-TemplateScalar (Get-RequiredProperty $config test_id) 'test_id'
$displayName = Assert-NonEmptyString (Get-RequiredProperty $config display_name) 'display_name'
if ($displayName.Contains('__')) { throw 'display_name contains a reserved template marker.' }
$targetHost = Assert-NonEmptyString (Get-RequiredProperty $config target_host) 'target_host'
if ($targetHost -notmatch '^https://[^/]+/?$') { throw 'target_host must be an explicit HTTPS origin with no path.' }
$tokenScope = Assert-NonEmptyString (Get-RequiredProperty $config token_scope) 'token_scope'
if ($tokenScope -notmatch '^https://[^/]+(?:/[^/]+)*/\.default$') { throw 'token_scope must be an explicit HTTPS /.default scope.' }
$engineInstances = [int](Get-RequiredProperty $config engine_instances)
if ($engineInstances -lt 1) { throw 'engine_instances must be positive.' }
$identityType = Assert-NonEmptyString (Get-RequiredProperty $config engine_identity_type) 'engine_identity_type'
if ($identityType -notin @('SystemAssigned', 'UserAssigned')) { throw 'engine_identity_type must be SystemAssigned or UserAssigned.' }
$identityClientId = ''
if ($identityType -eq 'UserAssigned') {
    $identityClientId = ConvertTo-TemplateScalar (Get-RequiredProperty $config engine_identity_client_id) 'engine_identity_client_id'
} elseif ($config.PSObject.Properties.Name -contains 'engine_identity_client_id' -and -not [string]::IsNullOrWhiteSpace([string]$config.engine_identity_client_id)) {
    throw 'engine_identity_client_id must be empty for SystemAssigned identity.'
}

$outputRoot = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$template = Join-Path $projectRoot 'load-tests/azure-load-test.yaml'
$outputYaml = Join-Path $outputRoot 'azure-load-test.yaml'
Expand-TemplateFile $template $outputYaml @{
    RIVERSIDE_LOAD_TEST_DISPLAY_NAME = $displayName
    RIVERSIDE_ENGINE_INSTANCES = [string]$engineInstances
    RIVERSIDE_TARGET_HOST = $targetHost.TrimEnd('/')
    RIVERSIDE_TOKEN_SCOPE = $tokenScope
    RIVERSIDE_MANAGED_IDENTITY_CLIENT_ID = $identityClientId
} @{ 'RIVERSIDE-LOAD-TEST-ID-REQUIRED' = $testId }

$files = [ordered]@{ 'azure-load-test.yaml' = Get-FileSha256 $outputYaml }
foreach ($name in @('locustfile.py', 'stages.json', 'synthetic-requests.jsonl', 'requirements.txt', 'locust.conf', 'success-criteria.json')) {
    $source = Join-Path $projectRoot "load-tests/$name"
    $destination = Join-Path $outputRoot $name
    Copy-Item -LiteralPath $source -Destination $destination -Force
    $files[$name] = Get-FileSha256 $destination
}
Write-JsonEvidence ([ordered]@{
    schema_version = '1.0.0'
    source_config_sha256 = Get-FileSha256 $ConfigPath
    test_id = $testId
    target_host = $targetHost.TrimEnd('/')
    token_scope = $tokenScope
    engine_instances = $engineInstances
    engine_identity_type = $identityType
    engine_identity_resource_id = if ($identityType -eq 'UserAssigned') { Assert-NonEmptyString (Get-RequiredProperty $config engine_identity_resource_id) 'engine_identity_resource_id' } else { $null }
    files = $files
}) (Join-Path $outputRoot 'materialization-manifest.json')
Write-Host "Materialized Azure Load Testing assets at $outputRoot. No Azure command was run."
