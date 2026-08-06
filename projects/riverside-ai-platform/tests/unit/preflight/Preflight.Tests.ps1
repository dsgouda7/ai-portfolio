$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
. (Join-Path $projectRoot 'scripts/Preflight.ps1')

function Write-Utf8Text {
    param([string]$Path, [string]$Content)
    $parent = Split-Path -Parent $Path
    if ($parent) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    [IO.File]::WriteAllText($Path, $Content, [Text.UTF8Encoding]::new($false))
}

function Write-TestJson {
    param([string]$Path, [object]$Value)
    Write-Utf8Text $Path (($Value | ConvertTo-Json -Depth 100) + "`n")
}

function Assert-Throws {
    param([scriptblock]$ScriptBlock)
    $threw = $false
    try {
        & $ScriptBlock
    } catch {
        $threw = $true
    }
    $threw | Should Be $true
}

function New-ValidPreflightFixture {
    param([string]$Root, [string]$Environment = 'staging')
    New-Item -ItemType Directory -Path $Root -Force | Out-Null
    $region = 'uksouth'
    $endpoint = "riverside-$Environment"
    $profilePath = Join-Path $Root 'profile.yaml'
    Write-Utf8Text $profilePath @"
config_version: 1.0.0
project_name: riverside-ai-platform
environment: $Environment
region: $region
identity:
  authentication: managed_identity
model:
  alias: riverside-editor
  release_manifest_uri: https://artifacts.contoso.test/model-release.json
  max_input_tokens: 4096
  max_output_tokens: 512
  precision: fp32
data:
  contract_version: 1.0.0
  index_name: riverside.$Environment.manuscripts
  index_version: 1.0.0
gateway:
  base_url: https://riverside-gateway.contoso.test
  route: /v1/chat/completions
  timeout_seconds: 110
  max_retries: 2
serving:
  endpoint_name: $endpoint
  blue_deployment: $endpoint-blue
  green_deployment: $endpoint-green
  request_timeout_seconds: 100
retrieval:
  top_k: 6
  search_type: hybrid
telemetry:
  enabled: true
  service_name: riverside-rag-orchestrator
  exporter_endpoint: https://otel.contoso.test
evaluation:
  release_report_uri: https://artifacts.contoso.test/evaluation.json
  required_domains:
    - data_quality
    - retrieval_quality
    - generation_citation_quality
    - adaptation_evidence
    - safety_authorization
    - operational_slos
    - cost
    - rollout_comparison
"@

    $subscriptionId = '11111111-1111-1111-1111-111111111111'
    $infraPath = Join-Path $Root 'infra.json'
    $subnetPrefix = "/subscriptions/$subscriptionId/resourceGroups/network-rg/providers/Microsoft.Network/virtualNetworks/riverside-vnet/subnets"
    $dnsPrefix = "/subscriptions/$subscriptionId/resourceGroups/dns-rg/providers/Microsoft.Network/privateDnsZones"
    Write-TestJson $infraPath @{
        parameters = @{
            environment = @{ value = $Environment }
            location = @{ value = $region }
            containerAppsInfrastructureSubnetResourceId = @{ value = "$subnetPrefix/container-apps" }
            networkAccessMode = @{ value = 'private' }
            privateEndpointSubnetResourceId = @{ value = "$subnetPrefix/private-endpoints" }
            privateDnsZoneResourceIds = @{ value = @{
                azuremlApi = "$dnsPrefix/privatelink.api.azureml.ms"
                azuremlNotebooks = "$dnsPrefix/privatelink.notebooks.azure.net"
                blob = "$dnsPrefix/privatelink.blob.core.windows.net"
                dfs = "$dnsPrefix/privatelink.dfs.core.windows.net"
                vault = "$dnsPrefix/privatelink.vaultcore.azure.net"
            } }
            provisionMachineLearningWorkspace = @{ value = $true }
            machineLearningWorkspaceName = @{ value = 'riverside-mlw' }
            provisionApiManagement = @{ value = $false }
            apiManagementName = @{ value = 'shared-riverside-apim' }
            assignGatewayInvokeRole = @{ value = $true }
            existingApiManagementGatewayPrincipalId = @{ value = '22222222-2222-2222-2222-222222222222' }
            provisionLoadTesting = @{ value = $false }
            loadTestingName = @{ value = 'riverside-load-testing' }
            databricksWorkspaceSubscriptionId = @{ value = $subscriptionId }
            databricksWorkspaceResourceGroupName = @{ value = 'databricks-rg' }
            databricksWorkspaceName = @{ value = 'riverside-databricks' }
            databricksWorkspaceUrl = @{ value = 'https://adb-1234567890123456.7.azuredatabricks.net' }
            databricksCatalogName = @{ value = 'riverside' }
            databricksSchemaName = @{ value = $Environment }
            databricksVectorSearchEndpointName = @{ value = 'riverside-vector-search' }
            databricksVectorSearchIndexName = @{ value = 'riverside-manuscripts' }
            databricksEmbeddingEndpointName = @{ value = 'riverside-embeddings' }
            embeddingDimensions = @{ value = 768 }
            modelReleaseId = @{ value = 'release-2026-08-05' }
        }
    }

    $mlConfigPath = Join-Path $Root 'azureml.json'
    Write-TestJson $mlConfigPath @{
        environment = $Environment
        region = $region
        endpoint_name = $endpoint
        blue_slot_name = 'blue'
        green_slot_name = 'green'
        blue_deployment_name = "$endpoint-blue"
        green_deployment_name = "$endpoint-green"
        blue_model_name = 'riverside-model-blue'
        blue_model_version = '20260805.1'
        green_model_name = 'riverside-model-green'
        green_model_version = '20260805.2'
        environment_asset_name = 'riverside-runtime'
        environment_asset_version = '20260805.1'
        application_deadline_seconds = 100
    }

    $materialized = Join-Path $Root 'materialized'
    $endpointPath = Join-Path $materialized 'endpoint.yml'
    Write-Utf8Text $endpointPath @"
name: $endpoint
tags:
  region: $region
"@
    foreach ($slot in @('blue', 'green')) {
        $modelVersion = if ($slot -eq 'blue') { '20260805.1' } else { '20260805.2' }
        Write-Utf8Text (Join-Path $materialized "deployments/$slot.yml") @"
name: $slot
endpoint_name: $endpoint
model: azureml:riverside-model-${slot}:$modelVersion
environment: azureml:riverside-runtime:20260805.1
request_settings:
  request_timeout_ms: 180000
environment_variables:
  RIVERSIDE_DEPLOYMENT_NAME: $endpoint-$slot
  RIVERSIDE_REGION: $region
  RIVERSIDE_INDEX_VERSION: 1.0.0
tags:
  application_deadline_seconds: "100"
"@
    }
    $files = [ordered]@{
        'endpoint.yml' = Get-FileSha256 $endpointPath
        'deployments/blue.yml' = Get-FileSha256 (Join-Path $materialized 'deployments/blue.yml')
        'deployments/green.yml' = Get-FileSha256 (Join-Path $materialized 'deployments/green.yml')
    }
    Write-TestJson (Join-Path $materialized 'materialization-manifest.json') @{
        environment = $Environment
        region = $region
        endpoint_name = $endpoint
        application_deadline_seconds = 100
        azureml_outer_container_timeout_seconds = 180
        files = $files
    }

    $contract = Read-JsonObject (Join-Path $projectRoot 'apim/parameters/named-values.json')
    $namedValues = [ordered]@{}
    foreach ($entry in $contract.named_values) { $namedValues[$entry.name] = 'configured-value' }
    $namedValues['riverside-environment'] = $Environment
    $namedValues['riverside-backend-pool-id'] = 'riverside-chat-pool'
    $namedValues['riverside-content-safety-enabled'] = 'false'
    $namedValues['riverside-semantic-cache-enabled'] = 'false'
    $namedValues['riverside-backend-timeout-seconds'] = '90'
    $apimPath = Join-Path $Root 'apim.json'
    Write-TestJson $apimPath @{
        apim_service_name = 'shared-riverside-apim'
        named_values = $namedValues
        backends = @{
            blue_name = 'riverside-chat-blue'
            green_name = 'riverside-chat-green'
            pool_name = 'riverside-chat-pool'
            blue_url = 'https://blue.contoso.test'
            green_url = 'https://green.contoso.test'
        }
    }

    return @{
        Profile = $profilePath
        Infra = $infraPath
        MlConfig = $mlConfigPath
        Materialized = $materialized
        Apim = $apimPath
    }
}

Describe 'Riverside offline preflight helpers' {
    It 'rejects unresolved and live-looking placeholder values' {
        Assert-Throws { Assert-NoPlaceholders '${DATABRICKS_HOST}' 'remote.workspace_url' }
        Assert-Throws { Assert-NoPlaceholders 'https://replace-with-workspace-url.azuredatabricks.net' 'workspace_url' }
        Assert-Throws { Assert-NoPlaceholders 'https://artifacts.example.invalid/model.json' 'manifest' }
    }

    It 'accepts a complete existing subnet resource ID and rejects a placeholder subscription' {
        $valid = '/subscriptions/11111111-1111-1111-1111-111111111111/resourceGroups/network-rg/providers/Microsoft.Network/virtualNetworks/vnet/subnets/apps'
        Assert-ResourceId $valid 'subnet' 'Microsoft.Network/virtualNetworks' | Should Be $valid
        Assert-Throws { Assert-ResourceId ($valid.Replace('11111111-1111-1111-1111-111111111111', '00000000-0000-0000-0000-000000000000')) 'subnet' }
    }

    It 'rejects an APIM backend pool ID that does not match the policy named value' {
        $fixture = New-ValidPreflightFixture (Join-Path $TestDrive 'apim-mismatch')
        $apim = Read-JsonObject $fixture.Apim
        $apim.backends.pool_name = 'different-pool'
        $profile = Read-RiversideYamlObject $fixture.Profile
        $infra = Read-JsonObject $fixture.Infra
        Assert-Throws { Assert-ApimConfiguration $apim $profile $infra $projectRoot }
    }

    It 'rejects placeholder Databricks workspace values' {
        $fixture = New-ValidPreflightFixture (Join-Path $TestDrive 'databricks-placeholder')
        $profile = Read-RiversideYamlObject $fixture.Profile
        $infra = Read-JsonObject $fixture.Infra
        $infra.parameters.databricksWorkspaceName.value = 'replace-with-databricks-workspace'
        Assert-Throws { Assert-InfrastructureParameters $profile $infra }
    }
}

Describe 'Riverside offline preflight orchestration' {
    It 'passes a consistent staging fixture without Azure access' {
        $fixture = New-ValidPreflightFixture (Join-Path $TestDrive 'valid')
        $result = Invoke-RiversidePreflight $fixture.Profile $fixture.Infra $fixture.MlConfig $fixture.Materialized $fixture.Apim $false
        $result.status | Should Be 'passed'
        $result.mode | Should Be 'offline-static'
        $result.azure_commands_run | Should Be 0
    }

    It 'rejects rendered endpoint drift after materialization' {
        $fixture = New-ValidPreflightFixture (Join-Path $TestDrive 'endpoint-drift')
        $endpointPath = Join-Path $fixture.Materialized 'endpoint.yml'
        (Get-Content $endpointPath -Raw).Replace('riverside-staging', 'wrong-endpoint') | Set-Content $endpointPath -NoNewline
        Assert-Throws { Invoke-RiversidePreflight $fixture.Profile $fixture.Infra $fixture.MlConfig $fixture.Materialized $fixture.Apim $false }
    }

    It 'rejects region drift after materialization' {
        $fixture = New-ValidPreflightFixture (Join-Path $TestDrive 'region-drift')
        $mlConfig = Read-JsonObject $fixture.MlConfig
        $mlConfig.region = 'eastus2'
        Write-TestJson $fixture.MlConfig $mlConfig
        Assert-Throws { Invoke-RiversidePreflight $fixture.Profile $fixture.Infra $fixture.MlConfig $fixture.Materialized $fixture.Apim $false }
    }

    It 'rejects model and environment reference drift' {
        $fixture = New-ValidPreflightFixture (Join-Path $TestDrive 'asset-drift')
        $bluePath = Join-Path $fixture.Materialized 'deployments/blue.yml'
        (Get-Content $bluePath -Raw).Replace('azureml:riverside-model-blue:20260805.1', 'azureml:other-model:9') | Set-Content $bluePath -NoNewline
        Assert-Throws { Invoke-RiversidePreflight $fixture.Profile $fixture.Infra $fixture.MlConfig $fixture.Materialized $fixture.Apim $false }
    }

    It 'requires explicit production opt-in' {
        $fixture = New-ValidPreflightFixture (Join-Path $TestDrive 'production') 'production'
        Assert-Throws { Invoke-RiversidePreflight $fixture.Profile $fixture.Infra $fixture.MlConfig $fixture.Materialized $fixture.Apim $false }
        (Invoke-RiversidePreflight $fixture.Profile $fixture.Infra $fixture.MlConfig $fixture.Materialized $fixture.Apim $true).status | Should Be 'passed'
    }
}
