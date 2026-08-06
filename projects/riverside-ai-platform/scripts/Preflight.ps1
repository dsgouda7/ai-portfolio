[CmdletBinding()]
param(
    [string]$ConfigPath,
    [string]$InfrastructureParametersPath,
    [string]$AzureMLConfigPath,
    [string]$MaterializedDirectory,
    [string]$ApimConfigPath,
    [ValidateSet('Offline', 'Static')] [string]$Mode = 'Offline',
    [switch]$ProductionOptIn
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

. "$PSScriptRoot/Common.ps1"

function Get-ObjectValue {
    param([object]$Object, [string]$Name, [string]$Context = 'input')
    if ($Object -is [System.Collections.IDictionary]) {
        if (-not $Object.Contains($Name)) { throw "Required property '$Context.$Name' is missing." }
        return $Object[$Name]
    }
    return Get-RequiredProperty $Object $Name $Context
}

function Get-ObjectNames {
    param([object]$Object)
    if ($Object -is [System.Collections.IDictionary]) { return @($Object.Keys) }
    return @($Object.PSObject.Properties.Name)
}

function ConvertFrom-RiversideYamlScalar {
    param([string]$Value)
    $text = $Value.Trim()
    if (($text.StartsWith('"') -and $text.EndsWith('"')) -or ($text.StartsWith("'") -and $text.EndsWith("'"))) {
        return $text.Substring(1, $text.Length - 2)
    }
    if ($text -ceq 'true') { return $true }
    if ($text -ceq 'false') { return $false }
    if ($text -ceq 'null') { return $null }
    $integer = 0
    if ([int]::TryParse($text, [ref]$integer)) { return $integer }
    return $text
}

function Read-RiversideYamlObject {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "YAML input does not exist: $Path" }
    $records = @()
    $lineNumber = 0
    foreach ($line in (Get-Content -LiteralPath $Path -Encoding utf8)) {
        $lineNumber++
        if ($line.Contains("`t")) { throw "YAML tabs are not supported at ${Path}:$lineNumber." }
        if ([string]::IsNullOrWhiteSpace($line) -or $line.TrimStart().StartsWith('#')) { continue }
        $indent = $line.Length - $line.TrimStart().Length
        if ($indent % 2 -ne 0) { throw "YAML indentation must use two-space levels at ${Path}:$lineNumber." }
        $records += [pscustomobject]@{ Indent = $indent; Text = $line.Trim(); Line = $lineNumber }
    }
    if ($records.Count -eq 0) { throw "YAML input is empty: $Path" }

    function Read-YamlBlock([object[]]$Items, [ref]$Index, [int]$Indent, [string]$SourcePath) {
        if ($Index.Value -ge $Items.Count -or $Items[$Index.Value].Indent -ne $Indent) {
            throw "Malformed YAML block in $SourcePath."
        }
        $isSequence = $Items[$Index.Value].Text.StartsWith('- ')
        if ($isSequence) {
            $values = @()
            while ($Index.Value -lt $Items.Count -and $Items[$Index.Value].Indent -eq $Indent) {
                $record = $Items[$Index.Value]
                if (-not $record.Text.StartsWith('- ')) { throw "Mixed YAML mapping and sequence at ${SourcePath}:$($record.Line)." }
                $itemText = $record.Text.Substring(2).Trim()
                if ([string]::IsNullOrWhiteSpace($itemText)) { throw "Nested YAML sequence items are not supported at ${SourcePath}:$($record.Line)." }
                $values += ConvertFrom-RiversideYamlScalar $itemText
                $Index.Value++
            }
            return ,$values
        }

        $mapping = [ordered]@{}
        while ($Index.Value -lt $Items.Count -and $Items[$Index.Value].Indent -eq $Indent) {
            $record = $Items[$Index.Value]
            if ($record.Text -notmatch '^([A-Za-z0-9_]+):(?:\s*(.*))?$') { throw "Unsupported YAML mapping at ${SourcePath}:$($record.Line)." }
            $name = $Matches[1]
            $tail = $Matches[2]
            if ($mapping.Contains($name)) { throw "Duplicate YAML key '$name' at ${SourcePath}:$($record.Line)." }
            $Index.Value++
            if ([string]::IsNullOrWhiteSpace($tail)) {
                if ($Index.Value -ge $Items.Count -or $Items[$Index.Value].Indent -le $Indent) { throw "YAML key '$name' has no value at ${SourcePath}:$($record.Line)." }
                if ($Items[$Index.Value].Indent -ne ($Indent + 2)) { throw "Unexpected YAML indentation after '$name' at ${SourcePath}:$($record.Line)." }
                $mapping[$name] = Read-YamlBlock $Items $Index ($Indent + 2) $SourcePath
            } else {
                $mapping[$name] = ConvertFrom-RiversideYamlScalar $tail
            }
        }
        return $mapping
    }

    $position = 0
    $result = Read-YamlBlock $records ([ref]$position) 0 $Path
    if ($position -ne $records.Count) { throw "YAML contains an unexpected indentation transition at ${Path}:$($records[$position].Line)." }
    if ($result -isnot [System.Collections.IDictionary]) { throw "YAML root must be a mapping: $Path" }
    return $result
}

function Test-RiversidePlaceholder {
    param([object]$Value)
    if ($Value -isnot [string]) { return $false }
    $text = $Value.Trim()
    return [string]::IsNullOrWhiteSpace($text) -or
        $text -match '\$\{[A-Z][A-Z0-9_]*\}' -or
        $text -match '__RIVERSIDE_[A-Z0-9_]+__' -or
        $text -match '(?i)(replace[-_ ]?with|example\.invalid|placeholder|changeme|todo|required>)' -or
        $text -match '^<[^>]+>$' -or
        $text -eq '00000000-0000-0000-0000-000000000000'
}

function Assert-NoPlaceholders {
    param([object]$Value, [string]$Path = 'input')
    if ($null -eq $Value -or $Value -is [string] -or $Value -is [System.ValueType]) {
        if (Test-RiversidePlaceholder $Value) { throw "Placeholder or unresolved value is not allowed at '$Path'." }
        return
    }
    if ($Value -is [System.Collections.IDictionary]) {
        foreach ($name in $Value.Keys) { Assert-NoPlaceholders $Value[$name] "$Path.$name" }
        return
    }
    if ($Value -is [System.Collections.IEnumerable] -and $Value -isnot [string]) {
        $index = 0
        foreach ($item in $Value) { Assert-NoPlaceholders $item "$Path[$index]"; $index++ }
        return
    }
    if (@($Value.PSObject.Properties).Count -gt 0) {
        foreach ($property in $Value.PSObject.Properties) { Assert-NoPlaceholders $property.Value "$Path.$($property.Name)" }
        return
    }
}

function Assert-GuidValue {
    param([object]$Value, [string]$Name)
    $text = Assert-NonEmptyString ([string]$Value) $Name
    $parsed = [guid]::Empty
    if (-not [guid]::TryParse($text, [ref]$parsed) -or $parsed -eq [guid]::Empty) { throw "'$Name' must be a non-zero GUID." }
    return $text
}

function Assert-AbsoluteUriValue {
    param([object]$Value, [string]$Name, [switch]$RequireHttps)
    $text = Assert-NonEmptyString ([string]$Value) $Name
    $parsed = $null
    if (-not [uri]::TryCreate($text, [UriKind]::Absolute, [ref]$parsed) -or [string]::IsNullOrWhiteSpace($parsed.Host)) { throw "'$Name' must be an absolute URI." }
    if ($RequireHttps -and $parsed.Scheme -cne 'https') { throw "'$Name' must use HTTPS." }
    return $text
}

function Assert-ResourceId {
    param([object]$Value, [string]$Name, [string]$ExpectedType = '')
    $text = Assert-NonEmptyString ([string]$Value) $Name
    Assert-NoPlaceholders $text $Name
    if ($text -notmatch '^/subscriptions/([^/]+)/resourceGroups/([^/]+)/providers/([^/]+/[^/]+)(?:/[^/]+)(?:/.*)?$') { throw "'$Name' must be a complete Azure resource ID." }
    Assert-GuidValue $Matches[1] "$Name.subscription" | Out-Null
    if ($ExpectedType -and $Matches[3] -cne $ExpectedType) { throw "'$Name' must identify resource type '$ExpectedType'." }
    return $text
}

function Assert-RiversideConfigShape {
    param([object]$Config)
    $required = @('config_version', 'project_name', 'environment', 'region', 'identity', 'model', 'data', 'gateway', 'serving', 'retrieval', 'telemetry', 'evaluation')
    $names = Get-ObjectNames $Config
    foreach ($name in $required) { if ($name -notin $names) { throw "Configuration schema requires '$name'." } }
    $unexpected = @($names | Where-Object { $_ -notin $required })
    if ($unexpected.Count -gt 0) { throw "Configuration schema rejects additional properties: $($unexpected -join ', ')." }
    if ((Get-ObjectValue $Config config_version) -cne '1.0.0') { throw 'Configuration schema requires config_version 1.0.0.' }
    if ((Get-ObjectValue $Config project_name) -cne 'riverside-ai-platform') { throw 'Configuration schema requires project_name riverside-ai-platform.' }
    if ((Get-ObjectValue $Config environment) -notin @('dev', 'staging', 'production')) { throw 'Configuration schema rejects environment.' }
    if ([string](Get-ObjectValue $Config region) -cnotmatch '^[a-z0-9-]{2,64}$') { throw 'Configuration schema rejects region.' }
    $gateway = Get-ObjectValue $Config gateway
    $serving = Get-ObjectValue $Config serving
    $model = Get-ObjectValue $Config model
    $data = Get-ObjectValue $Config data
    $identity = Get-ObjectValue $Config identity
    $retrieval = Get-ObjectValue $Config retrieval
    $telemetry = Get-ObjectValue $Config telemetry
    $evaluation = Get-ObjectValue $Config evaluation
    foreach ($entry in @(
        @{ Object = $identity; Names = @('authentication'); Section = 'identity' },
        @{ Object = $gateway; Names = @('base_url', 'route', 'timeout_seconds', 'max_retries'); Section = 'gateway' },
        @{ Object = $serving; Names = @('endpoint_name', 'blue_deployment', 'green_deployment', 'request_timeout_seconds'); Section = 'serving' },
        @{ Object = $model; Names = @('alias', 'release_manifest_uri', 'max_input_tokens', 'max_output_tokens', 'precision'); Section = 'model' },
        @{ Object = $data; Names = @('contract_version', 'index_name', 'index_version'); Section = 'data' },
        @{ Object = $retrieval; Names = @('top_k', 'search_type'); Section = 'retrieval' },
        @{ Object = $telemetry; Names = @('enabled', 'service_name', 'exporter_endpoint'); Section = 'telemetry' },
        @{ Object = $evaluation; Names = @('release_report_uri', 'required_domains'); Section = 'evaluation' }
    )) {
        $sectionNames = Get-ObjectNames $entry.Object
        foreach ($name in $entry.Names) { if ($name -notin $sectionNames) { throw "Configuration schema requires '$($entry.Section).$name'." } }
        $sectionUnexpected = @($sectionNames | Where-Object { $_ -notin $entry.Names })
        if ($sectionUnexpected.Count -gt 0) { throw "Configuration schema rejects additional $($entry.Section) properties: $($sectionUnexpected -join ', ')." }
    }
    if ((Get-ObjectValue $identity authentication identity) -notin @('managed_identity', 'workload_identity')) { throw 'Configuration schema rejects identity.authentication.' }
    if ((Get-ObjectValue $gateway route gateway) -cne '/v1/chat/completions') { throw 'Configuration schema rejects gateway.route.' }
    Assert-AbsoluteUriValue (Get-ObjectValue $gateway base_url gateway) 'gateway.base_url' | Out-Null
    Assert-AbsoluteUriValue (Get-ObjectValue $model release_manifest_uri model) 'model.release_manifest_uri' | Out-Null
    Assert-AbsoluteUriValue (Get-ObjectValue $telemetry exporter_endpoint telemetry) 'telemetry.exporter_endpoint' | Out-Null
    Assert-AbsoluteUriValue (Get-ObjectValue $evaluation release_report_uri evaluation) 'evaluation.release_report_uri' | Out-Null
    $gatewayTimeout = [int](Get-ObjectValue $gateway timeout_seconds gateway)
    $applicationDeadline = [int](Get-ObjectValue $serving request_timeout_seconds serving)
    if ($gatewayTimeout -lt 1 -or $gatewayTimeout -gt 120 -or $applicationDeadline -lt 1 -or $applicationDeadline -gt 120) { throw 'Configuration schema requires deadlines from 1 through 120 seconds.' }
    if ([int](Get-ObjectValue $gateway max_retries gateway) -lt 0 -or [int](Get-ObjectValue $gateway max_retries gateway) -gt 3) { throw 'Configuration schema rejects gateway.max_retries.' }
    if ([int](Get-ObjectValue $model max_input_tokens model) -lt 1 -or [int](Get-ObjectValue $model max_input_tokens model) -gt 8192) { throw 'Configuration schema rejects model.max_input_tokens.' }
    if ([int](Get-ObjectValue $model max_output_tokens model) -lt 1 -or [int](Get-ObjectValue $model max_output_tokens model) -gt 2048) { throw 'Configuration schema rejects model.max_output_tokens.' }
    if ([string](Get-ObjectValue $model alias model) -cnotmatch '^[a-z0-9][a-z0-9._-]{0,127}$') { throw 'Configuration schema rejects model.alias.' }
    if ((Get-ObjectValue $model precision model) -notin @('fp32', 'fp16', 'bf16', 'int8', 'int4')) { throw 'Configuration schema rejects model.precision.' }
    if ((Get-ObjectValue $data contract_version data) -cne '1.0.0') { throw 'Configuration schema rejects data.contract_version.' }
    if ([string](Get-ObjectValue $data index_name data) -cnotmatch '^[a-z0-9][a-z0-9._-]{0,127}$') { throw 'Configuration schema rejects data.index_name.' }
    if ([string](Get-ObjectValue $data index_version data) -cnotmatch '^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$') { throw 'Configuration schema rejects data.index_version.' }
    foreach ($name in @('endpoint_name', 'blue_deployment', 'green_deployment')) {
        if ([string](Get-ObjectValue $serving $name serving) -cnotmatch '^[a-z0-9][a-z0-9-]{0,127}$') { throw "Configuration schema rejects serving.$name." }
    }
    if ([int](Get-ObjectValue $retrieval top_k retrieval) -lt 1 -or [int](Get-ObjectValue $retrieval top_k retrieval) -gt 20) { throw 'Configuration schema rejects retrieval.top_k.' }
    if ((Get-ObjectValue $retrieval search_type retrieval) -notin @('similarity', 'mmr', 'hybrid')) { throw 'Configuration schema rejects retrieval.search_type.' }
    if ((Get-ObjectValue $telemetry service_name telemetry) -notin @('riverside-gateway', 'riverside-rag-orchestrator', 'riverside-model-endpoint')) { throw 'Configuration schema rejects telemetry.service_name.' }
    if ((Get-ObjectValue $telemetry enabled telemetry) -isnot [bool]) { throw 'Configuration schema requires telemetry.enabled to be boolean.' }
    $requiredDomains = @('data_quality', 'retrieval_quality', 'generation_citation_quality', 'adaptation_evidence', 'safety_authorization', 'operational_slos', 'cost', 'rollout_comparison')
    $actualDomains = @((Get-ObjectValue $evaluation required_domains evaluation))
    if ($actualDomains.Count -ne $requiredDomains.Count -or @($requiredDomains | Where-Object { $_ -notin $actualDomains }).Count -gt 0 -or @($actualDomains | Select-Object -Unique).Count -ne $actualDomains.Count) { throw 'Configuration schema rejects evaluation.required_domains.' }
}

function Get-YamlLineValue {
    param([string]$Path, [string]$Pattern, [string]$Name)
    $content = Get-Content -LiteralPath $Path -Raw -Encoding utf8
    $matches = [regex]::Matches($content, $Pattern, [Text.RegularExpressions.RegexOptions]::Multiline)
    if ($matches.Count -ne 1) { throw "Expected one '$Name' value in $Path; found $($matches.Count)." }
    return $matches[0].Groups[1].Value.Trim().Trim('"', "'")
}

function Assert-AzureMLMaterialization {
    param([object]$Profile, [object]$AzureMLConfig, [string]$Directory)
    $manifest = Read-JsonObject (Join-Path $Directory 'materialization-manifest.json')
    foreach ($property in $manifest.files.PSObject.Properties) {
        Assert-Digest (Join-Path $Directory $property.Name) ([string]$property.Value) "materialized.$($property.Name)" | Out-Null
    }
    $profileServing = Get-ObjectValue $Profile serving
    $profileData = Get-ObjectValue $Profile data
    $expectedEndpoint = [string](Get-ObjectValue $profileServing endpoint_name serving)
    $expectedEnvironment = [string](Get-ObjectValue $Profile environment)
    $expectedRegion = [string](Get-ObjectValue $Profile region)
    $expectedDeadline = [int](Get-ObjectValue $profileServing request_timeout_seconds serving)
    if ($expectedEndpoint -cne "riverside-$expectedEnvironment") { throw 'Resolved serving endpoint does not match the Bicep-derived endpoint name.' }
    foreach ($comparison in @(
        @{ Actual = [string]$AzureMLConfig.endpoint_name; Expected = $expectedEndpoint; Name = 'Azure ML config endpoint' },
        @{ Actual = [string]$manifest.endpoint_name; Expected = $expectedEndpoint; Name = 'materialization manifest endpoint' },
        @{ Actual = [string]$AzureMLConfig.region; Expected = $expectedRegion; Name = 'Azure ML config region' },
        @{ Actual = [string]$manifest.region; Expected = $expectedRegion; Name = 'materialization manifest region' },
        @{ Actual = [string]$AzureMLConfig.application_deadline_seconds; Expected = [string]$expectedDeadline; Name = 'Azure ML config deadline' },
        @{ Actual = [string]$manifest.application_deadline_seconds; Expected = [string]$expectedDeadline; Name = 'materialization manifest deadline' }
    )) {
        if ($comparison.Actual -cne $comparison.Expected) { throw "$($comparison.Name) '$($comparison.Actual)' does not match '$($comparison.Expected)'." }
    }
    $outerTimeout = [int]$manifest.azureml_outer_container_timeout_seconds
    $gatewayTimeout = [int](Get-ObjectValue (Get-ObjectValue $Profile gateway) timeout_seconds gateway)
    if ($expectedDeadline -ge $gatewayTimeout -or ($gatewayTimeout * 1000) -ge ($outerTimeout * 1000)) { throw 'Deadline order must be application < gateway < Azure ML outer timeout.' }

    $endpointFile = Join-Path $Directory 'endpoint.yml'
    if ((Get-YamlLineValue $endpointFile '^name:\s*(.+)$' 'endpoint name') -cne $expectedEndpoint) { throw 'Rendered Azure ML endpoint name is inconsistent.' }
    if ((Get-YamlLineValue $endpointFile '^\s*region:\s*(.+)$' 'endpoint region') -cne $expectedRegion) { throw 'Rendered Azure ML endpoint region is inconsistent.' }

    foreach ($slot in @('blue', 'green')) {
        $file = Join-Path $Directory "deployments/$slot.yml"
        $slotName = [string](Get-ObjectValue $AzureMLConfig "${slot}_slot_name")
        $deploymentName = [string](Get-ObjectValue $AzureMLConfig "${slot}_deployment_name")
        $profileDeploymentName = [string](Get-ObjectValue $profileServing "${slot}_deployment" serving)
        if ($deploymentName -cne $profileDeploymentName -or $deploymentName -cne "$expectedEndpoint-$slot") { throw "Resolved $slot deployment name does not match the profile and Bicep-derived name." }
        $modelName = [string](Get-ObjectValue $AzureMLConfig "${slot}_model_name")
        $modelVersion = [string](Get-ObjectValue $AzureMLConfig "${slot}_model_version")
        $expectedModel = "azureml:${modelName}:${modelVersion}"
        $expectedEnvironmentReference = "azureml:$($AzureMLConfig.environment_asset_name):$($AzureMLConfig.environment_asset_version)"
        if ((Get-YamlLineValue $file '^name:\s*(.+)$' "$slot deployment slot") -cne $slotName) { throw "Rendered $slot slot name is inconsistent." }
        if ((Get-YamlLineValue $file '^endpoint_name:\s*(.+)$' "$slot endpoint") -cne $expectedEndpoint) { throw "Rendered $slot endpoint is inconsistent." }
        if ((Get-YamlLineValue $file '^model:\s*(.+)$' "$slot model") -cne $expectedModel) { throw "Rendered $slot model reference is inconsistent." }
        if ((Get-YamlLineValue $file '^environment:\s*(.+)$' "$slot environment") -cne $expectedEnvironmentReference) { throw "Rendered $slot environment reference is inconsistent." }
        if ([int](Get-YamlLineValue $file '^\s*request_timeout_ms:\s*(.+)$' "$slot outer timeout") -ne ($outerTimeout * 1000)) { throw "Rendered $slot outer timeout is inconsistent." }
        if ((Get-YamlLineValue $file '^\s*RIVERSIDE_DEPLOYMENT_NAME:\s*(.+)$' "$slot deployment name") -cne $deploymentName) { throw "Rendered $slot deployment name is inconsistent." }
        if ((Get-YamlLineValue $file '^\s*RIVERSIDE_REGION:\s*(.+)$' "$slot region") -cne $expectedRegion) { throw "Rendered $slot region is inconsistent." }
        if ((Get-YamlLineValue $file '^\s*RIVERSIDE_INDEX_VERSION:\s*(.+)$' "$slot index version") -cne [string](Get-ObjectValue $profileData index_version data)) { throw "Rendered $slot index version is inconsistent." }
        if ([int](Get-YamlLineValue $file '^\s*application_deadline_seconds:\s*(.+)$' "$slot application deadline") -ne $expectedDeadline) { throw "Rendered $slot application deadline is inconsistent." }
    }
}

function Assert-InfrastructureParameters {
    param([object]$Profile, [object]$Parameters)
    $values = $Parameters.parameters
    $environment = [string]$values.environment.value
    if ($environment -cne [string](Get-ObjectValue $Profile environment)) { throw 'Infrastructure environment does not match the resolved profile.' }
    if ([string]$values.location.value -cne [string](Get-ObjectValue $Profile region)) { throw 'Infrastructure region does not match the resolved profile.' }
    $containerAppsSubnet = Assert-ResourceId $values.containerAppsInfrastructureSubnetResourceId.value 'parameters.containerAppsInfrastructureSubnetResourceId' 'Microsoft.Network/virtualNetworks'
    if ($containerAppsSubnet -notmatch '/subnets/[^/]+$') { throw 'parameters.containerAppsInfrastructureSubnetResourceId must identify a subnet.' }
    if ([string]$values.networkAccessMode.value -ceq 'private') {
        $privateEndpointSubnet = Assert-ResourceId $values.privateEndpointSubnetResourceId.value 'parameters.privateEndpointSubnetResourceId' 'Microsoft.Network/virtualNetworks'
        if ($privateEndpointSubnet -notmatch '/subnets/[^/]+$') { throw 'parameters.privateEndpointSubnetResourceId must identify a subnet.' }
        foreach ($name in @('azuremlApi', 'azuremlNotebooks', 'blob', 'dfs', 'vault')) {
            Assert-ResourceId $values.privateDnsZoneResourceIds.value.$name "parameters.privateDnsZoneResourceIds.$name" 'Microsoft.Network/privateDnsZones' | Out-Null
        }
    }
    if (-not [bool]$values.provisionMachineLearningWorkspace.value) { Assert-NoPlaceholders $values.machineLearningWorkspaceName.value 'parameters.machineLearningWorkspaceName' }
    if (-not [bool]$values.provisionApiManagement.value) {
        Assert-NoPlaceholders $values.apiManagementName.value 'parameters.apiManagementName'
        if ([bool]$values.assignGatewayInvokeRole.value) { Assert-GuidValue $values.existingApiManagementGatewayPrincipalId.value 'parameters.existingApiManagementGatewayPrincipalId' | Out-Null }
    }
    if (-not [bool]$values.provisionLoadTesting.value) { Assert-NoPlaceholders $values.loadTestingName.value 'parameters.loadTestingName' }
    Assert-GuidValue $values.databricksWorkspaceSubscriptionId.value 'parameters.databricksWorkspaceSubscriptionId' | Out-Null
    foreach ($name in @('databricksWorkspaceResourceGroupName', 'databricksWorkspaceName', 'databricksCatalogName', 'databricksSchemaName', 'databricksVectorSearchEndpointName', 'databricksVectorSearchIndexName', 'databricksEmbeddingEndpointName', 'modelReleaseId')) {
        Assert-NoPlaceholders $values.$name.value "parameters.$name"
    }
    $workspaceUrl = [uri]$values.databricksWorkspaceUrl.value
    Assert-NoPlaceholders $workspaceUrl.AbsoluteUri 'parameters.databricksWorkspaceUrl'
    if ($workspaceUrl.Scheme -cne 'https' -or $workspaceUrl.Host -notmatch '\.azuredatabricks\.net$') { throw 'Databricks workspace URL must be an explicit HTTPS azuredatabricks.net URL.' }
    if ([int]$values.embeddingDimensions.value -lt 1) { throw 'Databricks embeddingDimensions must be positive.' }
}

function Assert-ApimConfiguration {
    param([object]$ApimConfig, [object]$Profile, [object]$InfrastructureParameters, [string]$ProjectRoot)
    Assert-NoPlaceholders $ApimConfig 'apim'
    $contract = Read-JsonObject (Join-Path $ProjectRoot 'apim/parameters/named-values.json')
    $providedNames = @($ApimConfig.named_values.PSObject.Properties.Name)
    $requiredNames = @($contract.named_values | ForEach-Object { $_.name })
    foreach ($name in $requiredNames) { if ($name -notin $providedNames) { throw "APIM named value '$name' is missing." } }
    $extra = @($providedNames | Where-Object { $_ -notin $requiredNames })
    if ($extra.Count -gt 0) { throw "Unexpected APIM named values: $($extra -join ', ')." }
    if ($ApimConfig.named_values.'riverside-backend-pool-id' -cne $ApimConfig.backends.pool_name) { throw 'APIM backend pool named value must match backends.pool_name.' }
    if ($ApimConfig.named_values.'riverside-environment' -cne [string](Get-ObjectValue $Profile environment)) { throw 'APIM environment named value must match the resolved profile.' }
    if ($ApimConfig.apim_service_name -cne $InfrastructureParameters.parameters.apiManagementName.value) { throw 'APIM service name must match the infrastructure parameters.' }
    if ([int]$ApimConfig.named_values.'riverside-backend-timeout-seconds' -ge [int](Get-ObjectValue (Get-ObjectValue $Profile serving) request_timeout_seconds serving)) { throw 'APIM backend timeout must be lower than the application deadline.' }
    foreach ($name in @('blue_url', 'green_url')) {
        $uri = [uri]$ApimConfig.backends.$name
        if ($uri.Scheme -cne 'https' -or -not $uri.IsAbsoluteUri) { throw "APIM backends.$name must be an absolute HTTPS URL." }
    }
    if ($ApimConfig.backends.blue_name -eq $ApimConfig.backends.green_name -or $ApimConfig.backends.pool_name -in @($ApimConfig.backends.blue_name, $ApimConfig.backends.green_name)) { throw 'APIM backend IDs must be distinct.' }
    if ($ApimConfig.named_values.'riverside-content-safety-enabled' -ceq 'true') { Assert-NoPlaceholders $ApimConfig.named_values.'riverside-content-safety-backend-id' 'apim.named_values.riverside-content-safety-backend-id' }
    if ($ApimConfig.named_values.'riverside-semantic-cache-enabled' -ceq 'true') { Assert-NoPlaceholders $ApimConfig.named_values.'riverside-embeddings-backend-id' 'apim.named_values.riverside-embeddings-backend-id' }
}

function Invoke-RiversidePreflight {
    param(
        [string]$ResolvedConfigPath,
        [string]$InfraPath,
        [string]$MlConfigPath,
        [string]$RenderedMlDirectory,
        [string]$ApimPath,
        [bool]$AllowProduction
    )
    foreach ($requiredPath in @($ResolvedConfigPath, $InfraPath, $MlConfigPath, $ApimPath)) {
        if ([string]::IsNullOrWhiteSpace($requiredPath) -or -not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) { throw "Required preflight input does not exist: $requiredPath" }
    }
    if ([string]::IsNullOrWhiteSpace($RenderedMlDirectory) -or -not (Test-Path -LiteralPath $RenderedMlDirectory -PathType Container)) { throw "Required materialized directory does not exist: $RenderedMlDirectory" }
    $profile = Read-RiversideYamlObject $ResolvedConfigPath
    Assert-RiversideConfigShape $profile
    Assert-NoPlaceholders $profile 'config'
    $infra = Read-JsonObject $InfraPath
    $mlConfig = Read-JsonObject $MlConfigPath
    $apim = Read-JsonObject $ApimPath
    Assert-NoPlaceholders $mlConfig 'azureml_config'
    Assert-InfrastructureParameters $profile $infra
    Assert-AzureMLMaterialization $profile $mlConfig $RenderedMlDirectory
    $projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
    Assert-ApimConfiguration $apim $profile $infra $projectRoot
    $environments = @([string](Get-ObjectValue $profile environment), [string]$infra.parameters.environment.value, [string]$mlConfig.environment)
    if ($environments | Where-Object { $_ -ceq 'production' }) {
        if (-not $AllowProduction) { throw 'Production preflight requires explicit -ProductionOptIn.' }
        if ($environments | Where-Object { $_ -cne 'production' }) { throw 'Production environment labels must agree across all inputs.' }
    }
    [pscustomobject]@{
        mode = 'offline-static'
        status = 'passed'
        environment = [string](Get-ObjectValue $profile environment)
        region = [string](Get-ObjectValue $profile region)
        endpoint = [string](Get-ObjectValue (Get-ObjectValue $profile serving) endpoint_name serving)
        azure_commands_run = 0
    }
}

if ($MyInvocation.InvocationName -ne '.') {
    $result = Invoke-RiversidePreflight -ResolvedConfigPath $ConfigPath -InfraPath $InfrastructureParametersPath -MlConfigPath $AzureMLConfigPath -RenderedMlDirectory $MaterializedDirectory -ApimPath $ApimConfigPath -AllowProduction $ProductionOptIn.IsPresent
    $result | ConvertTo-Json -Depth 10
}
