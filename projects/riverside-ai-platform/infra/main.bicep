targetScope = 'resourceGroup'

@allowed([
  'dev'
  'staging'
  'production'
])
@description('Riverside deployment environment. The production project has no local profile.')
param environment string

@description('Primary Azure region for the environment.')
param location string = resourceGroup().location

@description('Short base name used to derive deterministic Azure resource names.')
param baseName string = 'riverside'

@allowed([
  'public'
  'restricted'
  'private'
])
@description('Network posture for Storage, Key Vault, Azure ML, and newly created APIM resources.')
param networkAccessMode string = 'public'

@description('CIDR allowlist for restricted mode.')
param allowedIpCidrs array = []

@description('Subnet allowlist for restricted mode. Service endpoints must be configured on these subnets.')
param allowedSubnetResourceIds array = []

@description('Subnet resource ID used for private endpoints in private mode.')
param privateEndpointSubnetResourceId string = ''

@description('Existing private DNS zone IDs. Required fields in private mode are blob, dfs, vault, azuremlApi, and azuremlNotebooks.')
param privateDnsZoneResourceIds object = {
  azuremlApi: ''
  azuremlNotebooks: ''
  blob: ''
  dfs: ''
  vault: ''
}

@description('Additional governance and cost tags. Required project and environment tags are merged over these values.')
param tags object = {}

@minValue(30)
@maxValue(730)
@description('Log Analytics retention in days.')
param logRetentionInDays int = 90

@description('Keep Log Analytics and Application Insights ingestion and query public. Disable only when an Azure Monitor Private Link Scope is already configured.')
param allowAzureMonitorPublicAccess bool = true

@description('Storage replication SKU.')
param storageSkuName string = 'Standard_ZRS'

@description('Create Azure Machine Learning workspace resources. When false, machineLearningWorkspaceName must identify an existing workspace.')
param provisionMachineLearningWorkspace bool = true

@description('Optional Azure Machine Learning workspace name override.')
param machineLearningWorkspaceName string = ''

@description('Create the stable managed online endpoint shell. Blue/green deployments remain release assets.')
param provisionOnlineEndpoint bool = true

@description('Configure diagnostics on the Azure ML workspace, including when it is existing.')
param enableMachineLearningDiagnostics bool = true

@description('Create an endpoint-scoped role assignment allowing the generated gateway identity to invoke Azure ML.')
param assignGatewayInvokeRole bool = false

@description('Create API Management. False is the default because APIM is expensive and is commonly shared.')
param provisionApiManagement bool = false

@description('Optional API Management service name override. Supply an existing globally unique name when provisionApiManagement is false.')
param apiManagementName string = ''

@description('Managed-identity principal ID already attached to a reused API Management service. Required for endpoint RBAC when provisionApiManagement is false.')
param existingApiManagementGatewayPrincipalId string = ''

@description('API Management publisher name used only when creating the service.')
param apiManagementPublisherName string = 'Riverside AI Platform Operations'

@description('API Management publisher email used only when creating the service. This is not a credential.')
param apiManagementPublisherEmail string = 'azure-admin@example.invalid'

@description('API Management SKU name.')
param apiManagementSkuName string = 'Developer'

@description('API Management capacity. Consumption requires zero.')
param apiManagementSkuCapacity int = 1

@allowed([
  'None'
  'External'
  'Internal'
])
@description('API Management VNet mode for newly created services.')
param apiManagementVirtualNetworkType string = 'None'

@description('Dedicated API Management subnet resource ID for External or Internal VNet mode.')
param apiManagementSubnetResourceId string = ''

@description('Configure diagnostics on API Management. Enable for existing APIM only when this deployment should own its diagnostic setting name.')
param enableApiManagementDiagnostics bool = false

@description('Create Azure Load Testing for this environment.')
param provisionLoadTesting bool = false

@description('Optional Azure Load Testing resource name override.')
param loadTestingName string = ''

@description('Configure diagnostics on Azure Load Testing. Enable for an existing resource only when this deployment should own its diagnostic setting name.')
param enableLoadTestingDiagnostics bool = false

@description('Subscription containing the existing Azure Databricks workspace.')
param databricksWorkspaceSubscriptionId string = subscription().subscriptionId

@description('Resource group containing the existing Azure Databricks workspace.')
param databricksWorkspaceResourceGroupName string = resourceGroup().name

@description('Existing Azure Databricks workspace name. No workspace or token is created by this template.')
param databricksWorkspaceName string = 'replace-with-existing-databricks'

@description('Existing Azure Databricks workspace URL. No token is accepted.')
param databricksWorkspaceUrl string = 'https://replace-with-workspace-url.azuredatabricks.net'

@description('Unity Catalog catalog consumed by Riverside.')
param databricksCatalogName string = 'riverside'

@description('Unity Catalog schema consumed by Riverside.')
param databricksSchemaName string = environment

@description('Existing Databricks Vector Search endpoint name.')
param databricksVectorSearchEndpointName string = 'riverside-vector-search'

@description('Existing Databricks Vector Search index name.')
param databricksVectorSearchIndexName string = 'riverside_manuscripts'

@description('Immutable model release manifest URI exported to application configuration.')
param releaseManifestUri string = 'https://artifacts.example.invalid/riverside/model-release.json'

@description('Machine-readable evaluation report URI exported to application configuration.')
param evaluationReportUri string = 'https://artifacts.example.invalid/riverside/evaluation-release-report.json'

@description('Application-owned OTLP collector endpoint. Application Insights uses Azure Monitor OpenTelemetry with Entra authentication and is exported separately by resource ID.')
param otelExporterEndpoint string = 'https://otel-collector.example.invalid'

@description('Delegated subnet for the internal Azure Container Apps managed environment. APIM must have private network reachability and DNS resolution to this environment.')
param containerAppsInfrastructureSubnetResourceId string

@description('Entra application ID URI requested by APIM and validated by the RAG orchestrator.')
param orchestratorBackendAudience string

@description('Initial image used while provisioning the Container App. azd replaces this image during deployment.')
param orchestratorImage string = 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'

@description('Optional globally unique ACR name override.')
param containerRegistryName string = ''

@minValue(0)
@maxValue(20)
@description('Minimum ready RAG orchestrator replicas.')
param orchestratorMinReplicas int = 1

@minValue(1)
@maxValue(50)
@description('Maximum RAG orchestrator replicas.')
param orchestratorMaxReplicas int = 5

@description('Existing Databricks model-serving endpoint used to create query embeddings.')
param databricksEmbeddingEndpointName string

@minValue(1)
@maxValue(65536)
@description('Pinned embedding vector dimensions.')
param embeddingDimensions int

@description('Immutable model release identifier used as a bounded telemetry dimension.')
param modelReleaseId string

@allowed([
  'blue'
  'green'
])
@description('Azure ML deployment slot selected by this application revision.')
param activeModelSlot string = 'blue'

var resourceToken = take(toLower(uniqueString(subscription().subscriptionId, resourceGroup().id, environment)), 6)
var resourceNamePrefix = take(toLower('${baseName}-${environment}-${resourceToken}'), 32)
var storageAccountName = take(toLower(replace('${baseName}${environment}${resourceToken}', '-', '')), 24)
var keyVaultName = take(toLower('${baseName}-${environment}-${resourceToken}-kv'), 24)
var resolvedMachineLearningWorkspaceName = empty(machineLearningWorkspaceName)
  ? take(toLower('${baseName}-${environment}-${resourceToken}-mlw'), 33)
  : machineLearningWorkspaceName
var onlineEndpointName = 'riverside-${environment}'
var blueDeploymentName = '${onlineEndpointName}-blue'
var greenDeploymentName = '${onlineEndpointName}-green'
var resolvedApiManagementName = empty(apiManagementName)
  ? take(toLower('${baseName}-${environment}-${resourceToken}-apim'), 50)
  : apiManagementName
var gatewayInvokerPrincipalId = provisionApiManagement
  ? identities.outputs.gateway.principalId
  : existingApiManagementGatewayPrincipalId
var resolvedLoadTestingName = empty(loadTestingName)
  ? take(toLower('${baseName}-${environment}-${resourceToken}-load'), 64)
  : loadTestingName
var resolvedContainerRegistryName = empty(containerRegistryName)
  ? take(toLower(replace('${baseName}${environment}${resourceToken}acr', '-', '')), 50)
  : containerRegistryName
var activeModelDeploymentName = activeModelSlot == 'blue' ? blueDeploymentName : greenDeploymentName
var commonTags = union(tags, {
  environment: environment
  managedBy: 'azd-bicep'
  project: 'riverside-ai-platform'
})
var privateDnsZones = {
  azureml: filter([
    privateDnsZoneResourceIds.azuremlApi
    privateDnsZoneResourceIds.azuremlNotebooks
  ], zoneResourceId => !empty(zoneResourceId))
  blob: filter([
    privateDnsZoneResourceIds.blob
  ], zoneResourceId => !empty(zoneResourceId))
  dfs: filter([
    privateDnsZoneResourceIds.dfs
  ], zoneResourceId => !empty(zoneResourceId))
  vault: filter([
    privateDnsZoneResourceIds.vault
  ], zoneResourceId => !empty(zoneResourceId))
}

assert restrictedNetworkRulesProvided = networkAccessMode != 'restricted' || length(allowedIpCidrs) + length(allowedSubnetResourceIds) > 0
assert privateEndpointSubnetProvided = networkAccessMode != 'private' || !empty(privateEndpointSubnetResourceId)
assert privateDnsZonesProvided = networkAccessMode != 'private' || (
  length(privateDnsZones.blob) == 1 &&
  length(privateDnsZones.dfs) == 1 &&
  length(privateDnsZones.vault) == 1 &&
  length(privateDnsZones.azureml) == 2
)
assert apiManagementSubnetProvided = !provisionApiManagement || apiManagementVirtualNetworkType == 'None' || !empty(apiManagementSubnetResourceId)
assert privateApiManagementUsesVnet = !provisionApiManagement || networkAccessMode != 'private' || apiManagementVirtualNetworkType == 'Internal'
assert existingApiManagementIdentityProvided = provisionApiManagement || !assignGatewayInvokeRole || !empty(existingApiManagementGatewayPrincipalId)
assert containerAppsSubnetProvided = !empty(containerAppsInfrastructureSubnetResourceId)
assert orchestratorAudienceProvided = !empty(orchestratorBackendAudience)

module identities './modules/identities.bicep' = {
  name: 'identities'
  params: {
    location: location
    namePrefix: resourceNamePrefix
    tags: commonTags
  }
}

module monitoring './modules/monitoring.bicep' = {
  name: 'monitoring'
  params: {
    allowAzureMonitorPublicAccess: allowAzureMonitorPublicAccess
    location: location
    namePrefix: resourceNamePrefix
    retentionInDays: logRetentionInDays
    tags: commonTags
  }
}

module storage './modules/storage.bicep' = {
  name: 'storage'
  params: {
    allowedIpCidrs: allowedIpCidrs
    allowedSubnetResourceIds: allowedSubnetResourceIds
    blobContributorPrincipalIds: [
      identities.outputs.platform.principalId
      identities.outputs.workspace.principalId
    ]
    blobReaderPrincipalIds: [
      identities.outputs.endpoint.principalId
    ]
    location: location
    logAnalyticsWorkspaceId: monitoring.outputs.logAnalyticsWorkspaceId
    networkAccessMode: networkAccessMode
    storageAccountName: storageAccountName
    storageSkuName: storageSkuName
    tags: commonTags
  }
}

module keyVault './modules/key-vault.bicep' = {
  name: 'key-vault'
  params: {
    allowedIpCidrs: allowedIpCidrs
    allowedSubnetResourceIds: allowedSubnetResourceIds
    keyVaultName: keyVaultName
    location: location
    logAnalyticsWorkspaceId: monitoring.outputs.logAnalyticsWorkspaceId
    networkAccessMode: networkAccessMode
    secretOfficerPrincipalIds: [
      identities.outputs.workspace.principalId
    ]
    secretReaderPrincipalIds: [
      identities.outputs.platform.principalId
      identities.outputs.endpoint.principalId
    ]
    tags: commonTags
  }
}

module machineLearning './modules/machine-learning.bicep' = {
  name: 'machine-learning'
  params: {
    allowPublicNetworkAccess: networkAccessMode != 'private'
    applicationInvokerPrincipalIds: [
      identities.outputs.platform.principalId
    ]
    applicationInsightsId: monitoring.outputs.applicationInsightsId
    assignGatewayInvokeRole: assignGatewayInvokeRole
    enableDiagnostics: enableMachineLearningDiagnostics
    endpointIdentityResourceId: identities.outputs.endpoint.resourceId
    gatewayPrincipalId: gatewayInvokerPrincipalId
    keyVaultId: keyVault.outputs.keyVaultId
    location: location
    logAnalyticsWorkspaceId: monitoring.outputs.logAnalyticsWorkspaceId
    onlineEndpointName: onlineEndpointName
    provisionOnlineEndpoint: provisionOnlineEndpoint
    provisionWorkspace: provisionMachineLearningWorkspace
    storageAccountId: storage.outputs.storageAccountId
    tags: commonTags
    workspaceIdentityResourceId: identities.outputs.workspace.resourceId
    workspaceName: resolvedMachineLearningWorkspaceName
  }
}

module apiManagement './modules/api-management.bicep' = {
  name: 'api-management'
  params: {
    allowPublicNetworkAccess: networkAccessMode != 'private'
    apiManagementName: resolvedApiManagementName
    enableDiagnostics: enableApiManagementDiagnostics
    gatewayIdentityResourceId: identities.outputs.gateway.resourceId
    location: location
    logAnalyticsWorkspaceId: monitoring.outputs.logAnalyticsWorkspaceId
    provisionApiManagement: provisionApiManagement
    publisherEmail: apiManagementPublisherEmail
    publisherName: apiManagementPublisherName
    skuCapacity: apiManagementSkuCapacity
    skuName: apiManagementSkuName
    subnetResourceId: apiManagementSubnetResourceId
    tags: commonTags
    virtualNetworkType: apiManagementVirtualNetworkType
  }
}

module loadTesting './modules/load-testing.bicep' = {
  name: 'load-testing'
  params: {
    enableDiagnostics: enableLoadTestingDiagnostics
    loadTestingName: resolvedLoadTestingName
    location: location
    logAnalyticsWorkspaceId: monitoring.outputs.logAnalyticsWorkspaceId
    platformIdentityResourceId: identities.outputs.platform.resourceId
    provisionLoadTesting: provisionLoadTesting
    tags: commonTags
  }
}

module databricks './modules/databricks-integration.bicep' = {
  name: 'databricks-integration'
  params: {
    catalogName: databricksCatalogName
    schemaName: databricksSchemaName
    vectorSearchEndpointName: databricksVectorSearchEndpointName
    vectorSearchIndexName: databricksVectorSearchIndexName
    workspaceName: databricksWorkspaceName
    workspaceResourceGroupName: databricksWorkspaceResourceGroupName
    workspaceSubscriptionId: databricksWorkspaceSubscriptionId
    workspaceUrl: databricksWorkspaceUrl
  }
}

module containerAppHost './modules/container-app-host.bicep' = {
  name: 'container-app-host'
  params: {
    activeModelDeploymentName: activeModelDeploymentName
    apiManagementPrincipalId: gatewayInvokerPrincipalId
    applicationIdentityClientId: identities.outputs.platform.clientId
    applicationIdentityResourceId: identities.outputs.platform.resourceId
    backendAudience: orchestratorBackendAudience
    blueDeploymentName: blueDeploymentName
    configPath: '/app/config/${environment}.yaml'
    containerRegistryName: resolvedContainerRegistryName
    databricksHost: databricks.outputs.workspaceUrl
    databricksTimeoutSeconds: 30
    embeddingDimensions: embeddingDimensions
    embeddingEndpointName: databricksEmbeddingEndpointName
    environment: environment
    evaluationReportUri: evaluationReportUri
    gatewayBaseUrl: apiManagement.outputs.gatewayBaseUrl
    greenDeploymentName: greenDeploymentName
    infrastructureSubnetResourceId: containerAppsInfrastructureSubnetResourceId
    location: location
    logAnalyticsWorkspaceName: monitoring.outputs.logAnalyticsWorkspaceName
    maxReplicas: orchestratorMaxReplicas
    minReplicas: orchestratorMinReplicas
    modelEndpointTimeoutSeconds: environment == 'production' ? 90 : (environment == 'staging' ? 70 : 40)
    modelEndpointUrl: replace(machineLearning.outputs.onlineEndpointScoringUri, '/score', '')
    modelReleaseId: modelReleaseId
    namePrefix: resourceNamePrefix
    orchestratorImage: orchestratorImage
    otelExporterEndpoint: otelExporterEndpoint
    releaseManifestUri: releaseManifestUri
    servingEndpointName: onlineEndpointName
    tags: commonTags
    vectorSearchEndpointName: databricksVectorSearchEndpointName
  }
}

module storageBlobPrivateEndpoint './modules/private-endpoint.bicep' = if (networkAccessMode == 'private') {
  name: 'storage-blob-private-endpoint'
  params: {
    groupIds: [
      'blob'
    ]
    location: location
    privateDnsZoneResourceIds: privateDnsZones.blob
    privateEndpointName: '${resourceNamePrefix}-blob-pe'
    subnetResourceId: privateEndpointSubnetResourceId
    tags: commonTags
    targetResourceId: storage.outputs.storageAccountId
  }
}

module storageDfsPrivateEndpoint './modules/private-endpoint.bicep' = if (networkAccessMode == 'private') {
  name: 'storage-dfs-private-endpoint'
  params: {
    groupIds: [
      'dfs'
    ]
    location: location
    privateDnsZoneResourceIds: privateDnsZones.dfs
    privateEndpointName: '${resourceNamePrefix}-dfs-pe'
    subnetResourceId: privateEndpointSubnetResourceId
    tags: commonTags
    targetResourceId: storage.outputs.storageAccountId
  }
}

module keyVaultPrivateEndpoint './modules/private-endpoint.bicep' = if (networkAccessMode == 'private') {
  name: 'key-vault-private-endpoint'
  params: {
    groupIds: [
      'vault'
    ]
    location: location
    privateDnsZoneResourceIds: privateDnsZones.vault
    privateEndpointName: '${resourceNamePrefix}-vault-pe'
    subnetResourceId: privateEndpointSubnetResourceId
    tags: commonTags
    targetResourceId: keyVault.outputs.keyVaultId
  }
}

module machineLearningPrivateEndpoint './modules/private-endpoint.bicep' = if (networkAccessMode == 'private') {
  name: 'machine-learning-private-endpoint'
  params: {
    groupIds: [
      'amlworkspace'
    ]
    location: location
    privateDnsZoneResourceIds: privateDnsZones.azureml
    privateEndpointName: '${resourceNamePrefix}-aml-pe'
    subnetResourceId: privateEndpointSubnetResourceId
    tags: commonTags
    targetResourceId: machineLearning.outputs.workspaceId
  }
}

output AZURE_LOCATION string = location
output AZURE_RESOURCE_GROUP string = resourceGroup().name
output AZURE_SUBSCRIPTION_ID string = subscription().subscriptionId
output AZUREML_WORKSPACE_NAME string = machineLearning.outputs.workspaceName
output RIVERSIDE_SERVING_ENDPOINT_NAME string = machineLearning.outputs.onlineEndpointName
output RIVERSIDE_BLUE_DEPLOYMENT_NAME string = blueDeploymentName
output RIVERSIDE_GREEN_DEPLOYMENT_NAME string = greenDeploymentName
output RIVERSIDE_GATEWAY_BASE_URL string = apiManagement.outputs.gatewayBaseUrl
output RIVERSIDE_RELEASE_MANIFEST_URI string = releaseManifestUri
output RIVERSIDE_EVALUATION_REPORT_URI string = evaluationReportUri
output OTEL_EXPORTER_OTLP_ENDPOINT string = otelExporterEndpoint
output APPLICATIONINSIGHTS_RESOURCE_ID string = monitoring.outputs.applicationInsightsId
output LOG_ANALYTICS_WORKSPACE_ID string = monitoring.outputs.logAnalyticsWorkspaceId
output RIVERSIDE_STORAGE_ACCOUNT_ID string = storage.outputs.storageAccountId
output RIVERSIDE_ADLS_DFS_ENDPOINT string = storage.outputs.dfsEndpoint
output RIVERSIDE_ARTIFACTS_URI string = storage.outputs.artifactsContainerUri
output RIVERSIDE_EVALUATIONS_URI string = storage.outputs.evaluationsContainerUri
output RIVERSIDE_KEY_VAULT_URI string = keyVault.outputs.keyVaultUri
output RIVERSIDE_LOAD_TESTING_RESOURCE_ID string = loadTesting.outputs.loadTestingId
output RIVERSIDE_PLATFORM_IDENTITY_CLIENT_ID string = identities.outputs.platform.clientId
output RIVERSIDE_ENDPOINT_IDENTITY_CLIENT_ID string = identities.outputs.endpoint.clientId
output RIVERSIDE_GATEWAY_IDENTITY_CLIENT_ID string = identities.outputs.gateway.clientId
output RIVERSIDE_GATEWAY_INVOKER_PRINCIPAL_ID string = gatewayInvokerPrincipalId
output DATABRICKS_WORKSPACE_RESOURCE_ID string = databricks.outputs.workspaceResourceId
output DATABRICKS_WORKSPACE_URL string = databricks.outputs.workspaceUrl
output DATABRICKS_CATALOG string = databricks.outputs.catalogName
output DATABRICKS_SCHEMA string = databricks.outputs.schemaName
output DATABRICKS_VECTOR_SEARCH_ENDPOINT string = databricks.outputs.vectorSearchEndpointName
output DATABRICKS_VECTOR_SEARCH_INDEX string = databricks.outputs.vectorSearchIndexName
output RIVERSIDE_ORCHESTRATOR_NAME string = containerAppHost.outputs.applicationName
output RIVERSIDE_ORCHESTRATOR_RESOURCE_ID string = containerAppHost.outputs.applicationId
output RIVERSIDE_ORCHESTRATOR_FQDN string = containerAppHost.outputs.applicationFqdn
output RIVERSIDE_ORCHESTRATOR_URL string = containerAppHost.outputs.applicationUrl
output RIVERSIDE_BACKEND_AUDIENCE string = orchestratorBackendAudience
output RIVERSIDE_APIM_PRINCIPAL_ID string = gatewayInvokerPrincipalId
output RIVERSIDE_MODEL_RELEASE_ID string = modelReleaseId
output RIVERSIDE_ACTIVE_DEPLOYMENT_NAME string = activeModelDeploymentName
output AZURE_CONTAINER_REGISTRY_ENDPOINT string = containerAppHost.outputs.containerRegistryEndpoint
