targetScope = 'resourceGroup'

@description('Azure region for the Container Apps host and registry.')
param location string

@description('Stable resource-name prefix derived by the composition template.')
param namePrefix string

@description('Riverside environment selected by azd.')
param environment string

@description('Resource tags applied to the host resources.')
param tags object

@description('User-assigned identity resource ID used by the application and ACR pull.')
param applicationIdentityResourceId string

@description('User-assigned identity client ID exported to DefaultAzureCredential.')
param applicationIdentityClientId string

@description('Object ID of the APIM managed identity trusted by the application.')
param apiManagementPrincipalId string

@description('Entra application ID URI requested by APIM and validated by the application.')
param backendAudience string

@description('Delegated subnet used by the internal Container Apps managed environment.')
param infrastructureSubnetResourceId string

@description('Log Analytics workspace name receiving platform and console logs.')
param logAnalyticsWorkspaceName string

@description('Container image pinned by sha256 digest. Mutable tags are rejected even for initial provisioning.')
param orchestratorImage string

@description('Container registry name. A dedicated Entra-only Basic registry is created.')
param containerRegistryName string

@minValue(0)
@maxValue(20)
@description('Minimum ready application replicas.')
param minReplicas int

@minValue(1)
@maxValue(50)
@description('Maximum application replicas.')
param maxReplicas int

@minValue(1)
@maxValue(1000)
@description('Concurrent HTTP requests per replica before scale-out.')
param concurrentRequestsPerReplica int = 20

@description('Riverside profile path baked into the application image.')
param configPath string

@description('Azure ML managed online endpoint origin.')
param modelEndpointUrl string

@description('Azure ML blue or green deployment selected for this revision.')
param activeModelDeploymentName string

@description('Immutable model release identifier used only as a bounded telemetry dimension.')
param modelReleaseId string

@minValue(1)
@maxValue(119)
@description('Azure ML request timeout, which must remain shorter than the application deadline.')
param modelEndpointTimeoutSeconds int

@description('Existing Azure Databricks workspace HTTPS origin.')
param databricksHost string

@description('Existing Databricks Vector Search endpoint name.')
param vectorSearchEndpointName string

@description('Existing Databricks model-serving endpoint used for embeddings.')
param embeddingEndpointName string

@minValue(1)
@maxValue(65536)
@description('Pinned output dimension of the embedding endpoint.')
param embeddingDimensions int

@minValue(1)
@maxValue(119)
@description('Databricks request timeout, which must remain shorter than the application deadline.')
param databricksTimeoutSeconds int = 30

@description('OTLP HTTP collector origin. The application appends /v1/traces and /v1/metrics.')
param otelExporterEndpoint string

@description('Release manifest URI consumed by the selected profile.')
param releaseManifestUri string

@description('Evaluation report URI consumed by the selected profile.')
param evaluationReportUri string

@description('Public APIM gateway origin exported into the selected profile.')
param gatewayBaseUrl string

@description('Stable Azure ML online endpoint name exported into the selected profile.')
param servingEndpointName string

@description('Configured blue deployment name exported into the selected profile.')
param blueDeploymentName string

@description('Configured green deployment name exported into the selected profile.')
param greenDeploymentName string

var acrPullRoleId = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
var applicationPrincipalId = reference(applicationIdentityResourceId, '2023-01-31').principalId
var applicationName = take(toLower('${namePrefix}-rag'), 32)
var managedEnvironmentName = take(toLower('${namePrefix}-cae'), 32)
var orchestratorImageParts = split(toLower(orchestratorImage), '@sha256:')
var orchestratorImageDigest = last(orchestratorImageParts)
var orchestratorImageDigestNonHex = replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(orchestratorImageDigest, '0', ''), '1', ''), '2', ''), '3', ''), '4', ''), '5', ''), '6', ''), '7', ''), '8', ''), '9', ''), 'a', ''), 'b', ''), 'c', ''), 'd', ''), 'e', ''), 'f', '')

assert replicaRangeIsValid = minReplicas <= maxReplicas
assert subnetIsProvided = !empty(infrastructureSubnetResourceId)
assert backendIdentityIsConfigured = !empty(backendAudience) && !empty(apiManagementPrincipalId)
assert orchestratorImageIsImmutable = length(orchestratorImageParts) == 2 && !empty(orchestratorImageParts[0]) && length(orchestratorImageDigest) == 64 && empty(orchestratorImageDigestNonHex)

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' existing = {
  name: logAnalyticsWorkspaceName
}

resource registry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: containerRegistryName
  location: location
  tags: tags
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: false
    anonymousPullEnabled: false
    dataEndpointEnabled: false
    networkRuleBypassOptions: 'AzureServices'
    publicNetworkAccess: 'Enabled'
    zoneRedundancy: 'Disabled'
  }
}

resource registryDiagnostics 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = {
  name: '${registry.name}-diagnostics'
  scope: registry
  properties: {
    workspaceId: logAnalytics.id
    logs: [
      {
        categoryGroup: 'allLogs'
        enabled: true
      }
    ]
    metrics: [
      {
        category: 'AllMetrics'
        enabled: true
      }
    ]
  }
}

resource registryPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(registry.id, applicationPrincipalId, acrPullRoleId)
  scope: registry
  properties: {
    principalId: applicationPrincipalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: acrPullRoleId
  }
}

resource managedEnvironment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: managedEnvironmentName
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'azure-monitor'
    }
    peerAuthentication: {
      mtls: {
        enabled: true
      }
    }
    vnetConfiguration: {
      infrastructureSubnetId: infrastructureSubnetResourceId
      internal: true
    }
    zoneRedundant: environment == 'production'
  }
}

resource managedEnvironmentDiagnostics 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = {
  name: '${managedEnvironment.name}-diagnostics'
  scope: managedEnvironment
  properties: {
    workspaceId: logAnalytics.id
    logs: [
      { category: 'ContainerAppConsoleLogs', enabled: true }
      { category: 'ContainerAppSystemLogs', enabled: true }
      { category: 'ContainerAppHttpLogs', enabled: true }
    ]
    metrics: [
      {
        category: 'AllMetrics'
        enabled: true
      }
    ]
  }
}

resource application 'Microsoft.App/containerApps@2024-03-01' = {
  name: applicationName
  location: location
  tags: union(tags, {
    'azd-service-name': 'rag-orchestrator'
  })
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${applicationIdentityResourceId}': {}
    }
  }
  properties: {
    environmentId: managedEnvironment.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        allowInsecure: false
        external: true
        targetPort: 8000
        transport: 'http'
      }
      registries: [
        {
          identity: applicationIdentityResourceId
          server: registry.properties.loginServer
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'rag-orchestrator'
          image: orchestratorImage
          env: [
            { name: 'AZURE_CLIENT_ID', value: applicationIdentityClientId }
            { name: 'AZURE_ENV_NAME', value: environment }
            { name: 'AZURE_LOCATION', value: location }
            { name: 'RIVERSIDE_ENVIRONMENT', value: environment }
            { name: 'RIVERSIDE_CONFIG', value: configPath }
            { name: 'RIVERSIDE_BACKEND_TENANT_ID', value: tenant().tenantId }
            { name: 'RIVERSIDE_BACKEND_AUDIENCE', value: backendAudience }
            { name: 'RIVERSIDE_APIM_PRINCIPAL_ID', value: apiManagementPrincipalId }
            { name: 'RIVERSIDE_ENDPOINT_PROVIDER', value: 'azure_ml' }
            { name: 'RIVERSIDE_ENDPOINT_URL', value: modelEndpointUrl }
            { name: 'RIVERSIDE_ENDPOINT_ROUTE', value: '/score' }
            { name: 'RIVERSIDE_ENDPOINT_TOKEN_SCOPE', value: 'https://ml.azure.com/.default' }
            { name: 'RIVERSIDE_ENDPOINT_TIMEOUT_SECONDS', value: string(modelEndpointTimeoutSeconds) }
            { name: 'RIVERSIDE_ENDPOINT_MAX_RETRIES', value: '1' }
            { name: 'RIVERSIDE_AZUREML_DEPLOYMENT', value: activeModelDeploymentName }
            { name: 'DATABRICKS_HOST', value: databricksHost }
            { name: 'RIVERSIDE_VECTOR_SEARCH_ENDPOINT', value: vectorSearchEndpointName }
            { name: 'RIVERSIDE_EMBEDDING_ENDPOINT', value: embeddingEndpointName }
            { name: 'RIVERSIDE_EMBEDDING_DIMENSIONS', value: string(embeddingDimensions) }
            { name: 'RIVERSIDE_DATABRICKS_TIMEOUT_SECONDS', value: string(databricksTimeoutSeconds) }
            { name: 'RIVERSIDE_MODEL_RELEASE_ID', value: modelReleaseId }
            { name: 'RIVERSIDE_ACTIVE_DEPLOYMENT_NAME', value: activeModelDeploymentName }
            { name: 'RIVERSIDE_RELEASE_MANIFEST_URI', value: releaseManifestUri }
            { name: 'RIVERSIDE_EVALUATION_REPORT_URI', value: evaluationReportUri }
            { name: 'RIVERSIDE_GATEWAY_BASE_URL', value: gatewayBaseUrl }
            { name: 'RIVERSIDE_SERVING_ENDPOINT_NAME', value: servingEndpointName }
            { name: 'RIVERSIDE_BLUE_DEPLOYMENT_NAME', value: blueDeploymentName }
            { name: 'RIVERSIDE_GREEN_DEPLOYMENT_NAME', value: greenDeploymentName }
            { name: 'OTEL_EXPORTER_OTLP_ENDPOINT', value: otelExporterEndpoint }
          ]
          probes: [
            {
              type: 'Liveness'
              httpGet: {
                path: '/health'
                port: 8000
                scheme: 'HTTP'
              }
              initialDelaySeconds: 10
              periodSeconds: 10
            }
            {
              type: 'Readiness'
              httpGet: {
                path: '/ready'
                port: 8000
                scheme: 'HTTP'
              }
              initialDelaySeconds: 5
              periodSeconds: 5
            }
          ]
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
        }
      ]
      scale: {
        minReplicas: minReplicas
        maxReplicas: maxReplicas
        rules: [
          {
            name: 'http-concurrency'
            http: {
              metadata: {
                concurrentRequests: string(concurrentRequestsPerReplica)
              }
            }
          }
        ]
      }
    }
  }
  dependsOn: [
    registryPull
  ]
}

output applicationId string = application.id
output applicationName string = application.name
output applicationFqdn string = application.properties.configuration.ingress.fqdn
output applicationUrl string = 'https://${application.properties.configuration.ingress.fqdn}'
output containerRegistryEndpoint string = registry.properties.loginServer
output managedEnvironmentId string = managedEnvironment.id
