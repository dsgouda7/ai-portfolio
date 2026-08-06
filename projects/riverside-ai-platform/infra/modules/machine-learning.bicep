targetScope = 'resourceGroup'

@description('Create the Azure Machine Learning workspace. When false, the named workspace must already exist.')
param provisionWorkspace bool

@description('Azure Machine Learning workspace name.')
param workspaceName string

@description('Create the managed online endpoint shell. Deployments and traffic remain owned by the Azure ML release assets.')
param provisionOnlineEndpoint bool

@description('Managed online endpoint name.')
param onlineEndpointName string

@description('Azure region for Azure Machine Learning resources.')
param location string

@description('Storage account resource ID associated with a newly created workspace.')
param storageAccountId string

@description('Key Vault resource ID associated with a newly created workspace.')
param keyVaultId string

@description('Application Insights resource ID associated with a newly created workspace.')
param applicationInsightsId string

@description('User-assigned identity resource ID used by the workspace control plane and system datastores.')
param workspaceIdentityResourceId string

@description('User-assigned identity resource ID used by the online endpoint.')
param endpointIdentityResourceId string

@description('Gateway principal granted endpoint-scoped invocation permissions.')
param gatewayPrincipalId string

@description('Application principals granted endpoint-scoped invocation permissions.')
param applicationInvokerPrincipalIds array = []

@description('Create the endpoint-scoped AzureML Data Scientist role assignment for the gateway identity.')
param assignGatewayInvokeRole bool = true

@description('Enable or disable public inbound access to the workspace and managed online endpoint.')
param allowPublicNetworkAccess bool

@description('Enable workspace diagnostic settings. Existing workspaces are modified when provisionWorkspace is false.')
param enableDiagnostics bool = true

@description('Log Analytics workspace resource ID receiving Azure ML diagnostics.')
param logAnalyticsWorkspaceId string

@description('Resource tags applied to Azure Machine Learning resources.')
param tags object

var azureMlDataScientistRoleId = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'f1a07417-d97a-45cb-824c-7a7467783830')
var publicNetworkAccess = allowPublicNetworkAccess ? 'Enabled' : 'Disabled'

resource workspace 'Microsoft.MachineLearningServices/workspaces@2025-06-01' = if (provisionWorkspace) {
  name: workspaceName
  location: location
  kind: 'Default'
  sku: {
    name: 'Basic'
    tier: 'Basic'
  }
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${workspaceIdentityResourceId}': {}
    }
  }
  tags: tags
  properties: {
    applicationInsights: applicationInsightsId
    description: 'Riverside AI Platform model registry and managed online endpoint workspace.'
    friendlyName: 'Riverside ${tags.environment} Azure ML workspace'
    keyVault: keyVaultId
    primaryUserAssignedIdentity: workspaceIdentityResourceId
    publicNetworkAccess: publicNetworkAccess
    storageAccount: storageAccountId
    systemDatastoresAuthMode: 'Identity'
    v1LegacyMode: false
  }
}

resource workspaceReference 'Microsoft.MachineLearningServices/workspaces@2025-06-01' existing = {
  name: workspaceName
}

resource workspaceDiagnostics 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = if (enableDiagnostics) {
  name: '${workspaceName}-diagnostics'
  scope: workspaceReference
  properties: {
    workspaceId: logAnalyticsWorkspaceId
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
  dependsOn: [
    workspace
  ]
}

resource onlineEndpoint 'Microsoft.MachineLearningServices/workspaces/onlineEndpoints@2025-06-01' = if (provisionOnlineEndpoint) {
  name: '${workspaceName}/${onlineEndpointName}'
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${endpointIdentityResourceId}': {}
    }
  }
  tags: tags
  properties: {
    authMode: 'AADToken'
    description: 'Riverside stable model alias endpoint. Blue/green deployments are managed separately.'
    publicNetworkAccess: publicNetworkAccess
  }
  dependsOn: [
    workspace
  ]
}

resource onlineEndpointReference 'Microsoft.MachineLearningServices/workspaces/onlineEndpoints@2025-06-01' existing = {
  name: '${workspaceName}/${onlineEndpointName}'
}

resource gatewayEndpointInvoker 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (assignGatewayInvokeRole) {
  name: guid(onlineEndpointReference.id, gatewayPrincipalId, azureMlDataScientistRoleId)
  scope: onlineEndpointReference
  properties: {
    principalId: gatewayPrincipalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: azureMlDataScientistRoleId
  }
  dependsOn: [
    onlineEndpoint
  ]
}

resource applicationEndpointInvokers 'Microsoft.Authorization/roleAssignments@2022-04-01' = [for principalId in applicationInvokerPrincipalIds: {
  name: guid(onlineEndpointReference.id, principalId, azureMlDataScientistRoleId)
  scope: onlineEndpointReference
  properties: {
    principalId: principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: azureMlDataScientistRoleId
  }
  dependsOn: [
    onlineEndpoint
  ]
}]

output workspaceId string = workspaceReference.id
output workspaceName string = workspaceName
output onlineEndpointId string = onlineEndpointReference.id
output onlineEndpointName string = onlineEndpointName
output onlineEndpointScoringUri string = onlineEndpointReference.properties.scoringUri
