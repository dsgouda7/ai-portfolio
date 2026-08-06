targetScope = 'resourceGroup'

@description('Globally unique storage account name.')
param storageAccountName string

@description('Azure region for the storage account.')
param location string

@description('Storage replication SKU.')
param storageSkuName string = 'Standard_ZRS'

@allowed([
  'public'
  'restricted'
  'private'
])
@description('Public allows all networks, restricted applies supplied firewall rules, and private disables public network access.')
param networkAccessMode string

@description('CIDR rules used only in restricted mode.')
param allowedIpCidrs array = []

@description('Subnet resource IDs used only in restricted mode. Service endpoints must be configured on those subnets.')
param allowedSubnetResourceIds array = []

@description('Log Analytics workspace resource ID receiving storage diagnostics.')
param logAnalyticsWorkspaceId string

@description('Principals allowed to create and update platform data and artifacts.')
param blobContributorPrincipalIds array = []

@description('Principals allowed read-only access to platform data and artifacts.')
param blobReaderPrincipalIds array = []

@description('Resource tags applied to storage resources.')
param tags object

var storageBlobDataContributorRoleId = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'ba92f5b4-2d11-453d-a403-e96b0029c9fe')
var storageBlobDataReaderRoleId = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '2a2b9908-6ea1-4ae2-8e65-a410df84e7d1')

resource storageAccount 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageAccountName
  location: location
  kind: 'StorageV2'
  sku: {
    name: storageSkuName
  }
  tags: tags
  properties: {
    accessTier: 'Hot'
    allowBlobPublicAccess: false
    allowCrossTenantReplication: false
    allowSharedKeyAccess: false
    defaultToOAuthAuthentication: true
    dnsEndpointType: 'Standard'
    encryption: {
      keySource: 'Microsoft.Storage'
      requireInfrastructureEncryption: true
      services: {
        blob: {
          enabled: true
          keyType: 'Account'
        }
        file: {
          enabled: true
          keyType: 'Account'
        }
      }
    }
    isHnsEnabled: true
    minimumTlsVersion: 'TLS1_2'
    networkAcls: {
      bypass: 'AzureServices'
      defaultAction: networkAccessMode == 'public' ? 'Allow' : 'Deny'
      ipRules: [for cidr in allowedIpCidrs: {
        action: 'Allow'
        value: cidr
      }]
      virtualNetworkRules: [for subnetResourceId in allowedSubnetResourceIds: {
        action: 'Allow'
        id: subnetResourceId
      }]
    }
    publicNetworkAccess: networkAccessMode == 'private' ? 'Disabled' : 'Enabled'
    supportsHttpsTrafficOnly: true
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  parent: storageAccount
  name: 'default'
  properties: {
    deleteRetentionPolicy: {
      enabled: true
      days: 30
    }
  }
}

resource rawContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: 'raw'
  properties: {
    publicAccess: 'None'
  }
}

resource artifactsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: 'artifacts'
  properties: {
    publicAccess: 'None'
  }
}

resource evaluationsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: 'evaluations'
  properties: {
    publicAccess: 'None'
  }
}

resource blobDiagnostics 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = {
  name: '${storageAccount.name}-blob-diagnostics'
  scope: blobService
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
}

resource blobContributorAssignments 'Microsoft.Authorization/roleAssignments@2022-04-01' = [for principalId in blobContributorPrincipalIds: {
  name: guid(storageAccount.id, principalId, storageBlobDataContributorRoleId)
  scope: storageAccount
  properties: {
    principalId: principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: storageBlobDataContributorRoleId
  }
}]

resource blobReaderAssignments 'Microsoft.Authorization/roleAssignments@2022-04-01' = [for principalId in blobReaderPrincipalIds: {
  name: guid(storageAccount.id, principalId, storageBlobDataReaderRoleId)
  scope: storageAccount
  properties: {
    principalId: principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: storageBlobDataReaderRoleId
  }
}]

output storageAccountId string = storageAccount.id
output storageAccountName string = storageAccount.name
output blobEndpoint string = storageAccount.properties.primaryEndpoints.blob
output dfsEndpoint string = storageAccount.properties.primaryEndpoints.dfs
output rawContainerUri string = '${storageAccount.properties.primaryEndpoints.dfs}raw'
output artifactsContainerUri string = '${storageAccount.properties.primaryEndpoints.dfs}artifacts'
output evaluationsContainerUri string = '${storageAccount.properties.primaryEndpoints.dfs}evaluations'
