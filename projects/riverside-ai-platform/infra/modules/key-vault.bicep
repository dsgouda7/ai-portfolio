targetScope = 'resourceGroup'

@description('Globally unique Key Vault name.')
param keyVaultName string

@description('Azure region for Key Vault.')
param location string

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

@description('Log Analytics workspace resource ID receiving Key Vault diagnostics.')
param logAnalyticsWorkspaceId string

@description('Principals allowed to manage secret values. This module creates no secrets.')
param secretOfficerPrincipalIds array = []

@description('Principals allowed to read secret values.')
param secretReaderPrincipalIds array = []

@description('Resource tags applied to Key Vault.')
param tags object

var keyVaultSecretsOfficerRoleId = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'b86a8fe4-44ce-4948-aee5-eccb2c155cd7')
var keyVaultSecretsUserRoleId = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '4633458b-17de-408a-b874-0445c86b69e6')

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: keyVaultName
  location: location
  tags: tags
  properties: {
    enablePurgeProtection: true
    enableRbacAuthorization: true
    enableSoftDelete: true
    enabledForDeployment: false
    enabledForDiskEncryption: false
    enabledForTemplateDeployment: false
    networkAcls: {
      bypass: 'AzureServices'
      defaultAction: networkAccessMode == 'public' ? 'Allow' : 'Deny'
      ipRules: [for cidr in allowedIpCidrs: {
        value: cidr
      }]
      virtualNetworkRules: [for subnetResourceId in allowedSubnetResourceIds: {
        id: subnetResourceId
        ignoreMissingVnetServiceEndpoint: false
      }]
    }
    publicNetworkAccess: networkAccessMode == 'private' ? 'Disabled' : 'Enabled'
    sku: {
      family: 'A'
      name: 'standard'
    }
    softDeleteRetentionInDays: 90
    tenantId: tenant().tenantId
  }
}

resource keyVaultDiagnostics 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = {
  name: '${keyVault.name}-diagnostics'
  scope: keyVault
  properties: {
    workspaceId: logAnalyticsWorkspaceId
    logs: [
      {
        categoryGroup: 'audit'
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

resource secretOfficerAssignments 'Microsoft.Authorization/roleAssignments@2022-04-01' = [for principalId in secretOfficerPrincipalIds: {
  name: guid(keyVault.id, principalId, keyVaultSecretsOfficerRoleId)
  scope: keyVault
  properties: {
    principalId: principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: keyVaultSecretsOfficerRoleId
  }
}]

resource secretReaderAssignments 'Microsoft.Authorization/roleAssignments@2022-04-01' = [for principalId in secretReaderPrincipalIds: {
  name: guid(keyVault.id, principalId, keyVaultSecretsUserRoleId)
  scope: keyVault
  properties: {
    principalId: principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: keyVaultSecretsUserRoleId
  }
}]

output keyVaultId string = keyVault.id
output keyVaultName string = keyVault.name
output keyVaultUri string = keyVault.properties.vaultUri
