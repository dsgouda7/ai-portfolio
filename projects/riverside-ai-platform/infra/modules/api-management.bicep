targetScope = 'resourceGroup'

@description('Create API Management. When false, the named service must already exist and is not otherwise modified unless diagnostics are enabled.')
param provisionApiManagement bool

@description('Globally unique API Management service name.')
param apiManagementName string

@description('Azure region for API Management.')
param location string

@description('Publisher display name required when creating API Management.')
param publisherName string

@description('Publisher contact email required when creating API Management. This is configuration, not a credential.')
param publisherEmail string

@description('API Management SKU name. Prefer an existing service for costly production SKUs.')
param skuName string = 'Developer'

@minValue(0)
@description('API Management capacity. Consumption requires zero; dedicated SKUs generally require at least one.')
param skuCapacity int = 1

@allowed([
  'None'
  'External'
  'Internal'
])
@description('API Management VNet mode. Internal requires a supported SKU and a dedicated subnet.')
param virtualNetworkType string = 'None'

@description('Dedicated APIM subnet resource ID when virtualNetworkType is External or Internal.')
param subnetResourceId string = ''

@description('Allow public APIM gateway access. Disable only when private connectivity and DNS are ready.')
param allowPublicNetworkAccess bool = true

@description('Attach the gateway user-assigned identity to a newly created API Management service.')
param gatewayIdentityResourceId string

@description('Enable diagnostics. When provisionApiManagement is false, this modifies the existing service diagnostic settings.')
param enableDiagnostics bool = true

@description('Log Analytics workspace resource ID receiving API Management diagnostics.')
param logAnalyticsWorkspaceId string

@description('Resource tags applied to API Management.')
param tags object

var virtualNetworkProperties = virtualNetworkType == 'None' ? {} : {
  virtualNetworkConfiguration: {
    subnetResourceId: subnetResourceId
  }
}

resource apiManagement 'Microsoft.ApiManagement/service@2024-05-01' = if (provisionApiManagement) {
  name: apiManagementName
  location: location
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${gatewayIdentityResourceId}': {}
    }
  }
  sku: {
    capacity: skuCapacity
    name: skuName
  }
  properties: union({
    publisherEmail: publisherEmail
    publisherName: publisherName
    publicNetworkAccess: allowPublicNetworkAccess ? 'Enabled' : 'Disabled'
    virtualNetworkType: virtualNetworkType
  }, virtualNetworkProperties)
}

resource apiManagementReference 'Microsoft.ApiManagement/service@2024-05-01' existing = {
  name: apiManagementName
}

resource apiManagementDiagnostics 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = if (enableDiagnostics) {
  name: '${apiManagementName}-diagnostics'
  scope: apiManagementReference
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
    apiManagement
  ]
}

output apiManagementId string = apiManagementReference.id
output apiManagementName string = apiManagementName
output gatewayBaseUrl string = 'https://${apiManagementName}.azure-api.net'
