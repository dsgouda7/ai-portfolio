targetScope = 'resourceGroup'

@description('Azure region for monitoring resources.')
param location string

@description('Stable resource-name prefix derived by the composition template.')
param namePrefix string

@description('Resource tags applied to monitoring resources.')
param tags object

@minValue(30)
@maxValue(730)
@description('Log Analytics retention in days.')
param retentionInDays int = 90

@description('Allow Log Analytics and Application Insights ingestion and query over public endpoints. Disable only when an Azure Monitor Private Link Scope is configured outside this module.')
param allowAzureMonitorPublicAccess bool = true

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${namePrefix}-law'
  location: location
  tags: tags
  properties: {
    features: {
      enableLogAccessUsingOnlyResourcePermissions: true
    }
    publicNetworkAccessForIngestion: allowAzureMonitorPublicAccess ? 'Enabled' : 'Disabled'
    publicNetworkAccessForQuery: allowAzureMonitorPublicAccess ? 'Enabled' : 'Disabled'
    retentionInDays: retentionInDays
    sku: {
      name: 'PerGB2018'
    }
  }
}

resource applicationInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: '${namePrefix}-appi'
  location: location
  kind: 'web'
  tags: tags
  properties: {
    Application_Type: 'web'
    DisableIpMasking: false
    DisableLocalAuth: true
    IngestionMode: 'LogAnalytics'
    WorkspaceResourceId: logAnalytics.id
    publicNetworkAccessForIngestion: allowAzureMonitorPublicAccess ? 'Enabled' : 'Disabled'
    publicNetworkAccessForQuery: allowAzureMonitorPublicAccess ? 'Enabled' : 'Disabled'
  }
}

output logAnalyticsWorkspaceId string = logAnalytics.id
output logAnalyticsWorkspaceName string = logAnalytics.name
output applicationInsightsId string = applicationInsights.id
output applicationInsightsName string = applicationInsights.name
