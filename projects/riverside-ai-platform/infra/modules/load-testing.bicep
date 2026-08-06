targetScope = 'resourceGroup'

@description('Create Azure Load Testing. When false, the output points to the named existing resource.')
param provisionLoadTesting bool

@description('Azure Load Testing resource name.')
param loadTestingName string

@description('Azure region for Azure Load Testing.')
param location string

@description('Platform user-assigned identity resource ID attached to a newly created load testing resource.')
param platformIdentityResourceId string

@description('Enable diagnostics. When provisionLoadTesting is false, the named resource must already exist.')
param enableDiagnostics bool = true

@description('Log Analytics workspace resource ID receiving load test diagnostics.')
param logAnalyticsWorkspaceId string

@description('Resource tags applied to Azure Load Testing.')
param tags object

resource loadTesting 'Microsoft.LoadTestService/loadTests@2022-12-01' = if (provisionLoadTesting) {
  name: loadTestingName
  location: location
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${platformIdentityResourceId}': {}
    }
  }
  properties: {
    description: 'Bounded Riverside release and operational load tests.'
  }
}

resource loadTestingReference 'Microsoft.LoadTestService/loadTests@2022-12-01' existing = {
  name: loadTestingName
}

resource loadTestingDiagnostics 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = if (enableDiagnostics) {
  name: '${loadTestingName}-diagnostics'
  scope: loadTestingReference
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
    loadTesting
  ]
}

output loadTestingId string = loadTestingReference.id
output loadTestingName string = loadTestingName
