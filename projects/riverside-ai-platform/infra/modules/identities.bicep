targetScope = 'resourceGroup'

@description('Azure region for the managed identities.')
param location string

@description('Stable resource-name prefix derived by the composition template.')
param namePrefix string

@description('Resource tags applied to every managed identity.')
param tags object

resource platformIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${namePrefix}-platform-id'
  location: location
  tags: tags
}

resource workspaceIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${namePrefix}-aml-workspace-id'
  location: location
  tags: tags
}

resource endpointIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${namePrefix}-aml-endpoint-id'
  location: location
  tags: tags
}

resource gatewayIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${namePrefix}-gateway-id'
  location: location
  tags: tags
}

output platform object = {
  clientId: platformIdentity.properties.clientId
  principalId: platformIdentity.properties.principalId
  resourceId: platformIdentity.id
}

output workspace object = {
  clientId: workspaceIdentity.properties.clientId
  principalId: workspaceIdentity.properties.principalId
  resourceId: workspaceIdentity.id
}

output endpoint object = {
  clientId: endpointIdentity.properties.clientId
  principalId: endpointIdentity.properties.principalId
  resourceId: endpointIdentity.id
}

output gateway object = {
  clientId: gatewayIdentity.properties.clientId
  principalId: gatewayIdentity.properties.principalId
  resourceId: gatewayIdentity.id
}
