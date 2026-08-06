targetScope = 'resourceGroup'

@description('Private endpoint resource name.')
param privateEndpointName string

@description('Azure region for the private endpoint.')
param location string

@description('Subnet resource ID used by the private endpoint.')
param subnetResourceId string

@description('Target Azure resource ID exposed through Private Link.')
param targetResourceId string

@description('Private Link group IDs requested from the target service.')
param groupIds array

@description('Existing private DNS zone resource IDs linked to the endpoint. DNS zones are not created here.')
param privateDnsZoneResourceIds array = []

@description('Resource tags applied to the private endpoint.')
param tags object

resource privateEndpoint 'Microsoft.Network/privateEndpoints@2024-07-01' = {
  name: privateEndpointName
  location: location
  tags: tags
  properties: {
    privateLinkServiceConnections: [
      {
        name: '${privateEndpointName}-connection'
        properties: {
          groupIds: groupIds
          privateLinkServiceId: targetResourceId
          requestMessage: 'Riverside AI Platform managed private endpoint.'
        }
      }
    ]
    subnet: {
      id: subnetResourceId
    }
  }
}

resource privateDnsZoneGroup 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2024-07-01' = if (length(privateDnsZoneResourceIds) > 0) {
  parent: privateEndpoint
  name: 'default'
  properties: {
    privateDnsZoneConfigs: [for (zoneResourceId, index) in privateDnsZoneResourceIds: {
      name: 'zone-${index}'
      properties: {
        privateDnsZoneId: zoneResourceId
      }
    }]
  }
}

output privateEndpointId string = privateEndpoint.id
