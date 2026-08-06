@description('Name of an existing API Management service.')
param apimServiceName string

@description('Backend entity name for the blue RAG orchestrator deployment.')
param blueBackendName string = 'riverside-chat-blue'

@description('Backend entity name for the green RAG orchestrator deployment.')
param greenBackendName string = 'riverside-chat-green'

@description('Backend pool entity name referenced by the API policy.')
param poolBackendName string = 'riverside-chat-pool'

@description('HTTPS URL supplied by deployment for the blue backend. No endpoint is committed in source.')
param blueBackendUrl string

@description('HTTPS URL supplied by deployment for the green backend. No endpoint is committed in source.')
param greenBackendUrl string

@minValue(1)
@maxValue(100)
param blueWeight int = 100

@minValue(1)
@maxValue(100)
param greenWeight int = 1

@minValue(1)
param bluePriority int = 1

@minValue(1)
param greenPriority int = 2

@minValue(1)
@maxValue(100)
param breakerFailureCount int = 3

@description('ISO 8601 observation interval for the circuit breaker.')
param breakerFailureInterval string = 'PT1M'

@description('ISO 8601 duration for which a tripped backend remains unavailable.')
param breakerTripDuration string = 'PT1M'

resource apim 'Microsoft.ApiManagement/service@2024-05-01' existing = {
  name: apimServiceName
}

resource blueBackend 'Microsoft.ApiManagement/service/backends@2024-05-01' = {
  parent: apim
  name: blueBackendName
  properties: {
    description: 'Riverside blue RAG orchestrator deployment.'
    type: 'Single'
    protocol: 'http'
    url: blueBackendUrl
    circuitBreaker: {
      rules: [
        {
          name: 'transient-and-overload'
          failureCondition: {
            count: breakerFailureCount
            interval: breakerFailureInterval
            statusCodeRanges: [
              {
                min: 429
                max: 429
              }
              {
                min: 500
                max: 599
              }
            ]
          }
          tripDuration: breakerTripDuration
          acceptRetryAfter: true
        }
      ]
    }
    tls: {
      validateCertificateChain: true
      validateCertificateName: true
    }
  }
}

resource greenBackend 'Microsoft.ApiManagement/service/backends@2024-05-01' = {
  parent: apim
  name: greenBackendName
  properties: {
    description: 'Riverside green RAG orchestrator deployment.'
    type: 'Single'
    protocol: 'http'
    url: greenBackendUrl
    circuitBreaker: {
      rules: [
        {
          name: 'transient-and-overload'
          failureCondition: {
            count: breakerFailureCount
            interval: breakerFailureInterval
            statusCodeRanges: [
              {
                min: 429
                max: 429
              }
              {
                min: 500
                max: 599
              }
            ]
          }
          tripDuration: breakerTripDuration
          acceptRetryAfter: true
        }
      ]
    }
    tls: {
      validateCertificateChain: true
      validateCertificateName: true
    }
  }
}

resource backendPool 'Microsoft.ApiManagement/service/backends@2024-05-01' = {
  parent: apim
  name: poolBackendName
  properties: {
    description: 'Priority and weighted pool for Riverside blue/green deployments.'
    type: 'Pool'
    pool: {
      services: [
        {
          id: blueBackend.id
          priority: bluePriority
          weight: blueWeight
        }
        {
          id: greenBackend.id
          priority: greenPriority
          weight: greenWeight
        }
      ]
    }
  }
}

output backendPoolName string = backendPool.name
