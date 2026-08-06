targetScope = 'resourceGroup'

@description('Subscription containing the existing Azure Databricks workspace.')
param workspaceSubscriptionId string

@description('Resource group containing the existing Azure Databricks workspace.')
param workspaceResourceGroupName string

@description('Existing Azure Databricks workspace name. This module never creates the workspace.')
param workspaceName string

@description('Existing Azure Databricks workspace URL, such as https://adb-<id>.<region>.azuredatabricks.net. No token is accepted.')
param workspaceUrl string

@description('Unity Catalog catalog consumed by the Riverside data plane.')
param catalogName string

@description('Unity Catalog schema consumed by the Riverside data plane.')
param schemaName string

@description('Existing Databricks Vector Search endpoint name.')
param vectorSearchEndpointName string

@description('Existing Databricks Vector Search index name.')
param vectorSearchIndexName string

var workspaceResourceId = resourceId(
  workspaceSubscriptionId,
  workspaceResourceGroupName,
  'Microsoft.Databricks/workspaces',
  workspaceName
)

output workspaceResourceId string = workspaceResourceId
output workspaceName string = workspaceName
output workspaceUrl string = workspaceUrl
output catalogName string = catalogName
output schemaName string = schemaName
output vectorSearchEndpointName string = vectorSearchEndpointName
output vectorSearchIndexName string = vectorSearchIndexName
