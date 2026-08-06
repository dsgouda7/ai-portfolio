-- Parameterized reference DDL. Bind :catalog and :schema as Databricks SQL parameters.
-- ADLS access is granted through a Unity Catalog external location backed by an
-- Azure Databricks access connector managed identity; this asset contains no credential.

CREATE SCHEMA IF NOT EXISTS IDENTIFIER(:catalog || '.' || :schema);

CREATE TABLE IF NOT EXISTS IDENTIFIER(:catalog || '.' || :schema || '.raw_documents') (
  contract_version STRING NOT NULL,
  kind STRING NOT NULL,
  tenant_id STRING NOT NULL,
  document_id STRING NOT NULL,
  source_uri STRING NOT NULL,
  source_version STRING NOT NULL,
  content_hash STRING NOT NULL,
  acl STRUCT<visibility: STRING, principals: ARRAY<STRING>, groups: ARRAY<STRING>> NOT NULL,
  region STRING NOT NULL,
  classification STRING NOT NULL,
  ingested_at STRING NOT NULL,
  pipeline_version STRING NOT NULL,
  deletion_state STRUCT<status: STRING, requested_at: STRING, deleted_at: STRING> NOT NULL,
  media_type STRING NOT NULL,
  byte_size BIGINT NOT NULL,
  storage_uri STRING NOT NULL,
  source_name STRING NOT NULL,
  metadata VARIANT
) USING DELTA
TBLPROPERTIES (
  'delta.enableChangeDataFeed' = 'true',
  'riverside.contract' = 'raw-document',
  'riverside.contract.version' = '1.0.0'
);

CREATE TABLE IF NOT EXISTS IDENTIFIER(:catalog || '.' || :schema || '.parsed_documents') (
  contract_version STRING NOT NULL,
  kind STRING NOT NULL,
  tenant_id STRING NOT NULL,
  document_id STRING NOT NULL,
  source_uri STRING NOT NULL,
  source_version STRING NOT NULL,
  content_hash STRING NOT NULL,
  raw_content_hash STRING NOT NULL,
  acl STRUCT<visibility: STRING, principals: ARRAY<STRING>, groups: ARRAY<STRING>> NOT NULL,
  region STRING NOT NULL,
  classification STRING NOT NULL,
  ingested_at STRING NOT NULL,
  parsed_at STRING NOT NULL,
  pipeline_version STRING NOT NULL,
  deletion_state STRUCT<status: STRING, requested_at: STRING, deleted_at: STRING> NOT NULL,
  parser STRUCT<name: STRING, version: STRING> NOT NULL,
  language STRING NOT NULL,
  title STRING NOT NULL,
  text STRING NOT NULL,
  metadata VARIANT
) USING DELTA
TBLPROPERTIES (
  'delta.enableChangeDataFeed' = 'true',
  'riverside.contract' = 'parsed-document',
  'riverside.contract.version' = '1.0.0'
);

CREATE TABLE IF NOT EXISTS IDENTIFIER(:catalog || '.' || :schema || '.document_quarantine') (
  event_id STRING NOT NULL,
  run_id STRING NOT NULL,
  tenant_id STRING NOT NULL,
  document_id STRING,
  source_uri STRING NOT NULL,
  source_version STRING,
  raw_content_hash STRING,
  media_type STRING,
  byte_size BIGINT,
  region STRING NOT NULL,
  classification STRING NOT NULL,
  acl STRUCT<visibility: STRING, principals: ARRAY<STRING>, groups: ARRAY<STRING>> NOT NULL,
  deletion_state STRUCT<status: STRING, requested_at: STRING, deleted_at: STRING> NOT NULL,
  ingested_at STRING NOT NULL,
  pipeline_version STRING NOT NULL,
  error_code STRING NOT NULL,
  error_message STRING NOT NULL,
  retryable BOOLEAN NOT NULL
) USING DELTA
TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true');

CREATE TABLE IF NOT EXISTS IDENTIFIER(:catalog || '.' || :schema || '.ingestion_quality_reports') (
  run_id STRING NOT NULL,
  pipeline_version STRING NOT NULL,
  started_at STRING NOT NULL,
  completed_at STRING NOT NULL,
  status STRING NOT NULL,
  source_count BIGINT NOT NULL,
  parsed_count BIGINT NOT NULL,
  quarantine_count BIGINT NOT NULL,
  duplicate_input_count BIGINT NOT NULL,
  duplicate_content_count BIGINT NOT NULL,
  duplicate_existing_count BIGINT NOT NULL,
  schema_drift_count BIGINT NOT NULL,
  parse_success_rate DOUBLE NOT NULL,
  quarantine_rate DOUBLE NOT NULL,
  required_field_completeness DOUBLE NOT NULL,
  acl_coverage DOUBLE NOT NULL,
  classification_coverage DOUBLE NOT NULL,
  region_coverage DOUBLE NOT NULL,
  lineage_completeness DOUBLE NOT NULL,
  deletion_lineage_coverage DOUBLE NOT NULL,
  freshest_source_at STRING,
  freshness_lag_seconds DOUBLE,
  gates VARIANT NOT NULL
) USING DELTA;
