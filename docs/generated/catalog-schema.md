# Catalog schema

Schema version: `1`.

The catalog is a rebuildable derived index; verified manifests remain authoritative.

## Tables

- `artifacts`
- `catalog_metadata`
- `datasets`
- `feature_decisions`
- `features`
- `fold_metrics`
- `lineage_edges`
- `manifests`
- `run_events`
- `runs`
- `stages`
- `validation_checks`

## Live DDL

```sql
CREATE TABLE IF NOT EXISTS catalog_metadata (
  singleton BOOLEAN PRIMARY KEY, schema_version INTEGER NOT NULL,
  manifest_schema_version VARCHAR NOT NULL
);
CREATE TABLE IF NOT EXISTS manifests (
  manifest_sha256 VARCHAR PRIMARY KEY, manifest_uri VARCHAR UNIQUE NOT NULL,
  run_id VARCHAR NOT NULL, case_fingerprint VARCHAR NOT NULL, layer VARCHAR NOT NULL,
  layer_fingerprint VARCHAR NOT NULL, state VARCHAR NOT NULL, dataset VARCHAR NOT NULL,
  method VARCHAR NOT NULL, model VARCHAR NOT NULL, seed BIGINT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL, completed_at TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS runs (
  manifest_sha256 VARCHAR PRIMARY KEY, run_id VARCHAR NOT NULL, case_fingerprint VARCHAR NOT NULL,
  layer VARCHAR NOT NULL, state VARCHAR NOT NULL, created_at TIMESTAMPTZ NOT NULL,
  completed_at TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS datasets (
  manifest_sha256 VARCHAR PRIMARY KEY, dataset VARCHAR NOT NULL,
  dataset_fingerprint VARCHAR, source_identity_json VARCHAR NOT NULL
);
CREATE TABLE IF NOT EXISTS stages (
  manifest_sha256 VARCHAR NOT NULL, ordinal INTEGER NOT NULL, stage VARCHAR NOT NULL,
  state VARCHAR NOT NULL, started_at TIMESTAMPTZ NOT NULL, finished_at TIMESTAMPTZ,
  duration_ms DOUBLE, failure_class VARCHAR, error_type VARCHAR,
  PRIMARY KEY (manifest_sha256, ordinal)
);
CREATE TABLE IF NOT EXISTS artifacts (
  manifest_sha256 VARCHAR NOT NULL, relative_path VARCHAR NOT NULL, name VARCHAR NOT NULL,
  media_type VARCHAR NOT NULL, artifact_sha256 VARCHAR NOT NULL, size_bytes BIGINT NOT NULL,
  row_count BIGINT, column_count BIGINT, required BOOLEAN NOT NULL,
  PRIMARY KEY (manifest_sha256, relative_path)
);
CREATE TABLE IF NOT EXISTS validation_checks (
  manifest_sha256 VARCHAR NOT NULL, stage_ordinal INTEGER NOT NULL, check_ordinal INTEGER NOT NULL,
  check_id VARCHAR NOT NULL, severity VARCHAR NOT NULL, passed BOOLEAN NOT NULL,
  required BOOLEAN NOT NULL, message VARCHAR NOT NULL,
  PRIMARY KEY (manifest_sha256, stage_ordinal, check_ordinal)
);
CREATE TABLE IF NOT EXISTS lineage_edges (
  child_manifest_sha256 VARCHAR NOT NULL, ordinal INTEGER NOT NULL,
  parent_manifest_sha256 VARCHAR NOT NULL, parent_run_id VARCHAR NOT NULL,
  parent_layer VARCHAR NOT NULL,
  PRIMARY KEY (child_manifest_sha256, ordinal)
);
CREATE TABLE IF NOT EXISTS features (
  manifest_sha256 VARCHAR NOT NULL, candidate_id VARCHAR NOT NULL, name VARCHAR NOT NULL,
  batch_index INTEGER, code_path VARCHAR, specification_json VARCHAR NOT NULL,
  PRIMARY KEY (manifest_sha256, candidate_id)
);
CREATE TABLE IF NOT EXISTS feature_decisions (
  manifest_sha256 VARCHAR NOT NULL, decision_scope VARCHAR NOT NULL,
  candidate_id VARCHAR NOT NULL, state VARCHAR NOT NULL, reason_code VARCHAR NOT NULL,
  selected BOOLEAN, feature_name VARCHAR, evidence_json VARCHAR NOT NULL,
  PRIMARY KEY (manifest_sha256, decision_scope, candidate_id)
);
CREATE TABLE IF NOT EXISTS fold_metrics (
  manifest_sha256 VARCHAR NOT NULL, ordinal BIGINT NOT NULL, fold BIGINT,
  arm VARCHAR, candidate_id VARCHAR, score DOUBLE, record_json VARCHAR NOT NULL,
  PRIMARY KEY (manifest_sha256, ordinal)
);
CREATE TABLE IF NOT EXISTS run_events (
  event_id VARCHAR PRIMARY KEY, run_id VARCHAR NOT NULL, case_id VARCHAR NOT NULL,
  event_type VARCHAR NOT NULL, occurred_at TIMESTAMPTZ NOT NULL, state VARCHAR,
  stage VARCHAR, layer VARCHAR, manifest_sha256 VARCHAR, disposition VARCHAR,
  attempt INTEGER NOT NULL, failure_json VARCHAR, fingerprint VARCHAR,
  details_json VARCHAR NOT NULL
);
```
