# Durable package layouts

Synthetic directory examples generated from live package contracts. `_SUCCESS` contains the committed manifest SHA-256.

## Bronze reference

```text
_SUCCESS
bronze.json
manifest.json
source_reference.json
```

## Bronze snapshot

```text
_SUCCESS
bronze.json
manifest.json
test.parquet
train.parquet
```

## Silver

```text
_SUCCESS
canonical_features.parquet
canonical_target.parquet
checks.json
fold_assignments.parquet
manifest.json
profile.json
row_ids.parquet
```

## Gold

```text
_SUCCESS
accepted_features.parquet
candidate_features.parquet
candidates.json
checks.json
decisions.json
dependencies.json
generated_code/*.py
manifest.json
provenance.json
request.json
```

## Platinum

```text
_SUCCESS
aggregate_metrics.json
checks.json
fold_assignments.parquet
fold_metrics.parquet
manifest.json
predictions.parquet
report.json
request.json
selection_decisions.json
uncertainty.json
```
