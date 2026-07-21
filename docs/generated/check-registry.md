# Validation check registry

Generated from `feature_forge.verification.checks`.

| ID | Layer | Severity | Required | Owner | Description |
|---|---|---|---|---|---|
| `BRONZE.SOURCE.CHECKSUM` | bronze | error | yes | verification | Source checksum is a valid SHA-256 digest. |
| `BRONZE.SOURCE.READABLE` | bronze | error | yes | verification | Source contains readable training rows. |
| `BRONZE.TARGET.DECLARED` | bronze | error | yes | verification | Target metadata names a non-empty column. |
| `GOLD.CANDIDATES.DECIDED` | gold | error | yes | verification | Every generated candidate has exactly one decision. |
| `GOLD.ROW_IDS.ALIGNED` | gold | error | yes | verification | Gold row IDs align exactly with Silver. |
| `PLATINUM.FOLDS.IDENTICAL` | platinum | error | yes | verification | Baseline and enhanced evidence use identical folds. |
| `PLATINUM.METRICS.RECONSTRUCTABLE` | platinum | error | yes | verification | Aggregate metrics reconstruct from fold records. |
| `PLATINUM.PREDICTIONS.JOINABLE` | platinum | error | yes | verification | Predictions join exactly to authoritative Silver row IDs. |
| `PLATINUM.UPSTREAM.VERIFIED` | platinum | error | yes | verification | Gold and Silver evidence identities are present and verified. |
| `SILVER.LEAKAGE.NO_TARGET_PROXY` | silver | error | yes | verification | No feature exactly duplicates the target. |
| `SILVER.ROW_ID.UNIQUE` | silver | error | yes | verification | Row IDs are unique and aligned with canonical rows. |
| `SILVER.SCHEMA.VALID` | silver | error | yes | verification | The canonical feature schema is non-empty and unique. |
| `SILVER.SPLIT.NON_OVERLAP` | silver | error | yes | verification | Fold assignments are exhaustive, ordered, and unique. |
| `SILVER.TARGET.COMPLETE` | silver | error | yes | verification | The canonical target contains no missing values. |
| `SILVER.TARGET.EXCLUDED` | silver | error | yes | verification | The target column is absent from canonical features. |
