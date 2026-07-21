# Hamilton node inventory

Generated from the side-effect-free documentation profile.

| Node | Layer | Owner | Cost | Persistence | Sensitivity | Dependencies |
|---|---|---|---|---|---|---|
| `artifact_store` | - | - | - | - | - | - |
| `bronze_checks` | bronze | verification | cheap | none | metadata | `bronze_snapshot` |
| `bronze_manifest` | bronze | verification | cheap | none | metadata | `bronze_checks`, `bronze_snapshot`, `dataset_request`, `environment_snapshot`, `source_metadata` |
| `bronze_materialization` | bronze | storage | io | boundary | dataset-derived | `artifact_store`, `bronze_checks`, `bronze_manifest`, `bronze_snapshot`, `dataset_request`, `execution_profile`, `raw_dataset` |
| `bronze_snapshot` | bronze | data | cheap | boundary | dataset-derived | `dataset_request`, `raw_dataset` |
| `build_platinum_request` | - | - | - | - | - | `case_fingerprint`, `evaluation_policy`, `evaluator`, `gold`, `gold_ref`, `model_name`, `run_id`, `selection_policy`, `silver`, `uncertainty_policy` |
| `candidate_execution_batches` | gold | methods | expensive | none | dataset-derived | `candidate_feature_specs`, `sandbox` |
| `candidate_feature_specs` | gold | methods | expensive | none | dataset-derived | `method`, `method_generation_request`, `silver_package` |
| `candidate_selection` | gold | methods | cheap | none | metadata | `candidate_verification` |
| `candidate_verification` | gold | verification | cheap | none | metadata | `candidate_execution_batches` |
| `canonical_features` | silver | data | cheap | boundary | dataset-derived | `bronze_snapshot`, `raw_dataset` |
| `canonical_target` | silver | data | cheap | boundary | target | `bronze_snapshot`, `raw_dataset` |
| `case_fingerprint` | - | - | - | - | - | - |
| `dataset_fingerprint_value` | silver | data | cheap | none | metadata | `bronze_snapshot`, `dataset_request` |
| `dataset_profile` | silver | data | cheap | boundary | dataset-derived | `bronze_snapshot`, `canonical_features`, `canonical_target`, `dataset_fingerprint_value` |
| `dataset_registry` | - | - | - | - | - | - |
| `dataset_request` | - | - | - | - | - | - |
| `environment` | - | - | - | - | - | - |
| `environment_snapshot` | - | - | - | - | - | - |
| `evaluation_policy` | - | - | - | - | - | - |
| `evaluator` | - | - | - | - | - | - |
| `evidence` | - | - | - | - | - | - |
| `execute_gold` | - | - | - | - | - | `environment`, `method`, `profile`, `request`, `sandbox`, `store` |
| `execute_platinum` | platinum | evaluation | expensive | boundary | - | `artifact_store`, `case_fingerprint`, `environment_snapshot`, `evaluation_policy`, `evaluator`, `execution_profile`, `gold_ref`, `model_name`, `run_id`, `selection_policy`, `silver_ref`, `uncertainty_policy` |
| `execution_profile` | - | - | - | - | - | - |
| `experiment_result_from_package` | - | - | - | - | - | `gold`, `manifest_uri`, `package`, `silver` |
| `fold_assignments` | silver | data | cheap | boundary | dataset-derived | `canonical_features`, `canonical_target`, `dataset_request`, `row_ids` |
| `gold` | - | - | - | - | - | - |
| `gold_feature_counts` | - | - | - | - | - | `evidence` |
| `gold_manifest` | gold | storage | cheap | none | metadata | `candidate_selection`, `environment_snapshot`, `method_generation_request` |
| `gold_materialization` | gold | storage | io | boundary | dataset-derived | `artifact_store`, `candidate_selection`, `environment_snapshot`, `execution_profile`, `gold_manifest`, `method_generation_request` |
| `gold_ref` | - | - | - | - | - | - |
| `gold_request` | - | - | - | - | - | - |
| `load_gold_package` | - | - | - | - | - | `ref`, `store` |
| `load_platinum_package` | - | - | - | - | - | `ref`, `store` |
| `load_silver_package` | - | - | - | - | - | `artifact_store`, `manifest_ref` |
| `manifest_ref` | - | - | - | - | - | - |
| `manifest_uri` | - | - | - | - | - | - |
| `method` | - | - | - | - | - | - |
| `method_generation_request` | gold | methods | expensive | none | dataset-derived | `gold_request` |
| `model_name` | - | - | - | - | - | - |
| `package` | - | - | - | - | - | - |
| `profile` | - | - | - | - | - | - |
| `raw_dataset` | bronze | data | io | boundary | dataset-derived | `dataset_registry`, `dataset_request`, `execution_profile` |
| `ref` | - | - | - | - | - | - |
| `replay_gold_package` | - | - | - | - | - | `ref`, `sandbox`, `silver`, `store` |
| `request` | - | - | - | - | - | - |
| `row_ids` | silver | data | cheap | boundary | dataset-derived | `canonical_features` |
| `run_id` | - | - | - | - | - | - |
| `sandbox` | - | - | - | - | - | - |
| `selection_policy` | - | - | - | - | - | - |
| `silver` | - | - | - | - | - | - |
| `silver_checks` | silver | verification | cheap | none | dataset-derived | `bronze_snapshot`, `canonical_features`, `canonical_target`, `dataset_request`, `fold_assignments`, `row_ids` |
| `silver_manifest` | silver | verification | cheap | none | metadata | `bronze_materialization`, `bronze_snapshot`, `dataset_fingerprint_value`, `dataset_request`, `environment_snapshot`, `silver_checks`, `source_metadata` |
| `silver_materialization` | silver | storage | io | boundary | dataset-derived | `artifact_store`, `bronze_snapshot`, `canonical_features`, `canonical_target`, `dataset_profile`, `dataset_request`, `execution_profile`, `fold_assignments`, `row_ids`, `silver_checks`, `silver_manifest` |
| `silver_package` | - | - | - | - | - | - |
| `silver_ref` | - | - | - | - | - | - |
| `source_metadata` | bronze | data | cheap | none | metadata | `bronze_snapshot` |
| `store` | - | - | - | - | - | - |
| `uncertainty_policy` | - | - | - | - | - | - |
