# Canonical pipeline DAGs

These portable Mermaid diagrams are generated from the normalized topology JSON.
The JSON and this page are freshness-checked on every docs build.

## Silver DAG

```mermaid
graph TD
    n0["artifact_store"]
    n1["bronze_checks"]
    n2["bronze_manifest"]
    n3["bronze_materialization"]
    n4["bronze_snapshot"]
    n5["canonical_features"]
    n6["canonical_target"]
    n7["dataset_fingerprint_value"]
    n8["dataset_profile"]
    n9["dataset_registry"]
    n10["dataset_request"]
    n11["environment_snapshot"]
    n12["execution_profile"]
    n13["fold_assignments"]
    n14["raw_dataset"]
    n15["row_ids"]
    n16["silver_checks"]
    n17["silver_manifest"]
    n18["silver_materialization"]
    n19["source_metadata"]
    n4 --> n1
    n1 --> n2
    n4 --> n2
    n10 --> n2
    n11 --> n2
    n19 --> n2
    n0 --> n3
    n1 --> n3
    n2 --> n3
    n4 --> n3
    n10 --> n3
    n12 --> n3
    n14 --> n3
    n10 --> n4
    n14 --> n4
    n4 --> n5
    n14 --> n5
    n4 --> n6
    n14 --> n6
    n4 --> n7
    n10 --> n7
    n4 --> n8
    n5 --> n8
    n6 --> n8
    n7 --> n8
    n5 --> n13
    n6 --> n13
    n10 --> n13
    n15 --> n13
    n9 --> n14
    n10 --> n14
    n12 --> n14
    n5 --> n15
    n4 --> n16
    n5 --> n16
    n6 --> n16
    n10 --> n16
    n13 --> n16
    n15 --> n16
    n3 --> n17
    n4 --> n17
    n7 --> n17
    n10 --> n17
    n11 --> n17
    n16 --> n17
    n19 --> n17
    n0 --> n18
    n4 --> n18
    n5 --> n18
    n6 --> n18
    n8 --> n18
    n10 --> n18
    n12 --> n18
    n13 --> n18
    n15 --> n18
    n16 --> n18
    n17 --> n18
    n4 --> n19
```

[Normalized topology JSON](graphs/silver.json)

## Gold DAG

```mermaid
graph TD
    n0["artifact_store"]
    n1["candidate_execution_batches"]
    n2["candidate_feature_specs"]
    n3["candidate_selection"]
    n4["candidate_verification"]
    n5["environment_snapshot"]
    n6["execution_profile"]
    n7["gold_manifest"]
    n8["gold_materialization"]
    n9["gold_request"]
    n10["method"]
    n11["method_generation_request"]
    n12["sandbox"]
    n13["silver_package"]
    n2 --> n1
    n12 --> n1
    n10 --> n2
    n11 --> n2
    n13 --> n2
    n4 --> n3
    n1 --> n4
    n3 --> n7
    n5 --> n7
    n11 --> n7
    n0 --> n8
    n3 --> n8
    n5 --> n8
    n6 --> n8
    n7 --> n8
    n11 --> n8
    n9 --> n11
```

[Normalized topology JSON](graphs/gold.json)

## Platinum DAG

```mermaid
graph TD
    n0["artifact_store"]
    n1["case_fingerprint"]
    n2["environment_snapshot"]
    n3["evaluation_policy"]
    n4["evaluator"]
    n5["execute_platinum"]
    n6["execution_profile"]
    n7["gold_ref"]
    n8["model_name"]
    n9["run_id"]
    n10["selection_policy"]
    n11["silver_ref"]
    n12["uncertainty_policy"]
    n0 --> n5
    n1 --> n5
    n2 --> n5
    n3 --> n5
    n4 --> n5
    n6 --> n5
    n7 --> n5
    n8 --> n5
    n9 --> n5
    n10 --> n5
    n11 --> n5
    n12 --> n5
```

[Normalized topology JSON](graphs/platinum.json)

## Case DAG

```mermaid
graph TD
    n0["artifact_store"]
    n1["bronze_checks"]
    n2["bronze_manifest"]
    n3["bronze_materialization"]
    n4["bronze_snapshot"]
    n5["candidate_execution_batches"]
    n6["candidate_feature_specs"]
    n7["candidate_selection"]
    n8["candidate_verification"]
    n9["canonical_features"]
    n10["canonical_target"]
    n11["case_fingerprint"]
    n12["dataset_fingerprint_value"]
    n13["dataset_profile"]
    n14["dataset_registry"]
    n15["dataset_request"]
    n16["environment_snapshot"]
    n17["evaluation_policy"]
    n18["evaluator"]
    n19["execute_platinum"]
    n20["execution_profile"]
    n21["fold_assignments"]
    n22["gold_manifest"]
    n23["gold_materialization"]
    n24["gold_ref"]
    n25["gold_request"]
    n26["method"]
    n27["method_generation_request"]
    n28["model_name"]
    n29["raw_dataset"]
    n30["row_ids"]
    n31["run_id"]
    n32["sandbox"]
    n33["selection_policy"]
    n34["silver_checks"]
    n35["silver_manifest"]
    n36["silver_materialization"]
    n37["silver_package"]
    n38["silver_ref"]
    n39["source_metadata"]
    n40["uncertainty_policy"]
    n4 --> n1
    n1 --> n2
    n4 --> n2
    n15 --> n2
    n16 --> n2
    n39 --> n2
    n0 --> n3
    n1 --> n3
    n2 --> n3
    n4 --> n3
    n15 --> n3
    n20 --> n3
    n29 --> n3
    n15 --> n4
    n29 --> n4
    n6 --> n5
    n32 --> n5
    n26 --> n6
    n27 --> n6
    n37 --> n6
    n8 --> n7
    n5 --> n8
    n4 --> n9
    n29 --> n9
    n4 --> n10
    n29 --> n10
    n4 --> n12
    n15 --> n12
    n4 --> n13
    n9 --> n13
    n10 --> n13
    n12 --> n13
    n0 --> n19
    n11 --> n19
    n16 --> n19
    n17 --> n19
    n18 --> n19
    n20 --> n19
    n24 --> n19
    n28 --> n19
    n31 --> n19
    n33 --> n19
    n38 --> n19
    n40 --> n19
    n9 --> n21
    n10 --> n21
    n15 --> n21
    n30 --> n21
    n7 --> n22
    n16 --> n22
    n27 --> n22
    n0 --> n23
    n7 --> n23
    n16 --> n23
    n20 --> n23
    n22 --> n23
    n27 --> n23
    n25 --> n27
    n14 --> n29
    n15 --> n29
    n20 --> n29
    n9 --> n30
    n4 --> n34
    n9 --> n34
    n10 --> n34
    n15 --> n34
    n21 --> n34
    n30 --> n34
    n3 --> n35
    n4 --> n35
    n12 --> n35
    n15 --> n35
    n16 --> n35
    n34 --> n35
    n39 --> n35
    n0 --> n36
    n4 --> n36
    n9 --> n36
    n10 --> n36
    n13 --> n36
    n15 --> n36
    n20 --> n36
    n21 --> n36
    n30 --> n36
    n34 --> n36
    n35 --> n36
    n4 --> n39
```

[Normalized topology JSON](graphs/case.json)
