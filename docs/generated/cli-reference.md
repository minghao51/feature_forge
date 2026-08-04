# CLI reference

Generated from the live argparse command tree.

## `feature-forge`

```text
usage: feature-forge [-h] [--format {json,human}]
                     [--artifact-root ARTIFACT_ROOT]
                     [--catalog-path CATALOG_PATH]
                     [--lifecycle-root LIFECYCLE_ROOT]
                     {verify,catalog,run} ...

positional arguments:
  {verify,catalog,run}

options:
  -h, --help            show this help message and exit
  --format {json,human}
  --artifact-root ARTIFACT_ROOT
  --catalog-path CATALOG_PATH
  --lifecycle-root LIFECYCLE_ROOT
```

## `feature-forge catalog`

```text
usage: feature-forge catalog [-h] {rebuild,status} ...

positional arguments:
  {rebuild,status}

options:
  -h, --help        show this help message and exit
```

## `feature-forge catalog rebuild`

```text
usage: feature-forge catalog rebuild [-h]

options:
  -h, --help  show this help message and exit
```

## `feature-forge catalog status`

```text
usage: feature-forge catalog status [-h]

options:
  -h, --help  show this help message and exit
```

## `feature-forge run`

```text
usage: feature-forge run [-h] {plan} ...

positional arguments:
  {plan}

options:
  -h, --help  show this help message and exit
```

## `feature-forge run plan`

```text
usage: feature-forge run plan [-h] --dataset DATASET --method METHOD
                              [--model MODEL] [--seed SEED] [--metric METRIC]
                              [--artifact-policy {legacy,layer_boundaries}]
                              [--fingerprints-json FINGERPRINTS_JSON]

options:
  -h, --help            show this help message and exit
  --dataset DATASET
  --method METHOD
  --model MODEL
  --seed SEED
  --metric METRIC
  --artifact-policy {legacy,layer_boundaries}
  --fingerprints-json FINGERPRINTS_JSON
```

## `feature-forge verify`

```text
usage: feature-forge verify [-h] {run,artifact,catalog} ...

positional arguments:
  {run,artifact,catalog}

options:
  -h, --help            show this help message and exit
```

## `feature-forge verify artifact`

```text
usage: feature-forge verify artifact [-h] uri

positional arguments:
  uri

options:
  -h, --help  show this help message and exit
```

## `feature-forge verify catalog`

```text
usage: feature-forge verify catalog [-h]

options:
  -h, --help  show this help message and exit
```

## `feature-forge verify run`

```text
usage: feature-forge verify run [-h] run_id

positional arguments:
  run_id

options:
  -h, --help  show this help message and exit
```

## Exit codes

| Name | Code |
|---|---:|
| `OK` | 0 |
| `INVALID` | 10 |
| `MISSING` | 11 |
| `INCOMPATIBLE` | 12 |
| `OPERATIONAL_ERROR` | 20 |
| `USAGE` | 64 |
