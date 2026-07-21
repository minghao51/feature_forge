# Installation and execution profiles

```bash
# Core library and development tooling
uv sync --group dev

# Hamilton, graph rendering, and DuckDB catalog
uv sync --extra pipeline --group dev

# Aggregate optional integrations (not required by CI)
uv sync --all-extras --group dev
```

| Profile | Network | Persistence | Hamilton cache | Artifact store required | Intended use |
|---|---:|---:|---:|---:|---|
| `development` | yes | yes | yes | no | local generation and experimentation |
| `ci` | no | no | no | no | deterministic tests |
| `production` | yes | yes | yes | yes | committed generation/evaluation |
| `replay` | no | yes | no | yes | provider-free reconstruction |
| `documentation` | no | no | no | no | graphs and generated references |

The default installation does not import Hamilton or DuckDB. Catalog commands and generated
pipeline references require `--extra pipeline`. A live provider is needed only for methods
that generate features. Replay, verification, catalog rebuild, documentation, and the
non-allow-failure docs workflow job do not need credentials. Repository branch-rule enforcement
has not been inspected.

Nested settings use the `FF_` prefix and `__` delimiter, for example
`FF_CATALOG__PATH` and `FF_TRACKER__FAILURE_POLICY`. Keep secrets in the supported secret
source; never place them in config files, fingerprints, command examples, or artifacts.
