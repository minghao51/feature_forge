# Verification and catalog operations

```bash
uv run feature-forge verify run <run-id>
uv run feature-forge verify artifact bronze:<run-id>:records.jsonl.gz
uv run feature-forge verify catalog
uv run feature-forge catalog status
uv run --extra pipeline feature-forge catalog rebuild
```

Global options such as `--format json`, `--artifact-root`, `--catalog-path`, and
`--lifecycle-root` appear before the command. Stable exits are `0` success, `10` invalid,
`11` missing, `12` incompatible, `20` operational failure, and `64` usage error.

Verification and status are read-only. Rebuild is the explicit mutation: it scans layer roots
deterministically, verifies semantic lineage, builds a sibling DuckDB database, fsyncs it, and
atomically replaces the old catalog. Invalid or incomplete packages are reported and omitted;
they are not deleted.

The catalog is a derived manifest-centric index. Deleting it loses no authoritative evidence.
Every row traces to a verified manifest digest. Full rebuild and incremental indexing must
produce the same logical digest.

- [Detailed catalog behavior](../catalog_verification.md)
- [Generated catalog DDL](../generated/catalog-schema.md)
- [Generated CLI help](../generated/cli-reference.md)
