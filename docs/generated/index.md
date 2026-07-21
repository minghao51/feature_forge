# Generated pipeline references

These files are generated from live contracts, Hamilton graphs, validation metadata,
catalog DDL, and CLI definitions. Regenerate with:

```bash
uv run --extra pipeline python scripts/generate_medallion_docs.py --write
```

- [Contract reference](contracts.md)
- [Validation check registry](check-registry.md)
- [Catalog schema](catalog-schema.md)
- [CLI reference](cli-reference.md)
- [Hamilton node inventory](node-inventory.md)
- [Durable package layouts](package-layouts.md)

## DAGs

- [Silver DAG](dags.md#silver-dag)
- [Gold DAG](dags.md#gold-dag)
- [Platinum DAG](dags.md#platinum-dag)
- [End-to-end case DAG](dags.md#case-dag)

The canonical displays are portable Mermaid generated from normalized JSON topology;
both are cross-platform freshness contracts. Graphviz PNGs remain unchecked previews:
`--check` requires them but does not compare their bytes or claim that opaque image
payloads are secret-scannable.
