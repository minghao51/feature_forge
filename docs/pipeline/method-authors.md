# Method and plugin authors

Existing methods may continue to implement `MethodProtocol`; they do not import Hamilton.
The coarse compatibility adapter calls `fit`, reads `generated_scripts` and
`feature_metadata`, and converts the result into ordered Gold evidence.

Methods that need exact batch ownership may expose a duck-typed `gold_generation_batches()`
method. It returns an ordered list of mappings with a `code` string and named
`specifications`. This lets a failed batch retain its own candidate/error evidence and avoids
assigning later specifications to earlier code.

Generated code runs through `SandboxedExecutor`. It never receives authoritative `row_id`;
the adapter attaches IDs after execution. Each candidate receives provenance, a validation
decision, and an owning code path. Platinum, not the method, owns metric/effect selection.

Compatibility guarantees:

- no Hamilton import requirement for third-party methods;
- legacy `ExperimentResult` fields retain their order and meaning;
- richer result fields remain optional for legacy consumers;
- method/prompt/config/code changes invalidate Gold, not Bronze or Silver;
- replay never falls back to generation when evidence is missing or corrupt.

See [Methods](../methods.md) for plugin registration and [Gold replay](../gold_dataflow.md) for
the durable adapter contract.
