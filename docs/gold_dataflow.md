# Gold offline replay

Gold is the durable boundary around generated feature evidence. It consumes a
verified Silver package, executes generated code in the existing sandbox, and
commits row-ID-keyed candidate and validation-accepted feature matrices.

Gold uses two identities. The input fingerprint is available before generation
and excludes the outer evaluation model and metric. The materialization
fingerprint additionally binds generated specifications, code hashes,
provenance, and decisions. Final metric/effect selection is a Platinum concern.

Every package contains its request, candidates, provenance, decisions,
accepted features, checks, dependency metadata, and human-readable Python
batches. The candidate matrix may be omitted only when the request explicitly
disables it; all explanatory evidence remains required.

Offline replay verifies the Gold manifest and upstream Silver manifest, then
executes stored batches in order against a cumulative working frame. The
authoritative Silver `row_id` is attached after sandbox execution and is never
exposed to generated code. Schema, dtypes, row IDs, and values are compared
with explicit tolerances.

Replay is fail-closed: provider construction and text/JSON completions raise,
and missing or corrupt evidence never falls back to generation.

`execute_gold(...)` is the explicit durable entrypoint. It loads Silver from
the verified manifest named by `GoldRequest`, performs the staged Hamilton
boundaries, publishes Gold atomically, replays it offline, and returns exact
candidate/execution/acceptance/failure counts. The existing platform experiment
path remains the default and does not opt into Gold merely because a non-default
network/cache profile was selected.

Methods may continue to expose the coarse `generated_scripts` and
`feature_metadata` properties. A richer duck-typed adapter may instead expose
`gold_generation_batches()`, returning ordered mappings with `code` and named
`specifications`; this gives failed batches explicit specification ownership and
prevents a failed early batch from consuming evidence that belongs to a later
successful batch.
