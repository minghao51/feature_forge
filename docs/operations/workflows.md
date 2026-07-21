# Run, plan, replay, resume, and cancellation

## Plan without side effects

```bash
uv run feature-forge --format json run plan \
  --dataset titanic --method openfe --model xgboost --seed 42
```

For `--artifact-policy layer_boundaries`, also pass `--fingerprints-json` containing the
resolved per-run Bronze/Silver/Gold/Platinum fingerprints. Planning performs no provider call,
catalog mutation, cache write, or artifact write.

## Run and resume

`ExperimentalPlatform.run()` remains the execution API. The legacy path is the default.
Select `artifact_policy="layer_boundaries"`, provide resolved layer fingerprints and a layer
executor, and use `ResumePolicy(enabled=True)` to reuse the longest contiguous chain of
verified compatible packages. A missing, corrupt, wrong-version, or wrong-lineage package
stops reuse; recovery never silently broadens into upstream regeneration.

Feature Forge does **not** ship a turnkey durable layer materializer. A full boundary run
requires an application-owned, module-level `LayerExecutor` that writes valid packages and
returns their `ManifestRef`s. Module level matters when process execution must pickle the
callback. The tested helpers below cover the platform/store/ref/sandbox wiring; `run_durable`
becomes executable only after the application supplies that callback.

```python
--8<-- "examples/medallion_operations.py"
```

For a no-write smoke, build the exact four-layer mapping and call `plan_durable_run`. Use
`inspect_resume` to inspect reuse, `replay_gold` with existing verified Silver/Gold refs, and
`recover_abandoned` only after selecting explicit retention ages.

`fail_fast=False` lets independent cases finish after another case fails. `fail_fast=True`
cancels or skips remaining cases predictably. There is no interactive remote cancel service;
cancellation is an outer-scheduler terminal state and process futures are cancelled during
fail-fast shutdown.

## Replay

Replay starts from a verified Gold manifest, re-executes stored batches in order, checks row
IDs/schema/values, and then evaluates against persisted Silver folds. The replay guard rejects
provider construction and both text and JSON completion. Missing evidence is an error, not a
request to regenerate.

Run journals record planned, running, reused, skipped, executed, partial, failed, cancelled,
and terminal outcomes. Trackers are downstream observers only.
