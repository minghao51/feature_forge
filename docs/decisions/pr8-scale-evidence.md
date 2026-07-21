# PR 8 scale evidence

**Status:** `MEASURED / OWNER APPROVAL OUTSTANDING`
**Decision gate:** GV-03 and the scale portion of GV-01
**Generator:** `scripts/measure_pr8_readiness.py`

This evidence uses synthetic JSONL records shaped like the safe, aggregate
catalog information a future browser snapshot might consume. It does not read
DuckDB, manifests, artifacts, datasets, prompts, predictions, code, filesystem
roots, credentials, or tracker payloads. The corpus is disposable; the catalog
and verified manifests remain authoritative.

## Profiles

The profile sizes are the starting proposal from the PR 8 handoff. They are not
an owner-approved forecast yet.

| Profile | Runs | Features | Lineage edges | Stages | Checks | Fold metrics |
|---|---:|---:|---:|---:|---:|---:|
| Small | 100 | 2,000 | 5,000 | 400 | 800 | 500 |
| Expected | 5,000 | 100,000 | 300,000 | 20,000 | 40,000 | 25,000 |
| Stress | 50,000 | 1,000,000 | 3,000,000 | 200,000 | 400,000 | 250,000 |

The generator uses seed `19` by default and emits stable identifiers, bounded
numeric values, enumerated states, and no free text. It writes records in a
fixed type order and canonical JSON encoding. Its output is suitable for
offline size and generation measurements, not for production export.

## Reproduction

Run from the repository root:

```bash
uv run python scripts/measure_pr8_readiness.py \
  --profile small \
  --results docs/generated/pr8/small.json
uv run python scripts/measure_pr8_readiness.py \
  --profile expected \
  --results docs/generated/pr8/expected.json
```

To retain the disposable corpus and flat static reference for inspection:

```bash
uv run python scripts/measure_pr8_readiness.py \
  --profile small \
  --corpus /tmp/feature-forge-pr8-small.jsonl \
  --markdown /tmp/feature-forge-pr8-small.md \
  --results /tmp/feature-forge-pr8-small.json
```

Each result records the profile, seed, row counts, corpus byte count and digest,
flat-Markdown byte count, elapsed generation/render measurements, and limited
runtime metadata. It intentionally omits wall-clock timestamps and hostnames so
the report remains reviewable and comparisons are not contaminated by mutable
identity fields.

## Recorded measurements

These results were generated on 2026-07-15 with CPython 3.13.1 on Darwin arm64.
Elapsed values are local measurements and must be refreshed on the decision
owner's approved environment.

| Profile | Records | Corpus bytes | Flat Markdown bytes | Corpus generation | Flat render | Evidence |
|---|---:|---:|---:|---:|---:|---|
| Small | 8,800 | 1,126,366 | 317,997 | 0.018493 s | 0.005341 s | [`small.json`](../generated/pr8/small.json) |
| Expected | 490,000 | 63,420,882 | 17,696,029 | 1.072098 s | 0.314792 s | [`expected.json`](../generated/pr8/expected.json) |

The expected corpus is approximately 60.5 MiB and the flat Markdown reference
approximately 16.9 MiB. This demonstrates that a naïve flat representation
would become large at the proposed expected volume, but it does not demonstrate
that MkDocs navigation/search fails or that a browser is the right solution.

The stress profile remains available but was not materialized in this pass; no
stress result is being inferred from the expected measurement.

## Static baseline interpretation

The flat Markdown output is a deliberately simple static reference baseline. It
is not a browser implementation and it does not prove that MkDocs fails. The
following must be measured and reviewed for an actual decision:

1. current generated-reference size and strict offline MkDocs build time;
2. a timed reviewer task against those references for each candidate IX item;
3. the profile that represents the approved planning horizon; and
4. a performance budget for the smallest viable surface.

The current handoff baseline is approximately 21 KiB of generated reference
Markdown and approximately 1.3 seconds for the strict offline build. Those
values are context only until refreshed with the commands in the decision record.

## Current result

This work establishes a reproducible measurement mechanism and safe synthetic
profiles. It does not establish that any profile is representative, that static
references are materially inadequate, or that a browser catalog should be built.
GV-03 remains `NOT_MET` until a named owner approves the profile, planning
horizon, and performance budget. IX-01 and GV-01 likewise remain `NOT_MET`
until a representative timed task shows a meaningful static limitation.
