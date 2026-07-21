# PR 7 Handoff: Documentation and Operational Runbook

**Status:** Implemented and verified within the library boundary 2026-07-15; full durable
materialization remains application-integrated through `LayerExecutor`
**Depends on:** Final contracts, CLI, catalog, and lifecycle behavior
**Index:** [`2026-07-14-medallion-pr-index.md`](2026-07-14-medallion-pr-index.md)

## Objective

Make the implemented medallion pipeline operable and reviewable from the
canonical MkDocs site. Documentation must be generated or checked against live
contracts and commands rather than restating the original design.

## Confirmed PR 6 inputs

- Catalog schema version 1 and its live DDL are in
  `src/feature_forge/storage/catalog.py`; generated schema documentation must
  derive from or freshness-check this source.
- CLI parsing and stable exits `0/10/11/12/20/64` are in
  `src/feature_forge/cli.py`. Generate help snippets for the six command
  families without constructing a provider, tracker, cache, or mutable catalog.
- Verification result, catalog report, and issue schemas are in
  `src/feature_forge/contracts/catalog.py`.
- Hamilton telemetry and its tag/redaction boundary are in
  `src/feature_forge/observability/hamilton_adapter.py`; tracker references and
  optional/required policy are in `experiment/tracker.py` and `config.py`.
- `docs/catalog_verification.md` is the PR 6 operational baseline. PR 7 should
  integrate it into the full run/replay/resume/recovery narrative rather than
  duplicating it.

## Required documentation

### Architecture and contracts

- outer control plane vs Hamilton inner DAG;
- Bronze/Silver/Gold/Platinum responsibilities;
- layer-specific fingerprint and invalidation matrix;
- manifest, descriptor, package, and `_SUCCESS` semantics;
- source/reference vs snapshot behavior;
- cache vs durable evidence distinction;
- plugin/method author impact.

### Operations

- install profiles and dependency groups;
- run, dry-run, replay, resume, cancel, and recovery workflows;
- verification CLI and exit-code reference;
- catalog rebuild/status/reconciliation;
- stale locks/staging and retention handling;
- tracker failure behavior;
- security, redaction, and sensitive-data boundaries;
- backup/migration/rollback guidance.

### Generated references

- current DAGs for Silver, Gold, Platinum, and end-to-end case flow;
- contract/schema reference generated from live models where practical;
- check registry with ID, severity, layer, required/optional status, and owner;
- catalog schema/reference;
- CLI help snapshots or checked command examples;
- package directory examples containing no real secrets or proprietary data.

### Migration

- legacy imperative profile to Hamilton profile;
- old vs enriched `ExperimentResult`;
- existing descriptive artifact storage vs durable package store;
- method author migration and compatibility fallback;
- rollback boundaries for each stage.

## CI freshness

- Add a deterministic generation/check command for DAGs and generated
  references.
- CI must fail when committed generated docs differ from live code.
- Strict MkDocs must run in a non-allow-failure workflow job. Branch-rule enforcement must
  not be claimed without inspecting repository settings.
- Builds must require no network, provider credentials, tracker login, or
  mutable catalog state.
- Resolve the current navigation exclusions intentionally: include, archive, or
  explicitly document non-nav plan pages.

## State and handoff cleanup

- Update `.planning/OVERVIEW.md`, `.planning/STATE.md`, and relevant style notes.
- Mark completed PR handoffs with verified results.
- Replace stale claims and commands throughout README/docs.
- Keep the original master specification as historical design context and the
  phase index as implementation tracking truth.

## Expected file areas

- `docs/`
- `mkdocs.yml`
- README and migration guide
- generated-doc scripts
- `.github/workflows/docs.yml` and relevant CI checks
- `.planning/`
- phase handoffs

## Tests/checks to add

- Generated references are byte-stable from a clean checkout.
- Strict docs build succeeds offline.
- Every documented CLI command exists and parses.
- Internal links and API references resolve.
- Examples use supported dependency groups and profiles.
- No generated output contains configured test secrets.
- Navigation and page status are intentional.

## Acceptance criteria

- **Original turnkey criterion not met as written and explicitly narrowed:** a new contributor
  cannot execute a full durable Bronze-to-Platinum materialization from this library alone.
  Feature Forge intentionally requires an application-owned, module-level `LayerExecutor`.
- Within the library boundary, a contributor can install, plan, inspect resume, replay existing
  verified Gold, recover abandoned local candidates, verify, and rebuild the catalog using
  current docs. A regression verifies the executor callback crosses the platform and spawn
  serialization boundaries; it does not claim the fake callback materializes real packages.
- Generated DAG/contract/check references match executable code.
- Documentation CI detects stale generated assets.
- No examples require live provider/network access unless explicitly labeled
  optional and excluded from the offline docs workflow job.
- State and handoff documents match the final checkout.

## Explicitly out of scope

- Astro/browser catalog implementation.
- Redesign of unrelated historical notebooks.
- General documentation rewrite outside medallion/runtime drift.

## Verification

```bash
uv run ruff check .
uv run mypy src
uv run python scripts/check_repo_hygiene.py
uv run python scripts/check_docs_references.py
uv run python scripts/build_docs_offline.py
uv run pytest -m "not llm" -q
uv run python scripts/run_pip_audit.py
git diff --check
```

Run the final generated-reference freshness command introduced by this PR and
record its exact invocation in this handoff.

## Completion handoff requirements

Report documentation inventory, generated sources and commands, CI freshness
behavior, navigation decisions, migration/rollback coverage, remaining known
risks, and whether the PR 8 decision gate is met.

## Completion record

- Canonical MkDocs sections now cover the outer scheduler/Hamilton boundary,
  layer identity and invalidation, durable contracts, method authors, profiles,
  run/plan/replay/resume/cancellation, catalog verification, crash recovery,
  retention, tracker failure, sensitive-data boundaries, backup, and rollback.
- `scripts/generate_medallion_docs.py` provides deterministic `--write` and
  `--check` modes. It generates platform-stable normalized Silver, Gold, Platinum,
  and full-case topology JSON plus canonical portable Mermaid DAGs; all discoverable contract
  references/schemas; a centralized 15-check
  registry; live catalog DDL; recursive CLI help/exits; Hamilton node metadata;
  package layouts; and a manifest hashing every generated text/source output. Mermaid is
  generated by reading the normalized JSON and both are freshness-checked. PNG bytes remain
  optional previews and are intentionally excluded because external Graphviz renderers are
  not cross-platform byte-stable; PNG presence is enforced.
- Generation runs under a scrubbed environment with network creation denied. Tests prove two
  clean generations of every text/source output are byte-equal and scan those outputs for
  credential sentinels and local absolute paths. Opaque PNG pixels cannot be honestly scanned.
- A non-allow-failure `docs` workflow job installs the pipeline extra, freshness-checks generated
  references, runs documentation contract tests, checks hygiene/references, and
  runs strict MkDocs through `scripts/build_docs_offline.py`, which scrubs application
  configuration and denies in-process socket creation for MkDocs/plugins. Tests exercise the
  wrapper and a blocked plugin network attempt. Branch protection was not inspected, so this
  does not claim repository required-check enforcement.
- Navigation now has explicit Pipeline, Operations, Generated Reference,
  Migration, Implementation Plan, and Historical Plans sections. The four former
  non-nav plan pages are intentionally grouped under Historical Plans.
- README, Quick Start, API-key guidance, project documentation URL, Makefile,
  pre-commit configuration, planning overview/style/state, and historical notebook
  links were reconciled with live uv/MkDocs/CLI behavior. Stale Quarto automation
  and invalid extras were removed.
- Migration preserves the legacy default, optional enriched result fields,
  descriptive storage compatibility, Hamilton-free third-party plugins, and
  independent rollback at every layer/catalog/default-profile boundary.
- Tested runnable helpers cover resolved fingerprints, dry-run planning, verified resume
  inspection, offline Gold replay with manifest refs and the sandbox, explicit artifact-store
  recovery, and the application-supplied top-level executor boundary. A module-level fake
  executor proves platform dispatch and spawn picklability only. End-to-end durable package
  materialization remains an application integration requirement, so the original turnkey
  new-contributor criterion is not recorded as complete. Readers reject future artifact schemas;
  no automatic/in-place persisted-artifact migration is shipped.
- The PR 8 gate is not met. Static generated Markdown is about 21 KiB across the
  reference pages and strict MkDocs builds locally in about 1.3 seconds. There is
  no representative production-volume evidence, no two demonstrated interactive
  requirements, no public snapshot schema, and no owner/security/hosting approvals.

### Generated-reference commands

```bash
uv run --extra pipeline python scripts/generate_medallion_docs.py --write
uv run --extra pipeline python scripts/generate_medallion_docs.py --check
```

### Focused verification

- Generated write/check and static compile: passed.
- Documentation contract suite: 20 passed.
- Strict MkDocs: passed with every Markdown page intentionally navigated.
- Full pipeline-enabled non-LLM lane: 877 passed at 88% coverage.
- Ruff: passed. Strict mypy: passed across 112 source files and separately for the runnable
  operations example and offline MkDocs wrapper.
- Generated freshness, repository hygiene, documentation references, strict
  MkDocs, dependency audit, lockfile consistency, and diff whitespace: passed.
  The audit found no known vulnerabilities and skipped only the editable local
  distribution.
