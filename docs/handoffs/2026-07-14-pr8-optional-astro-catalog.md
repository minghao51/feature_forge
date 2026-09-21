# PR 8 Handoff: Optional Astro Experiment Catalog

**Status:** Deferred; evidence-only readiness artifacts added, decision gate not met as of 2026-07-15
**Depends on:** Completed PR 7 and explicit owner decision
**Index:** [`2026-07-14-medallion-pr-index.md`](2026-07-14-medallion-pr-index.md)
**Decision-readiness handoff:** [`2026-07-14-pr8-decision-readiness.md`](2026-07-14-pr8-decision-readiness.md)

Complete the decision-readiness handoff first. This implementation handoff
remains blocked until its R6 outcome is `APPROVE`; evidence gathering and
bounded synthetic-data spikes do not authorize Astro implementation.

## Decision gate

Do not implement this PR unless all conditions are true:

- MkDocs cannot meet a demonstrated interactive exploration requirement.
- The DuckDB-derived snapshot schema is stable and versioned.
- The expected run/feature volume makes static Markdown tables insufficient.
- An owner accepts a second frontend toolchain and maintenance burden.
- Security approves the exact browser-export allowlist.
- Hosting/deployment ownership is explicit.

If the gate is not met, mark this handoff rejected/deferred and continue using
MkDocs plus generated static snapshots.

## PR 7 decision evidence (2026-07-15)

The master specification requires at least two real interaction requirements,
and this handoff requires every governance item. Neither threshold is met.

Evidence-only readiness work now provides deterministic synthetic scale results
and a proposed deny-by-default snapshot schema, but it does not change this
status or authorize Astro implementation.

| Gate | Evidence | Result |
|---|---|---|
| MkDocs cannot meet the need | Generated reference Markdown is about 21 KiB; strict local build is about 1.3 seconds | Not demonstrated |
| Stable versioned browser snapshot | Catalog schema v1 exists, but no public allowlisted snapshot schema exists | Not met |
| Static tables fail at expected scale | No representative production run/feature-volume measurement exists | Not met |
| Second-toolchain owner | No named owner or maintenance acceptance | Not met |
| Security allowlist approval | No approved browser-export field allowlist | Not met |
| Hosting/deployment ownership | No target, owner, or rollback procedure | Not met |

No master interaction requirement is evidenced: there is no required
hundreds/thousands-run filtering, interactive lineage, client-side comparison,
separate public portal, or catalog-snapshot deployment. Reopen only with measured
volume, at least two approved interaction use cases, a versioned allowlisted
snapshot proposal, named maintenance/hosting owners, security approval, and
explicit performance/deployment/rollback budgets.

Current readiness artifacts:

- [`pr8-browser-catalog-decision.md`](../../docs/decisions/pr8-browser-catalog-decision.md)
- [`pr8-scale-evidence.md`](../../docs/decisions/pr8-scale-evidence.md)
- [`pr8-snapshot-schema-v1.md`](../../docs/decisions/pr8-snapshot-schema-v1.md)

## Objective if approved

Build a read-only static browser catalog from allowlisted, versioned JSON
snapshots derived from the rebuildable DuckDB catalog. The browser must never
read raw experiment artifacts or become authoritative state.

## Scope

- Define a versioned public snapshot schema.
- Export only allowlisted dataset/run/stage/feature/metric/check summaries.
- Add dataset, run, feature, lineage, and comparison views.
- Link back to canonical MkDocs operational/contract documentation.
- Generate static assets suitable for the approved hosting target.
- Keep snapshot generation deterministic and explicit.
- Add content-size limits, pagination/aggregation strategy, and redaction.

## Security boundaries

Never export:

- provider credentials or environment secrets;
- raw prompts/responses unless separately classified and explicitly approved;
- raw source datasets, candidate matrices, predictions with sensitive row
  identifiers, or agent memory files;
- arbitrary artifact paths or filesystem roots;
- tracker tokens or private tracker URLs;
- non-allowlisted manifest metadata.

Treat free-text feature descriptions, generated code, and error messages as
potentially sensitive. Their inclusion requires an explicit policy.

## Required views if approved

- datasets and logical Silver identities;
- runs/cases and stage status;
- layer lineage and reuse;
- accepted/rejected feature summaries and decision reasons;
- baseline/enhanced metrics with uncertainty;
- verification/check health;
- comparison across compatible runs;
- stale/corrupt state surfaced from derived catalog status.

## Tests

- Snapshot schema validation and version rejection.
- Deterministic export from the same catalog.
- Secret/sensitive-field negative tests.
- Browser bundle contains no raw artifact payloads or local absolute paths.
- Missing/corrupt catalog entries render safely.
- Large snapshot strategy stays within recorded bundle/performance budgets.
- Static build and link checks pass offline.

## Acceptance criteria

- The owner-approved interaction need is demonstrably met.
- Every browser record traces to a catalog row and verified manifest.
- Deleting the browser output loses no authoritative state.
- Snapshot export is allowlisted, versioned, deterministic, and security-tested.
- MkDocs remains canonical for contracts and operations.
- Build/deployment ownership and rollback are documented.

## Explicitly out of scope

- Browser writes, artifact mutation, or run control.
- Direct DuckDB access from the browser.
- Authentication system unless separately approved and scoped.
- Replacement of MkDocs.
- Serving raw artifacts.

## Verification if approved

Record the actual frontend commands selected by the approved implementation in
this handoff. They must include lint/type/build/test, snapshot schema tests,
secret scanning, offline link validation, and repository-wide Python gates.

## Completion handoff requirements

Record the decision evidence, owner, approved export fields, snapshot version,
hosting target, performance budget, security review, deployment/rollback
procedure, and whether the catalog remains optional at install/runtime.
