# PR 8 Decision-Readiness Handoff: Optional Browser Catalog

**Status:** Ready for evidence gathering; browser implementation is not authorized
**Prepared:** 2026-07-15
**Depends on:** PR 7 verified within its recorded library boundary
**Implementation handoff:** [`2026-07-14-pr8-optional-astro-catalog.md`](2026-07-14-pr8-optional-astro-catalog.md)
**Index:** [`2026-07-14-medallion-pr-index.md`](2026-07-14-medallion-pr-index.md)

## Readiness progress — 2026-07-15

| Work package | State | Evidence | Gate impact |
|---|---|---|---|
| R0 decision contract | Partial; decision, security, and hosting owners remain `TBD` | `docs/decisions/pr8-browser-catalog-decision.md` | R0 not exited; implementation remains blocked |
| R1 scale corpus | Measurement mechanism complete; small and expected profiles measured, owner approval and timed user tasks outstanding | `scripts/measure_pr8_readiness.py`, `docs/decisions/pr8-scale-evidence.md`, `docs/generated/pr8/small.json`, `docs/generated/pr8/expected.json` | GV-03 and IX-01 remain `NOT_MET` |
| R3 snapshot proposal | Schema and synthetic-only fixture spike complete; security/data-owner approval outstanding | `docs/decisions/pr8-snapshot-schema-v1.md`, `docs/decisions/schemas/pr8-public-snapshot-v1.schema.json`, `scripts/emit_pr8_snapshot_spike.py` | GV-02 and GV-05 remain `NOT_MET` |

Recorded checks for this slice: `uv run ruff check` on the new scripts/tests,
five focused readiness tests, byte-stable synthetic snapshot regeneration, JSON
syntax validation, and `git diff --check`. No production catalog or browser
export was used. R2 interaction evidence and R5 ownership/hosting evidence are
still unstarted, so R6 remains `DEFER`.

## Purpose

Resolve the evidence, ownership, security, scale, and deployment decisions that
currently block PR 8. This handoff authorizes measurement, requirements work,
schema proposals, and bounded architecture spikes. It does **not** authorize an
Astro application, a browser export containing real experiment data, a second
frontend toolchain, or deployment.

If the evidence does not prove that a browser catalog is justified, close PR 8
as deferred or rejected and retain MkDocs plus generated static references.

## Current baseline

As of 2026-07-15:

- interaction evidence is **0 of 5 demonstrated**; at least 2 are required;
- governance approvals are **0 of 6 complete**; all 6 are required;
- generated reference Markdown is approximately 21 KiB;
- strict offline MkDocs builds in approximately 1.3 seconds;
- catalog schema v1 exists, but no public browser snapshot schema exists;
- no representative production run/feature-volume measurement exists;
- no browser-catalog product owner, frontend maintainer, security approver,
  hosting owner, or rollback owner is recorded;
- no hosting target, browser bundle budget, deployment budget, or support SLO
  is approved;
- PR 7's application-owned `LayerExecutor` boundary is unrelated to this work
  and must not expand as part of PR 8 readiness.

## Non-negotiable approval rule

PR 8 implementation may begin only when both conditions are satisfied:

1. at least **two** interaction requirements below are demonstrated with
   representative evidence and acceptance scenarios; and
2. **all six** governance gates have named owners, dated approvals, and linked
   evidence.

Anything less results in `DEFER` or `REJECT`. A prototype, stakeholder interest,
or attractive mockup is not approval evidence.

## Interaction requirements

| ID | Candidate requirement | Evidence required | State |
|---|---|---|---|
| IX-01 | Filter/search hundreds or thousands of runs and features | Representative corpus, timed user task, proof generated MkDocs is materially inadequate | Not demonstrated |
| IX-02 | Explore multi-hop lineage interactively | Approved workflow with at least three lineage hops and static-reference comparison | Not demonstrated |
| IX-03 | Compare compatible runs client-side | Defined cohort, filters, metrics/uncertainty behavior, timed static-vs-interactive task | Not demonstrated |
| IX-04 | Publish a separate read-only portal | Named audience, access boundary, update cadence, reason MkDocs is insufficient | Not demonstrated |
| IX-05 | Deploy versioned catalog snapshots independently | Approved lifecycle, hosting requirement, rollback scenario, operational owner | Not demonstrated |

## Governance gates

| ID | Required gate | Required evidence | Owner | State |
|---|---|---|---|---|
| GV-01 | Demonstrated MkDocs limitation | IX evidence plus measured static baseline | TBD | Not met |
| GV-02 | Stable versioned public snapshot | Reviewed schema, compatibility policy, deterministic export, size limits | TBD | Not met |
| GV-03 | Representative scale justifies browser UI | Run/feature/lineage counts and performance budget | TBD | Not met |
| GV-04 | Second-toolchain ownership | Product owner and frontend maintainer accept maintenance/SLO burden | TBD | Not met |
| GV-05 | Security allowlist approval | Field allowlist, threat model, negative fixtures, signed review | TBD | Not met |
| GV-06 | Hosting/deployment ownership | Target, deploy/support/rollback owners, retention policy | TBD | Not met |

## Work packages

### R0 — Freeze the decision contract

Actions:

- Confirm the five IX candidates and six GV gates.
- Name the decision owner, security reviewer, and hosting/platform reviewer.
- Define where approval evidence is recorded and who may change it.
- Record an approval expiry date; stale approvals must be refreshed.

Deliverables:

- `docs/decisions/pr8-browser-catalog-decision.md`;
- owner/approver table with names or team aliases;
- decision date, expiry date, and review record.

Exit criteria:

- all decision roles are named;
- implementation contributors cannot approve their own gate;
- missing evidence is `NOT_MET`, never inferred as approval.

### R1 — Build a representative non-sensitive scale corpus

Actions:

- Define owner-approved small, expected, and stress profiles for runs, stages,
  features, lineage edges, checks, and fold metrics.
- Generate deterministic synthetic catalog/package evidence containing no raw
  prompts, datasets, predictions, code, filesystem roots, or credentials.
- Measure Markdown bytes, row counts, offline build time, navigation latency,
  search usability, and timed reviewer tasks.
- Record the planning horizon and measurement environment.

Suggested starting profiles, pending owner approval:

| Profile | Runs | Features | Lineage edges | Purpose |
|---|---:|---:|---:|---|
| Small | 100 | 2,000 | 5,000 | Current/local use |
| Expected | 5,000 | 100,000 | 300,000 | Planning baseline |
| Stress | 50,000 | 1,000,000 | 3,000,000 | Upper-bound behavior |

Deliverables:

- deterministic synthetic-corpus generator;
- `docs/decisions/pr8-scale-evidence.md`;
- machine-readable results under `docs/generated/pr8/`;
- exact commands, seeds, runtime/hardware metadata, and aggregate results.

Exit criteria:

- representative volume is owner-approved;
- measurements reproduce offline;
- no sensitive/proprietary record is present;
- MkDocs limitations are tied to a timed task or explicit budget.

### R2 — Validate interaction requirements with users

Actions:

- Define one measurable acceptance scenario per candidate IX item.
- Identify the actual user role for each scenario.
- Run the scenario against current MkDocs/generated references first.
- Record success, time, error rate, and blockers.
- Approve only needs not solved acceptably by better static generation, CLI
  output, or MkDocs navigation/search.

Deliverables:

- `docs/decisions/pr8-interaction-evidence.md`;
- scenario scripts and acceptance criteria;
- static-baseline results and reviewed conclusions;
- explicit list of approved IX IDs.

Exit criteria:

- at least two IX IDs pass, or the outcome is `DEFER/REJECT`;
- each passing IX item has a measurable browser acceptance test;
- requirements remain frontend-technology neutral.

### R3 — Propose snapshot schema v1 and security allowlist

Actions:

- Derive snapshots from the rebuildable catalog only; the browser build must
  never read raw artifacts.
- Define a strict, versioned, extra-forbid snapshot schema.
- Classify every field as approved, omitted, aggregated, or pseudonymized.
- Define deterministic ordering, partitioning, compatibility, size caps, and
  future-version rejection.
- Threat-model names, configuration, free text, errors, descriptions, code,
  paths, identifiers, tracker references, and timestamps.
- Add sentinels for provider keys/tokens, authorization headers, absolute paths,
  private URLs, prompts, raw rows, predictions, and agent memory.
- Require security approval of the exact field allowlist/transforms.

Generated code, raw errors, prompts/responses, artifact paths, row IDs, raw
predictions, memory, environment dumps, and tracker URLs are excluded by default.

Deliverables:

- `docs/decisions/pr8-snapshot-schema-v1.md`;
- proposed JSON Schema under `docs/decisions/schemas/`;
- field allowlist and threat model;
- deterministic synthetic-only exporter spike;
- dated security approval or rejection.

Exit criteria:

- schema version and compatibility policy are explicit;
- every exported field is allowlisted;
- deterministic and secret-negative tests pass;
- deleting snapshots loses no authoritative state;
- security approval is recorded.

### R4 — Compare the smallest viable surfaces

Compare at least:

1. improved MkDocs/generated pages;
2. static HTML/JavaScript without a new framework; and
3. Astro or another approved static-site toolchain.

Actions:

- Use only synthetic approved-schema snapshots.
- Exercise every approved IX scenario.
- Measure build time, bundle bytes, memory, interaction latency, accessibility,
  dependency count, vulnerability surface, and maintenance steps.
- Confirm no browser spike reads DuckDB or artifacts directly.
- Delete/archive unselected spikes after the decision.

Deliverables:

- `docs/decisions/pr8-surface-comparison.md`;
- reproducible measurements and relevant screenshots;
- recommendation plus rejected alternatives;
- proposed frontend dependency/update policy.

Exit criteria:

- the selected surface materially improves all approved IX scenarios;
- bundle/performance/accessibility budgets pass;
- owner accepts the toolchain burden;
- evidence precedes technology choice.

### R5 — Close ownership, hosting, deployment, and rollback

Actions:

- Select hosting target and access model.
- Name product, frontend maintenance, security, deployment, support, and
  rollback owners.
- Define publication cadence, retention, preview, production, rollback, and
  snapshot-revocation flows.
- Define analytics/logging policy; default to none unless approved.
- Define support SLO, dependency updates, and end-of-life procedure.
- Confirm browser dependencies remain outside default Python install/runtime.

Deliverables:

- `docs/decisions/pr8-operations-ownership.md`;
- hosting/access decision and RACI;
- deployment/rollback runbook;
- approved budgets and SLOs.

Exit criteria:

- every governance owner accepts the role;
- deployment and rollback are rehearsable;
- browser loss cannot affect execution, evidence, catalog rebuild,
  verification, or MkDocs;
- browser dependencies remain optional.

### R6 — Issue the decision

The decision owner must choose exactly one:

- `APPROVE`: at least two IX and all GV items pass; authorize implementation;
- `DEFER`: evidence/ownership is incomplete; record missing items/review date;
- `REJECT`: measured benefit does not justify cost/risk; retain MkDocs.

Deliverables:

- completed IX/GV matrices and evidence links;
- signed decision, rationale, date, expiry/review date;
- updated PR index, implementation handoff, ADR 0004, and `.planning/STATE.md`.

`APPROVE` requires:

- at least two IX rows `PASS`;
- all six GV rows `PASS`;
- no owner/security field `TBD`;
- schema and budgets frozen for implementation;
- implementation boundaries and rollback agreed.

## Optional implementation after approval

Do not start these phases before R6 returns `APPROVE`.

### I1 — Snapshot exporter and contracts

- Implement the approved catalog-to-snapshot exporter and schema.
- Verify recursive manifest traceability and exact allowlist enforcement.
- Add deterministic, size-cap, version-rejection, and secret-negative tests.
- Keep export explicit; verification remains read-only.

### I2 — Read-only browser views

- Implement only approved IX scenarios.
- Keep MkDocs canonical for contracts/operations.
- Exclude writes, run control, authentication, and direct DuckDB access unless
  separately approved.

### I3 — Performance, accessibility, and security

- Test expected/stress snapshots against budgets.
- Run frontend lint/type/unit/build/accessibility/link checks offline.
- Scan bundle/snapshots for secrets, paths, raw payloads, unapproved fields.

### I4 — Deployment and rollback

- Deploy to the approved preview target first.
- Validate access, cache behavior, revocation, rollback, and deletion.
- Record exact commands, versions, owners, monitoring, and evidence.

## Required readiness tests

- Synthetic corpus is deterministic across clean processes.
- Benchmark outputs contain no secrets, raw data, absolute paths, or mutable
  timestamps outside an explicit measurement envelope.
- Snapshot schema rejects unknown/future fields and versions.
- Export allowlist is deny-by-default and complete.
- Same input yields byte-stable snapshots.
- Sensitive sentinels are absent from snapshots and spike bundles.
- IX scenarios are reproducible with machine-readable results.
- Surface comparison runs offline from synthetic snapshots.
- Decision validation rejects `APPROVE` with fewer than two IX or any missing GV.
- Astro/Node never becomes a default/runtime Python dependency.

## Verification baseline

Exact new commands are added only when their scripts exist. Readiness changes
must retain:

```bash
uv run ruff check .
uv run mypy src
uv run pytest -m "not llm" -q
uv run --extra pipeline python scripts/generate_medallion_docs.py --check
uv run python scripts/build_docs_offline.py
uv run python scripts/check_repo_hygiene.py
uv run python scripts/check_docs_references.py
uv run python scripts/run_pip_audit.py
uv lock --check
git diff --check
```

Any R4 frontend spike must record its package manager, pinned runtime, lockfile,
lint/type/test/build commands, dependency audit, and offline build. Do not add
placeholder frontend commands to CI.

## Risks and controls

| Risk | Control |
|---|---|
| Prototype momentum bypasses gate | Synthetic-data spikes confer no implementation approval |
| Snapshot becomes shadow truth | Derived/read-only export; deletion/rebuild tested |
| Sensitive free text leaks | Deny-by-default allowlist, sentinels, security sign-off |
| Scale estimates are invented | Owner-approved profiles and planning horizon |
| Astro precedes requirements | Technology-neutral IX work precedes R4 |
| Second toolchain is unowned | GV-04/GV-06 require named owners |
| PR 7 executor boundary expands | Layer materialization is explicitly out of scope |
| Default install gains frontend deps | Isolation and import tests remain required |

## Explicitly out of scope for readiness

- Changing legacy execution defaults.
- Shipping a turnkey `LayerExecutor`.
- Browser writes, run control, cancellation, or artifact mutation.
- Direct browser access to DuckDB/manifests/artifacts.
- Authentication implementation.
- Production-data export.
- Replacing MkDocs.
- Deployment before R6 approval.

## Handoff update protocol

After each work package, record:

- status/completion date;
- exact evidence files/commands;
- measured results/environment metadata;
- named reviewers/owners and approval state;
- unresolved issues/expired assumptions;
- whether the gate stays `DEFER`, becomes `REJECT`, or is eligible for R6.

Do not mark the implementation handoff `Approved` until R6 records at least two
passing IX requirements and all six passing GV gates.
