# PR 8 browser catalog decision record

**Status:** `DEFER` — evidence gathering in progress; implementation is not authorized.

**Decision owner:** `TBD`
**Security reviewer:** `TBD`
**Hosting/platform reviewer:** `TBD`
**Review date:** `2026-07-15`
**Approval expiry:** `TBD`

This is the working evidence ledger for the optional browser catalog. It is not
an approval record. Missing owners, approvals, or evidence remain `NOT_MET`.
Contributors who implement a readiness artifact cannot approve the corresponding
governance gate.

## Decision rule

PR 8 can move from `DEFER` to `APPROVE` only when at least two interaction
requirements pass with representative acceptance evidence and all six governance
gates have named, independent owners, dated approvals, and linked evidence.
Otherwise the repository retains canonical MkDocs and generated static references.

## Interaction requirements

| ID | Requirement | Evidence | State | Owner |
|---|---|---|---|---|
| IX-01 | Filter/search hundreds or thousands of runs and features | `docs/decisions/pr8-scale-evidence.md` plus a timed reviewer task | `NOT_MET` | `TBD` |
| IX-02 | Explore multi-hop lineage interactively | Three-hop workflow and static-reference comparison | `NOT_MET` | `TBD` |
| IX-03 | Compare compatible runs client-side | Cohort definition, uncertainty behavior, and timed task | `NOT_MET` | `TBD` |
| IX-04 | Publish a separate read-only portal | Audience, access boundary, cadence, and MkDocs gap | `NOT_MET` | `TBD` |
| IX-05 | Deploy versioned catalog snapshots independently | Lifecycle, hosting, rollback, and owner evidence | `NOT_MET` | `TBD` |

## Governance gates

| ID | Gate | Evidence | Owner | Approval | State |
|---|---|---|---|---|---|
| GV-01 | Demonstrated MkDocs limitation | IX evidence and measured static baseline | `TBD` | `TBD` | `NOT_MET` |
| GV-02 | Stable versioned public snapshot | Schema, compatibility, deterministic export, size limits | `TBD` | `TBD` | `NOT_MET` |
| GV-03 | Representative scale justifies browser UI | Approved volume profiles and performance budget | `TBD` | `TBD` | `NOT_MET` |
| GV-04 | Second-toolchain ownership | Product and frontend maintenance acceptance | `TBD` | `TBD` | `NOT_MET` |
| GV-05 | Security allowlist approval | Allowlist, threat model, negative fixtures, signed review | `TBD` | `TBD` | `NOT_MET` |
| GV-06 | Hosting/deployment ownership | Target, support, rollback, and retention ownership | `TBD` | `TBD` | `NOT_MET` |

## Evidence ledger

| Date | Work package | Result | Evidence |
|---|---|---|---|
| 2026-07-15 | Baseline | Current generated references are about 21 KiB; strict offline MkDocs build was about 1.3 seconds | Existing PR 8 handoff baseline; refresh with the commands below before approval |
| 2026-07-15 | R1 | Deterministic synthetic scale measurement added; owner approval and user tasks remain outstanding | `scripts/measure_pr8_readiness.py`, `docs/decisions/pr8-scale-evidence.md`, `docs/generated/pr8/` |
| 2026-07-15 | R3 | Deny-by-default public snapshot schema proposed; security approval is outstanding | `docs/decisions/pr8-snapshot-schema-v1.md`, `docs/decisions/schemas/pr8-public-snapshot-v1.schema.json` |

## Required review record

Before any `APPROVE` decision, append a dated review entry containing the
decision owner, independent gate approvers, evidence links, approved budgets,
snapshot version, expiry date, and the exact `APPROVE`, `DEFER`, or `REJECT`
outcome. Do not infer approval from a prototype or an unassigned role.

## Readiness commands

```bash
uv run python scripts/measure_pr8_readiness.py \
  --profile small \
  --results docs/generated/pr8/small.json
uv run python scripts/measure_pr8_readiness.py \
  --profile expected \
  --results docs/generated/pr8/expected.json
uv run python scripts/build_docs_offline.py
```

The corpus and flat Markdown files are temporary unless explicit `--corpus` or
`--markdown` paths are supplied. No production catalog, artifact, prompt,
prediction, code, path, credential, or tracker payload is an input to these
measurements.
