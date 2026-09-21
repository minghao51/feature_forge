# Implementation Handoffs

Chronological record of session implementation handoffs: the per-PR working
plans that drive (and record) agent-assisted implementation. These are
**provenance, not guidance** — they describe what was planned and executed
at a point in time, including superseded reasoning. Where a handoff
conflicts with `README.md`, `docs/plan/00_index.md`, or an accepted ADR, the
latter win (decision order in `AGENTS.md`).

Handoffs are not part of the published site (`mkdocs.yml` `exclude_docs`),
same as `docs/archive/`, `docs/decisions/`, and `docs/deferred-design.md`.

**Directory history:** these documents previously lived in `.claude/handoffs/`
(gitignored, workstation-local). On 2026-09-18 they were migrated here so
plan history is versioned with the code. Files dated before 2026-09 may
retain originating-workstation absolute paths in narrative text; repo
references use `docs/handoffs/`.

**Relationship to `docs/plan/`:** numbered plan documents (e.g.
`docs/plan/23_*.md`) define scope and acceptance criteria; handoffs are the
execution companions for their PR slices. Some older handoffs were promoted
directly into `docs/plan/` (`15_notebook_updates_handoff.md`,
`21_hamilton_default_execution_handoff.md`); those remain in `docs/plan/`.

## Index

| Handoff | Covers | Status |
|---------|--------|--------|
| `2026-09-21-pr1-salvage-ci-recovery-closeout.md` | Remote PR #1 audit follow-up — immutable archive, July handoff recovery, CI recovery slices, and superseded-PR closeout | Ready for execution |
| `2026-09-18-plan23-pr6-llm-cache-identity-v2.md` | Plan 23 PR 6 — LLM cache request identity v2 (ADR 0020): `cache_identity()`, `CACHE_KEY_SCHEMA_VERSION = 2`, provenance, flipping the 8 strict xfails | Ready for implementation |
| `2026-09-18-post-audit-fix-batch.md` | Verified post-audit fix batch across plan-23 PRs 1–5 (leakage fail-closed, typed worker-death errors, provenance/portability pins), slice commits, push | Complete 2026-09-18 |
| `2026-09-18-plan23-pr5-sandbox-containment.md` | Plan 23 PR 5 — strict OS containment and bounded worker lifecycle (ADR 0019) | Complete 2026-09-18 |
| `2026-09-16-plan23-pr4-platinum-v2-evidence-uncertainty.md` | Plan 23 PR 4 — Platinum evidence v2 and uncertainty (ADR 0018 decisions 5–7) | Complete 2026-09-18 |
| `2026-09-16-plan23-pr3-fold-local-preprocessing.md` | Plan 23 PR 3 — fold-local preprocessing and selection policies (ADR 0018 decisions 4–6) | Complete 2026-09-16 |
| `2026-09-15-plan23-pr2-partition-aware-discovery.md` | Plan 23 PR 2 — partition-aware discovery (ADR 0018 decisions 1–3) | Complete 2026-09-16 |
| `2026-09-15-evaluation-security-audit-remediation.md` | Plan 23 PR 1 — characterization tests, OS-isolation spike, ADRs 0018–0020 | Complete |
| `2026-09-14-fail-fast-cancellation-complete.md` | Plan 22 completion record — fail-fast scheduling and cooperative cancellation (ADR 0017) | Complete 2026-09-14 |
| `2026-09-14-legacy-removal-and-fail-fast-plan.md` | Legacy-removal decision and fail-fast planning (ADR 0016/0017) | Superseded by completion record above |
| `2026-09-10-hamilton-default-continuation.md` | Hamilton-default execution continuation (plan 21) | Superseded by `docs/plan/21_*` |
| `2026-05-03-feature-forge-unified-artifacts-implementation.md` | Unified artifact store implementation (early phases) | Historical |
| `2026-05-02-feature-forge-phases-4-13-complete.md` | Early project phases 4–13 completion record | Historical |
| `2026-05-02-feature-forge-phases-4-13.md` | Early project phases 4–13 plan | Historical |

## Conventions

- One handoff per implementation slice (PR or verified fix batch), dated
  `YYYY-MM-DD-<topic>.md`.
- State scope, verified findings with file:line anchors, constraints, the
  agent workflow (worker/reviewer model pins where used), validation gates,
  acceptance criteria, and explicit out-of-scope items.
- New handoffs are written here directly; do not recreate `.claude/handoffs/`.
