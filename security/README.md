# Security policy operations

Operational runbook for the dependency-audit CI lanes and the maintainer
checklist for branch protection. The lanes themselves are defined in
`.github/workflows/ci.yml` (`security`, `security-tooling`); the allowlist
policy header lives at the top of `pip_audit_allowlist.txt`.

Worked example for everything below: the 2026-09-21/22 REPORT_LOG entries
("Slice D: split runtime/tooling dependency audits" and "CI fully green").

## When an audit lane goes red

pip-audit's advisory database drifts continuously (51→53 advisories moved
during the 2026-09-21 session alone), so a red lane is not automatically a
regression. Reproduce locally before touching the lockfile.

1. **Classify the surface.** Identify which closure the advisory belongs to:
   - **Runtime** (`security` lane): the base install closure — project
     dependencies plus transitives, no dependency groups, no extras.
   - **Tooling, dev env** (`security-tooling` lane, step 2): tests, docs,
     notebooks, and pip-audit itself.
   - **Tooling, extras** (`security-tooling` lane, step 4): the
     `--all-extras` closure (litellm/torch/boto3/...).

   Local reproduction, mirroring the lanes exactly:

   ```bash
   # runtime closure
   uv export --format requirements-txt --no-header --no-hashes --no-annotate \
     --no-emit-project --no-default-groups -o /tmp/runtime-requirements.txt
   uv run python -m pip_audit -r /tmp/runtime-requirements.txt

   # dev environment
   uv run python scripts/run_pip_audit.py

   # all-extras closure
   uv export --format requirements-txt --no-header --no-hashes --no-annotate \
     --no-emit-project --no-default-groups --all-extras -o /tmp/extras-requirements.txt
   uv run python scripts/run_pip_audit.py --requirements /tmp/extras-requirements.txt
   ```

   Provenance of the affected package (`uv tree --invert <name>`) decides the
   surface: a package only reachable via the `dev` group or an optional extra
   can never redden the runtime lane.

2. **Prefer a targeted upgrade.** Fix with
   `uv lock --upgrade-package <name>` (no blanket `uv upgrade`), then re-run
   the reproduction commands above and `uv run pytest` for the full local
   gate battery. If an exact/upper pin from a transitive constrainer blocks
   the fixed version, upgrade the constrainer deliberately as well (2026-09-21
   example: `litellm` and the jupyter stack were upgraded to unblock three
   advisories). Runtime-closure advisories must always be fixed.

3. **Allowlist only as a last resort.** An entry in
   `pip_audit_allowlist.txt` is acceptable only when no fixed version can be
   pinned at all. It must carry all six fields per the file's header: the
   vulnerability ID (first whitespace token), the package name, and the four
   metadata keys `owner`, `expiry`, `reason`, `evidence`. An entry missing a
   key, or whose expiry trigger has fired, is a policy violation. The runtime
   lane never consumes the allowlist — a runtime advisory must be fixed, not
   inherited from the tooling lane.

## Documented residual: diskcache CVE-2025-69872

The single allowlisted entry. `diskcache` is absent from the base runtime
closure (verified via `uv export --no-default-groups`); it reaches only the
dev group and the optional observability/all extras, and the
pickle-deserialization path is only ever fed entries written by this process
itself (see the entry's `evidence` for the full exposure statement).

- **Expiry condition:** remove the entry when a diskcache release newer than
  5.6.3 ships without pickle-by-default (upstream
  GHSA-w8v5-vhqr-4h9v / PYSEC-2026-2447, empty `fix_versions` today).
- **Re-check:** re-run the tooling-lane reproduction commands above
  periodically and after any `uv lock` change; the advisory's `fix_versions`
  turning non-empty (or `uv run pip-audit` reporting a fix for
  `diskcache`) means the entry must be removed and the upgrade applied.

## Maintainer checklist: branch protection for `main`

Not agent-executable; requires maintainer GitHub authorization. `main`
currently has no branch protection at all (verified via API, 2026-09-22), so
no check is required before merge.

Minimum required checks (names as they appear in the branch-protection UI):

- `test (3.13)` — representative pytest matrix cell (adding `test (3.11)` /
  `test (3.12)` is the natural strengthening)
- `security` — runtime dependency audit
- `security-tooling` — dev-env + all-extras dependency audit (allowlisted)
- `hygiene` — repo hygiene + docs reference checks

Note on `docs`: the "Deploy Docs" workflow (`build`/`deploy` jobs) triggers
only on pushes to `main`, so it cannot be a required PR check unless the
workflow is first extended with a `pull_request` trigger; `mkdocs build
--strict` is enforced locally in the handoff gate battery meanwhile.

Sketch (adjust `contexts` before running; exact check names are listed under
the branch's protection settings):

```bash
gh api -X PUT repos/minghao51/feature_forge/branches/main/protection --input - <<'JSON'
{
  "required_status_checks": {
    "strict": false,
    "contexts": ["test (3.13)", "security", "security-tooling", "hygiene"]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": null,
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
JSON
```

After enabling, push a trivial change and confirm the required checks report
correctly before relying on them.
