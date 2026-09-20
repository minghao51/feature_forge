# Sandbox OS-Isolation Spike (2026-09-15)

Time-boxed probe for plan 23 §5.2 and ADR 0019 (PR 1, "Characterization and
ADRs"). Purpose: empirically determine which OS-enforced containment mechanism
the production sandbox can use on this Linux host, and prove the plan's
acceptance signal — that restricted generated code cannot read a sentinel
outside its allowed input/output roots. Machine-readable evidence:
`experiments/sandbox_isolation_spike/2026-09-15/report.json`.

## Context

- Plan 23 §5.2 requires OS-enforced sandbox containment: the production
  sandbox must prove that generated code cannot read a sentinel outside its
  allowed input/output roots, deny other filesystem access and socket creation
  at the OS boundary, and fail closed when strict enforcement is unavailable.
- Audit finding behind this requirement: the current sandbox relies on AST
  checks, monkeypatches, and process isolation that do **not** restrict host
  filesystem reads — a generated payload can read arbitrary files.
- ADR 0019 must select the mechanism (threat model, strict/degraded modes,
  filesystem + network guarantees) after this time-boxed spike (plan 23 §3).

## Method

Standalone stdlib-only probe `scripts/spike_sandbox_isolation.py`
(`--report <path>`, `--sentinel-timeout <s>`), run as:

```bash
uv run python scripts/spike_sandbox_isolation.py \
    --report experiments/sandbox_isolation_spike/2026-09-15/report.json
```

Probes: (a) `platform.uname` identity; (b) `landlock_create_ruleset` (syscall
444, `LANDLOCK_CREATE_RULESET_VERSION`) for ABI version; (c) forked child
`unshare(CLONE_NEWUSER|CLONE_NEWNS)`; (d) `find_library('seccomp')` +
`prctl(PR_GET_SECCOMP)` errno probe; (e) `shutil.which('bwrap')` /
`which('firejail')`; (f) live Landlock containment demo when ABI ≥ 1: a forked
child creates a random-byte sentinel outside the allowed roots, builds a
ruleset handling all ABI-valid filesystem access bits, adds `PATH_BENEATH`
rules (read/execute for `/usr`, `/lib`, `/lib64`, `/etc`, `sys.prefix`, and the
standard-library directory; read/write/create/remove for two temp input/output
roots), commits with `PR_SET_NO_NEW_PRIVS` + `landlock_restrict_self`, then
attempts `open(sentinel)` and a loopback `connect()`; (g) JSON report with
provenance (script path, plan reference, syscall numbers, timeout).

## Results

Host: Linux x86_64, WSL2 kernel `6.18.33.2-microsoft-standard-WSL2`, Python
3.13.14 (2026-09-15).

| Probe | Result |
|-------|--------|
| Landlock | **Available, ABI 7** (≥ 4 ⇒ network bind/connect scoping supported) |
| Unprivileged user namespaces | Available — `unshare(CLONE_NEWUSER\|CLONE_NEWNS)` succeeded |
| seccomp | Available — `PR_GET_SECCOMP` returned mode 0 (active); `libseccomp.so.2` present |
| bubblewrap | `/usr/bin/bwrap` |
| firejail | Not installed |
| Live containment demo | **Sentinel read outside roots DENIED (EACCES, errno 13)**; imports and temp-root writes remained usable; loopback `connect()` permitted (demo ruleset handled filesystem rights only) |

Demo fidelity notes: the sentinel is created *before* restriction so a missing
file (ENOENT) cannot masquerade as a denial; the child's restriction kept
runtime reads and the two temp roots working, so the EACCES is attributable to
the path scoping, not a broken allowlist. The demo did not exercise the
Landlock network scope; the loopback connect succeeding is expected because
the demo ruleset handled only filesystem access bits — with the detected ABI 7
(≥ 4), `LANDLOCK_ACCESS_NET_*` port rules are available natively, and seccomp
remains a defense-in-depth complement on older kernels.

## Analysis — recommendation for ADR 0019

- **Primary (Linux strict mode): Landlock.** Kernel-enforced, unprivileged
  (no namespaces or root), path-scoped; the live demo demonstrates the exact
  sentinel guarantee plan 23 §5.2 asks for, and the compose order (load
  runtime modules and input frame → `PR_SET_NO_NEW_PRIVS` →
  `landlock_restrict_self`) matches the plan's preferred design.
- **Socket denial:** prefer Landlock network scoping (ABI ≥ 4, available here
  as ABI 7); use a seccomp filter (or bwrap `--unshare-net`) as fallback on
  kernels with Landlock ABI < 4. The demo confirms socket calls are otherwise
  unimpeded — filesystem rules alone do not deny networking.
- **Fallback path: user namespaces / bubblewrap.** Unprivileged userns works
  on this host and `bwrap` is installed; keep as the documented fallback for
  hosts where the Landlock LSM is disabled (e.g. some container/WSL kernels).
- **macOS / Windows: degraded development profile.** No Landlock equivalent;
  strict mode fails closed in production (plan 23 §5.2), and development
  proceeds under the explicitly named degraded profile with provenance.
- **Runtime ABI masking.** Rulesets must handle only access bits valid for the
  detected ABI (the probe's `create_ruleset` fails with EINVAL otherwise);
  enforcement code should derive its handled-access mask from the runtime ABI
  version probe.

## Open questions for PR 5 (strict OS enforcement)

1. Allowlist breadth: `/usr`, `/lib(64)`, `/etc` are broad; a narrower
   `sys.prefix` + loader-discovered shared-library set reduces exposure but
   needs per-host probing and startup-cost measurement.
2. Network policy shape: deny-all connect/bind vs port-scoped rules; where the
   seccomp fallback lives (worker-side filter vs `bwrap` wrapper).
3. `REFER` (ABI ≥ 2) semantics if the worker ever moves results across
   directory boundaries under restriction.
4. Fail-closed error surface: exception type/message and provenance fields
   when strict enforcement is unavailable (ties into §5.4 provenance).
5. Host variability: WSL2/container kernels can disable the Landlock LSM;
   strict-mode entry must re-probe at runtime (never cache across hosts) and
   tests need a platform-specific reproducer marker per plan 23 §6 PR 1.
