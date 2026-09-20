# ADR 0019: Sandbox containment and bounded worker lifecycle

- **Status:** Accepted (maintainer accepted on 2026-09-15)
- **Date:** 2026-09-15
- **Related:** ADR 0001, ADR 0013,
  `docs/plan/23_evaluation_integrity_security_hardening.md` (§5),
  `docs/spikes/2026-09-15-sandbox-os-isolation.md`,
  `experiments/sandbox_isolation_spike/2026-09-15/report.json`,
  `tests/unit/test_plan23_sandbox_hardening.py`, `REPORT_LOG.md`

## Context

The 2026-09-14 audit confirmed that the sandbox isolates generated code in a
process but does not restrict that process from reading files available to the
host user: the static AST policy misses file-capable NumPy APIs
(`np.fromfile`, `np.load`, `np.memmap`, …), so generated code can attempt
host-file reads, and the worker keeps the host user's filesystem permissions.
Separately, the worker lifecycle is unbounded:
`methods/malmas/pipeline/core._exec_sandbox` wraps
`asyncio.to_thread(sandbox.execute, ...)` in an outer guard, and cancelling
the outer task cannot stop the running thread or guarantee cleanup. Both gaps
have characterization tests (plan 23 PR 1).

The time-boxed isolation spike
(`docs/spikes/2026-09-15-sandbox-os-isolation.md`) measured on Linux 6.18
(WSL2, x86_64): **Landlock ABI 7** available; unprivileged user namespaces
available; seccomp/`libseccomp.so.2` available; `bwrap` present. A live
Landlock ruleset denied an out-of-root sentinel read with `EACCES` while
Python imports and temp-root writes kept working; a filesystem-only ruleset
cannot deny sockets, but Landlock ABI ≥ 4 adds native TCP scoping — it
denies TCP bind/connect only; AF_UNIX and NETLINK sockets remain
kernel-permitted and their Python-level construction (including
`socketpair`) is blocked only by the worker's runtime shim — while below
ABI 4 a seccomp filter denies `socket(2)` outright. macOS and Windows have no
Landlock equivalent.

On 2026-09-15 the maintainer accepted this ADR, unlocking the sandbox runtime
changes (plan 23 PR 5).

## Decision

1. **Threat model.** LLM-generated code is untrusted. Strict containment must
   prevent, at the operating-system boundary: reading or writing host files
   outside the granted input/output roots (secrets included), network
   access, privilege escalation, and unbounded run time. AST checks,
   monkeypatches, environment scrubbing, and working-directory changes are
   defense in depth and are never described as filesystem isolation.
2. **Supported platforms.** Strict enforcement on Linux with Landlock ABI ≥ 1
   (preferred mechanism); strict mode **includes** the network-denial
   mechanism — Landlock TCP bind/connect denial where ABI ≥ 4 (AF_UNIX and
   NETLINK remain kernel-permitted; Python-level socket construction,
   including `socketpair`, is blocked by the worker's runtime shim),
   otherwise a seccomp filter denying `socket(2)` outright — so the
   "strict" label is never granted without network denial.
   macOS and Windows receive a clearly named degraded development
   profile only; the production profile **fails closed** when strict
   enforcement is unavailable, and degraded execution records that fact in
   provenance.
3. **Strict design.** Load trusted runtime modules and the input frame
   **before** applying restrictions; apply a Landlock ruleset permitting only
   the specific temporary input/output paths plus required runtime-library
   reads; deny other filesystem reads/writes; deny TCP bind/connect via
   Landlock network rules where ABI ≥ 4 (AF_UNIX/NETLINK stay
   kernel-permitted; the runtime shim blocks Python-level socket
   construction, including `socketpair`), otherwise a seccomp filter
   denying `socket(2)` outright; set
   `no_new_privs` before `landlock_restrict_self`; mask handled-access bits to
   the detected ABI.
4. **Immediate AST defense (lands first).** Extend the canonical static
   policy to reject file-capable APIs exposed by allowed libraries — NumPy
   load/save/fromfile/memmap/data-source APIs and equivalent Pandas
   readers/writers — matching normalized attribute paths **and** terminal
   attribute names so simple aliasing cannot bypass the rule. Clear unneeded
   environment variables in the worker and keep runtime monkeypatches of
   known library I/O surfaces as defense in depth. Never expose secrets.
5. **Bounded worker lifecycle.** An async sandbox entry point starts and owns
   the child process from the event loop/main thread (never
   `multiprocessing.Process.start()` inside `asyncio.to_thread`); sync and
   async paths share one process handle/cleanup helper; worker launch, input
   load, generated-code execution, output publication, process join, and
   cleanup are each bounded; on timeout the child/process group is killed,
   waited on, queues closed, and every temporary file removed. One
   authoritative deadline is used and its effective value logged. Async
   callers await the async entry point directly and never rely on cancelling
   `asyncio.to_thread` to stop a running sandbox.
6. **Qualification.** Adversarial tests must prove that a generated program
   cannot read an external sentinel (including through allowed libraries),
   cannot create a network connection, and leaves no worker process, thread,
   queue, or temporary artifact after timeout or task cancellation — on every
   operating system claimed strict; degraded platforms carry an explicit
   fail-closed production test.

## Consequences

- Containment moves from promise-level checks to kernel enforcement; bounded
  lifecycle removes thread/process/temp-file leaks on cancellation.
- Strict mode is Linux-only; macOS/Windows users must explicitly opt into the
  degraded profile, and production runs there fail closed until a native
  mechanism is qualified.
- Landlock requires careful path grants (runtime library trees) and ABI
  masking; implementation complexity lands in plan 23 PR 5, which may add a
  small direct dependency (e.g. `python-landlock`) over raw ctypes if it
  proves more reliable — recorded here, decided at implementation time.
- The existing AST policy remains the first line (cheap, deterministic
  rejections): layered, not replaced.
- **Rollback.** Each plan 23 PR is independently revertible; the expanded AST
  defense remains valuable even if strict OS enforcement is reverted, and the
  sandbox never silently weakens — a reverted strict mode fails closed in
  production rather than falling back quietly to process-only isolation.
