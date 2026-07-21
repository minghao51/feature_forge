# Operations runbook

This runbook covers the local, manifest-backed pipeline. Required workflows are offline and
deterministic unless explicitly identified as provider-backed generation.

- [Installation and execution profiles](profiles.md)
- [Run, plan, replay, resume, and cancellation](workflows.md)
- [Verification and catalog operations](verification-catalog.md)
- [Recovery, retention, security, backup, and rollback](recovery-security.md)

Use `--format json` before a CLI command for machine-readable output. Read-only verification
must not create a catalog, cache, provider, tracker session, or artifact directory.
