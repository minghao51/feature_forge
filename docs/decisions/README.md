# Decisions (ADRs)

Architecture Decision Records for Feature Forge. An ADR is a short,
immutable record of one decision: the context, the choice, and its
consequences. Accepted ADRs sit at level 2 of the decision order in
`AGENTS.md` — above exploratory docs, below the package contract.

## Conventions

- File naming: `NNNN-slug.md`, zero-padded, sequential (`0001-`,
  `0002-`, ...). `0000-template.md` is the template and is never numbered
  as a real record.
- Start a new ADR by copying `0000-template.md`.
- Status lifecycle: `proposed` → `accepted` → `superseded by ADR-NNNN`.
  Never delete or rewrite an accepted ADR; supersede it with a new one that
  links back.
- Any change to the architectural boundaries listed in `AGENTS.md` requires
  an ADR **before** implementation.
- Keep ADRs decision-sized: one decision per record, a few paragraphs each.

## Index

| ADR | Title | Status |
|-----|-------|--------|
| [0001](0001-architectural-boundaries.md) | Architectural boundaries for the experimentation platform | Accepted |
| [0002](0002-three-tier-agent-memory.md) | Three-tier agent memory for MALMAS | Accepted |
| [0003](0003-dynamic-agent-router.md) | Dynamic agent router with selectable strategies | Accepted |
| [0004](0004-methods-namespace-unified-protocol.md) | Methods namespace, unified protocol, and sklearn-compatible public API | Accepted |
| [0005](0005-prompt-colocation-pydantic.md) | Prompt colocation with Pydantic-validated rendering | Accepted (rendering superseded by 0008) |
| [0006](0006-experiment-execution-seam.md) | Experiment execution seam behind the platform facade | Accepted |
| [0007](0007-observability-stack.md) | Observability stack — structlog + OpenTelemetry + Langfuse | Accepted |
| [0008](0008-jinja2-prompt-templating-full-externalization.md) | Jinja2 prompt templating and full prompt externalization | Accepted |
| [0009](0009-versioned-prompts-cache-provenance.md) | Versioned prompt files and cache provenance | Accepted |
| [0010](0010-intel-openmp-bootstrap.md) | Intel Extension acceleration and OpenMP runtime isolation | Accepted |
| [0011](0011-structured-outputs.md) | Strict structured outputs with pydantic and gateway negotiation | Accepted |
| [0012](0012-hamilton-default-case-dataflow.md) | Hamilton as the case dataflow engine | Superseded by ADR 0016 |
| [0013](0013-medallion-lite-boundaries.md) | Medallion-lite stage boundaries | Accepted |
| [0014](0014-atomic-artifact-packages.md) | Atomic artifact packages | Accepted |
| [0015](0015-hamilton-cache-policy.md) | Hamilton cache policy | Accepted |
| [0016](0016-remove-legacy-execution-engine.md) | Remove the legacy execution engine after qualification | Accepted |
| [0017](0017-failure-policy-and-cancellation-contract.md) | Failure policy and cooperative cancellation contract | Accepted |
| [0018](0018-independent-discovery-evaluation-protocol.md) | Independent discovery and evaluation protocol | Accepted |
| [0019](0019-sandbox-containment-bounded-lifecycle.md) | Sandbox containment and bounded worker lifecycle | Accepted |
| [0020](0020-llm-cache-request-identity-v2.md) | LLM cache request identity v2 | Accepted |
