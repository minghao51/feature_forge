# Project guidance

## Purpose and decision order

Build and maintain the Feature Forge experimentation platform: a modular,
experiment-first refactor of the MALMAS research codebase for LLM-based
multi-agent automated feature engineering. When guidance conflicts, follow
this order:

1. The package contract in `README.md` and the active plan index in
   `docs/plan/00_index.md`.
2. Accepted records in `docs/decisions/` (ADRs).
3. The exploratory design digests in `docs/` (`methods.md`,
   `MALMAS_Technical_Roadmap.md`, `deferred-design.md`); superseded and
   deprecated material lives only in `docs/archive/`.
4. Quick context: `.planning/OVERVIEW.md` (architecture),
   `.planning/STATE.md` (status), `.planning/STYLE.md` (conventions) —
   maintained, but subordinate to 1–3.

## Architectural boundaries

- Every method is a first-class, independently runnable experiment unit;
  the plugin-based `MethodRegistry` / `BaseMethod` is the only extension
  point for new methods.
- All LLM calls go through `LLMClient` with enforced DiskCache (SHA-256
  keys). Never bypass the cache.
- LLM-generated feature code runs only through the sandbox
  (AST validation + process isolation).
- Experiment behavior (datasets × methods × seeds × models × rounds) stays
  configurable via pydantic-settings (YAML + env + `.env`).
- Prefer the smallest architecture that supports a measured experiment.
  Record any change to these boundaries as an ADR in `docs/decisions/`
  before implementing it.

## Engineering conventions

- Start from the documentation map (`docs/README.md`); superseded material
  lives only in `docs/archive/`.
- Use Python 3.11+, `uv` (`uv run` for execution, `uv add` for deps),
  typed Pydantic boundaries, and async LLM calls.
- Keep prompts lean, versioned, and testable. Return concise reasoning
  summaries, not hidden chain-of-thought.
- Never modify supplied data under `data/`. Write run artifacts and traces
  under `experiments/` with resolved configuration and provenance.
- Avoid provider abstractions, tracking platforms, or heavy infrastructure
  until an experiment demonstrates the need.
- Update `REPORT_LOG.md` with material findings and disclose substantive
  AI assistance.
- Read a file before editing it; prefer surgical edits; never commit unless
  explicitly asked.

## Output style

- Be concise; bullet points over paragraphs.
- Reference file paths with line numbers when relevant.
- No preamble, no postamble. Answer the question directly.

## Validation

Run before handing off a change:

```bash
uv sync --all-groups --extra intel  # plain `--all-groups` would prune sklearnex
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run python scripts/check_repo_hygiene.py
uv run python scripts/check_docs_references.py
```

Reliably capturing the pytest summary from piped output (worker-process
stderr can interleave after it): `uv run pytest 2>&1 | grep -E "[0-9]+ (passed|failed)"`;
use `--junitxml=<path>` when machine-readable results are needed.

A change is complete when relevant tests pass, public behavior is
documented, and experiment outputs contain resolved configuration and
reproducibility metadata.
