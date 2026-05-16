# Agent Behavior Rules (Single Source of Truth)

## Workflow

1. **Analyze first** — Before making changes, understand the relevant code, its imports, and the patterns used.
2. **Minimal scope** — Change only what's needed to fulfill the request. No refactoring beyond scope.
3. **Check skills** — Before starting any task, check if a relevant skill exists and follow it.
4. **Verify** — After changes, run lint (`ruff check`) and type-check (`mypy`) if applicable. Ask the user for the specific command if unsure.
5. **No commits** — Never commit changes unless explicitly asked.

## 3. Technical Stack
- **Python:**
  - Package manager: `uv`.
  - Execution: Always `uv run <command>`. Never `python`.
  - Installing package : `uv add`
  - Sync: `uv sync`.

## Output Style

- Be concise. bulletpoints over paragraphs.
- Reference file paths with line numbers when relevant.
- No preamble, no postamble. Answer the question directly.

## Universal File Operations

- **Read before edit** — Always read a file before editing it.
- **Prefer Edit tool** over Write for existing files (surgical changes).
- **Prefer editing existing files** over creating new ones.

## Project Context References

- Architecture, modules, and stack: @.planning/OVERVIEW.md
- Current status and risks: @.planning/STATE.md
- Coding conventions and workflow: @.planning/STYLE.md
