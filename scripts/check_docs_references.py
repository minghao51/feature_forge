"""Fail CI when docs contain stale paths or broken local @references."""

from __future__ import annotations

from pathlib import Path

DOCS_TO_CHECK = [
    Path("README.md"),
    Path("docs/index.md"),
    Path("docs/methods.md"),
    Path("AGENTS.md"),
    Path(".planning/OVERVIEW.md"),
    Path(".planning/STYLE.md"),
    Path(".planning/STATE.md"),
]

LEGACY_PATH_HINTS = {
    "src/feature_forge/pipeline/": "src/feature_forge/methods/malmas/pipeline/",
    "src/feature_forge/agents/": "src/feature_forge/methods/malmas/agents/",
    "src/feature_forge/memory/": "src/feature_forge/methods/malmas/memory/",
    "src/feature_forge/baselines/": "src/feature_forge/methods/",
}


def main() -> int:
    violations: list[str] = []
    for doc_path in DOCS_TO_CHECK:
        if not doc_path.exists():
            continue
        text = doc_path.read_text(encoding="utf-8")
        for legacy, replacement in LEGACY_PATH_HINTS.items():
            if legacy in text:
                violations.append(f"{doc_path}: replace '{legacy}' with '{replacement}'")

        for raw_line in text.splitlines():
            if "@" not in raw_line:
                continue
            for token in raw_line.split():
                if not token.startswith("@"):
                    continue
                maybe_path = token[1:].strip(",:;()[]{}<>\"'")
                if not maybe_path:
                    continue
                if "/" not in maybe_path and "." not in maybe_path:
                    continue
                ref_path = Path(maybe_path)
                if not ref_path.exists():
                    violations.append(f"{doc_path}: broken local reference '@{maybe_path}'")

    if violations:
        print("Found stale documentation references:")
        for line in violations:
            print(f"- {line}")
        return 1

    print("Docs reference check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
