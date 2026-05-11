#!/usr/bin/env python3
"""Fail if forbidden transient files are tracked by git."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
FORBIDDEN_SUBSTRINGS = (
    "__pycache__/",
    ".DS_Store",
)
FORBIDDEN_SUFFIXES = (
    ".pyc",
    ".pyo",
)


def main() -> int:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = [line.strip() for line in result.stdout.splitlines() if line.strip()]

    violations = []
    for path in tracked:
        if any(part in path for part in FORBIDDEN_SUBSTRINGS) or path.endswith(FORBIDDEN_SUFFIXES):
            violations.append(path)

    if violations:
        print("Tracked transient artifacts found. Remove them from git index:")
        for path in sorted(violations):
            print(f"- {path}")
        return 1

    print("Repo hygiene check passed: no forbidden tracked artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
