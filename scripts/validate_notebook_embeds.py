#!/usr/bin/env python3
"""Validate notebook docs embeds and generated HTML consistency."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DOCS_NOTEBOOKS_DIR = REPO_ROOT / "docs" / "notebooks"
HTML_DIR = DOCS_NOTEBOOKS_DIR / "html"

IFRAME_PATTERN = re.compile(r'<iframe[^>]*src="([^"]+)"[^>]*>', re.IGNORECASE)
A_PATTERN = re.compile(r'<a[^>]*href="([^"]+)"[^>]*>\s*Open in New Tab\s*</a>', re.IGNORECASE)


def _assert_common_hardening(path: Path, content: str) -> list[str]:
    errors: list[str] = []
    if "onclick=" in content:
        errors.append(f"{path}: inline onclick is forbidden")
    if "data-notebook-toggle" not in content:
        errors.append(f"{path}: missing data-notebook-toggle control")
    if 'sandbox="allow-scripts allow-same-origin"' not in content:
        errors.append(f"{path}: iframe sandbox is missing or altered")
    if 'referrerpolicy="no-referrer"' not in content:
        errors.append(f"{path}: missing iframe referrerpolicy=no-referrer")
    return errors


def main() -> int:
    errors: list[str] = []
    md_files = sorted(p for p in DOCS_NOTEBOOKS_DIR.glob("*.md") if p.name != "index.md")

    for md in md_files:
        content = md.read_text(encoding="utf-8")
        errors.extend(_assert_common_hardening(md, content))

        iframe_match = IFRAME_PATTERN.search(content)
        anchor_match = A_PATTERN.search(content)

        if not iframe_match or not anchor_match:
            errors.append(f"{md}: missing iframe or 'Open in New Tab' anchor")
            continue

        iframe_src = iframe_match.group(1)
        anchor_href = anchor_match.group(1)
        if iframe_src != anchor_href:
            errors.append(f"{md}: iframe src and anchor href mismatch")

        html_name = iframe_src.rsplit("/", 1)[-1]
        html_path = HTML_DIR / html_name
        if not html_path.exists():
            errors.append(f"{md}: missing rendered HTML target {html_path}")

    if errors:
        print("Notebook embed validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Validated {len(md_files)} notebook stubs and iframe targets successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
