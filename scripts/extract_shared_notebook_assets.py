#!/usr/bin/env python3
"""Post-process Quarto notebook HTML to extract shared CSS/JS into external files.

After ``quarto render``, each HTML file is self-contained (~2.1 MB).
This script identifies CSS <style> and JS <script> blocks that are
identical across multiple notebooks, writes them into shared files under
``docs/notebooks/html/shared/``, and replaces the inline blocks with
``<link>`` / ``<script src>`` references.

Typical savings: ~7 MB (730 KB x 10 files).
"""

from __future__ import annotations

import re
from hashlib import sha256
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
HTML_DIR = REPO_ROOT / "docs" / "notebooks" / "html"
SHARED_DIR = HTML_DIR / "shared"

STYLE_PAT = re.compile(r"(<style[^>]*>)(.*?)(</style>)", re.DOTALL)
SCRIPT_PAT = re.compile(r"(<script[^>]*>)(.*?)(</script>)", re.DOTALL)

MIN_SHARED_COUNT = 2


def _hash(content: str) -> str:
    return sha256(content.encode()).hexdigest()[:12]


def collect_blocks(html_files: list[Path]) -> tuple[dict, dict, dict, dict]:
    style_content: dict[str, str] = {}
    script_content: dict[str, str] = {}
    style_count: dict[str, int] = {}
    script_count: dict[str, int] = {}

    for path in html_files:
        html = path.read_text(encoding="utf-8")
        seen_s: set[str] = set()
        for m in STYLE_PAT.finditer(html):
            h = _hash(m.group(2))
            if h not in style_content:
                style_content[h] = m.group(2)
            seen_s.add(h)
        for h in seen_s:
            style_count[h] = style_count.get(h, 0) + 1

        seen_j: set[str] = set()
        for m in SCRIPT_PAT.finditer(html):
            body = m.group(2)
            if not body.strip():
                continue
            h = _hash(body)
            if h not in script_content:
                script_content[h] = body
            seen_j.add(h)
        for h in seen_j:
            script_count[h] = script_count.get(h, 0) + 1

    return style_content, script_content, style_count, script_count


def write_shared(
    style_content: dict[str, str],
    script_content: dict[str, str],
    style_count: dict[str, int],
    script_count: dict[str, int],
) -> tuple[dict[str, str], dict[str, str]]:
    SHARED_DIR.mkdir(parents=True, exist_ok=True)

    css_map: dict[str, str] = {}
    for h, content in style_content.items():
        if style_count[h] < MIN_SHARED_COUNT:
            continue
        fname = f"shared-{h}.css"
        (SHARED_DIR / fname).write_text(content, encoding="utf-8")
        css_map[h] = fname

    js_map: dict[str, str] = {}
    for h, content in script_content.items():
        if script_count[h] < MIN_SHARED_COUNT:
            continue
        fname = f"shared-{h}.js"
        (SHARED_DIR / fname).write_text(content, encoding="utf-8")
        js_map[h] = fname

    return css_map, js_map


def rewrite_html(
    html_files: list[Path],
    css_map: dict[str, str],
    js_map: dict[str, str],
    style_count: dict[str, int],
    script_count: dict[str, int],
) -> None:
    css_inject = "\n".join(
        f'<link rel="stylesheet" href="shared/{fname}">' for fname in css_map.values()
    )
    js_inject = "\n".join(f'<script src="shared/{fname}"></script>' for fname in js_map.values())

    for path in html_files:
        html = path.read_text(encoding="utf-8")

        def _replace_style(m: re.Match) -> str:
            h = _hash(m.group(2))
            if h in css_map and style_count.get(h, 0) >= MIN_SHARED_COUNT:
                return ""
            return m.group(0)

        html = STYLE_PAT.sub(_replace_style, html)

        def _replace_script(m: re.Match) -> str:
            body = m.group(2)
            if not body.strip():
                return m.group(0)
            h = _hash(body)
            if h in js_map and script_count.get(h, 0) >= MIN_SHARED_COUNT:
                return ""
            return m.group(0)

        html = SCRIPT_PAT.sub(_replace_script, html)

        if css_inject and "</head>" in html:
            html = html.replace("</head>", f"{css_inject}\n</head>", 1)
        if js_inject and "</body>" in html:
            html = html.replace("</body>", f"{js_inject}\n</body>", 1)

        path.write_text(html, encoding="utf-8")


def main() -> None:
    html_files = sorted(HTML_DIR.glob("*.html"))
    if not html_files:
        print("No HTML files found in docs/notebooks/html/")
        return

    before = sum(f.stat().st_size for f in html_files)
    print(f"Processing {len(html_files)} notebooks ({before / 1024 / 1024:.1f} MB)...")

    style_content, script_content, style_count, script_count = collect_blocks(html_files)
    css_map, js_map = write_shared(style_content, script_content, style_count, script_count)

    print(f"  Extracted {len(css_map)} CSS blocks, {len(js_map)} JS blocks")
    rewrite_html(html_files, css_map, js_map, style_count, script_count)

    after_html = sum(f.stat().st_size for f in html_files)
    shared_size = sum(f.stat().st_size for f in SHARED_DIR.glob("*"))
    after = after_html + shared_size
    print(f"  HTML: {before / 1024 / 1024:.1f} MB -> {after_html / 1024 / 1024:.1f} MB")
    print(f"  Shared assets: {shared_size / 1024:.0f} KB")
    print(
        f"  Total: {before / 1024 / 1024:.1f} MB -> {after / 1024 / 1024:.1f} MB ({(before - after) / 1024 / 1024:.1f} MB saved)"
    )


if __name__ == "__main__":
    main()
