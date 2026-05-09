#!/usr/bin/env python3
"""Generate MkDocs notebook stubs from .qmd frontmatter.

Reads notebooks/*.qmd files, extracts YAML frontmatter, and generates
docs/notebooks/*.md stubs with iframe embeds pointing to rendered HTML.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).parent.parent
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
DOCS_NOTEBOOKS_DIR = REPO_ROOT / "docs" / "notebooks"
HTML_OUTPUT_DIR = DOCS_NOTEBOOKS_DIR / "html"


def parse_frontmatter(qmd_path: Path) -> dict:
    """Extract YAML frontmatter from a .qmd file."""
    content = qmd_path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if not match:
        return {}
    return yaml.safe_load(match.group(1)) or {}


def slugify(name: str) -> str:
    """Convert '01_quick_start' -> '01-quick-start'."""
    return name.replace("_", "-")


def get_base_path() -> str:
    """Extract base path from mkdocs.yml site_url for absolute iframe src."""
    mkdocs_path = REPO_ROOT / "mkdocs.yml"
    if not mkdocs_path.exists():
        return "/feature-forge"
    mkdocs = yaml.safe_load(mkdocs_path.read_text(encoding="utf-8"))
    site_url = mkdocs.get("site_url", "")
    from urllib.parse import urlparse
    parsed = urlparse(site_url)
    return parsed.path.rstrip("/") or "/feature-forge"


def generate_stub(qmd_path: Path, base_path: str) -> str:
    """Generate MkDocs stub markdown for a single notebook."""
    front = parse_frontmatter(qmd_path)
    title = front.get("title", qmd_path.stem.replace("_", " ").title())
    description = front.get("description", "")
    html_name = f"{qmd_path.stem}.html"
    iframe_src = f"{base_path}/notebooks/html/{html_name}"

    lines = [
        f"# {title}",
        "",
        f"{description}",
        "",
        "## Notebook",
        "",
        '<div style="margin: 0 -0.8rem">',
        f'  <iframe src="{iframe_src}"',
        '    style="width:100%; height:700px; border:1px solid var(--md-default-fg-color--lightest); border-radius:4px;"',
        '    loading="lazy"></iframe>',
        "</div>",
        "",
        "## Run Locally",
        "",
        "```bash",
        "# Ensure your DeepSeek API key is set",
        "export FF_LLM__API_KEY=sk-...",
        "",
        "# Render this notebook",
        f'uv run quarto render notebooks/{qmd_path.name}',
        "",
        "# Or render all notebooks",
        "uv run quarto render notebooks/",
        "```",
    ]
    return "\n".join(lines)


def generate_index(stubs: list[tuple[str, str, str]]) -> str:
    """Generate the notebooks index page."""
    lines = [
        "# Notebooks",
        "",
        "Interactive tutorials demonstrating Feature Forge capabilities.",
        "All notebooks are rendered with [Quarto](https://quarto.org) and embedded below.",
        "",
        "| Notebook | Description |",
        "|----------|-------------|",
    ]
    for slug, title, description in stubs:
        lines.append(f"| [{title}]({slug}.md) | {description} |")
    lines.extend([
        "",
        "## Prerequisites",
        "",
        "```bash",
        "export FF_LLM__API_KEY=sk-your-deepseek-key",
        "uv sync --extra docs",
        "```",
    ])
    return "\n".join(lines)


def main() -> None:
    DOCS_NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    base_path = get_base_path()

    qmd_files = sorted(NOTEBOOKS_DIR.glob("*.qmd"))
    stubs: list[tuple[str, str, str]] = []

    for qmd in qmd_files:
        if qmd.name.startswith("_"):
            continue
        front = parse_frontmatter(qmd)
        title = front.get("title", qmd.stem.replace("_", " ").title())
        description = front.get("description", "")
        slug = slugify(qmd.stem)

        stub_md = generate_stub(qmd, base_path)
        stub_path = DOCS_NOTEBOOKS_DIR / f"{slug}.md"
        stub_path.write_text(stub_md, encoding="utf-8")
        stubs.append((slug, title, description))
        print(f"  generated docs/notebooks/{slug}.md")

    # Generate index
    index_path = DOCS_NOTEBOOKS_DIR / "index.md"
    index_path.write_text(generate_index(stubs), encoding="utf-8")
    print(f"  generated docs/notebooks/index.md")

    print(f"\nDone. {len(stubs)} notebook stubs generated.")
    print("Next: uv run quarto render notebooks/ && uv run mkdocs build")


if __name__ == "__main__":
    main()
