#!/usr/bin/env python3
"""Deterministically generate medallion stage-DAG documentation.

The generator introspects the four Hamilton stage modules
(``feature_forge.dataflows.{bronze,silver,gold,platinum}``) and the canonical
``EXPECTED_NODE_NAMES`` inventory from
``feature_forge.dataflows.inventory``. For every stage it emits:

* a Mermaid ``graph TD`` DAG whose intra-stage edges are derived strictly from
  the node function signatures (a parameter whose name matches another node in
  the same stage becomes an upstream -> downstream edge);
* a node metadata table (layer / cost / persistence / sensitivity / owner /
  return type / upstream fan-in);
* a Bronze -> Silver -> Gold -> Platinum durable-boundary architecture diagram
  whose cross-stage edges are derived from the package-typed external inputs
  each stage consumes (e.g. ``SilverPackage``, ``GoldPackage``).

The output is fully deterministic: node and edge order is sorted, and no
timestamps are embedded, so the committed ``docs/generated/stage_dags.md`` can
be byte-compared against a fresh render (``--check``).
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from feature_forge.dataflows.inventory import EXPECTED_NODE_NAMES

__all__ = ["EXPECTED_NODE_NAMES", "build_markdown"]

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "generated" / "stage_dags.md"

STAGE_MODULE_PREFIX = "feature_forge.dataflows."
LAYER_KEYWORDS = ("bronze", "silver", "gold", "platinum")

# Stage order is taken from the inventory's deterministic declaration order.
STAGE_ORDER: list[str] = [
    name[len(STAGE_MODULE_PREFIX) :]
    for name in EXPECTED_NODE_NAMES
    if name.startswith(STAGE_MODULE_PREFIX)
]


def _load_stage_module(stage: str) -> ModuleType:
    """Import a stage module by its medallion layer name."""
    return importlib.import_module(f"{STAGE_MODULE_PREFIX}{stage}")


def node_function(module: ModuleType, name: str) -> Callable[..., Any]:
    """Return the public node function for ``name`` in ``module``."""
    return cast("Callable[..., Any]", getattr(module, name))


def node_tags(function: Callable[..., Any]) -> dict[str, str]:
    """Extract Hamilton tag metadata from a decorated node function.

    Hamilton records tag decorators on ``function.decorate_nodes``; each entry
    exposes a ``.tags`` mapping. The optional-cache decorator carries no tags.
    """
    tags: dict[str, str] = {}
    for decorator in getattr(function, "decorate_nodes", []):
        found = getattr(decorator, "tags", None)
        if isinstance(found, dict):
            tags.update({str(key): str(value) for key, value in found.items()})
    return tags


def _annotation_name(annotation: object) -> str:
    """Best-effort human-readable name for a type annotation."""
    if annotation is inspect.Signature.empty:
        return "Any"
    if isinstance(annotation, str):
        return annotation
    name = getattr(annotation, "__name__", None)
    if name:
        return str(name)
    return str(annotation).replace("typing.", "")


def node_return_annotation(function: Callable[..., Any]) -> str:
    """Return a readable name for a node's return annotation."""
    return _annotation_name(inspect.signature(function).return_annotation)


def _annotation_layer(annotation: object) -> str | None:
    """Map a parameter annotation to a medallion layer keyword, if any."""
    name = _annotation_name(annotation).lower()
    for layer in LAYER_KEYWORDS:
        if layer in name:
            return layer
    return None


def intra_stage_edges(stage: str, module: ModuleType) -> list[tuple[str, str]]:
    """Upstream -> downstream edges where a node's parameter is another node.

    A parameter name that matches a node declared in the same stage is treated
    as an intra-stage dependency (the parameter feeds the node).
    """
    expected = EXPECTED_NODE_NAMES[f"{STAGE_MODULE_PREFIX}{stage}"]
    edges: list[tuple[str, str]] = []
    for name in expected:
        function = node_function(module, name)
        for param in inspect.signature(function).parameters.values():
            if param.name in expected and param.name != name:
                edges.append((param.name, name))
    edges.sort()
    return edges


def cross_stage_edges() -> list[tuple[str, str]]:
    """Durable-boundary edges derived from consumed package-typed inputs."""
    edges: set[tuple[str, str]] = set()
    for stage in STAGE_ORDER:
        module = _load_stage_module(stage)
        expected = EXPECTED_NODE_NAMES[f"{STAGE_MODULE_PREFIX}{stage}"]
        for name in expected:
            function = node_function(module, name)
            for param in inspect.signature(function).parameters.values():
                upstream = _annotation_layer(param.annotation)
                if upstream is not None and upstream != stage and upstream in STAGE_ORDER:
                    edges.add((upstream, stage))
    return sorted(edges)


def _materialization_node(stage: str) -> str:
    """The durable-boundary materialization node for a stage."""
    expected = EXPECTED_NODE_NAMES[f"{STAGE_MODULE_PREFIX}{stage}"]
    for name in sorted(expected):
        if name.endswith("_materialization"):
            return name
    raise ValueError(f"No materialization boundary node found for stage {stage}")


def _mermaid_stage_graph(stage: str, module: ModuleType) -> str:
    """Render a per-stage Mermaid ``graph TD`` DAG."""
    expected = EXPECTED_NODE_NAMES[f"{STAGE_MODULE_PREFIX}{stage}"]
    nodes = sorted(expected)
    lines = ["graph TD"]
    for name in nodes:
        lines.append(f'    {name}["{name}"]')
    lines.append("")
    materialization = _materialization_node(stage)
    for upstream, downstream in intra_stage_edges(stage, module):
        lines.append(f"    {upstream} --> {downstream}")
    lines.append("")
    lines.append("    classDef boundary fill:#cfe3ff,stroke:#3a6ea5,stroke-width:2px;")
    lines.append(f"    class {materialization} boundary;")
    return "\n".join(lines)


def _stage_metadata_table(stage: str, module: ModuleType) -> str:
    """Render a node metadata table for a stage."""
    expected = EXPECTED_NODE_NAMES[f"{STAGE_MODULE_PREFIX}{stage}"]
    edges = intra_stage_edges(stage, module)
    fan_in: dict[str, int] = dict.fromkeys(expected, 0)
    for _upstream, downstream in edges:
        fan_in[downstream] += 1

    header = "| Node | Layer | Cost | Persistence | Sensitivity | Owner | Returns | Upstream deps |"
    divider = "| --- | --- | --- | --- | --- | --- | --- | --- |"
    rows = [header, divider]
    for name in sorted(expected):
        function = node_function(module, name)
        tags = node_tags(function)
        upstream = ", ".join(sorted(u for u, d in edges if d == name)) or "—"
        rows.append(
            "| `{name}` | {layer} | {cost} | {persistence} | {sensitivity} | "
            "{owner} | `{returns}` | {fan_in} ({upstream}) |".format(
                name=name,
                layer=tags.get("layer", "—"),
                cost=tags.get("cost", "—"),
                persistence=tags.get("persistence", "—"),
                sensitivity=tags.get("sensitivity", "—"),
                owner=tags.get("owner", "—"),
                returns=node_return_annotation(function),
                fan_in=fan_in[name],
                upstream=upstream,
            )
        )
    return "\n".join(rows)


def _mermaid_architecture_graph() -> str:
    """Render the Bronze -> Silver -> Gold -> Platinum durable-boundary DAG."""
    lines = ["graph LR"]
    for stage in STAGE_ORDER:
        node = _materialization_node(stage)
        lines.append(f'    {stage}["{stage.capitalize()}<br/>{node}"]')
    lines.append("")
    for upstream, downstream in cross_stage_edges():
        lines.append(f"    {upstream} --> {downstream}")
    lines.append("")
    lines.append("    classDef boundary fill:#cfe3ff,stroke:#3a6ea5,stroke-width:2px;")
    for stage in STAGE_ORDER:
        lines.append(f"    class {stage} boundary;")
    return "\n".join(lines)


def build_markdown() -> str:
    """Render the full stage-DAG documentation as deterministic Markdown."""
    sections: list[str] = []
    sections.append("# Medallion Stage DAGs")
    sections.append("")
    sections.append(
        "> Auto-generated by `scripts/generate_stage_dag_docs.py`. Do not edit "
        "by hand — regenerate with `uv run python scripts/generate_stage_dag_docs.py` "
        "and commit the output. The document is deterministic (no timestamps), so it "
        "can be byte-compared against a fresh render via `--check`."
    )
    sections.append("")

    total_nodes = sum(len(names) for names in EXPECTED_NODE_NAMES.values())
    sections.append(
        f"The platform decomposes each experiment case into four durable Hamilton "
        f"stages — **Bronze**, **Silver**, **Gold**, and **Platinum** — spanning "
        f"{total_nodes} tagged nodes across "
        f"{len(STAGE_ORDER)} modules "
        f"(`{STAGE_MODULE_PREFIX}{{bronze,silver,gold,platinum}}`)."
    )
    sections.append("")

    sections.append("## Durable-boundary architecture")
    sections.append("")
    sections.append(
        "Each stage persists a verified package through its `*_materialization` "
        "boundary node. Downstream stages consume the upstream durable packages as "
        "external inputs and never recompute them; cross-stage edges below are "
        "derived from the package-typed inputs each stage declares."
    )
    sections.append("")
    sections.append("```mermaid")
    sections.append(_mermaid_architecture_graph())
    sections.append("```")
    sections.append("")

    for stage in STAGE_ORDER:
        module = _load_stage_module(stage)
        expected = EXPECTED_NODE_NAMES[f"{STAGE_MODULE_PREFIX}{stage}"]
        sections.append(f"## {stage.capitalize()} stage")
        sections.append("")
        sections.append(
            f"Module `{STAGE_MODULE_PREFIX}{stage}` — {len(expected)} nodes. "
            "Intra-stage edges are derived from node function signatures."
        )
        sections.append("")
        sections.append("### DAG")
        sections.append("")
        sections.append("```mermaid")
        sections.append(_mermaid_stage_graph(stage, module))
        sections.append("```")
        sections.append("")
        sections.append("### Node metadata")
        sections.append("")
        sections.append(_stage_metadata_table(stage, module))
        sections.append("")

    return "\n".join(sections)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    * default: write the rendered document to ``docs/generated/stage_dags.md``;
    * ``--output PATH``: write to an explicit path instead;
    * ``--check``: compare against the default committed document and exit
      non-zero when stale — never mutates any file.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination path for the rendered document (default: %(default)s).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail (exit 1) if the default document is stale; do not write.",
    )
    args = parser.parse_args(argv)

    content = build_markdown()

    if args.check:
        target = DEFAULT_OUTPUT
        if not target.exists():
            print(f"check failed: {target} is missing", file=sys.stderr)
            return 1
        existing = target.read_text(encoding="utf-8")
        if existing == content:
            print(f"check ok: {target} is up to date")
            return 0
        print(
            "check failed: docs/generated/stage_dags.md is stale; "
            "run `uv run python scripts/generate_stage_dag_docs.py`",
            file=sys.stderr,
        )
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
