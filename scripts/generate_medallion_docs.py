#!/usr/bin/env python3
"""Generate or freshness-check deterministic medallion documentation references."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import pkgutil
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, PydanticInvalidForJsonSchema

from feature_forge.verification.documentation import (
    deny_network,
    scrubbed_environment,
    scrubbed_subprocess_environment,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
COMMITTED_ROOT = REPO_ROOT / "docs" / "generated"
DISPLAY_DAGS = ("silver-dag.png", "gold-dag.png", "platinum-dag.png", "case-dag.png")
EVIDENCE_SUBTREES = {"pr8"}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="write committed generated references")
    mode.add_argument(
        "--check", action="store_true", help="compare fresh output with committed files"
    )
    parser.add_argument("--output-root", type=Path, help=argparse.SUPPRESS)
    mode.add_argument("--render-dags-child", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _json(path: Path, payload: Any) -> None:
    _write(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


def _slug(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower().replace("__", "_")


def _contract_models() -> list[type[BaseModel]]:
    import feature_forge.contracts as contracts_package

    models: dict[str, type[BaseModel]] = {}
    module_names = sorted(
        module.name
        for module in pkgutil.iter_modules(contracts_package.__path__)
        if not module.name.startswith("_")
    )
    for module_name in module_names:
        module = importlib.import_module(f"feature_forge.contracts.{module_name}")
        for name, value in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(value, BaseModel)
                and value is not BaseModel
                and value.__module__ == module.__name__
            ):
                models[name] = value
    return [models[name] for name in sorted(models)]


def _generate_contracts(root: Path) -> None:
    rows = [
        "# Contract reference",
        "",
        "Generated from live Pydantic models.",
        "",
        "| Contract | Module | JSON schema |",
        "|---|---|---|",
    ]
    for model in _contract_models():
        schema_path = root / "schemas" / f"{_slug(model.__name__)}.schema.json"
        try:
            schema = model.model_json_schema()
        except PydanticInvalidForJsonSchema:
            rows.append(
                f"| `{model.__name__}` | `{model.__module__}` | Runtime-only Python types |"
            )
            continue
        _json(schema_path, schema)
        rows.append(
            f"| `{model.__name__}` | `{model.__module__}` | "
            f"[`{schema_path.name}`](schemas/{schema_path.name}) |"
        )
    _write(root / "contracts.md", "\n".join(rows))


def _generate_checks(root: Path) -> None:
    from feature_forge.verification.checks import CHECK_DEFINITIONS

    rows = [
        "# Validation check registry",
        "",
        "Generated from `feature_forge.verification.checks`.",
        "",
        "| ID | Layer | Severity | Required | Owner | Description |",
        "|---|---|---|---|---|---|",
    ]
    for item in sorted(CHECK_DEFINITIONS, key=lambda value: value.check_id):
        rows.append(
            f"| `{item.check_id}` | {item.layer.value} | {item.severity.value} | "
            f"{'yes' if item.required else 'no'} | {item.owner} | {item.description} |"
        )
    _write(root / "check-registry.md", "\n".join(rows))


def _generate_catalog(root: Path) -> None:
    from feature_forge.contracts.catalog import CATALOG_SCHEMA_VERSION
    from feature_forge.storage.catalog import CATALOG_DDL

    table_names = sorted(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", CATALOG_DDL))
    text = [
        "# Catalog schema",
        "",
        f"Schema version: `{CATALOG_SCHEMA_VERSION}`.",
        "",
        "The catalog is a rebuildable derived index; verified manifests remain authoritative.",
        "",
        "## Tables",
        "",
        *[f"- `{name}`" for name in table_names],
        "",
        "## Live DDL",
        "",
        "```sql",
        CATALOG_DDL.strip(),
        "```",
    ]
    _write(root / "catalog-schema.md", "\n".join(text))


def _parser_tree() -> list[tuple[str, Any]]:
    from argparse import _SubParsersAction

    from feature_forge.cli import _parser

    found: list[tuple[str, Any]] = []

    def visit(prefix: str, parser: Any) -> None:
        found.append((prefix, parser))
        for action in parser._actions:
            if isinstance(action, _SubParsersAction):
                for name, child in sorted(action.choices.items()):
                    visit(f"{prefix} {name}".strip(), child)

    visit("feature-forge", _parser())
    return found


def _generate_cli(root: Path) -> None:
    from feature_forge.contracts.catalog import CLIExitCode

    lines = ["# CLI reference", "", "Generated from the live argparse command tree.", ""]
    for command, parser in _parser_tree():
        lines.extend([f"## `{command}`", "", "```text", parser.format_help().rstrip(), "```", ""])
    lines.extend(["## Exit codes", "", "| Name | Code |", "|---|---:|"])
    lines.extend(f"| `{item.name}` | {int(item)} |" for item in CLIExitCode)
    _write(root / "cli-reference.md", "\n".join(lines))


def _generate_nodes(root: Path) -> None:
    from feature_forge.dataflows import ExecutionProfile, build_driver

    driver = build_driver(profile=ExecutionProfile.DOCUMENTATION)
    rows = [
        "# Hamilton node inventory",
        "",
        "Generated from the side-effect-free documentation profile.",
        "",
        "| Node | Layer | Owner | Cost | Persistence | Sensitivity | Dependencies |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, node in sorted(driver.graph.nodes.items()):
        tags = node.tags
        dependencies = (
            ", ".join(
                f"`{item.name}`" for item in sorted(node.dependencies, key=lambda value: value.name)
            )
            or "-"
        )
        rows.append(
            f"| `{name}` | {tags.get('layer', '-')} | {tags.get('owner', '-')} | "
            f"{tags.get('cost', '-')} | {tags.get('persistence', '-')} | "
            f"{tags.get('sensitivity', '-')} | {dependencies} |"
        )
    _write(root / "node-inventory.md", "\n".join(rows))


def _generate_graph_sources(root: Path) -> None:
    from feature_forge.dataflows import ExecutionProfile, build_driver
    from feature_forge.dataflows.driver import (
        GOLD_FINAL_VARS,
        PLATINUM_FINAL_VARS,
        SILVER_FINAL_VARS,
    )

    driver = build_driver(profile=ExecutionProfile.DOCUMENTATION)
    definitions = {
        "silver": SILVER_FINAL_VARS,
        "gold": GOLD_FINAL_VARS,
        "platinum": PLATINUM_FINAL_VARS,
        "case": [*SILVER_FINAL_VARS, *GOLD_FINAL_VARS, *PLATINUM_FINAL_VARS],
    }
    for graph_name, final_vars in definitions.items():
        included = _upstream_node_names(driver, final_vars)
        nodes = []
        for name in sorted(included):
            node = driver.graph.nodes[name]
            nodes.append(
                {
                    "name": name,
                    "dependencies": sorted(item.name for item in node.dependencies),
                    "tags": {key: str(value) for key, value in sorted(node.tags.items())},
                }
            )
        _json(
            root / "graphs" / f"{graph_name}.json",
            {"schema_version": "1", "final_vars": sorted(final_vars), "nodes": nodes},
        )


def _generate_portable_dags(root: Path) -> None:
    """Render canonical Mermaid displays from the normalized topology JSON."""
    lines = [
        "# Canonical pipeline DAGs",
        "",
        "These portable Mermaid diagrams are generated from the normalized topology JSON.",
        "The JSON and this page are freshness-checked on every docs build.",
    ]
    for graph_name in ("silver", "gold", "platinum", "case"):
        payload = json.loads((root / "graphs" / f"{graph_name}.json").read_text(encoding="utf-8"))
        nodes = payload["nodes"]
        node_ids = {node["name"]: f"n{index}" for index, node in enumerate(nodes)}
        lines.extend(["", f"## {graph_name.title()} DAG", "", "```mermaid", "graph TD"])
        for node in nodes:
            label = node["name"].replace('"', "&quot;")
            lines.append(f'    {node_ids[node["name"]]}["{label}"]')
        for node in nodes:
            for dependency in node["dependencies"]:
                if dependency in node_ids:
                    lines.append(f"    {node_ids[dependency]} --> {node_ids[node['name']]}")
        lines.extend(
            [
                "```",
                "",
                f"[Normalized topology JSON](graphs/{graph_name}.json)",
            ]
        )
    _write(root / "dags.md", "\n".join(lines))


def _upstream_node_names(driver: Any, final_vars: list[str]) -> set[str]:
    included: set[str] = set()

    def visit(node_name: str) -> None:
        if node_name in included:
            return
        included.add(node_name)
        for dependency in driver.graph.nodes[node_name].dependencies:
            visit(dependency.name)

    for final_var in final_vars:
        visit(final_var)
    return included


def _generate_packages(root: Path) -> None:
    from feature_forge.dataflows.gold import (
        GOLD_OPTIONAL_ARTIFACTS,
        GOLD_REQUIRED_ARTIFACTS,
    )
    from feature_forge.dataflows.platinum import PLATINUM_REQUIRED_ARTIFACTS
    from feature_forge.dataflows.silver import (
        BRONZE_OPTIONAL_ARTIFACTS,
        BRONZE_REFERENCE_REQUIRED_ARTIFACTS,
        BRONZE_SNAPSHOT_REQUIRED_ARTIFACTS,
        SILVER_REQUIRED_ARTIFACTS,
    )

    packages = {
        "Bronze reference": ["manifest.json", "_SUCCESS", *BRONZE_REFERENCE_REQUIRED_ARTIFACTS],
        "Bronze snapshot": [
            "manifest.json",
            "_SUCCESS",
            *BRONZE_SNAPSHOT_REQUIRED_ARTIFACTS,
            *BRONZE_OPTIONAL_ARTIFACTS,
        ],
        "Silver": ["manifest.json", "_SUCCESS", *SILVER_REQUIRED_ARTIFACTS],
        "Gold": ["manifest.json", "_SUCCESS", *GOLD_REQUIRED_ARTIFACTS, *GOLD_OPTIONAL_ARTIFACTS],
        "Platinum": ["manifest.json", "_SUCCESS", *PLATINUM_REQUIRED_ARTIFACTS],
    }
    lines = [
        "# Durable package layouts",
        "",
        "Synthetic directory examples generated from live package contracts. `_SUCCESS` contains the committed manifest SHA-256.",
    ]
    for name, paths in packages.items():
        lines.extend(["", f"## {name}", "", "```text", *sorted(set(paths)), "```"])
    _write(root / "package-layouts.md", "\n".join(lines))


def _render_dags(root: Path) -> None:
    from feature_forge.dataflows import ExecutionProfile, build_driver
    from feature_forge.dataflows.visualization import (
        render_case_dag,
        render_gold_dag,
        render_platinum_dag,
        render_silver_dag,
    )

    driver = build_driver(profile=ExecutionProfile.DOCUMENTATION)
    render_silver_dag(driver, root / "silver-dag.png")
    render_gold_dag(driver, root / "gold-dag.png")
    render_platinum_dag(driver, root / "platinum-dag.png")
    render_case_dag(driver, root / "case-dag.png")


def _generate_dags(root: Path) -> None:
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--render-dags-child", str(root)],
        cwd=REPO_ROOT,
        env=scrubbed_subprocess_environment(),
        check=True,
    )


def _generate_index(root: Path) -> None:
    _write(
        root / "index.md",
        """# Generated pipeline references

These files are generated from live contracts, Hamilton graphs, validation metadata,
catalog DDL, and CLI definitions. Regenerate with:

```bash
uv run --extra pipeline python scripts/generate_medallion_docs.py --write
```

- [Contract reference](contracts.md)
- [Validation check registry](check-registry.md)
- [Catalog schema](catalog-schema.md)
- [CLI reference](cli-reference.md)
- [Hamilton node inventory](node-inventory.md)
- [Durable package layouts](package-layouts.md)

## DAGs

- [Silver DAG](dags.md#silver-dag)
- [Gold DAG](dags.md#gold-dag)
- [Platinum DAG](dags.md#platinum-dag)
- [End-to-end case DAG](dags.md#case-dag)

The canonical displays are portable Mermaid generated from normalized JSON topology;
both are cross-platform freshness contracts. Graphviz PNGs remain unchecked previews:
`--check` requires them but does not compare their bytes or claim that opaque image
payloads are secret-scannable.
""",
    )


def _manifest(root: Path) -> None:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json" and path.suffix != ".png":
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    _json(
        root / "manifest.json",
        {
            "schema_version": "1",
            "freshness_files": files,
            "display_files": list(DISPLAY_DAGS),
        },
    )


def _scan_text_outputs(root: Path) -> None:
    forbidden_patterns = {
        "credential prefix": re.compile(r"(?:sk|rk|pk)-[A-Za-z0-9_-]{8,}"),
        "authorization header": re.compile(r"authorization\s*:\s*bearer", re.IGNORECASE),
        "local user path": re.compile(r"(?:/Users/|/home/|[A-Za-z]:\\\\Users\\\\)"),
    }
    violations: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix == ".png" or path.name == "manifest.json":
            continue
        text = path.read_text(encoding="utf-8")
        for label, pattern in forbidden_patterns.items():
            if pattern.search(text):
                violations.append(f"{path.relative_to(root)}: {label}")
    if violations:
        raise ValueError("Unsafe generated documentation: " + "; ".join(violations))


def generate(root: Path, *, render_display: bool) -> None:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    _generate_contracts(root)
    _generate_checks(root)
    _generate_catalog(root)
    _generate_cli(root)
    _generate_nodes(root)
    _generate_graph_sources(root)
    _generate_portable_dags(root)
    _generate_packages(root)
    if render_display:
        _generate_dags(root)
    _generate_index(root)
    _scan_text_outputs(root)
    _manifest(root)


def _compare(expected: Path, actual: Path) -> list[str]:
    ignored = {Path(name) for name in DISPLAY_DAGS}
    expected_files = {
        path.relative_to(expected)
        for path in expected.rglob("*")
        if path.is_file()
        and path.relative_to(expected) not in ignored
        and not EVIDENCE_SUBTREES.intersection(path.relative_to(expected).parts)
    }
    actual_files = {
        path.relative_to(actual)
        for path in actual.rglob("*")
        if path.is_file()
        and path.relative_to(actual) not in ignored
        and not EVIDENCE_SUBTREES.intersection(path.relative_to(actual).parts)
    }
    problems = [f"missing generated file: {path}" for path in sorted(expected_files - actual_files)]
    problems.extend(
        f"unexpected generated file: {path}" for path in sorted(actual_files - expected_files)
    )
    for path in sorted(expected_files & actual_files):
        if (expected / path).read_bytes() != (actual / path).read_bytes():
            problems.append(f"stale generated file: {path}")
    return problems


def main() -> int:
    args = _args()
    with scrubbed_environment(), deny_network():
        if args.render_dags_child is not None:
            _render_dags(args.render_dags_child)
            return 0
        if args.output_root is not None:
            generate(args.output_root, render_display=False)
            return 0
        if args.write:
            generate(COMMITTED_ROOT, render_display=True)
            print(f"Wrote generated references to {COMMITTED_ROOT.relative_to(REPO_ROOT)}")
            return 0
        with tempfile.TemporaryDirectory(prefix="feature-forge-docs-") as temporary:
            fresh = Path(temporary) / "generated"
            generate(fresh, render_display=False)
            problems = _compare(fresh, COMMITTED_ROOT)
        for display_file in DISPLAY_DAGS:
            if not (COMMITTED_ROOT / display_file).is_file():
                problems.append(f"missing display DAG: {display_file}")
    if problems:
        print("Generated documentation is stale:")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print("Generated documentation is fresh.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
