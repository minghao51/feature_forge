"""Contract tests for canonical and generated documentation."""

from __future__ import annotations

import importlib
import inspect
import json
import os
import pkgutil
import re
import socket
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import BaseModel

from examples import medallion_operations
from feature_forge.cli import _parser
from feature_forge.contracts import Layer, ManifestRef, RunState
from feature_forge.contracts.orchestration import EffectiveResourcePlan
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
from feature_forge.experiment.case_executor import CaseComputation, CaseComputationInput
from feature_forge.experiment.execution import (
    ExperimentCase,
    ExperimentResult,
    ProcessPoolExecutionAdapter,
)
from feature_forge.verification.checks import CHECKS_BY_ID
from feature_forge.verification.documentation import deny_network, scrubbed_environment
from scripts import build_docs_offline

ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs"
GENERATOR = ROOT / "scripts" / "generate_medallion_docs.py"
OFFLINE_BUILDER = ROOT / "scripts" / "build_docs_offline.py"


def _fake_layer_executor(
    case: ExperimentCase,
    layer: Layer,
    upstream: Mapping[Layer, ManifestRef],
) -> ManifestRef:
    """Picklable application-owned fake used to test the documented boundary."""
    del upstream
    return ManifestRef(layer=layer, run_id=case.run_id or "fake", sha256="f" * 64)


def _fake_layered_compute(
    payload: CaseComputationInput,
    resource_plan: EffectiveResourcePlan | None,
) -> ExperimentResult:
    del resource_plan
    if payload.layer_executor is None:
        raise ValueError("missing fake layer executor")
    ref = payload.layer_executor(payload.case, Layer.BRONZE, {})
    return ExperimentResult(
        dataset=payload.case.dataset,
        method=payload.case.method,
        model=payload.case.model,
        seed=payload.case.seed,
        run_id=payload.case.run_id,
        state=RunState.SUCCEEDED.value if ref.layer is Layer.BRONZE else RunState.FAILED.value,
    )


def _fake_layer_executor_worker(payload: CaseComputationInput) -> ExperimentResult:
    return _fake_layered_compute(payload, None)


def _generated_files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and path.suffix != ".png" and "pr8" not in path.relative_to(root).parts
    }


def _generate(destination: Path) -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONHASHSEED": "19",  # parent seed must not affect normalized topology output
            "FF_LLM__API_KEY": "sk-documentation-sentinel-never-emit",
            "WANDB_API_KEY": "documentation-tracker-sentinel-never-emit",
        }
    )
    subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--write",
            "--output-root",
            str(destination),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_generated_references_are_byte_stable_and_fresh(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _generate(first)
    _generate(second)
    assert _generated_files(first) == _generated_files(second)
    assert _generated_files(first) == _generated_files(DOCS / "generated")


def test_generated_references_cover_every_contract_and_registered_check() -> None:
    import feature_forge.contracts as contracts_package

    contract_reference = (DOCS / "generated/contracts.md").read_text(encoding="utf-8")
    module_names = sorted(
        item.name
        for item in pkgutil.iter_modules(contracts_package.__path__)
        if not item.name.startswith("_")
    )
    for module_name in module_names:
        module = importlib.import_module(f"feature_forge.contracts.{module_name}")
        for _, value in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(value, BaseModel)
                and value is not BaseModel
                and value.__module__ == module.__name__
            ):
                assert f"`{value.__name__}`" in contract_reference

    check_reference = (DOCS / "generated/check-registry.md").read_text(encoding="utf-8")
    for check_id in CHECKS_BY_ID:
        assert f"`{check_id}`" in check_reference
    runtime_sources = sorted((ROOT / "src/feature_forge").rglob("*.py"))
    runtime_ids = {
        check_id
        for source in runtime_sources
        for check_id in re.findall(
            r'"((?:BRONZE|SILVER|GOLD|PLATINUM)\.[A-Z0-9_.]+)"',
            source.read_text(encoding="utf-8"),
        )
    }
    assert runtime_ids == set(CHECKS_BY_ID)


def test_generated_package_layouts_derive_from_runtime_constants() -> None:
    package_reference = (DOCS / "generated/package-layouts.md").read_text(encoding="utf-8")
    declared_paths = {
        *BRONZE_REFERENCE_REQUIRED_ARTIFACTS,
        *BRONZE_SNAPSHOT_REQUIRED_ARTIFACTS,
        *BRONZE_OPTIONAL_ARTIFACTS,
        *SILVER_REQUIRED_ARTIFACTS,
        *GOLD_REQUIRED_ARTIFACTS,
        *GOLD_OPTIONAL_ARTIFACTS,
        *PLATINUM_REQUIRED_ARTIFACTS,
    }
    for path in declared_paths:
        assert path in package_reference


def test_display_dags_are_present_but_stable_sources_are_authoritative() -> None:
    manifest = json.loads((DOCS / "generated/manifest.json").read_text(encoding="utf-8"))
    assert manifest["display_files"] == [
        "silver-dag.png",
        "gold-dag.png",
        "platinum-dag.png",
        "case-dag.png",
    ]
    assert not [item for item in manifest["freshness_files"] if item["path"].endswith(".png")]
    for name in ("silver", "gold", "platinum", "case"):
        assert (DOCS / "generated" / f"{name}-dag.png").is_file()
        assert (DOCS / "generated/graphs" / f"{name}.json").is_file()
    canonical_dags = (DOCS / "generated/dags.md").read_text(encoding="utf-8")
    generated_index = (DOCS / "generated/index.md").read_text(encoding="utf-8")
    assert canonical_dags.count("```mermaid") == 4
    assert "unchecked previews" in generated_index
    for name in ("silver", "gold", "platinum", "case"):
        assert f"dags.md#{name}-dag" in generated_index


def test_generation_environment_is_scrubbed_and_network_is_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FF_LLM__API_KEY", "sk-must-not-survive")
    with scrubbed_environment():
        assert "FF_LLM__API_KEY" not in os.environ
    with deny_network(), pytest.raises(RuntimeError, match="forbids network"):
        socket.create_connection(("127.0.0.1", 9))


def test_offline_mkdocs_wrapper_builds_with_scrubbed_environment() -> None:
    environment = os.environ.copy()
    environment["FF_LLM__API_KEY"] = "sk-wrapper-sentinel-never-emit"
    subprocess.run(
        [sys.executable, str(OFFLINE_BUILDER)],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_offline_mkdocs_wrapper_denies_plugin_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def network_attempt(config: object) -> None:
        del config
        socket.create_connection(("127.0.0.1", 9))

    monkeypatch.setattr("mkdocs.commands.build.build", network_attempt)
    with pytest.raises(RuntimeError, match="forbids network"):
        build_docs_offline.build_docs()


def test_operations_helpers_plan_resume_replay_and_recover(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "run_titanic_openfe_xgboost_42"
    fingerprints = medallion_operations.layer_fingerprints(
        run_id,
        bronze="bronze-fingerprint",
        silver="silver-fingerprint",
        gold="gold-fingerprint",
        platinum="platinum-fingerprint",
    )
    plan = medallion_operations.plan_durable_run(tmp_path, fingerprints)
    assert plan[0]["state"] == "planned"
    assert plan[0]["execution_plan"]["persistent_writes"] is False

    resume = medallion_operations.inspect_resume(
        tmp_path,
        run_id,
        {Layer(name): value for name, value in fingerprints[run_id].items()},
    )
    assert resume.dry_run is True
    assert resume.next_layer is Layer.BRONZE
    assert medallion_operations.recover_abandoned(tmp_path) == {
        "removed_staging": [],
        "removed_locks": [],
        "retained": [],
    }

    silver_ref = ManifestRef(
        layer=Layer.SILVER,
        run_id=run_id,
        sha256="a" * 64,
    )
    gold_ref = ManifestRef(
        layer=Layer.GOLD,
        run_id=run_id,
        sha256="b" * 64,
    )
    silver_package = object()
    gold_package = object()
    monkeypatch.setattr(
        medallion_operations,
        "load_silver_package",
        lambda store, ref: silver_package,
    )

    def replay(store: object, ref: ManifestRef, silver: object, sandbox: object) -> object:
        assert ref == gold_ref
        assert silver is silver_package
        assert sandbox.limits.timeout_seconds == 5
        return gold_package

    monkeypatch.setattr(medallion_operations, "replay_gold_package", replay)
    assert medallion_operations.replay_gold(tmp_path, silver_ref, gold_ref) is gold_package


def test_run_durable_uses_picklable_application_layer_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "run_titanic_openfe_xgboost_42"
    fingerprints = medallion_operations.layer_fingerprints(
        run_id,
        bronze="bronze-fingerprint",
        silver="silver-fingerprint",
        gold="gold-fingerprint",
        platinum="platinum-fingerprint",
    )
    monkeypatch.setattr(
        CaseComputation,
        "_compute_layered",
        staticmethod(_fake_layered_compute),
    )
    result = medallion_operations.run_durable(
        tmp_path,
        fingerprints,
        _fake_layer_executor,
    )
    assert result[0]["state"] == RunState.SUCCEEDED.value

    payload = CaseComputationInput(
        case=ExperimentCase(
            dataset="titanic",
            method="openfe",
            model="xgboost",
            seed=42,
            run_id=run_id,
            artifact_policy="layer_boundaries",
        ),
        settings_data={},
        layer_executor=_fake_layer_executor,
    )
    spawned = ProcessPoolExecutionAdapter(max_workers=1).run(
        [payload],
        _fake_layer_executor_worker,
        progress=False,
    )
    assert spawned[0].state == RunState.SUCCEEDED.value


@pytest.mark.parametrize(
    "arguments",
    [
        ["verify", "run", "run-1"],
        ["verify", "artifact", "bronze:run-1:bronze.json"],
        ["verify", "catalog"],
        ["catalog", "rebuild"],
        ["catalog", "status"],
        ["run", "plan", "--dataset", "titanic", "--method", "openfe"],
        ["--format", "json", "verify", "catalog"],
    ],
)
def test_every_documented_cli_family_parses(arguments: list[str]) -> None:
    _parser().parse_args(arguments)


def test_generated_output_contains_no_secrets_or_local_paths() -> None:
    text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in (DOCS / "generated").rglob("*")
        if path.is_file() and path.suffix != ".png"
    )
    forbidden = (
        "sk-documentation-sentinel",
        "documentation-tracker-sentinel",
        "/Users/",
        "/home/",
        "Authorization: Bearer",
        "WANDB_API_KEY",
    )
    assert not [value for value in forbidden if value in text]


def test_canonical_internal_links_resolve() -> None:
    missing: list[str] = []
    for source in sorted(DOCS.rglob("*.md")):
        text = re.sub(
            r"```.*?```",
            "",
            source.read_text(encoding="utf-8"),
            flags=re.DOTALL,
        )
        for target in re.findall(r"(?<!!)\[[^]]+\]\(([^)]+)\)", text):
            target = target.split("#", maxsplit=1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            resolved = (
                (DOCS / target.lstrip("/")) if target.startswith("/") else (source.parent / target)
            )
            if not resolved.resolve().exists():
                missing.append(f"{source.relative_to(ROOT)} -> {target}")
    assert not missing


def _nav_paths(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, list):
        return set().union(*(_nav_paths(item) for item in value)) if value else set()
    if isinstance(value, dict):
        return set().union(*(_nav_paths(item) for item in value.values())) if value else set()
    return set()


def test_every_markdown_page_has_intentional_navigation_status() -> None:
    config = yaml.safe_load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"))
    navigated = _nav_paths(config["nav"])
    pages = {path.relative_to(DOCS).as_posix() for path in DOCS.rglob("*.md")}
    assert pages == {path for path in navigated if path.endswith(".md")}


def test_canonical_uv_extras_groups_and_profiles_are_supported() -> None:
    import tomllib

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = set(project["project"]["optional-dependencies"])
    groups = set(project["dependency-groups"])
    profile_module = importlib.import_module("feature_forge.dataflows.profile")
    profiles = {item.value for item in profile_module.ExecutionProfile}
    canonical = [
        ROOT / "README.md",
        *(path for path in DOCS.rglob("*.md") if "plan" not in path.parts),
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in canonical)
    for extra in re.findall(r"--extra\s+([\w-]+)", text):
        assert extra in extras
    for group in re.findall(r"--group\s+([\w-]+)", text):
        assert group in groups
    profile_page = (DOCS / "operations/profiles.md").read_text(encoding="utf-8")
    for profile in profiles:
        assert f"`{profile}`" in profile_page
