"""Command-line planning and safe Hamilton operations."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from feature_forge.exceptions import FeatureForgeError


def _add_format(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--format", dest="leaf_format", choices=("json", "human"))


def _add_roots(parser: argparse.ArgumentParser, *, cache: bool = False) -> None:
    parser.add_argument("--artifact-root")
    if cache:
        parser.add_argument("--cache-path")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="feature-forge")
    parser.add_argument("--format", choices=("json", "human"), default="human")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run")
    run_commands = run.add_subparsers(dest="run_command", required=True)
    plan = run_commands.add_parser("plan")
    _add_format(plan)
    plan.add_argument("--dataset", action="append", required=True)
    plan.add_argument("--method", action="append", required=True)
    plan.add_argument("--model", action="append", default=[])
    plan.add_argument("--seed", action="append", type=int, default=[])
    plan.add_argument("--mode")
    plan.add_argument("--cv-folds", type=int)
    plan.add_argument("--metric")
    plan.add_argument("--artifact-root")

    cache = commands.add_parser("cache")
    cache_commands = cache.add_subparsers(dest="cache_command", required=True)
    status = cache_commands.add_parser("status")
    _add_format(status)
    _add_roots(status, cache=True)
    inspect_cache = cache_commands.add_parser("inspect")
    _add_format(inspect_cache)
    _add_roots(inspect_cache, cache=True)
    inspect_cache.add_argument("--limit", type=int, default=100)
    inspect_cache.add_argument("--node")
    inspect_cache.add_argument("--layer", choices=("bronze", "silver", "gold", "platinum"))
    gc = cache_commands.add_parser("gc")
    _add_format(gc)
    _add_roots(gc, cache=True)
    gc.add_argument("--max-age-days", type=float)
    gc.add_argument("--max-size-mb", type=float)
    gc.add_argument("--dry-run", action="store_true")
    clear = cache_commands.add_parser("clear")
    _add_format(clear)
    _add_roots(clear, cache=True)
    clear.add_argument("--force", action="store_true")
    clear.add_argument("--dry-run", action="store_true")

    artifacts = commands.add_parser("artifacts")
    artifact_commands = artifacts.add_subparsers(dest="artifacts_command", required=True)
    list_artifacts = artifact_commands.add_parser("list")
    _add_format(list_artifacts)
    _add_roots(list_artifacts)
    list_artifacts.add_argument("--layer", choices=("bronze", "silver", "gold", "platinum"))
    list_artifacts.add_argument("--case")
    verify = artifact_commands.add_parser("verify")
    _add_format(verify)
    _add_roots(verify)
    verify.add_argument("--layer", choices=("bronze", "silver", "gold", "platinum"))
    verify.add_argument("--case")
    verify.add_argument("--run-id")
    return parser


def _run_plan(args: argparse.Namespace) -> dict[str, Any]:
    from feature_forge.platform import ExperimentalPlatform

    config: dict[str, Any] = {}
    if args.metric is not None:
        config["metric"] = args.metric
    if args.artifact_root is not None:
        config["dataflow"] = {"artifact_root": args.artifact_root}
    plans = ExperimentalPlatform(config=config or None).plan(
        datasets=args.dataset,
        methods=args.method,
        models=args.model or None,
        mode=args.mode,
        cv_folds=args.cv_folds,
        seeds=args.seed or None,
    )
    return {"schema_version": "1", "status": "valid", "plans": plans}


def _settings(args: argparse.Namespace) -> Any:
    from feature_forge.config import Settings

    dataflow: dict[str, Any] = {}
    if getattr(args, "artifact_root", None) is not None:
        dataflow["artifact_root"] = args.artifact_root
    if getattr(args, "cache_path", None) is not None:
        dataflow["cache"] = {"path": args.cache_path}
    return Settings(dataflow=dataflow) if dataflow else Settings()


def _cache_manager(args: argparse.Namespace) -> tuple[Any, Any]:
    from feature_forge.dataflows.driver import resolve_cache_path
    from feature_forge.storage.hamilton_cache import HamiltonCacheManager

    settings = _settings(args)
    return settings, HamiltonCacheManager(resolve_cache_path(settings=settings))


def _run_cache(args: argparse.Namespace) -> dict[str, Any]:
    settings, manager = _cache_manager(args)
    if args.cache_command == "status":
        result = manager.status()
    elif args.cache_command == "inspect":
        result = {
            "path": str(manager.path),
            "entries": [
                entry.to_dict()
                for entry in manager.inspect(limit=args.limit, node=args.node, layer=args.layer)
            ],
        }
    elif args.cache_command == "gc":
        max_age_days = (
            args.max_age_days
            if args.max_age_days is not None
            else settings.dataflow.cache.max_age_days
        )
        max_size_mb = (
            args.max_size_mb
            if args.max_size_mb is not None
            else settings.dataflow.cache.max_size_mb
        )
        result = manager.collect(
            max_age_days=max_age_days,
            max_size_mb=max_size_mb,
            dry_run=args.dry_run,
        )
    elif args.cache_command == "clear":
        if not args.force and not args.dry_run:
            raise ValueError("cache clear requires --force (or use --dry-run)")
        result = manager.collect(clear=True, dry_run=args.dry_run)
    else:  # pragma: no cover - argparse enforces choices
        raise ValueError(f"unknown cache command: {args.cache_command}")
    return {"schema_version": "1", "status": "valid", "cache": result}


def _artifact_rows(args: argparse.Namespace) -> tuple[Any, list[dict[str, Any]]]:
    from feature_forge.contracts.artifacts import ArtifactNamespace
    from feature_forge.contracts.stages import Layer
    from feature_forge.storage.local import LocalArtifactStore

    settings = _settings(args)
    store = LocalArtifactStore(settings.dataflow.artifact_root)
    requested_layer = Layer(args.layer) if args.layer is not None else None
    rows: list[dict[str, Any]] = []
    for layer, path in store.iter_package_paths():
        if requested_layer is not None and layer is not requested_layer:
            continue
        if getattr(args, "run_id", None) is not None and path.name != args.run_id:
            continue
        row: dict[str, Any] = {
            "layer": layer.value,
            "run_id": path.name,
            "path": str(path),
        }
        try:
            ref = store.get_manifest_ref(ArtifactNamespace(layer=layer, run_id=path.name))
            if ref is None:
                raise ValueError("manifest reference is missing")
            manifest = store.load_manifest(ref)
            if args.case is not None and manifest.case_fingerprint != args.case:
                continue
            row.update(
                {
                    "complete": True,
                    "case_fingerprint": manifest.case_fingerprint,
                    "reuse_fingerprint": manifest.reuse_fingerprint,
                    "layer_fingerprint": manifest.layer_fingerprint,
                    "manifest_ref": ref.model_dump(mode="json"),
                    "upstream_manifests": [
                        upstream.model_dump(mode="json") for upstream in manifest.upstream_manifests
                    ],
                }
            )
        except (FeatureForgeError, OSError, ValueError) as exc:
            if args.case is not None:
                continue
            row.update({"complete": False, "error": str(exc)})
        rows.append(row)
    return store, rows


def _semantic_verify(store: Any, row: dict[str, Any]) -> None:
    from feature_forge.contracts.artifacts import ManifestRef
    from feature_forge.contracts.stages import Layer
    from feature_forge.dataflows._io import (
        load_gold_package,
        load_platinum_package,
        load_silver_package,
    )

    ref = ManifestRef.model_validate(row["manifest_ref"])
    if ref.layer is Layer.SILVER:
        load_silver_package(store, ref)
    elif ref.layer is Layer.GOLD:
        load_gold_package(store, ref)
    elif ref.layer is Layer.PLATINUM:
        load_platinum_package(store, ref)


def _run_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    store, rows = _artifact_rows(args)
    if args.artifacts_command == "list":
        return {"schema_version": "1", "status": "valid", "packages": rows}
    if args.artifacts_command != "verify":  # pragma: no cover - argparse choices
        raise ValueError(f"unknown artifacts command: {args.artifacts_command}")

    verified: list[dict[str, Any]] = []
    for row in rows:
        result = {"layer": row["layer"], "run_id": row["run_id"], "valid": False}
        if not row.get("complete"):
            result["errors"] = [row.get("error", "package is incomplete")]
        else:
            try:
                _semantic_verify(store, row)
                result.update({"valid": True, "errors": []})
            except (FeatureForgeError, OSError, ValueError) as exc:
                result["errors"] = [str(exc)]
        verified.append(result)
    return {
        "schema_version": "1",
        "status": ("valid" if verified and all(row["valid"] for row in verified) else "invalid"),
        "packages": verified,
    }


def _dispatch(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "run":
        return _run_plan(args)
    if args.command == "cache":
        return _run_cache(args)
    if args.command == "artifacts":
        return _run_artifacts(args)
    raise ValueError(f"unknown command: {args.command}")  # pragma: no cover


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_format = getattr(args, "leaf_format", None) or args.format
    try:
        payload = _dispatch(args)
        exit_code = 0 if payload.get("status") != "invalid" else 2
    except (FeatureForgeError, KeyError, OSError, ValueError) as exc:
        payload = {"schema_version": "1", "status": "invalid", "error": str(exc)}
        exit_code = 2
    if output_format == "json":
        sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    else:
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
