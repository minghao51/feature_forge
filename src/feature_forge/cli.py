"""Stable command-line interface for planning and local evidence operations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, NoReturn

from feature_forge.contracts.catalog import CLIExitCode, VerificationResult, VerificationStatus


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(int(CLIExitCode.USAGE), f"{self.prog}: error: {message}\n")


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="feature-forge")
    parser.add_argument("--format", choices=("json", "human"), default="human")
    parser.add_argument("--artifact-root", default=".feature_forge_artifacts")
    parser.add_argument(
        "--catalog-path", default=".feature_forge_artifacts/control/catalog/catalog.duckdb"
    )
    parser.add_argument("--lifecycle-root", default=".feature_forge_artifacts/control")
    commands = parser.add_subparsers(dest="command", required=True)

    verify = commands.add_parser("verify")
    verify_commands = verify.add_subparsers(dest="verify_command", required=True)
    run = verify_commands.add_parser("run")
    run.add_argument("run_id")
    artifact = verify_commands.add_parser("artifact")
    artifact.add_argument("uri")
    verify_commands.add_parser("catalog")

    catalog = commands.add_parser("catalog")
    catalog_commands = catalog.add_subparsers(dest="catalog_command", required=True)
    catalog_commands.add_parser("rebuild")
    catalog_commands.add_parser("status")

    run_group = commands.add_parser("run")
    run_commands = run_group.add_subparsers(dest="run_command", required=True)
    plan = run_commands.add_parser("plan")
    plan.add_argument("--dataset", action="append", required=True)
    plan.add_argument("--method", action="append", required=True)
    plan.add_argument("--model", action="append", default=[])
    plan.add_argument("--seed", action="append", type=int, default=[])
    plan.add_argument("--metric")
    plan.add_argument("--artifact-policy", choices=("legacy", "layer_boundaries"), default="legacy")
    plan.add_argument("--fingerprints-json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload, code = _dispatch(args)
    except FileNotFoundError as exc:
        payload = {"schema_version": "1", "status": "missing", "error": str(exc)}
        code = CLIExitCode.MISSING
    except ValueError as exc:
        payload = {"schema_version": "1", "status": "invalid", "error": str(exc)}
        code = CLIExitCode.INVALID
    except Exception as exc:
        payload = {"schema_version": "1", "status": "operational_error", "error": str(exc)}
        code = CLIExitCode.OPERATIONAL_ERROR
    _render(payload, output_format=args.format)
    return int(code)


def _dispatch(args: argparse.Namespace) -> tuple[dict[str, Any], CLIExitCode]:
    if args.command == "run":
        return _run_plan(args), CLIExitCode.OK
    from feature_forge.verification.service import VerificationService

    service = VerificationService(
        artifact_root=args.artifact_root,
        catalog_path=args.catalog_path,
        lifecycle_root=args.lifecycle_root,
    )
    if args.command == "verify":
        if args.verify_command == "run":
            result = service.verify_run(args.run_id)
        elif args.verify_command == "artifact":
            result = service.verify_artifact(args.uri)
        else:
            result = service.verify_catalog()
        return result.model_dump(mode="json"), _exit_for(result)
    if args.catalog_command == "status":
        result = service.verify_catalog()
        return result.model_dump(mode="json"), _exit_for(result)
    from feature_forge.storage.catalog import LocalCatalog
    from feature_forge.storage.local import LocalArtifactStore

    report = LocalCatalog(args.catalog_path).rebuild(
        LocalArtifactStore(args.artifact_root), lifecycle_root=args.lifecycle_root
    )
    code = CLIExitCode.OK if report.status is VerificationStatus.VALID else CLIExitCode.INVALID
    return report.model_dump(mode="json"), code


def _run_plan(args: argparse.Namespace) -> dict[str, Any]:
    from feature_forge.platform import ExperimentalPlatform

    fingerprints = None
    if args.fingerprints_json is not None:
        fingerprints = json.loads(args.fingerprints_json.read_text(encoding="utf-8"))
    result = ExperimentalPlatform().run(
        datasets=args.dataset,
        methods=args.method,
        models=args.model or None,
        seeds=args.seed or None,
        metric=args.metric,
        artifact_policy=args.artifact_policy,
        artifact_root=args.artifact_root,
        layer_fingerprints=fingerprints,
        dry_run=True,
        progress=False,
    )
    return {"schema_version": "1", "status": "valid", "plans": result}


def _exit_for(result: VerificationResult) -> CLIExitCode:
    return {
        VerificationStatus.VALID: CLIExitCode.OK,
        VerificationStatus.INVALID: CLIExitCode.INVALID,
        VerificationStatus.MISSING: CLIExitCode.MISSING,
        VerificationStatus.INCOMPATIBLE: CLIExitCode.INCOMPATIBLE,
        VerificationStatus.OPERATIONAL_ERROR: CLIExitCode.OPERATIONAL_ERROR,
    }[result.status]


def _render(payload: dict[str, Any], *, output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return
    status = payload.get("status", "unknown")
    print(f"status: {status}")
    for issue in payload.get("issues", []):
        print(f"- {issue['code']}: {issue['message']}")
    if "plans" in payload:
        print(json.dumps(payload["plans"], indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
