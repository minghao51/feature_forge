#!/usr/bin/env python3
"""Generate deterministic synthetic PR 8 scale evidence without production data."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter


@dataclass(frozen=True)
class CorpusProfile:
    """Synthetic volume profile used for a repeatable readiness measurement."""

    name: str
    runs: int
    features: int
    lineage_edges: int

    @property
    def stages(self) -> int:
        return self.runs * 4

    @property
    def checks(self) -> int:
        return self.runs * 8

    @property
    def fold_metrics(self) -> int:
        return self.runs * 5

    def counts(self) -> dict[str, int]:
        return {
            "runs": self.runs,
            "features": self.features,
            "lineage_edges": self.lineage_edges,
            "stages": self.stages,
            "checks": self.checks,
            "fold_metrics": self.fold_metrics,
        }


PROFILES = {
    "small": CorpusProfile("small", runs=100, features=2_000, lineage_edges=5_000),
    "expected": CorpusProfile("expected", runs=5_000, features=100_000, lineage_edges=300_000),
    "stress": CorpusProfile("stress", runs=50_000, features=1_000_000, lineage_edges=3_000_000),
}
SEED = 19
STAGES = ("bronze", "silver", "gold", "platinum")
LAYERS = ("bronze", "silver", "gold", "platinum")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--corpus",
        type=Path,
        help="optional JSONL path for the synthetic corpus; otherwise use a temporary file",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        help="optional flat Markdown path used to measure a static table baseline",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def _run_id(index: int) -> str:
    return f"run_{index + 1:08d}"


def _feature_id(index: int) -> str:
    return f"feature_{index + 1:08d}"


def _score(index: int, seed: int) -> float:
    """Return a deterministic bounded score without using mutable randomness."""
    return round(((index * 1_103_515_245 + seed) % 100_000) / 100_000, 6)


def iter_records(profile: CorpusProfile, seed: int) -> Iterator[dict[str, object]]:
    """Yield only synthetic, non-sensitive catalog-shaped records in stable order."""
    for index in range(profile.runs):
        run_id = _run_id(index)
        yield {
            "record_type": "run",
            "run_id": run_id,
            "dataset_id": f"dataset_{index % 12:02d}",
            "method_id": f"method_{index % 5:02d}",
            "model_id": f"model_{index % 4:02d}",
            "seed": seed + index,
            "state": "succeeded" if index % 17 else "failed",
        }

    for index in range(profile.features):
        yield {
            "record_type": "feature",
            "feature_id": _feature_id(index),
            "run_id": _run_id(index % profile.runs),
            "disposition": "selected" if index % 3 else "rejected",
            "decision_code": "accepted" if index % 3 else "quality_gate",
        }

    for index in range(profile.lineage_edges):
        child = index % profile.runs
        parent = (child - (index % 7) - 1) % profile.runs
        yield {
            "record_type": "lineage_edge",
            "child_run_id": _run_id(child),
            "parent_run_id": _run_id(parent),
            "parent_layer": LAYERS[index % len(LAYERS)],
            "ordinal": index,
        }

    for index in range(profile.stages):
        yield {
            "record_type": "stage",
            "run_id": _run_id(index // len(STAGES)),
            "stage": STAGES[index % len(STAGES)],
            "state": "succeeded" if index % 19 else "failed",
        }

    for index in range(profile.checks):
        yield {
            "record_type": "check",
            "run_id": _run_id(index // 8),
            "check_id": f"CHECK.{index % 8 + 1:02d}",
            "severity": "error" if index % 8 == 0 else "info",
            "passed": index % 19 != 0,
            "required": index % 8 == 0,
        }

    for index in range(profile.fold_metrics):
        yield {
            "record_type": "fold_metric",
            "run_id": _run_id(index // 5),
            "fold": index % 5,
            "arm": "enhanced" if index % 2 else "baseline",
            "metric_id": ("r2", "rmse", "accuracy")[index % 3],
            "value": _score(index, seed),
        }


def _record_bytes(record: dict[str, object]) -> bytes:
    return (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def write_corpus(profile: CorpusProfile, destination: Path, seed: int) -> tuple[str, int]:
    """Write a stable JSONL corpus and return its SHA-256 and byte count."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    byte_count = 0
    with destination.open("wb") as handle:
        for record in iter_records(profile, seed):
            payload = _record_bytes(record)
            handle.write(payload)
            digest.update(payload)
            byte_count += len(payload)
    return digest.hexdigest(), byte_count


def write_markdown(profile: CorpusProfile, destination: Path, seed: int) -> int:
    """Render a deliberately flat static reference for size/task comparison."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    byte_count = 0
    with destination.open("wb") as handle:
        header = (
            b"# Synthetic catalog reference\n\n| Type | Primary ID | State/value |\n|---|---|---|\n"
        )
        handle.write(header)
        byte_count += len(header)
        for record in iter_records(profile, seed):
            record_type = str(record["record_type"])
            primary_id = str(
                record.get("run_id")
                or record.get("feature_id")
                or record.get("child_run_id")
                or record.get("check_id")
                or record.get("metric_id")
                or "edge"
            )
            state = str(
                record.get("state") or record.get("value") or record.get("disposition") or "-"
            )
            payload = f"| {record_type} | {primary_id} | {state} |\n".encode()
            handle.write(payload)
            byte_count += len(payload)
    return byte_count


def measure(
    profile: CorpusProfile,
    *,
    results: Path,
    corpus: Path,
    markdown: Path,
    seed: int,
) -> dict[str, object]:
    """Generate and measure one profile."""
    generation_started = perf_counter()
    corpus_sha256, corpus_bytes = write_corpus(profile, corpus, seed)
    generation_seconds = perf_counter() - generation_started

    markdown_started = perf_counter()
    markdown_bytes = write_markdown(profile, markdown, seed)
    markdown_seconds = perf_counter() - markdown_started

    report: dict[str, object] = {
        "schema_version": "1",
        "measurement": {
            "profile": profile.name,
            "seed": seed,
            "counts": profile.counts(),
            "record_count": sum(profile.counts().values()),
            "corpus": {"bytes": corpus_bytes, "sha256": corpus_sha256},
            "flat_markdown": {"bytes": markdown_bytes},
            "elapsed_seconds": {
                "corpus_generation": round(generation_seconds, 6),
                "flat_markdown_render": round(markdown_seconds, 6),
            },
        },
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "system": platform.system(),
            "machine": platform.machine(),
        },
    }
    results.parent.mkdir(parents=True, exist_ok=True)
    results.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    args = _args()
    profile = PROFILES[args.profile]
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.corpus is None or args.markdown is None:
        temporary = tempfile.TemporaryDirectory(prefix="feature-forge-pr8-")
    try:
        root = Path(temporary.name) if temporary is not None else Path()
        corpus = args.corpus or root / f"{profile.name}.jsonl"
        markdown = args.markdown or root / f"{profile.name}.md"
        report = measure(
            profile, results=args.results, corpus=corpus, markdown=markdown, seed=args.seed
        )
        measurement = report["measurement"]
        assert isinstance(measurement, dict)
        print(json.dumps(measurement, indent=2, sort_keys=True))
        return 0
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
