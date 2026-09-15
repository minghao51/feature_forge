"""Expected Hamilton node inventories and metadata validation."""

from __future__ import annotations

import inspect
from types import ModuleType

EXPECTED_NODE_NAMES: dict[str, frozenset[str]] = {
    "feature_forge.dataflows.bronze": frozenset(
        {
            "dataset_computation_request",
            "raw_dataset",
            "bronze_snapshot",
            "source_metadata",
            "bronze_checks",
            "bronze_manifest",
            "bronze_materialization",
        }
    ),
    "feature_forge.dataflows.silver": frozenset(
        {
            "canonical_features",
            "canonical_target",
            "row_ids",
            "fold_assignments",
            "dataset_fingerprint_value",
            "dataset_profile",
            "silver_checks",
            "silver_manifest",
            "silver_materialization",
        }
    ),
    "feature_forge.dataflows.gold": frozenset(
        {
            "method_generation_request",
            "candidate_feature_specs",
            "candidate_execution_batches",
            "candidate_verification",
            "candidate_selection",
            "gold_manifest",
            "gold_materialization",
        }
    ),
    "feature_forge.dataflows.platinum": frozenset(
        {
            "platinum_request",
            "baseline_fold_evidence",
            "candidate_fold_evidence",
            "aggregate_metrics",
            "uncertainty_summary",
            "selection_decisions",
            "platinum_checks",
            "platinum_manifest",
            "platinum_materialization",
        }
    ),
}

REQUIRED_TAGS = {"layer", "cost", "persistence", "owner"}


def node_layer(node_name: str) -> str | None:
    """Return the declared layer for an expected node name."""
    matches = [
        module_name.rsplit(".", maxsplit=1)[-1]
        for module_name, names in EXPECTED_NODE_NAMES.items()
        if node_name in names
    ]
    return matches[0] if len(matches) == 1 else None


def discovered_node_names(module: ModuleType) -> frozenset[str]:
    """Return public functions Hamilton can discover in a stage module."""
    return frozenset(
        name
        for name, value in inspect.getmembers(module, inspect.isfunction)
        if not name.startswith("_") and value.__module__ == module.__name__
    )


def validate_node_inventory(module: ModuleType) -> None:
    """Fail closed when a public stage function is untagged or unexpected."""
    expected = EXPECTED_NODE_NAMES.get(module.__name__)
    if expected is None:
        raise ValueError(f"No expected node inventory registered for {module.__name__}")
    discovered = discovered_node_names(module)
    if discovered != expected:
        raise ValueError(
            f"{module.__name__} node inventory mismatch: "
            f"unexpected={sorted(discovered - expected)}, "
            f"missing={sorted(expected - discovered)}"
        )
    for name in sorted(expected):
        function = getattr(module, name)
        decorators = getattr(function, "decorate_nodes", [])
        tags = {key for decorator in decorators for key in getattr(decorator, "tags", {})}
        if not REQUIRED_TAGS <= tags:
            raise ValueError(
                f"{module.__name__}.{name} is missing node tags: {REQUIRED_TAGS - tags}"
            )
