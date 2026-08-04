"""Canonical definitions for persisted medallion validation checks."""

from __future__ import annotations

from dataclasses import dataclass

from feature_forge.contracts.stages import CheckSeverity, Layer


@dataclass(frozen=True)
class CheckDefinition:
    """Stable documentation and ownership metadata for one validation check."""

    check_id: str
    layer: Layer
    description: str
    owner: str = "verification"
    severity: CheckSeverity = CheckSeverity.ERROR
    required: bool = True


CHECK_DEFINITIONS: tuple[CheckDefinition, ...] = (
    CheckDefinition(
        "BRONZE.SOURCE.READABLE", Layer.BRONZE, "Source contains readable training rows."
    ),
    CheckDefinition(
        "BRONZE.SOURCE.CHECKSUM", Layer.BRONZE, "Source checksum is a valid SHA-256 digest."
    ),
    CheckDefinition(
        "BRONZE.TARGET.DECLARED", Layer.BRONZE, "Target metadata names a non-empty column."
    ),
    CheckDefinition(
        "SILVER.ROW_ID.UNIQUE", Layer.SILVER, "Row IDs are unique and aligned with canonical rows."
    ),
    CheckDefinition(
        "SILVER.TARGET.EXCLUDED",
        Layer.SILVER,
        "The target column is absent from canonical features.",
    ),
    CheckDefinition(
        "SILVER.TARGET.COMPLETE", Layer.SILVER, "The canonical target contains no missing values."
    ),
    CheckDefinition(
        "SILVER.SPLIT.NON_OVERLAP",
        Layer.SILVER,
        "Fold assignments are exhaustive, ordered, and unique.",
    ),
    CheckDefinition(
        "SILVER.SCHEMA.VALID", Layer.SILVER, "The canonical feature schema is non-empty and unique."
    ),
    CheckDefinition(
        "SILVER.LEAKAGE.NO_TARGET_PROXY", Layer.SILVER, "No feature exactly duplicates the target."
    ),
    CheckDefinition("GOLD.ROW_IDS.ALIGNED", Layer.GOLD, "Gold row IDs align exactly with Silver."),
    CheckDefinition(
        "GOLD.CANDIDATES.DECIDED", Layer.GOLD, "Every generated candidate has exactly one decision."
    ),
    CheckDefinition(
        "PLATINUM.FOLDS.IDENTICAL",
        Layer.PLATINUM,
        "Baseline and enhanced evidence use identical folds.",
    ),
    CheckDefinition(
        "PLATINUM.PREDICTIONS.JOINABLE",
        Layer.PLATINUM,
        "Predictions join exactly to authoritative Silver row IDs.",
    ),
    CheckDefinition(
        "PLATINUM.METRICS.RECONSTRUCTABLE",
        Layer.PLATINUM,
        "Aggregate metrics reconstruct from fold records.",
    ),
    CheckDefinition(
        "PLATINUM.UPSTREAM.VERIFIED",
        Layer.PLATINUM,
        "Gold and Silver evidence identities are present and verified.",
    ),
)

CHECKS_BY_ID = {definition.check_id: definition for definition in CHECK_DEFINITIONS}

if len(CHECKS_BY_ID) != len(CHECK_DEFINITIONS):  # pragma: no cover - import-time invariant
    raise RuntimeError("Validation check IDs must be unique")
