"""Versioned, layer-specific identity helpers."""

from __future__ import annotations

from typing import Any

from feature_forge.storage.hashing import fingerprint

IDENTITY_SCHEMA_VERSION = "1"


def bronze_fingerprint(*, source_checksum: str, source_contract_version: str = "1") -> str:
    """Identify source bytes and the contract used to interpret them."""
    return fingerprint(
        {
            "kind": "bronze",
            "identity_schema_version": IDENTITY_SCHEMA_VERSION,
            "source_checksum": source_checksum,
            "source_contract_version": source_contract_version,
        }
    )


def silver_fingerprint(
    *,
    bronze_fingerprint_value: str,
    target_name: str,
    task: str,
    canonicalization_config: Any,
    split_policy: Any,
    split_seed: int,
    schema_version: str = "1",
) -> str:
    """Identify canonical data and authoritative fold assignments."""
    return fingerprint(
        {
            "kind": "silver",
            "identity_schema_version": IDENTITY_SCHEMA_VERSION,
            "bronze_fingerprint": bronze_fingerprint_value,
            "target_name": target_name,
            "task": task,
            "canonicalization_config": canonicalization_config,
            "split_policy": split_policy,
            "split_seed": split_seed,
            "schema_version": schema_version,
        }
    )


def gold_input_fingerprint(
    *,
    silver_fingerprint_value: str,
    method_name: str,
    method_version: str,
    method_config: Any,
    prompt_bundle_fingerprint: str,
    generated_contract_version: str,
    selection_policy: Any,
) -> str:
    """Identify all inputs capable of changing accepted generated features."""
    return fingerprint(
        {
            "kind": "gold-input",
            "identity_schema_version": IDENTITY_SCHEMA_VERSION,
            "silver_fingerprint": silver_fingerprint_value,
            "method_name": method_name,
            "method_version": method_version,
            "method_config": method_config,
            "prompt_bundle_fingerprint": prompt_bundle_fingerprint,
            "generated_contract_version": generated_contract_version,
            "selection_policy": selection_policy,
        }
    )


def gold_materialization_fingerprint(
    *,
    gold_input_fingerprint_value: str,
    candidates: Any,
    provenance: Any,
    decisions: Any,
) -> str:
    """Identify immutable generated evidence after provider work completes."""
    return fingerprint(
        {
            "kind": "gold-materialization",
            "identity_schema_version": IDENTITY_SCHEMA_VERSION,
            "gold_input_fingerprint": gold_input_fingerprint_value,
            "candidates": candidates,
            "provenance": provenance,
            "decisions": decisions,
        }
    )


def platinum_input_fingerprint(
    *,
    gold_fingerprint_value: str,
    model_name: str,
    model_version: str,
    model_config: Any,
    metric: str,
    fold_fingerprint: str,
    evaluation_policy: Any,
    uncertainty_policy: Any,
) -> str:
    """Identify evaluation inputs without presentation or tracker settings."""
    return fingerprint(
        {
            "kind": "platinum-input",
            "identity_schema_version": IDENTITY_SCHEMA_VERSION,
            "gold_fingerprint": gold_fingerprint_value,
            "model_name": model_name,
            "model_version": model_version,
            "model_config": model_config,
            "metric": metric,
            "fold_fingerprint": fold_fingerprint,
            "evaluation_policy": evaluation_policy,
            "uncertainty_policy": uncertainty_policy,
        }
    )
