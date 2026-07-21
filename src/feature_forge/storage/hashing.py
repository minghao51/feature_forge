"""Deterministic, secret-free identity and file hashing helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, SecretStr

_SECRET_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "bearer_token",
    "credential",
    "credentials",
    "password",
    "passwd",
    "secret",
    "token",
}
_SECRET_KEY_SUFFIXES = (
    "_api_key",
    "_authorization",
    "_credential",
    "_credentials",
    "_password",
    "_secret",
    "_token",
)


def _is_secret_key(key: str | None) -> bool:
    if key is None:
        return False
    normalized = key.lower().replace("-", "_")
    return normalized in _SECRET_KEYS or normalized.endswith(_SECRET_KEY_SUFFIXES)


def normalize_secret_free(value: Any, *, key: str | None = None) -> Any:
    """Return JSON-compatible data with secret values replaced by presence flags."""
    if _is_secret_key(key):
        return {"configured": value is not None and value != ""}
    if isinstance(value, SecretStr):
        return {"configured": bool(value.get_secret_value())}
    if isinstance(value, BaseModel):
        return normalize_secret_free(value.model_dump(mode="python"), key=key)
    if isinstance(value, Enum):
        return normalize_secret_free(value.value, key=key)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        raw_value = bytes(value)
        return {"sha256": hashlib.sha256(raw_value).hexdigest(), "size_bytes": len(raw_value)}
    if isinstance(value, Mapping):
        normalized_mapping: dict[str, Any] = {}
        for item_key, item_value in sorted(value.items(), key=lambda item: str(item[0])):
            normalized_key = str(item_key)
            if normalized_key in normalized_mapping:
                raise ValueError(
                    f"mapping keys collide after string normalization: {normalized_key}"
                )
            normalized_mapping[normalized_key] = normalize_secret_free(
                item_value,
                key=normalized_key,
            )
        return normalized_mapping
    if isinstance(value, (set, frozenset)):
        normalized_items = [normalize_secret_free(item) for item in value]
        return sorted(
            normalized_items,
            key=lambda item: json.dumps(
                item,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ),
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytearray)):
        return [normalize_secret_free(item) for item in value]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a value deterministically after secret redaction."""
    normalized = normalize_secret_free(value)
    return json.dumps(
        normalized,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Return the SHA-256 digest for bytes."""
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it entirely into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    """Hash a canonical, secret-free payload."""
    return sha256_bytes(canonical_json_bytes(value))


def dataset_fingerprint(
    *,
    source_checksum: str,
    canonicalization_config: Any,
    schema_version: str,
    target_name: str,
    task: str,
    split_policy: Any,
    split_seed: int,
) -> str:
    """Build the stable Silver identity of a canonical dataset."""
    from feature_forge.contracts.identity import bronze_fingerprint, silver_fingerprint

    return silver_fingerprint(
        bronze_fingerprint_value=bronze_fingerprint(source_checksum=source_checksum),
        target_name=target_name,
        task=task,
        canonicalization_config=canonicalization_config,
        split_policy=split_policy,
        split_seed=split_seed,
        schema_version=schema_version,
    )


def case_fingerprint(
    *,
    dataset_fingerprint_value: str,
    method_name: str,
    method_distribution_version: str,
    method_config: Any,
    prompt_bundle_fingerprint: str,
    model_name: str,
    evaluation_config: Any,
    seed: int,
    source_commit: str | None,
    lockfile_fingerprint: str | None,
) -> str:
    """Build the stable identity of an experiment case."""
    return fingerprint(
        {
            "kind": "case",
            "dataset_fingerprint": dataset_fingerprint_value,
            "method_name": method_name,
            "method_distribution_version": method_distribution_version,
            "method_config": method_config,
            "prompt_bundle_fingerprint": prompt_bundle_fingerprint,
            "model_name": model_name,
            "evaluation_config": evaluation_config,
            "seed": seed,
            "source_commit": source_commit,
            "lockfile_fingerprint": lockfile_fingerprint,
        }
    )
