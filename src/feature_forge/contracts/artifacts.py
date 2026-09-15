"""Persisted artifact descriptors and namespace references."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import AfterValidator, BeforeValidator, Field

from feature_forge.contracts.base import ContractModel
from feature_forge.contracts.stages import Layer

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9_-])?$")
_WINDOWS_RESERVED_NAMES = {
    "AUX",
    "CON",
    "NUL",
    "PRN",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def validate_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or value != path.as_posix()
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
    ):
        raise ValueError("relative_path must be a normalized relative path")
    if any(
        part.split(".", maxsplit=1)[0].upper() in _WINDOWS_RESERVED_NAMES for part in path.parts
    ):
        raise ValueError("relative_path contains a reserved platform filename")
    return value


def validate_identifier(value: str) -> str:
    if (
        not _SAFE_ID_PATTERN.fullmatch(value)
        or value.split(".", maxsplit=1)[0].upper() in _WINDOWS_RESERVED_NAMES
    ):
        raise ValueError("identifier contains unsupported path characters")
    return value


def _validate_sha256(value: str) -> str:
    if not _SHA256_PATTERN.fullmatch(value):
        raise ValueError("sha256 must be a lowercase 64-character hexadecimal digest")
    return value


# Shared annotated scalar types. ``RunId``/``RelativePath`` run the identifier
# / path check before pydantic applies ``min_length``; ``Sha256`` runs its hex
# check after. Reuse these instead of per-class ``@field_validator`` methods.
RunId = Annotated[str, BeforeValidator(validate_identifier), Field(min_length=1)]
RelativePath = Annotated[str, BeforeValidator(validate_relative_path), Field(min_length=1)]
Sha256 = Annotated[str, AfterValidator(_validate_sha256)]


class ArtifactNamespace(ContractModel):
    """Stable location for one run and one durable layer."""

    layer: Layer
    run_id: RunId


class ArtifactRef(ContractModel):
    """Reference to one artifact within a committed namespace."""

    layer: Layer
    run_id: RunId
    relative_path: RelativePath


class ManifestRef(ContractModel):
    """Reference to a committed layer manifest."""

    layer: Layer
    run_id: RunId
    relative_path: Literal["manifest.json"] = "manifest.json"
    sha256: Sha256


class ArtifactDescriptor(ContractModel):
    """Hash and schema metadata for one file in a durable package."""

    schema_version: Literal["1"] = "1"
    name: str = Field(min_length=1)
    layer: Layer
    media_type: str = Field(min_length=1)
    relative_path: RelativePath
    sha256: Sha256
    size_bytes: int = Field(ge=0)
    row_count: int | None = Field(default=None, ge=0)
    column_count: int | None = Field(default=None, ge=0)
    schema_fingerprint: str | None = None
    required: bool = True
