"""Persisted artifact descriptors and namespace references."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Literal

from pydantic import Field, field_validator

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


class ArtifactNamespace(ContractModel):
    """Stable location for one run and one durable layer."""

    layer: Layer
    run_id: str = Field(min_length=1)

    @field_validator("run_id")
    @classmethod
    def _valid_run_id(cls, value: str) -> str:
        return validate_identifier(value)


class ArtifactRef(ContractModel):
    """Reference to one artifact within a committed namespace."""

    layer: Layer
    run_id: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)

    @field_validator("run_id")
    @classmethod
    def _valid_run_id(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("relative_path")
    @classmethod
    def _valid_path(cls, value: str) -> str:
        return validate_relative_path(value)


class ManifestRef(ContractModel):
    """Reference to a committed layer manifest."""

    layer: Layer
    run_id: str = Field(min_length=1)
    relative_path: Literal["manifest.json"] = "manifest.json"
    sha256: str

    @field_validator("run_id")
    @classmethod
    def _valid_run_id(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("relative_path")
    @classmethod
    def _valid_path(cls, value: str) -> str:
        return validate_relative_path(value)

    @field_validator("sha256")
    @classmethod
    def _valid_sha256(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("sha256 must be a lowercase 64-character hexadecimal digest")
        return value


class ArtifactDescriptor(ContractModel):
    """Hash and schema metadata for one file in a durable package."""

    schema_version: Literal["1"] = "1"
    name: str = Field(min_length=1)
    layer: Layer
    media_type: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)
    sha256: str
    size_bytes: int = Field(ge=0)
    row_count: int | None = Field(default=None, ge=0)
    column_count: int | None = Field(default=None, ge=0)
    schema_fingerprint: str | None = None
    required: bool = True

    @field_validator("relative_path")
    @classmethod
    def _valid_path(cls, value: str) -> str:
        return validate_relative_path(value)

    @field_validator("sha256")
    @classmethod
    def _valid_sha256(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("sha256 must be a lowercase 64-character hexadecimal digest")
        return value
