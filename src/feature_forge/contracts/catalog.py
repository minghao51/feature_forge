"""Versioned contracts for the derived local catalog and verification CLI."""

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Any, Literal

from pydantic import Field

from feature_forge.contracts.base import ContractModel

CATALOG_SCHEMA_VERSION = 1


class VerificationStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"
    MISSING = "missing"
    INCOMPATIBLE = "incompatible"
    OPERATIONAL_ERROR = "operational_error"


class CLIExitCode(IntEnum):
    OK = 0
    INVALID = 10
    MISSING = 11
    INCOMPATIBLE = 12
    OPERATIONAL_ERROR = 20
    USAGE = 64


class CatalogIssue(ContractModel):
    schema_version: Literal["1"] = "1"
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    path: str | None = None
    run_id: str | None = None


class VerificationResult(ContractModel):
    schema_version: Literal["1"] = "1"
    subject: str
    status: VerificationStatus
    issues: list[CatalogIssue] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class CatalogReport(ContractModel):
    schema_version: Literal["1"] = "1"
    catalog_schema_version: int = CATALOG_SCHEMA_VERSION
    status: VerificationStatus
    indexed_manifests: int = Field(default=0, ge=0)
    issues: list[CatalogIssue] = Field(default_factory=list)
    logical_digest: str | None = None
