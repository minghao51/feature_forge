"""Shared strict configuration for versioned persisted contracts."""

from pydantic import BaseModel, ConfigDict


class ContractModel(BaseModel):
    """Base model that rejects unknown fields instead of hiding schema drift."""

    model_config = ConfigDict(extra="forbid")
