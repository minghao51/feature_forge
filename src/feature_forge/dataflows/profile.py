"""Execution profiles for optional Hamilton dataflows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ExecutionProfile(StrEnum):
    """Network, persistence, and cache policy for a dataflow run."""

    DEVELOPMENT = "development"
    CI = "ci"
    PRODUCTION = "production"
    REPLAY = "replay"
    DOCUMENTATION = "documentation"


@dataclass(frozen=True)
class ProfilePolicy:
    """Resolved policy used by Hamilton boundary nodes."""

    network_allowed: bool
    persistence_enabled: bool
    cache_enabled: bool
    requires_artifact_store: bool


PROFILE_POLICIES: dict[ExecutionProfile, ProfilePolicy] = {
    ExecutionProfile.DEVELOPMENT: ProfilePolicy(
        network_allowed=True,
        persistence_enabled=True,
        cache_enabled=True,
        requires_artifact_store=False,
    ),
    ExecutionProfile.CI: ProfilePolicy(
        network_allowed=False,
        persistence_enabled=False,
        cache_enabled=False,
        requires_artifact_store=False,
    ),
    ExecutionProfile.PRODUCTION: ProfilePolicy(
        network_allowed=True,
        persistence_enabled=True,
        cache_enabled=True,
        requires_artifact_store=True,
    ),
    ExecutionProfile.REPLAY: ProfilePolicy(
        network_allowed=False,
        persistence_enabled=True,
        cache_enabled=False,
        requires_artifact_store=True,
    ),
    ExecutionProfile.DOCUMENTATION: ProfilePolicy(
        network_allowed=False,
        persistence_enabled=False,
        cache_enabled=False,
        requires_artifact_store=False,
    ),
}


def get_profile_policy(profile: ExecutionProfile) -> ProfilePolicy:
    """Return the immutable policy for an execution profile."""
    return PROFILE_POLICIES[profile]
