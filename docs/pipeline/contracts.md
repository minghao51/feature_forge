# Contracts and durable evidence

Every persisted record has a schema version and rejects unknown fields. Writers emit the
current version; unknown future versions fail clearly. Historical manifests are immutable:
Feature Forge ships no automatic or in-place artifact migration. Rebuild evidence from its
authoritative upstream source, or use a compatible old reader/tool to write a new package and
run namespace.

A committed package contains `manifest.json`, `_SUCCESS`, and every artifact declared by the
manifest. `_SUCCESS` stores the SHA-256 of the exact committed manifest. Verification checks
the marker, manifest identity, descriptor paths, sizes, hashes, schema/media metadata, and
layer-specific semantic invariants.

An `ArtifactDescriptor` describes one normalized relative path. A `ManifestRef` binds layer,
run ID, fixed manifest path, and digest. A `RunManifest` binds request/environment identity,
layer fingerprint, upstream manifests, stage outcomes, checks, and artifacts. Cache metadata,
catalog rows, tracker records, and generated documentation are not durable evidence.

- [Generated contract schemas](../generated/contracts.md)
- [Generated package layouts](../generated/package-layouts.md)
- [Generated check registry](../generated/check-registry.md)
- [Artifact commit ADR](../adr/0003-artifact-commit.md)

Secrets must not enter contracts, fingerprints, manifests, errors, catalog rows, generated
references, or telemetry. Environment capture includes dependency and plugin identities but
never raw credentials.
