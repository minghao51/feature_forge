"""Compatibility adapter from MethodProtocol to replayable Gold evidence."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
import pandas as pd

from feature_forge.contracts import (
    CheckResult,
    FeatureCandidate,
    FeatureDecision,
    FeatureDecisionState,
    FeatureProvenance,
    GoldRequest,
    SilverPackage,
)
from feature_forge.evaluation.sandbox import SandboxedExecutor
from feature_forge.methods.base import MethodProtocol
from feature_forge.storage.hashing import fingerprint, sha256_bytes


@dataclass(frozen=True)
class GoldMethodEvidence:
    candidates: list[FeatureCandidate]
    provenance: list[FeatureProvenance]
    decisions: list[FeatureDecision]
    candidate_features: pd.DataFrame
    accepted_features: pd.DataFrame
    checks: list[CheckResult]
    code: dict[str, str]
    dependencies: dict[str, Any]


@dataclass(frozen=True)
class GoldGenerationBatch:
    """One ordered code batch and the specifications it owns."""

    batch_index: int
    code: str
    specifications: list[dict[str, Any]]


@dataclass(frozen=True)
class GoldGenerationPlan:
    """Provider-complete, secret-free plan ready for sandbox execution."""

    silver: SilverPackage
    request: GoldRequest
    batches: list[GoldGenerationBatch]
    metadata: list[dict[str, Any]]


def installed_method_version(method: MethodProtocol) -> str:
    module_root = type(method).__module__.split(".", maxsplit=1)[0]
    try:
        return version(module_root.replace("_", "-"))
    except PackageNotFoundError:
        return "0+unknown"


def prepare_method_generation(
    method: MethodProtocol,
    silver: SilverPackage,
    request: GoldRequest,
) -> GoldGenerationPlan:
    """Fit once and bind each specification to its owning code batch."""
    request = GoldRequest.model_validate(request.model_dump(mode="python"))
    X = silver.canonical_features.copy()
    target_name = silver.bronze.target
    y = silver.canonical_target[target_name].copy()
    method.fit(X, y)
    staged = getattr(method, "gold_generation_batches", None)
    if staged is not None:
        raw_batches = staged() if callable(staged) else staged
        if not isinstance(raw_batches, list) or not raw_batches:
            raise ValueError("gold_generation_batches must be a non-empty list")
        staged_batches: list[GoldGenerationBatch] = []
        staged_metadata: list[dict[str, Any]] = []
        for batch_index, raw_batch in enumerate(raw_batches):
            if not isinstance(raw_batch, dict) or not isinstance(raw_batch.get("code"), str):
                raise ValueError("Each staged Gold batch requires string code")
            raw_specs = raw_batch.get("specifications", [])
            if not isinstance(raw_specs, list) or any(
                not isinstance(item, dict) or not item.get("name") for item in raw_specs
            ):
                raise ValueError("Staged Gold specifications require named mappings")
            specifications = [dict(item) for item in raw_specs]
            staged_metadata.extend(specifications)
            staged_batches.append(
                GoldGenerationBatch(
                    batch_index=batch_index,
                    code=raw_batch["code"],
                    specifications=specifications,
                )
            )
        return GoldGenerationPlan(
            silver=silver,
            request=request,
            batches=staged_batches,
            metadata=staged_metadata,
        )
    scripts = list(method.generated_scripts)
    if not scripts:
        raise ValueError(
            f"Method '{request.method_name}' does not expose replayable generated code"
        )

    metadata = [
        item for item in method.feature_metadata if isinstance(item, dict) and item.get("name")
    ]
    batches: list[GoldGenerationBatch] = []
    for batch_index, code in enumerate(scripts):
        owned = [
            item
            for item in metadata
            if item.get("batch_index") == batch_index
            or item.get("code") == code
            or item.get("iteration") == batch_index
        ]
        if not owned and len(metadata) == len(scripts):
            owned = [metadata[batch_index]]
        batches.append(
            GoldGenerationBatch(
                batch_index=batch_index,
                code=code,
                specifications=owned,
            )
        )
    return GoldGenerationPlan(
        silver=silver,
        request=request,
        batches=batches,
        metadata=metadata,
    )


def execute_method_plan(
    plan: GoldGenerationPlan,
    sandbox: SandboxedExecutor,
) -> GoldMethodEvidence:
    """Execute a provider-complete plan in ordered sandbox batches."""
    silver = plan.silver
    request = plan.request
    X = silver.canonical_features.copy()
    metadata = {str(item["name"]): item for item in plan.metadata}
    working = X.copy()
    parts: list[pd.DataFrame] = []
    candidates: list[FeatureCandidate] = []
    provenance: list[FeatureProvenance] = []
    decisions: list[FeatureDecision] = []
    code_files: dict[str, str] = {}
    seen: set[str] = set(X.columns)

    for batch in plan.batches:
        batch_index = batch.batch_index
        code = batch.code
        path = f"generated_code/batch_{batch_index:04d}.py"
        code_files[path] = code
        try:
            output = sandbox.execute(
                code,
                working,
                source="gold_generation",
                agent_name=request.method_name,
            )
        except Exception as exc:
            failed_specs = batch.specifications or [{"name": f"batch_{batch_index:04d}_execution"}]
            for spec in failed_specs:
                name = str(spec["name"])
                candidate_id = (
                    "cand-"
                    + fingerprint(
                        {
                            "gold_input": request.gold_input_fingerprint,
                            "batch": batch_index,
                            "name": name,
                            "specification": spec,
                            "code_sha256": sha256_bytes(code.encode()),
                        }
                    )[:24]
                )
                candidates.append(
                    FeatureCandidate(
                        candidate_id=candidate_id,
                        name=name,
                        batch_index=batch_index,
                        code_path=path,
                        specification=spec,
                    )
                )
                provenance.append(
                    FeatureProvenance(
                        candidate_id=candidate_id,
                        method=request.method_name,
                        method_version=request.method_version,
                        round_index=batch_index,
                        prompt_bundle_fingerprint=request.prompt_bundle_fingerprint,
                        code_sha256=sha256_bytes(code.encode()),
                        dependencies=list(spec.get("dependencies", [])),
                        source_columns=list(spec.get("base_columns", [])),
                    )
                )
                decisions.append(
                    FeatureDecision(
                        candidate_id=candidate_id,
                        state=FeatureDecisionState.ERROR,
                        reason_code=type(exc).__name__,
                        reason=str(exc) or "Sandbox execution failed",
                    )
                )
                seen.add(name)
            continue
        batch_new = pd.DataFrame(index=working.index)
        for name in output.columns:
            if name in seen:
                continue
            values = output[name].reset_index(drop=True)
            spec = metadata.get(str(name), {"name": str(name)})
            candidate_id = (
                "cand-"
                + fingerprint(
                    {
                        "gold_input": request.gold_input_fingerprint,
                        "batch": batch_index,
                        "name": str(name),
                        "specification": spec,
                        "code_sha256": sha256_bytes(code.encode()),
                    }
                )[:24]
            )
            candidate = FeatureCandidate(
                candidate_id=candidate_id,
                name=str(name),
                batch_index=batch_index,
                code_path=path,
                specification=spec,
            )
            finite = not (
                pd.api.types.is_numeric_dtype(values)
                and bool(np.isinf(values.to_numpy(dtype=float, na_value=np.nan)).any())
            )
            state = FeatureDecisionState.ACCEPTED if finite else FeatureDecisionState.REJECTED
            candidates.append(candidate)
            provenance.append(
                FeatureProvenance(
                    candidate_id=candidate_id,
                    method=request.method_name,
                    method_version=request.method_version,
                    round_index=batch_index,
                    prompt_bundle_fingerprint=request.prompt_bundle_fingerprint,
                    code_sha256=sha256_bytes(code.encode()),
                    dependencies=list(spec.get("dependencies", [])),
                    source_columns=list(spec.get("base_columns", [])),
                )
            )
            decisions.append(
                FeatureDecision(
                    candidate_id=candidate_id,
                    state=state,
                    reason_code="validation_passed" if finite else "non_finite_values",
                    reason="Passed Gold structural validation"
                    if finite
                    else "Numeric output contains infinite values",
                )
            )
            batch_new[str(name)] = values.to_numpy()
            seen.add(str(name))
        accepted_batch_names = [
            candidate.name
            for candidate, decision in zip(candidates, decisions, strict=True)
            if candidate.batch_index == batch_index
            and decision.state is FeatureDecisionState.ACCEPTED
        ]
        if not batch_new.empty:
            parts.append(batch_new)
            working = pd.concat(
                [working.reset_index(drop=True), batch_new[accepted_batch_names]], axis=1
            )
        elif not any(candidate.code_path == path for candidate in candidates):
            # Do not publish an unreferenced code artifact. Exact package loading
            # requires every persisted batch to explain at least one candidate.
            code_files.pop(path)

    candidate_frame = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=X.index)
    accepted_names = [
        candidate.name
        for candidate, decision in zip(candidates, decisions, strict=True)
        if decision.state is FeatureDecisionState.ACCEPTED
    ]
    row_ids = silver.row_ids["row_id"].reset_index(drop=True)
    candidate_with_ids = pd.concat(
        [row_ids.rename("row_id"), candidate_frame.reset_index(drop=True)], axis=1
    )
    accepted_with_ids = pd.concat(
        [row_ids.rename("row_id"), candidate_frame[accepted_names].reset_index(drop=True)], axis=1
    )
    checks = [
        CheckResult(
            check_id="GOLD.ROW_IDS.ALIGNED",
            passed=len(candidate_with_ids) == len(row_ids),
            message="Gold row IDs align exactly with Silver",
            affected_artifacts=["candidate_features.parquet", "accepted_features.parquet"],
        ),
        CheckResult(
            check_id="GOLD.CANDIDATES.DECIDED",
            passed=len(candidates) == len(decisions),
            message="Every candidate has exactly one decision",
            affected_artifacts=["candidates.json", "decisions.json"],
        ),
    ]
    return GoldMethodEvidence(
        candidates=candidates,
        provenance=provenance,
        decisions=decisions,
        candidate_features=candidate_with_ids,
        accepted_features=accepted_with_ids,
        checks=checks,
        code=code_files,
        dependencies={"method_version": request.method_version},
    )


def verify_method_evidence(evidence: GoldMethodEvidence) -> GoldMethodEvidence:
    """Validate evidence cardinality and return a distinct verified boundary."""
    candidate_ids = [item.candidate_id for item in evidence.candidates]
    if (
        len(candidate_ids) != len(set(candidate_ids))
        or len(evidence.provenance) != len(candidate_ids)
        or len(evidence.decisions) != len(candidate_ids)
        or {item.candidate_id for item in evidence.provenance} != set(candidate_ids)
        or {item.candidate_id for item in evidence.decisions} != set(candidate_ids)
    ):
        raise ValueError("Gold evidence cardinality is inconsistent")
    return GoldMethodEvidence(
        candidates=list(evidence.candidates),
        provenance=list(evidence.provenance),
        decisions=list(evidence.decisions),
        candidate_features=evidence.candidate_features.copy(),
        accepted_features=evidence.accepted_features.copy(),
        checks=list(evidence.checks),
        code=dict(evidence.code),
        dependencies=dict(evidence.dependencies),
    )


def select_method_evidence(evidence: GoldMethodEvidence) -> GoldMethodEvidence:
    """Materialize validation-accepted columns at the Gold selection boundary."""
    accepted_ids = {
        item.candidate_id
        for item in evidence.decisions
        if item.state is FeatureDecisionState.ACCEPTED
    }
    accepted_names = [
        item.name for item in evidence.candidates if item.candidate_id in accepted_ids
    ]
    return GoldMethodEvidence(
        candidates=list(evidence.candidates),
        provenance=list(evidence.provenance),
        decisions=list(evidence.decisions),
        candidate_features=evidence.candidate_features.copy(),
        accepted_features=evidence.candidate_features[["row_id", *accepted_names]].copy(),
        checks=list(evidence.checks),
        code=dict(evidence.code),
        dependencies=dict(evidence.dependencies),
    )


def generate_method_evidence(
    method: MethodProtocol,
    silver: SilverPackage,
    request: GoldRequest,
    sandbox: SandboxedExecutor,
) -> GoldMethodEvidence:
    """Compatibility entrypoint for non-Hamilton callers."""
    plan = prepare_method_generation(method, silver, request)
    return select_method_evidence(verify_method_evidence(execute_method_plan(plan, sandbox)))
