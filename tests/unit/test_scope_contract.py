"""Row-local scope contract coverage (plan 23 §4.2, ADR 0018)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar

import pandas as pd

from feature_forge.contracts import (
    BronzeRecord,
    EnvironmentSnapshot,
    EvaluationProtocol,
    FeatureDecisionState,
    GoldRequest,
    Layer,
    ManifestRef,
    RunManifest,
    RunRequest,
    RunState,
    SilverPackage,
    gold_input_fingerprint,
)
from feature_forge.dataflows.gold import candidate_execution_batches, candidate_feature_specs
from feature_forge.evaluation.sandbox import SandboxedExecutor
from feature_forge.evaluation.scope import row_local_violations

SUM_AB_SCRIPT = (
    "def generate_features(df):\n"
    "    return pd.DataFrame({'sum_ab': df['a'] + df['b']}, index=df.index)\n"
)
Z_SCORE_SCRIPT = (
    "def generate_features(df):\n"
    "    z = (df['a'] - df['a'].mean()) / df['a'].std()\n"
    "    return pd.DataFrame({'z_a': z}, index=df.index)\n"
)
POSITION_SCRIPT = (
    "def generate_features(df):\n"
    "    return pd.DataFrame(\n"
    "        {'pos': [value * i for i, value in enumerate(df['a'])]}, index=df.index\n"
    "    )\n"
)
FIRST_ROW_FLAG_SCRIPT = (
    "def generate_features(df):\n"
    "    return pd.DataFrame({'first': (df.index == 0) * df['a']}, index=df.index)\n"
)
RANK_SCRIPT = (
    "def generate_features(df):\n"
    "    return pd.DataFrame({'rank_a': df['a'].rank()}, index=df.index)\n"
)


class _InlineSandbox(SandboxedExecutor):
    """Deterministic in-process stand-in for the process-isolated sandbox."""

    def execute(
        self,
        code: str,
        df: pd.DataFrame,
        *,
        source: str = "unknown",
        agent_name: str = "unknown",
    ) -> pd.DataFrame:
        namespace: dict[str, object] = {"pd": pd}
        exec(compile(code, "<inline>", "exec"), namespace)
        generated = namespace["generate_features"]
        assert callable(generated)
        result = generated(df)
        assert isinstance(result, pd.DataFrame)
        return result


def _probe(code: str, frame: pd.DataFrame, name: str) -> dict[str, str]:
    sandbox = _InlineSandbox()
    full_output = sandbox.execute(code, frame, source="gold_scope_probe")
    return row_local_violations(sandbox, code, frame, full_output, [name])


def test_row_local_candidate_passes() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": [4, 5, 6, 7, 8, 9]})

    assert _probe(SUM_AB_SCRIPT, frame, "sum_ab") == {}


def test_whole_frame_statistics_candidate_violates() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6]})

    assert _probe(Z_SCORE_SCRIPT, frame, "z_a") == {"z_a": "row-subset probe mismatch"}


def test_position_dependent_candidate_violates() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6]})

    assert _probe(POSITION_SCRIPT, frame, "pos") == {"pos": "row-subset probe mismatch"}


def test_rank_candidate_violates() -> None:
    # rank() is permutation-invariant, so the row-permutation probe cannot detect
    # it; the row-subset probe (strict sub-multiset of values) catches it.
    frame = pd.DataFrame({"a": [5, 3, 8, 1, 9, 2]})

    assert _probe(RANK_SCRIPT, frame, "rank_a") == {"rank_a": "row-subset probe mismatch"}


def test_permutation_probe_catches_what_subset_misses() -> None:
    # A first-row flag survives the row-subset probe (row 0 stays first after
    # the reset), but the seeded permutation moves row 0 away from position 0,
    # so only the row-permutation probe detects the position dependence.
    frame = pd.DataFrame({"a": [3, 1, 4, 1, 5, 9]})

    assert _probe(FIRST_ROW_FLAG_SCRIPT, frame, "first") == {
        "first": "row-permutation probe mismatch"
    }


def test_tiny_frames_skip_probes() -> None:
    # Documented limitation: with fewer than 3 rows no probe can run, so even a
    # whole-frame statistic goes unverified. Gold under holdout always has at
    # least 2 * cv_folds >= 4 total Silver rows, so the contract still applies.
    frame = pd.DataFrame({"a": [1, 2]})

    assert _probe(Z_SCORE_SCRIPT, frame, "z_a") == {}


class _WholeFrameMethod:
    generated_scripts: ClassVar[list[str]] = [Z_SCORE_SCRIPT]
    feature_metadata: ClassVar[list[dict[str, object]]] = [{"name": "z_a", "base_columns": ["a"]}]


class _MixedMethod:
    generated_scripts: ClassVar[list[str]] = [
        "def generate_features(df):\n"
        "    return pd.DataFrame(\n"
        "        {\n"
        "            'sum_ab': df['a'] + df['b'],\n"
        "            'z_a': (df['a'] - df['a'].mean()) / df['a'].std(),\n"
        "        },\n"
        "        index=df.index,\n"
        "    )\n"
    ]
    feature_metadata: ClassVar[list[dict[str, object]]] = [
        {"name": ["sum_ab", "z_a"], "base_columns": ["a", "b"]}
    ]


def _environment() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        python_version="3.12",
        operating_system="test",
        architecture="test",
        feature_forge_version="0+test",
    )


def _silver() -> SilverPackage:
    now = datetime.now(UTC)
    manifest = RunManifest(
        layer=Layer.SILVER,
        package_kind="silver",
        layer_fingerprint="silver-fingerprint",
        run_id="silver-run",
        case_fingerprint="case-fingerprint",
        state=RunState.SUCCEEDED,
        request=RunRequest(dataset="fixture", method="silver", model="none", seed=42),
        environment=_environment(),
        created_at=now,
        completed_at=now,
    )
    row_ids = [f"row-{index:09d}" for index in range(6)]
    return SilverPackage(
        manifest=manifest,
        bronze=BronzeRecord(
            source="fixture",
            source_checksum="source",
            target="target",
            task="classification",
            snapshot_mode="snapshot",
            row_count=6,
            column_count=3,
        ),
        canonical_features=pd.DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": [4, 5, 6, 7, 8, 9]}),
        canonical_target=pd.DataFrame({"row_id": row_ids, "target": [0, 1, 0, 1, 0, 1]}),
        row_ids=pd.DataFrame({"row_id": row_ids}),
        fold_assignments=pd.DataFrame({"row_id": row_ids, "fold": [0, 1, 0, 1, 0, 1]}),
        profile={"dataset_fingerprint": "silver-fingerprint"},
        checks=[],
    )


def _request(protocol: EvaluationProtocol) -> GoldRequest:
    value = gold_input_fingerprint(
        silver_fingerprint_value="silver-fingerprint",
        method_name="deterministic",
        method_version="1",
        method_config={},
        prompt_bundle_fingerprint="fixture",
        generated_contract_version="1",
        evaluation_protocol=protocol,
        selection_policy={"policy": "validation"},
    )
    return GoldRequest(
        run_id="gold-run",
        case_fingerprint="case-fingerprint",
        silver_manifest=ManifestRef(layer=Layer.SILVER, run_id="silver-run", sha256="a" * 64),
        silver_fingerprint="silver-fingerprint",
        gold_input_fingerprint=value,
        method_name="deterministic",
        method_version="1",
        prompt_bundle_fingerprint="fixture",
        evaluation_protocol=protocol,
    )


def _executed(method: Any, protocol: EvaluationProtocol) -> dict[str, Any]:
    evidence = candidate_feature_specs(method, _silver(), _request(protocol))
    return candidate_execution_batches(evidence, _InlineSandbox())


def _decisions(evidence: dict[str, Any]) -> dict[str, Any]:
    names = {item.candidate_id: item.name for item in evidence["candidates"]}
    return {names[item.candidate_id]: item for item in evidence["decisions"]}


def test_gold_holdout_rejects_whole_frame_candidate() -> None:
    evidence = _executed(_WholeFrameMethod(), "holdout")
    decision = _decisions(evidence)["z_a"]

    assert decision.state is FeatureDecisionState.REJECTED
    assert decision.reason_code == "not_row_local"
    assert decision.reason.startswith("row-subset probe mismatch (row-local scope contract")
    assert "z_a" not in evidence["accepted"].columns
    assert len(evidence["accepted"].columns) == 1  # row_id only
    assert evidence["candidate_features"] is None or list(
        evidence["candidate_features"].columns
    ) == ["row_id"]


def test_gold_compatibility_accepts_it() -> None:
    evidence = _executed(_WholeFrameMethod(), "compatibility")
    decision = _decisions(evidence)["z_a"]

    assert decision.state is FeatureDecisionState.ACCEPTED
    assert decision.reason_code == "accepted"
    assert "z_a" in evidence["accepted"].columns


def test_gold_holdout_mixed_batch_keeps_only_row_local_columns() -> None:
    evidence = _executed(_MixedMethod(), "holdout")
    decisions = _decisions(evidence)

    assert decisions["sum_ab"].state is FeatureDecisionState.ACCEPTED
    assert decisions["z_a"].state is FeatureDecisionState.REJECTED
    assert decisions["z_a"].reason_code == "not_row_local"
    assert list(evidence["accepted"].columns) == ["row_id", "sum_ab"]
    assert sum(item.state is FeatureDecisionState.ACCEPTED for item in evidence["decisions"]) == 1
    assert sum(item.reason_code == "not_row_local" for item in evidence["decisions"]) == 1
