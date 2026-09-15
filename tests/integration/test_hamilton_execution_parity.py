"""Hamilton sequential/process recomputation parity tests (plan 21 §16 gate 2).

Covers the sequential/process execution-parity requirement that gated legacy
removal in ``docs/plan/21_hamilton_default_execution_handoff.md``:

* Sequential Hamilton execution matches a *real process worker recomputation*
  (``run_hamilton_case`` behind ``ProcessPoolExecutionAdapter``) on a fresh
  artifact/cache root — the worker recomputes from scratch, it never reuses
  the sequential run's durable packages (§14.2: "Sequential/process parity
  under ``spawn`` and the Linux default context").

The dual-engine metric-parity gate (plan 21 §16 gate 3) was retired with the
legacy engine removal (ADR 0016); its historical evidence is preserved under
``experiments/legacy_removal/2026-09-14/``.

Sequential-versus-process comparisons assert *exact* equality: both
sides run the identical deterministic engine on identical inputs, so any
difference at all would be a real parity defect that a tolerance could hide.

Sandbox note: the Gold stage executes generated code in a spawn-context
sandbox worker whose default ``evaluation.sandbox_timeout_seconds=5`` was
observed to expire under load, silently demoting the candidate to an error
decision. These tests raise the documented timeout to 60 seconds and assert
the Gold sandbox decision is ``accepted`` so candidate evaluation is proven.
"""

from __future__ import annotations

import json
import multiprocessing
import pathlib
from dataclasses import replace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from feature_forge import ExperimentalPlatform
from feature_forge.config import FailurePolicy, Settings
from feature_forge.data import DatasetRegistry
from feature_forge.experiment.execution import (
    CaseComputationInput,
    ExperimentCase,
    ExperimentResult,
    ProcessPoolExecutionAdapter,
)
from feature_forge.experiment.hamilton_executor import (
    HamiltonLayerExecutor,
    run_hamilton_case,
)
from feature_forge.methods import MethodRegistry
from feature_forge.methods.base import BaseMethod
from feature_forge.storage.local import LocalArtifactStore

# Fork-context pools (plan 22 dimension) fork from a multi-threaded pytest
# parent, which Python 3.12+ warns about; expected here, see
# tests/unit/test_fail_fast_process.py for the full rationale.
pytestmark = pytest.mark.filterwarnings(
    "ignore:This process.*multi-threaded.*fork:DeprecationWarning"
)

# The default 5-second sandbox timeout expires under load; these integration
# tests use a generous documented value so Gold candidate execution succeeds.
SANDBOX_TIMEOUT_SECONDS = 60.0

DATASET_NAME = "parity-xor"
METHOD_NAME = "parity_deterministic"
EXPLODING_METHOD_NAME = "parity_exploding"


class ParityDeterministicMethod(BaseMethod):
    """Offline deterministic method emitting one sandbox-safe numeric feature.

    The generated feature ``(a - b) ** 2`` carries signal for the XOR-style
    target of the local dataset (diagonal rows have small values, off-diagonal
    rows large ones), so the Gold candidate arm is a real evaluation and the
    parity comparisons are non-degenerate.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name=METHOD_NAME)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> ParityDeterministicMethod:
        self._artifacts["generated_code"] = (
            "import pandas as pd\n"
            "def generate_features(df):\n"
            "    return pd.DataFrame({'diag_sq': (df['a'] - df['b']) ** 2}, index=df.index)\n"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.assign(diag_sq=(X["a"] - X["b"]) ** 2)

    @property
    def feature_metadata(self) -> list[dict[str, Any]]:
        return [{"name": "diag_sq", "base_columns": ["a", "b"]}]


class ParityExplodingMethod(BaseMethod):
    """Deterministic terminal-failure method for fail-fast parity checks.

    ``fit`` always raises, so both engines normalize the identical terminal
    failure and fail-fast scheduling (ADR 0017) can be compared across the
    sequential and process adapters without providers or network.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name=EXPLODING_METHOD_NAME)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> ParityExplodingMethod:
        raise RuntimeError("parity deterministic failure")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise RuntimeError("unreachable: fit always fails")

    @property
    def feature_metadata(self) -> list[dict[str, Any]]:
        return []


def _write_xor_dataset(tmp_path: pathlib.Path) -> dict[str, Any]:
    """Write a local numeric deterministic dataset and return its registry info.

    The target is the XOR function of thresholded ``a`` and ``b`` over a
    seeded, rounded uniform grid: no axis-parallel split of the base features
    captures it, so the baseline and candidate arms genuinely differ. The
    values are materialized to CSV once; every engine reads the same file.
    """
    rng = np.random.default_rng(20260913)
    a = np.round(rng.uniform(0.0, 1.0, 40), 4)
    b = np.round(rng.uniform(0.0, 1.0, 40), 4)
    target = ((a > 0.5) ^ (b > 0.5)).astype(int)
    frame = pd.DataFrame({"a": a, "b": b, "target": target})
    sample_dir = tmp_path / "parity-xor-data"
    sample_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(sample_dir / "train.csv", index=False)
    (sample_dir / "metadata.json").write_text('{"target": "target", "task": "classification"}')
    return {
        "source": "local",
        "path": str(sample_dir),
        "target": "target",
        "task": "classification",
    }


def _hamilton_settings(tmp_path: pathlib.Path, tag: str) -> Settings:
    """Hamilton settings with a fresh, run-private artifact and cache root."""
    return Settings(
        metric="acc",
        evaluation={"sandbox_timeout_seconds": SANDBOX_TIMEOUT_SECONDS},
        dataflow={
            "engine": "hamilton",
            "artifact_root": tmp_path / f"artifacts-{tag}",
            "cache": {"path": tmp_path / f"cache-{tag}"},
        },
    )


def _hamilton_platform(
    tmp_path: pathlib.Path,
    tag: str,
    dataset_info: dict[str, Any],
    methods: dict[str, type[BaseMethod]] | None = None,
) -> ExperimentalPlatform:
    platform = ExperimentalPlatform(
        config={
            "metric": "acc",
            "evaluation": {"sandbox_timeout_seconds": SANDBOX_TIMEOUT_SECONDS},
            "dataflow": {
                "engine": "hamilton",
                "artifact_root": str(tmp_path / f"artifacts-{tag}"),
                "cache": {"path": str(tmp_path / f"cache-{tag}")},
            },
        }
    )
    platform.register_dataset(DATASET_NAME, dataset_info)
    for name, method_class in (methods or {METHOD_NAME: ParityDeterministicMethod}).items():
        platform.register_method(name, method_class)
    return platform


def _case_kwargs() -> dict[str, Any]:
    return {
        "datasets": [DATASET_NAME],
        "methods": [METHOD_NAME],
        "models": ["random_forest"],
        "cv_folds": 2,
        "seeds": [42],
        "progress": False,
    }


def _hamilton_payloads(
    settings: Settings, dataset_info: dict[str, Any]
) -> list[CaseComputationInput]:
    """Build serializable worker payloads with parent-allocated case identity.

    Mirrors the production parent seam (``ExperimentalPlatform`` planning step
    plus ``CaseComputationInput``) without touching platform internals. The
    payload carries the fresh worker artifact root, so the process worker can
    only recompute — no durable package from another root is reachable.
    """
    registry = DatasetRegistry()
    registry.register(DATASET_NAME, dataset_info)
    planner = HamiltonLayerExecutor(
        settings=settings,
        artifact_store=LocalArtifactStore(settings.dataflow.artifact_root),
        dataset_registry=registry,
        method_classes={
            **MethodRegistry.get_all_methods(),
            METHOD_NAME: ParityDeterministicMethod,
        },
    )
    case = ExperimentCase(
        dataset=DATASET_NAME,
        method=METHOD_NAME,
        model="random_forest",
        seed=42,
        cv_folds=2,
    )
    plan = planner.plan_case(case)
    scheduled = replace(
        case,
        case_key=plan.case_key,
        attempt_id=plan.attempt_id,
        run_id=plan.attempt_id,
    )
    return [
        CaseComputationInput(
            case=scheduled,
            settings_data=settings.model_dump(),
            dataset_overrides={DATASET_NAME: dataset_info},
            method_overrides={METHOD_NAME: ParityDeterministicMethod},
        )
    ]


def _metrics(result: ExperimentResult | dict[str, Any]) -> dict[str, Any]:
    """Extract comparable metric fields from either result shape."""
    if isinstance(result, dict):
        return {
            "cv_score": result["cv_score"],
            "gain": result["gain"],
            "baseline_score": result["baseline_score"],
            "num_features_generated": result["num_features_generated"],
            "error": result["error"],
        }
    return {
        "cv_score": result.cv_score,
        "gain": result.gain,
        "baseline_score": result.baseline_score,
        "num_features_generated": result.num_features_generated,
        "error": result.error,
    }


def _stage_fingerprints(result: ExperimentResult | dict[str, Any]) -> list[tuple[str, str]]:
    """Return ``(layer, layer_fingerprint)`` pairs in dependency order."""
    stages: list[Any] = result["stages"] if isinstance(result, dict) else list(result.stages)
    pairs: list[tuple[str, str]] = []
    for stage in stages:
        if isinstance(stage, dict):
            pairs.append((str(stage["layer"]), str(stage["layer_fingerprint"])))
        else:
            pairs.append((stage.layer.value, stage.layer_fingerprint))
    return pairs


def _assert_all_stages_executed(result: ExperimentResult) -> None:
    """Every layer recomputed in-run: no durable package was reused."""
    dispositions = [stage.disposition.value for stage in result.stages]
    assert dispositions == ["executed"] * 4


def _assert_gold_candidate_accepted(artifact_root: pathlib.Path) -> None:
    """The sandbox really executed the generated feature in this root.

    A timed-out or failed sandbox demotes the candidate to an error decision
    and the reported metrics silently degrade to baseline-only values; this
    assertion keeps the parity comparisons from passing in that degenerate
    state. A fresh artifact root hosts exactly one Gold attempt.
    """
    gold_runs = artifact_root / "03_gold" / "runs"
    run_dirs = [item for item in gold_runs.iterdir() if item.is_dir()]
    assert len(run_dirs) == 1, f"expected one Gold attempt, found {run_dirs}"
    decisions = json.loads((run_dirs[0] / "decisions.json").read_text(encoding="utf-8"))
    assert decisions, "Gold package recorded no sandbox decisions"
    assert all(item["state"] == "accepted" for item in decisions), decisions


class TestSequentialProcessRecomputationParity:
    """Sequential Hamilton equals a real process worker recomputation."""

    def test_sequential_matches_process_worker_on_fresh_roots(self, tmp_path: pathlib.Path) -> None:
        """Linux-default-context process worker recomputes the sequential result.

        The worker runs on its own fresh artifact/cache root, so parity proves
        recomputation equivalence rather than durable package reuse.
        """
        dataset_info = _write_xor_dataset(tmp_path)
        sequential = _hamilton_platform(tmp_path, "seq", dataset_info).run(**_case_kwargs())[0]

        worker_settings = _hamilton_settings(tmp_path, "proc")
        payloads = _hamilton_payloads(worker_settings, dataset_info)
        # mp_context=None keeps the executor's platform default (fork on Linux).
        adapter = ProcessPoolExecutionAdapter(max_workers=1, mp_context=None)
        assert adapter.mp_context is None
        worker_results = adapter.run(payloads, run_hamilton_case, progress=False)
        assert len(worker_results) == 1
        worker = worker_results[0]

        assert _metrics(sequential)["error"] is None
        assert _metrics(worker)["error"] is None
        # Same engine, same seeded deterministic pipeline: bit-identical.
        assert _metrics(worker) == _metrics(sequential)
        # Non-degenerate arms: the sandboxed candidate moved the score.
        assert _metrics(worker)["gain"] != 0.0
        assert 0.0 < _metrics(worker)["baseline_score"] < 1.0
        # Content identity across independent roots and processes. Gold and
        # Platinum layer fingerprints are attempt-scoped by design (the
        # candidate_id embeds run_id), so only the content-scoped Bronze and
        # Silver layers must match.
        worker_stages = _stage_fingerprints(worker)
        assert _stage_fingerprints(sequential)[:2] == worker_stages[:2]
        assert [layer for layer, _ in worker_stages] == [
            "bronze",
            "silver",
            "gold",
            "platinum",
        ]
        # The worker genuinely recomputed on its own roots.
        _assert_all_stages_executed(worker)
        _assert_gold_candidate_accepted(tmp_path / "artifacts-proc")
        _assert_gold_candidate_accepted(tmp_path / "artifacts-seq")
        assert worker.run_id != sequential["run_id"]
        assert (tmp_path / "artifacts-proc" / "04_platinum" / "runs").is_dir()

    def test_process_worker_parity_under_explicit_spawn_context(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The process seam holds when pool children are started via ``spawn``."""
        dataset_info = _write_xor_dataset(tmp_path)
        sequential = _hamilton_platform(tmp_path, "seq-spawn", dataset_info).run(**_case_kwargs())[
            0
        ]

        worker_settings = _hamilton_settings(tmp_path, "proc-spawn")
        payloads = _hamilton_payloads(worker_settings, dataset_info)
        adapter = ProcessPoolExecutionAdapter(
            max_workers=1, mp_context=multiprocessing.get_context("spawn")
        )
        assert adapter.mp_context is not None
        assert adapter.mp_context.get_start_method() == "spawn"
        worker_results = adapter.run(payloads, run_hamilton_case, progress=False)
        assert len(worker_results) == 1
        worker = worker_results[0]

        assert _metrics(worker)["error"] is None
        assert _metrics(worker) == _metrics(sequential)
        assert _metrics(worker)["gain"] != 0.0
        assert _stage_fingerprints(worker)[:2] == _stage_fingerprints(sequential)[:2]
        _assert_all_stages_executed(worker)
        _assert_gold_candidate_accepted(tmp_path / "artifacts-proc-spawn")


@pytest.mark.parametrize("mp_context_name", [None, "spawn"])
def test_adapter_mp_context_parameter_backward_compatible(
    mp_context_name: str | None,
) -> None:
    """Existing positional construction keeps working; context round-trips."""
    context = multiprocessing.get_context(mp_context_name) if mp_context_name else None
    adapter = ProcessPoolExecutionAdapter(2, context)
    assert adapter.max_workers == 2
    assert adapter.mp_context is context
    default_adapter = ProcessPoolExecutionAdapter()
    assert default_adapter.max_workers == 1
    assert default_adapter.mp_context is None


class TestExecutionPolicyParity:
    """Sequential/process parity under the failure policy (plan 22 §5 PR 3).

    Both engines run the identical deterministic matrix on fresh roots, so
    states, order, and deterministic scores must match exactly — including
    the typed cancelled tail under ``fail_fast``.
    """

    @staticmethod
    def _run_matrix(
        platform: ExperimentalPlatform,
        tracker: Any,
        *,
        parallel: bool,
        policy: FailurePolicy,
        seeds: list[int],
        methods: list[str],
        max_workers: int = 2,
    ) -> list[dict[str, Any]]:
        return platform.run(
            datasets=[DATASET_NAME],
            methods=methods,
            models=["random_forest"],
            cv_folds=2,
            seeds=seeds,
            progress=False,
            failure_policy=policy,
            cancellation_token=None,
            tracker=tracker,
            parallel=parallel,
            max_workers=max_workers,
        )

    def test_continue_policy_matches_sequential_and_process(self, tmp_path: pathlib.Path) -> None:
        """Plan 22 §6 test 1 with real chains: success+failure run in both."""
        dataset_info = _write_xor_dataset(tmp_path)
        policy_methods = {
            METHOD_NAME: ParityDeterministicMethod,
            EXPLODING_METHOD_NAME: ParityExplodingMethod,
        }
        sequential = self._run_matrix(
            _hamilton_platform(tmp_path, "seq-cont", dataset_info, policy_methods),
            MagicMock(),
            parallel=False,
            policy=FailurePolicy.CONTINUE,
            seeds=[42],
            methods=[METHOD_NAME, EXPLODING_METHOD_NAME],
        )
        tracker_process = MagicMock()
        process = self._run_matrix(
            _hamilton_platform(tmp_path, "proc-cont", dataset_info, policy_methods),
            tracker_process,
            parallel=True,
            policy=FailurePolicy.CONTINUE,
            seeds=[42],
            methods=[METHOD_NAME, EXPLODING_METHOD_NAME],
        )

        expected = [
            (METHOD_NAME, 42, "succeeded"),
            (EXPLODING_METHOD_NAME, 42, "failed"),
        ]
        assert [(row["method"], row["seed"], row["state"]) for row in sequential] == expected
        assert [(row["method"], row["seed"], row["state"]) for row in process] == expected
        # Deterministic engine: identical scores on independent roots.
        assert sequential[0]["cv_score"] == process[0]["cv_score"]
        assert sequential[0]["cv_score"] is not None
        # Tracker exactly once per actual completion, never for failures-as-
        # cancelled; failed cases still finish their tracker run.
        assert tracker_process.init_run.call_count == 2
        assert tracker_process.finish.call_count == 2

    def test_fail_fast_matches_sequential_and_process_with_cancelled_tail(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Plan 22 §5 PR 3 parity: identical rows including the cancelled tail.

        The terminal failure is the first matrix entry and the process run
        uses ``max_workers=1`` so no in-flight case can legitimately outlive
        the stop — both engines must then produce the exact same typed rows.
        (With a wider window the process path keeps up to ``max_workers``
        real in-flight results after the stop; that bounded-window tail is
        pinned by ``tests/unit/test_fail_fast_process.py`` instead.)
        """
        dataset_info = _write_xor_dataset(tmp_path)
        policy_methods = {
            METHOD_NAME: ParityDeterministicMethod,
            EXPLODING_METHOD_NAME: ParityExplodingMethod,
        }
        sequential = self._run_matrix(
            _hamilton_platform(tmp_path, "seq-ff", dataset_info, policy_methods),
            MagicMock(),
            parallel=False,
            policy=FailurePolicy.FAIL_FAST,
            seeds=[42],
            methods=[EXPLODING_METHOD_NAME, METHOD_NAME],
            max_workers=1,
        )
        tracker_process = MagicMock()
        process = self._run_matrix(
            _hamilton_platform(tmp_path, "proc-ff", dataset_info, policy_methods),
            tracker_process,
            parallel=True,
            policy=FailurePolicy.FAIL_FAST,
            seeds=[42],
            methods=[EXPLODING_METHOD_NAME, METHOD_NAME],
            max_workers=1,
        )

        expected_states = [
            (EXPLODING_METHOD_NAME, 42, "failed"),
            (METHOD_NAME, 42, "cancelled"),
        ]
        assert [(row["method"], row["seed"], row["state"]) for row in sequential] == expected_states
        assert [(row["method"], row["seed"], row["state"]) for row in process] == expected_states
        # The cancelled tail is typed and shaped identically on both paths.
        for seq_row, proc_row in zip(sequential[1:], process[1:], strict=True):
            assert seq_row["failure"]["failure_class"] == "cancelled"
            assert proc_row["failure"]["failure_class"] == "cancelled"
            assert "(fail_fast)" in seq_row["error"]
            assert "(fail_fast)" in proc_row["error"]
            assert proc_row["stages"] == []
            assert proc_row["cv_score"] is None
            assert proc_row["run_id"]
            assert proc_row["case_fingerprint"]
        # Tracker runs only for the case that actually started.
        assert tracker_process.init_run.call_count == 1
        assert tracker_process.finish.call_count == 1
