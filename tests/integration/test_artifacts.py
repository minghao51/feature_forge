"""Integration tests for unified artifact system across methods."""

from __future__ import annotations

import pandas as pd

from feature_forge.artifacts.comparison import compare_methods
from feature_forge.baselines.base import Baseline
from feature_forge.config import Settings
from feature_forge.evaluation.cv import CVEvaluator
from feature_forge.experiment.tracker import NoOpTracker
from feature_forge.llm.base import LLMResponse


def _make_llm(code: str):
    class FakeLLM:
        async def complete(self, messages, temperature=0.2, max_tokens=4096, **kw):
            return LLMResponse(content=code, model="fake")

    return FakeLLM()


def _make_df():
    X = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "b": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    return X, y


def _make_evaluator():
    cfg = Settings(evaluation={"cv_folds": 2})
    return CVEvaluator(config=cfg)


class TestLLMFESingleShotArtifacts:
    def test_artifacts_populated(self):
        from feature_forge.baselines.llmfe import LLMFEBaseline

        code = """
import pandas as pd

def generate_features(df):
    result = pd.DataFrame(index=df.index)
    result['feat1'] = df['a'] * 2
    result['feat2'] = df['b'] + df['a']
    return result
"""
        llm = _make_llm(code)
        baseline = LLMFEBaseline(llm_client=llm, mode="single_shot")
        X, y = _make_df()
        baseline.fit(X, y)

        artifacts = baseline.get_artifacts()
        assert "prompt" in artifacts
        assert "raw_response" in artifacts
        assert "generated_code" in artifacts
        assert isinstance(artifacts["generated_code"], str)

    def test_transform_after_fit(self):
        from feature_forge.baselines.llmfe import LLMFEBaseline

        code = """
import pandas as pd

def generate_features(df):
    result = pd.DataFrame(index=df.index)
    result['sum_ab'] = df['a'] + df['b']
    return result
"""
        llm = _make_llm(code)
        baseline = LLMFEBaseline(llm_client=llm)
        X, y = _make_df()
        baseline.fit(X, y)
        result = baseline.transform(X)
        assert "sum_ab" in result.columns


class TestLLMFEIterativeArtifacts:
    def test_iterations_list_populated(self):
        from feature_forge.baselines.llmfe import LLMFEBaseline

        code = """
import pandas as pd

def generate_features(df):
    result = pd.DataFrame(index=df.index)
    result['feat'] = df['a'] + 1
    return result
"""
        llm = _make_llm(code)
        baseline = LLMFEBaseline(
            llm_client=llm,
            mode="iterative",
            n_features=2,
            evaluator=_make_evaluator(),
        )
        X, y = _make_df()
        baseline.fit(X, y)

        artifacts = baseline.get_artifacts()
        assert "iterations" in artifacts
        assert isinstance(artifacts["iterations"], list)
        assert len(artifacts["iterations"]) == 2
        for it in artifacts["iterations"]:
            assert "iteration" in it
            assert "generated_code" in it


class TestCAAFEUnifiedArtifacts:
    def test_unified_artifacts(self):
        from feature_forge.baselines.caafe import CAAFEBaseline

        code = """
import pandas as pd

def generate_features(df):
    result = pd.DataFrame(index=df.index)
    result['cfeat'] = df['a'] * 3
    return result
"""
        llm = _make_llm(code)
        baseline = CAAFEBaseline(
            llm_client=llm,
            variant="unified",
            iterations=2,
            evaluator=_make_evaluator(),
        )
        X, y = _make_df()
        baseline.fit(X, y)

        artifacts = baseline.get_artifacts()
        assert artifacts.get("variant") == "unified"
        assert "iterations" in artifacts
        assert "dataset_description" in artifacts
        assert len(artifacts["iterations"]) == 2

    def test_unified_transform(self):
        from feature_forge.baselines.caafe import CAAFEBaseline

        code = """
import pandas as pd

def generate_features(df):
    result = pd.DataFrame(index=df.index)
    result['cf'] = df['a'] + df['b']
    return result
"""
        llm = _make_llm(code)
        baseline = CAAFEBaseline(
            llm_client=llm,
            variant="unified",
            iterations=1,
            evaluator=_make_evaluator(),
        )
        X, y = _make_df()
        baseline.fit(X, y)
        result = baseline.transform(X)
        assert isinstance(result, pd.DataFrame)


class TestCompareMethods:
    def test_compare_returns_all_methods(self):
        from feature_forge.baselines.llmfe import LLMFEBaseline

        code = """
import pandas as pd

def generate_features(df):
    return pd.DataFrame({'f': df['a'] + 1}, index=df.index)
"""
        llm = _make_llm(code)
        methods = {"llmfe": LLMFEBaseline(llm_client=llm)}
        X, y = _make_df()

        results = compare_methods(methods, X, y)
        assert "llmfe" in results
        assert "error" not in results["llmfe"]

    def test_compare_with_tracker(self):
        from feature_forge.baselines.llmfe import LLMFEBaseline

        code = """
import pandas as pd

def generate_features(df):
    return pd.DataFrame({'f': df['a'] * 2}, index=df.index)
"""
        llm = _make_llm(code)
        methods = {"llmfe": LLMFEBaseline(llm_client=llm)}
        X, y = _make_df()
        tracker = NoOpTracker(project="test")

        results = compare_methods(methods, X, y, tracker=tracker)
        assert "llmfe" in results

    def test_compare_handles_errors(self):
        class FailingBaseline(Baseline):
            def __init__(self):
                super().__init__("failing")

            def fit(self, X_train, y_train):
                raise RuntimeError("boom")

            def transform(self, X):
                return X

        methods = {"fail": FailingBaseline()}
        X, y = _make_df()

        results = compare_methods(methods, X, y)
        assert "error" in results["fail"]
        assert "boom" in results["fail"]["error"]


class TestMALMASArtifacts:
    def _make_fe(self):
        from feature_forge.api import MALMASFeatureEngineer

        return MALMASFeatureEngineer(llm_client=_make_llm("pass"))

    def test_get_artifacts_empty_before_fit(self):
        fe = self._make_fe()
        assert fe.get_artifacts() == {}

    def test_generated_scripts_empty_before_fit(self):
        fe = self._make_fe()
        assert fe.generated_scripts == []

    def test_feature_metadata_empty_before_fit(self):
        fe = self._make_fe()
        assert fe.feature_metadata == []


class TestDiskModeArtifacts:
    def test_llmfe_disk_mode_returns_lazy_ref(self):
        from feature_forge.artifacts.base import ArtifactConfig
        from feature_forge.artifacts.storage import LazyDataFrameRef
        from feature_forge.baselines.llmfe import LLMFEBaseline

        code = """
import pandas as pd
def generate_features(df):
    result = pd.DataFrame(index=df.index)
    result['f1'] = df['a'] * 2
    return result
"""
        llm = _make_llm(code)
        cfg = ArtifactConfig(storage_mode="disk", storage_format="parquet")
        baseline = LLMFEBaseline(
            llm_client=llm,
            mode="iterative",
            n_features=1,
            evaluator=_make_evaluator(),
            artifact_config=cfg,
        )
        X, y = _make_df()
        baseline.fit(X, y)

        # Iterative mode stores DataFrames inside iterations list
        artifacts = baseline.get_artifacts()
        assert "iterations" in artifacts
        it = artifacts["iterations"][0]
        # In disk mode, DataFrames inside iterations are stored as LazyDataFrameRef
        assert isinstance(it["all_new_features"], LazyDataFrameRef)
        # Verify lazy loading works
        loaded = it["all_new_features"].load()
        assert isinstance(loaded, pd.DataFrame)
        assert "f1" in loaded.columns

    def test_compare_methods_propagates_artifact_config(self):
        from feature_forge.artifacts.base import ArtifactConfig
        from feature_forge.baselines.llmfe import LLMFEBaseline

        code = """
import pandas as pd
def generate_features(df):
    return pd.DataFrame({'f': df['a'] + 1}, index=df.index)
"""
        llm = _make_llm(code)
        methods = {"llmfe": LLMFEBaseline(llm_client=llm)}
        X, y = _make_df()
        cfg = ArtifactConfig(storage_mode="disk")
        compare_methods(methods, X, y, artifact_config=cfg)
        # Verify the method received the config in-place
        assert methods["llmfe"].artifact_config.storage_mode == "disk"
