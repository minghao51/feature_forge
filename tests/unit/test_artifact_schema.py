"""Tests for artifact schema, diff, dashboard, and persistence."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from feature_forge.artifacts.base import ArtifactConfig, ArtifactExporter
from feature_forge.artifacts.dashboard import ArtifactDashboard
from feature_forge.artifacts.diff import ArtifactDiff
from feature_forge.artifacts.schema import (
    ArtifactBundle,
    ArtifactConfigSchema,
    FeatureMetadata,
    ProvenanceRecord,
)
from feature_forge.artifacts.storage import LazyDataFrameRef


class DummyExporter(ArtifactExporter):
    """Minimal exporter for testing."""

    def __init__(self, artifacts: dict[str, Any] | None = None) -> None:
        super().__init__()
        self._artifacts = artifacts or {}

    @property
    def generated_scripts(self) -> list[str]:
        return self._artifacts.get("scripts", [])

    def get_artifacts(self) -> dict[str, Any]:
        return self._artifacts

    @property
    def feature_metadata(self) -> list[dict[str, Any]]:
        return self._artifacts.get("metadata", [])


class TestArtifactConfigSchema:
    def test_default_values(self) -> None:
        cfg = ArtifactConfigSchema()
        assert cfg.storage_mode == "memory"
        assert cfg.storage_format == "parquet"
        assert cfg.spill_threshold_bytes == 50 * 1024 * 1024

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="storage_dir"):
            ArtifactConfigSchema(storage_dir="")

    def test_from_dataclass(self) -> None:
        dc = ArtifactConfig(storage_mode="disk", storage_dir="/tmp/test")
        schema = dc.to_schema()
        assert schema.storage_mode == "disk"
        assert schema.storage_dir == "/tmp/test"


class TestFeatureMetadata:
    def test_basic(self) -> None:
        fm = FeatureMetadata(name="feat_a", method="llmfe", gain=0.05)
        assert fm.name == "feat_a"
        assert fm.gain == 0.05

    def test_to_dataframe_empty(self) -> None:
        bundle = ArtifactBundle(method_name="test")
        df = bundle.to_feature_dataframe()
        assert df.empty

    def test_to_dataframe(self) -> None:
        bundle = ArtifactBundle(
            method_name="test",
            feature_metadata=[
                FeatureMetadata(name="a", method="m1", gain=0.1),
                FeatureMetadata(name="b", method="m1", gain=0.2),
            ],
        )
        df = bundle.to_feature_dataframe()
        assert len(df) == 2
        assert list(df["name"]) == ["a", "b"]


class TestArtifactBundle:
    def test_dedup_scripts(self) -> None:
        bundle = ArtifactBundle(
            method_name="test",
            generated_scripts=["code_a", "code_a", "code_b"],
        )
        assert len(bundle.generated_scripts) == 2

    def test_provenance_dataframe(self) -> None:
        bundle = ArtifactBundle(
            method_name="test",
            provenance_records=[
                ProvenanceRecord(feature_name="x", source_method="m1", cv_gain=0.1),
            ],
        )
        df = bundle.to_provenance_dataframe()
        assert len(df) == 1
        assert df["feature_name"].iloc[0] == "x"


class TestSaveLoadArtifacts:
    def test_roundtrip(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        exporter = DummyExporter(
            {
                "scripts": ["def f(): pass"],
                "metadata": [{"name": "feat", "gain": 0.1}],
                "my_df": df,
                "text": "hello",
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter.save_artifacts(tmpdir)
            restored = ArtifactExporter.load_artifacts(tmpdir)

        assert "my_df" in restored
        pd.testing.assert_frame_equal(restored["my_df"], df)
        assert restored["text"] == "hello"
        assert "__bundle__" in restored
        assert isinstance(restored["__bundle__"], ArtifactBundle)

    def test_lazy_dataframe_roundtrip(self) -> None:
        df = pd.DataFrame({"a": [1, 2]})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.parquet"
            df.to_parquet(path)
            ref = LazyDataFrameRef(str(path), "parquet")

            exporter = DummyExporter({"lazy": ref})
            exporter.save_artifacts(tmpdir)
            restored = ArtifactExporter.load_artifacts(tmpdir)

            assert isinstance(restored["lazy"], LazyDataFrameRef)
            pd.testing.assert_frame_equal(restored["lazy"].load(), df)

    def test_file_path_input(self) -> None:
        exporter = DummyExporter({"x": 1})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "out.json"
            exporter.save_artifacts(path)
            assert path.exists()
            restored = ArtifactExporter.load_artifacts(path)
        assert restored["x"] == 1


class TestArtifactDiff:
    def test_overlap(self) -> None:
        b1 = ArtifactBundle(
            method_name="m1",
            feature_metadata=[
                FeatureMetadata(name="a", method="m1", gain=0.1),
                FeatureMetadata(name="b", method="m1", gain=0.2),
            ],
        )
        b2 = ArtifactBundle(
            method_name="m2",
            feature_metadata=[
                FeatureMetadata(name="b", method="m2", gain=0.15),
                FeatureMetadata(name="c", method="m2", gain=0.3),
            ],
        )
        diff = ArtifactDiff({"m1": b1, "m2": b2})

        assert diff.all_features == ["a", "b", "c"]
        assert diff.shared_features() == ["b"]
        assert diff.unique_features("m1") == ["a"]
        assert diff.unique_features("m2") == ["c"]

    def test_gain_comparison(self) -> None:
        b1 = ArtifactBundle(
            method_name="m1",
            feature_metadata=[FeatureMetadata(name="a", method="m1", gain=0.1)],
        )
        b2 = ArtifactBundle(
            method_name="m2",
            feature_metadata=[FeatureMetadata(name="a", method="m2", gain=0.2)],
        )
        diff = ArtifactDiff({"m1": b1, "m2": b2})
        df = diff.gain_comparison()
        assert df.loc["a", "m1"] == 0.1
        assert df.loc["a", "m2"] == 0.2

    def test_summary(self) -> None:
        b1 = ArtifactBundle(
            method_name="m1",
            feature_metadata=[FeatureMetadata(name="a", method="m1", gain=0.1)],
        )
        diff = ArtifactDiff({"m1": b1})
        summary = diff.summary()
        assert summary["total_unique_features"] == 1
        assert summary["per_method"]["m1"]["total_features"] == 1

    def test_empty(self) -> None:
        diff = ArtifactDiff({})
        assert diff.all_features == []
        assert diff.summary()["total_unique_features"] == 0


class TestArtifactDashboard:
    def test_html_output(self) -> None:
        b1 = ArtifactBundle(
            method_name="m1",
            feature_metadata=[FeatureMetadata(name="a", method="m1", gain=0.1)],
            generated_scripts=["def f(): pass"],
        )
        dash = ArtifactDashboard({"m1": b1})
        html = dash.to_html()
        assert "Feature Forge Artifact Report" in html
        assert "a" in html
        assert "def f(): pass" in html

    def test_save(self) -> None:
        b1 = ArtifactBundle(method_name="m1")
        dash = ArtifactDashboard({"m1": b1})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.html"
            dash.save(path)
            assert path.exists()
            content = path.read_text()
            assert "<!DOCTYPE html>" in content

    def test_to_json(self) -> None:
        b1 = ArtifactBundle(
            method_name="m1",
            feature_metadata=[FeatureMetadata(name="a", method="m1", gain=0.1)],
        )
        dash = ArtifactDashboard({"m1": b1})
        j = dash.to_json()
        data = json.loads(j)
        assert data["total_unique_features"] == 1


class TestProvenanceRecords:
    def test_malmas_provenance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from feature_forge.api import FeatureForge

        # Mock LLM client to avoid API key requirement
        monkeypatch.setattr(FeatureForge, "_default_llm_client", lambda self: object())
        fe = FeatureForge()
        fe.pipeline_result = {
            "selected_features": ["f1"],
            "feature_codes": ["code"],
            "round_artifacts": [
                {
                    "round": 1,
                    "agents": ["unary"],
                    "generated_code": "code",
                    "specs": [{"name": "f1", "agent": "unary"}],
                    "gains": {"f1": 0.1},
                },
            ],
        }
        prov = fe.provenance_records
        assert len(prov) == 1
        assert prov[0]["feature_name"] == "f1"
        assert prov[0]["source_method"] == "malmas"
        assert prov[0]["source_agent"] == "unary"

    def test_malmas_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from feature_forge.api import FeatureForge

        monkeypatch.setattr(FeatureForge, "_default_llm_client", lambda self: object())
        fe = FeatureForge()
        assert fe.provenance_records == []


class TestLogArtifactsDedup:
    def test_deduplicates_code(self) -> None:
        class FakeTracker:
            def __init__(self) -> None:
                self.logged: dict[str, Any] = {}

            def log_artifacts_dict(self, artifacts: dict[str, Any], prefix: str = "") -> None:
                self.logged.update({f"{prefix}{k}": v for k, v in artifacts.items()})

        exporter = DummyExporter(
            {
                "code1": "def a(): pass",
                "code2": "def a(): pass",  # duplicate
                "code3": "def b(): pass",
            }
        )
        tracker = FakeTracker()
        exporter.log_artifacts(tracker)
        assert "code1" in tracker.logged
        assert "code2" not in tracker.logged
        assert "code3" in tracker.logged
