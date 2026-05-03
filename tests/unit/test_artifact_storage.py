"""Tests for artifact storage: LazyDataFrameRef and DataFrameStorage."""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import pytest

from feature_forge.artifacts.base import ArtifactConfig
from feature_forge.artifacts.storage import DataFrameStorage, LazyDataFrameRef


@pytest.fixture
def sample_df():
    return pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


class TestArtifactConfig:
    def test_defaults(self):
        cfg = ArtifactConfig()
        assert cfg.storage_mode == "memory"
        assert cfg.storage_format == "parquet"
        assert cfg.spill_threshold_bytes == 50 * 1024 * 1024

    def test_disk_mode_creates_dir(self, tmp_dir):
        d = os.path.join(tmp_dir, "artifacts")
        ArtifactConfig(storage_mode="disk", storage_dir=d)
        assert os.path.isdir(d)


class TestLazyDataFrameRef:
    def test_parquet_roundtrip(self, sample_df, tmp_dir):
        path = os.path.join(tmp_dir, "test.parquet")
        sample_df.to_parquet(path)
        ref = LazyDataFrameRef(path, "parquet")
        loaded = ref.load()
        pd.testing.assert_frame_equal(loaded, sample_df)

    def test_csv_roundtrip(self, sample_df, tmp_dir):
        path = os.path.join(tmp_dir, "test.csv")
        sample_df.to_csv(path)
        ref = LazyDataFrameRef(path, "csv")
        loaded = ref.load()
        pd.testing.assert_frame_equal(loaded, sample_df)

    def test_feather_roundtrip(self, sample_df, tmp_dir):
        path = os.path.join(tmp_dir, "test.feather")
        sample_df.to_feather(path)
        ref = LazyDataFrameRef(path, "feather")
        loaded = ref.load()
        pd.testing.assert_frame_equal(loaded, sample_df)

    def test_invalid_format_raises(self, tmp_dir):
        path = os.path.join(tmp_dir, "test.xyz")
        with open(path, "w") as f:
            f.write("")
        ref = LazyDataFrameRef(path, "xyz")
        with pytest.raises(ValueError, match="Unsupported format"):
            ref.load()

    def test_invalidate(self, sample_df, tmp_dir):
        path = os.path.join(tmp_dir, "test.parquet")
        sample_df.to_parquet(path)
        ref = LazyDataFrameRef(path, "parquet")
        _ = ref.load()
        assert ref._df is not None
        ref.invalidate()
        assert ref._df is None

    def test_repr(self, tmp_dir):
        ref = LazyDataFrameRef("/tmp/test.parquet", "parquet")
        assert "not loaded" in repr(ref)


class TestDataFrameStorage:
    def test_memory_mode_returns_df(self, sample_df):
        storage = DataFrameStorage(ArtifactConfig(storage_mode="memory"))
        result = storage.store("test", sample_df)
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, sample_df)

    def test_disk_mode_returns_lazy_ref(self, sample_df, tmp_dir):
        cfg = ArtifactConfig(storage_mode="disk", storage_dir=tmp_dir)
        storage = DataFrameStorage(cfg)
        result = storage.store("test_key", sample_df)
        assert isinstance(result, LazyDataFrameRef)
        loaded = result.load()
        pd.testing.assert_frame_equal(loaded, sample_df)

    def test_hybrid_mode_small_stays_memory(self, sample_df, tmp_dir):
        cfg = ArtifactConfig(
            storage_mode="hybrid",
            storage_dir=tmp_dir,
            spill_threshold_bytes=1024 * 1024 * 1024,
        )
        storage = DataFrameStorage(cfg)
        result = storage.store("small", sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_hybrid_mode_large_spills_to_disk(self, tmp_dir):
        big_df = pd.DataFrame({"x": range(100_000)})
        cfg = ArtifactConfig(
            storage_mode="hybrid",
            storage_dir=tmp_dir,
            spill_threshold_bytes=1,
        )
        storage = DataFrameStorage(cfg)
        result = storage.store("big", big_df)
        assert isinstance(result, LazyDataFrameRef)
        loaded = result.load()
        pd.testing.assert_frame_equal(loaded, big_df)

    def test_disk_csv_format(self, sample_df, tmp_dir):
        cfg = ArtifactConfig(storage_mode="disk", storage_format="csv", storage_dir=tmp_dir)
        storage = DataFrameStorage(cfg)
        result = storage.store("csv_test", sample_df)
        assert isinstance(result, LazyDataFrameRef)
        loaded = result.load()
        pd.testing.assert_frame_equal(loaded, sample_df)

    def test_disk_feather_format(self, sample_df, tmp_dir):
        cfg = ArtifactConfig(storage_mode="disk", storage_format="feather", storage_dir=tmp_dir)
        storage = DataFrameStorage(cfg)
        result = storage.store("feather_test", sample_df)
        assert isinstance(result, LazyDataFrameRef)
        loaded = result.load()
        pd.testing.assert_frame_equal(loaded, sample_df)
