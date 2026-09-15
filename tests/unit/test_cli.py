"""Machine-readable Hamilton plan CLI coverage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from feature_forge.cli import main


def test_run_plan_json_is_one_document_and_side_effect_free(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact_root = tmp_path / "artifacts"
    exit_code = main(
        [
            "run",
            "plan",
            "--format",
            "json",
            "--dataset",
            "titanic",
            "--method",
            "openfe",
            "--model",
            "random_forest",
            "--artifact-root",
            str(artifact_root),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["status"] == "valid"
    assert len(payload["plans"]) == 1
    assert payload["plans"][0]["unresolved_layers"] == [
        "bronze",
        "silver",
        "gold",
        "platinum",
    ]
    assert not artifact_root.exists()
