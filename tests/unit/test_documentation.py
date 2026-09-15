"""Tests for generated documentation and shipped-doc hygiene.

These focus on PR 5's generated stage-DAG documentation:

* the generated ``docs/generated/stage_dags.md`` is byte-identical to a fresh
  render (deterministic freshness via ``--check``);
* the generator supports ``--output PATH`` and a non-mutating ``--check``;
* shipped top-level docs never reference the removed ``ExperimentMatrix`` /
  ``ExperimentRunner`` classes as current API, and never claim WandB is the
  default tracker.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.generate_stage_dag_docs as gen  # noqa: E402

GENERATED_DOC = REPO_ROOT / "docs" / "generated" / "stage_dags.md"
SHIPPED_DOCS = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "index.md",
    REPO_ROOT / "docs" / "quick_start.md",
    REPO_ROOT / "docs" / "api_reference.md",
    REPO_ROOT / "docs" / "migration_guide.md",
]

REMOVED_CLASSES = ("ExperimentMatrix", "ExperimentRunner", "ExperimentCaseExecutor")
# A mention of a removed class is acceptable only inside a removal / deprecation
# note (e.g. "the ExperimentMatrix class was removed"); current-API usage is not.
REMOVAL_CONTEXT = re.compile(r"removed|deprecated|no longer|were removed", re.IGNORECASE)
WANDB_DEFAULT_CLAIM = re.compile(
    r"default\w*\s+(backend|tracker)\s+(is\s+)?(wandb|weights\s*[\s_and]*\s*biases)",
    re.IGNORECASE,
)
WANDB_IS_DEFAULT = re.compile(r"wandb\s+is\s+(the\s+)?default", re.IGNORECASE)


def test_generated_doc_is_fresh(tmp_path: Path) -> None:
    """A fresh render must byte-match the committed generated document."""
    rendered = gen.build_markdown()
    candidate = tmp_path / "stage_dags.md"
    candidate.write_text(rendered, encoding="utf-8")
    assert candidate.read_bytes() == GENERATED_DOC.read_bytes()


def test_generator_render_is_deterministic() -> None:
    assert gen.build_markdown() == gen.build_markdown()


def test_generated_doc_covers_every_expected_node() -> None:
    """Every node in EXPECTED_NODE_NAMES must appear in the generated doc."""
    text = gen.build_markdown()
    for names in gen.EXPECTED_NODE_NAMES.values():
        for node in names:
            assert node in text, f"generated doc is missing node {node}"


def test_generated_doc_has_four_stage_dags_and_architecture() -> None:
    """One Mermaid DAG per stage plus the durable-boundary architecture."""
    text = gen.build_markdown()
    assert text.count("```mermaid") == len(gen.STAGE_ORDER) + 1
    assert "## Durable-boundary architecture" in text
    for stage in gen.STAGE_ORDER:
        assert f"## {stage.capitalize()} stage" in text


def test_generator_output_flag_writes_rendered_doc(tmp_path: Path) -> None:
    """``--output PATH`` writes exactly the rendered markdown."""
    out = tmp_path / "out.md"
    assert gen.main(["--output", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == gen.build_markdown()


def test_generator_check_mode_passes_and_does_not_mutate() -> None:
    """``--check`` exits 0 against the committed doc and never writes."""
    before = GENERATED_DOC.read_bytes()
    assert gen.main(["--check"]) == 0
    assert GENERATED_DOC.read_bytes() == before


def test_generator_check_mode_detects_stale_and_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--check`` fails on a stale doc and passes on a fresh one."""
    stale = tmp_path / "stage_dags.md"
    stale.write_text("# stale\n", encoding="utf-8")
    monkeypatch.setattr(gen, "DEFAULT_OUTPUT", stale)
    assert gen.main(["--check"]) == 1

    fresh = tmp_path / "fresh.md"
    fresh.write_text(gen.build_markdown(), encoding="utf-8")
    monkeypatch.setattr(gen, "DEFAULT_OUTPUT", fresh)
    assert gen.main(["--check"]) == 0


def test_shipped_docs_omit_removed_classes() -> None:
    """Shipped docs must not present removed classes as current API."""
    for doc in SHIPPED_DOCS:
        text = doc.read_text(encoding="utf-8")
        for token in REMOVED_CLASSES:
            for match in re.finditer(re.escape(token), text):
                start = max(0, match.start() - 200)
                end = min(len(text), match.end() + 200)
                window = text[start:end]
                assert REMOVAL_CONTEXT.search(window), (
                    f"{doc.name} references {token} outside a removal/deprecation "
                    f"context: ...{window!r}..."
                )


def test_shipped_docs_do_not_claim_wandb_default() -> None:
    """Shipped docs must not claim WandB is the default tracker."""
    for doc in SHIPPED_DOCS:
        text = doc.read_text(encoding="utf-8")
        assert not WANDB_DEFAULT_CLAIM.search(text), (
            f"{doc.name} appears to claim WandB is the default tracker"
        )
        assert not WANDB_IS_DEFAULT.search(text), f"{doc.name} claims WandB is the default tracker"
