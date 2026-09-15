from __future__ import annotations

import pathlib
import re

import pytest
from jinja2 import exceptions as jinja_exceptions

from feature_forge.methods._prompting import PromptParams
from feature_forge.methods.caafe.prompts import CAAFEUnifiedParams
from feature_forge.methods.caafe.prompts import get_registry as get_caafe_registry
from feature_forge.methods.llmfe.prompts import LLMFEIterativeParams, LLMFESingleShotParams
from feature_forge.methods.llmfe.prompts import get_registry as get_llmfe_registry
from feature_forge.methods.malmas.memory.prompts import (
    SummarizeAgentParams,
    SummarizeGlobalParams,
)
from feature_forge.methods.malmas.prompts import get_registry as get_malmas_registry
from feature_forge.methods.malmus.prompts import MalmusIterativeParams
from feature_forge.methods.malmus.prompts import get_registry as get_malmus_registry


def test_malmas_prompt_registry_loads_yaml() -> None:
    prompt = get_malmas_registry().get("unary")
    assert prompt.system
    assert prompt.description


def test_method_prompt_registries_load_yaml() -> None:
    assert get_caafe_registry().get("unified").system
    assert get_llmfe_registry().get("single_shot").system
    assert get_malmus_registry().get("iterative").system


def test_missing_prompt_raises_key_error() -> None:
    with pytest.raises(KeyError, match="Prompt 'missing' not found"):
        get_malmas_registry().get("missing")


def test_llmfe_iterative_params_validate_iteration_bounds() -> None:
    with pytest.raises(ValueError, match="iteration must be <= n_iterations"):
        LLMFEIterativeParams(columns="a,b", task="classification", n_iterations=2, iteration=3)


def test_malmus_iterative_params_validate_iteration_bounds() -> None:
    with pytest.raises(ValueError, match="iteration must be <= n_iterations"):
        MalmusIterativeParams(columns="a,b", task="classification", n_iterations=2, iteration=3)


def test_caafe_params_validate_iteration_bounds() -> None:
    with pytest.raises(ValueError, match="iteration must be <= iterations"):
        CAAFEUnifiedParams(description="x", iterations=2, iteration=3)


def test_registry_render_substitutes_jinja_variables() -> None:
    params = MalmusIterativeParams(
        columns="a,b",
        task="classification",
        n_iterations=3,
        iteration=2,
        existing_features="f1",
        feedback="good",
    )
    rendered = get_malmus_registry().render("iterative", params)
    assert "a,b" in rendered
    assert "{{" not in rendered
    assert "{%" not in rendered


def test_registry_render_missing_variable_raises() -> None:
    params = LLMFESingleShotParams(columns="a", task="classification")
    # columns is declared, so a template variable absent from the params
    # model cannot be simulated via public API; instead render a template
    # with an undeclared variable directly through the Jinja env.
    registry = get_llmfe_registry()
    env = registry._build_env()
    with pytest.raises(jinja_exceptions.UndefinedError):
        env.from_string("{{ not_a_declared_var }}").render(**params.model_context())


def test_prompt_params_context_exposes_all_fields() -> None:
    class _P(PromptParams):
        a: int
        b: str = "x"

    assert _P(a=1).model_context() == {"a": 1, "b": "x"}


def test_render_messages_includes_system_and_user() -> None:
    params = SummarizeAgentParams(
        agent_name="unary",
        examples_text="ex",
        stats_text="st",
    )
    messages = get_malmas_registry().render_messages("summarize_agent", params)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "unary agent" in messages[0]["content"]
    assert "ex" in messages[1]["content"]
    assert "st" in messages[1]["content"]


def test_render_messages_global_summary_pair() -> None:
    params = SummarizeGlobalParams(combined_prompt="all agents data", task_description="titanic")
    messages = get_malmas_registry().render_messages("summarize_global", params)
    assert messages[0]["role"] == "system"
    assert "titanic" in messages[1]["content"]
    assert "all agents data" in messages[1]["content"]


def test_versioned_files_resolve_to_latest(tmp_path: pathlib.Path) -> None:
    from feature_forge.methods._prompting import PromptRegistry

    (tmp_path / "demo.v1.yaml").write_text("system: one {{ x }}\n")
    (tmp_path / "demo.v2.yaml").write_text("system: two {{ x }}\n")
    registry = PromptRegistry(tmp_path)

    class _P(PromptParams):
        x: str

    assert registry.render("demo", _P(x="!")) == "two !"
    assert registry.render("demo", _P(x="!"), version=1) == "one !"
    assert registry.get("demo").version == 2


def test_missing_version_lists_available(tmp_path: pathlib.Path) -> None:
    from feature_forge.methods._prompting import PromptRegistry

    (tmp_path / "demo.v1.yaml").write_text("system: one\n")
    registry = PromptRegistry(tmp_path)
    with pytest.raises(KeyError, match=r"available: \[1\]"):
        registry.get("demo", version=7)


def test_provenance_identity_and_content_hash() -> None:
    registry = get_malmus_registry()
    prov = registry.provenance("iterative")
    assert prov.prompt_name == "iterative"
    assert prov.prompt_version == 1
    assert len(prov.prompt_sha256) == 64
    # same content → same hash; different content → different hash
    again = registry.provenance("iterative")
    assert again.prompt_sha256 == prov.prompt_sha256
    other = registry.provenance("single_shot")
    assert other.prompt_sha256 != prov.prompt_sha256


def test_real_prompt_files_are_versioned() -> None:
    import feature_forge.methods.malmus.prompts as pkg
    from feature_forge.methods._prompting import prompts_dir

    yaml_files = [p.name for p in prompts_dir(pkg.__package__).glob("*.yaml")]
    assert yaml_files
    assert all(re.fullmatch(r"[a-z_]+\.v\d+\.yaml", name) for name in yaml_files)
