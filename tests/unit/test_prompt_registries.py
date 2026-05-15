from __future__ import annotations

import pytest

from feature_forge.methods.caafe.prompts import CAAFEUnifiedParams
from feature_forge.methods.caafe.prompts import get_registry as get_caafe_registry
from feature_forge.methods.llmfe.prompts import LLMFEIterativeParams
from feature_forge.methods.llmfe.prompts import get_registry as get_llmfe_registry
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
