"""Tests for the Intel/OpenMP runtime bootstrap."""

from __future__ import annotations

import os

import pytest


def test_kmp_duplicate_ok_set_on_package_import() -> None:
    # feature_forge/__init__.py sets this before any OpenMP runtime loads.
    import feature_forge  # noqa: F401

    assert os.environ.get("KMP_DUPLICATE_LIB_OK") == "TRUE"


def test_bootstrap_disabled_returns_false() -> None:
    from feature_forge.runtime.intel_bootstrap import bootstrap_intel_runtime

    assert bootstrap_intel_runtime(enabled=False) is False


def test_bootstrap_disabled_is_safe_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    # When disabled, the bootstrap must not patch and must leave activation
    # untouched, regardless of whether scikit-learn-intelex is installed.
    import feature_forge.runtime.intel_bootstrap as ib

    monkeypatch.setattr(ib, "_INTEL_ACTIVE", False)
    assert ib.bootstrap_intel_runtime(enabled=False) is False
    assert ib.is_intel_active() is False


def test_bootstrap_enabled_with_sklearnex_patches(monkeypatch: pytest.MonkeyPatch) -> None:
    # When scikit-learn-intelex is importable, the bootstrap must call
    # patch_sklearn() and report activation. This guards the patch-first order.
    import sys
    import types

    import feature_forge.runtime.intel_bootstrap as ib

    calls: list[str] = []
    fake = types.ModuleType("sklearnex")

    def patch_sklearn(verbose: bool = True, **kwargs: object) -> None:
        calls.append("patch_sklearn")

    # Attach a fake sklearnex patch entry point; mypy cannot know ModuleType
    # allows arbitrary attributes here.
    fake.patch_sklearn = patch_sklearn  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sklearnex", fake)
    monkeypatch.setattr(ib, "_INTEL_ACTIVE", False)

    from feature_forge.runtime.intel_bootstrap import bootstrap_intel_runtime

    assert bootstrap_intel_runtime(enabled=True) is True
    assert calls == ["patch_sklearn"]
    assert ib.is_intel_active() is True


def test_bootstrap_settings_error_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    # A broken settings file must never crash the package import; the
    # bootstrap fails closed (no patching, stays inactive).
    import feature_forge.config as config_module
    import feature_forge.runtime.intel_bootstrap as ib

    def boom() -> None:
        raise ValueError("malformed settings.yaml")

    monkeypatch.setattr(config_module, "get_settings", boom)
    monkeypatch.setattr(ib, "_INTEL_ACTIVE", False)
    assert ib.bootstrap_intel_runtime(enabled=None) is False
    assert ib.is_intel_active() is False


def test_runtime_info_shape() -> None:
    from feature_forge.runtime.intel_bootstrap import runtime_info

    info = runtime_info()
    assert isinstance(info["intel_active"], bool)
    if info["intel_active"]:
        assert isinstance(info["sklearnex_version"], str)
        assert info["sklearnex_version"]
