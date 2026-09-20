"""Kernel-level checks for the PR 5 strict/degraded sandbox profiles."""

from __future__ import annotations

import multiprocessing as mp
import os
import socket
import sys
import time
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

import feature_forge.evaluation.sandbox as sandbox_module
from feature_forge.config import SandboxProfile, Settings
from feature_forge.evaluation.sandbox import (
    SandboxedExecutor,
    _apply_strict_containment,
    _WorkerHandle,
)
from feature_forge.exceptions import SandboxContainmentError
from feature_forge.llm.base import LLMClient

_CODE = "def generate_features(df):\n    return df.assign(double=df.x * 2)\n"


def _kernel_probe_child(
    input_path: str, output_path: str, sentinel_path: str, result_pipe: mp.connection.Connection
) -> None:
    """Apply the real kernel policy, then probe paths and TCP in the child."""
    connection = result_pipe
    sentinel: int | str
    network: int | str
    try:
        containment = _apply_strict_containment(input_path, output_path)
        try:
            with open(sentinel_path, "rb"):
                sentinel = "allowed"
        except OSError as exc:
            sentinel = exc.errno if exc.errno is not None else -1
        try:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                client.connect(("127.0.0.1", 9))
                network = "allowed"
            except OSError as exc:
                network = exc.errno if exc.errno is not None else -1
            finally:
                client.close()
        except OSError as exc:
            network = exc.errno if exc.errno is not None else -1
        connection.send((containment.mechanism, sentinel, network))
    except BaseException as exc:  # pragma: no cover - child diagnostic
        connection.send((type(exc).__name__, str(exc), str(exc)))
    finally:
        connection.close()


def _require_landlock() -> None:
    if not sys.platform.startswith("linux"):
        pytest.skip("Landlock containment is Linux-specific")
    if sandbox_module._probe_landlock_abi() is None:
        pytest.skip("Landlock is unavailable on this host")


def test_strict_kernel_denies_external_sentinel_and_tcp(tmp_path: Path) -> None:
    """The kernel, rather than AST/monkeypatches, denies both probes."""
    _require_landlock()
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    sentinel_path = tmp_path / "outside-sentinel"
    input_path.touch()
    output_path.touch()
    sentinel_path.write_bytes(os.urandom(32))
    parent, child = mp.get_context("fork").Pipe(duplex=False)
    process = mp.get_context("fork").Process(
        target=_kernel_probe_child,
        args=(str(input_path), str(output_path), str(sentinel_path), child),
    )
    process.start()
    child.close()
    mechanism, sentinel, network = parent.recv()
    process.join(timeout=5)
    assert process.exitcode == 0
    assert mechanism.startswith("landlock")
    assert sentinel in (13, 1)  # EACCES or EPERM
    assert network in (13, 1)


def test_strict_grants_predeclared_output(tmp_path: Path) -> None:
    _require_landlock()
    result = SandboxedExecutor(timeout_seconds=15).execute(
        _CODE, pd.DataFrame({"x": [1.0, 2.0]}), source="containment-test"
    )
    assert result["double"].tolist() == [2.0, 4.0]


def test_strict_unavailable_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox_module, "_probe_landlock_abi", lambda: None)
    with pytest.raises(SandboxContainmentError, match="strict sandbox is unavailable"):
        SandboxedExecutor().execute(_CODE, pd.DataFrame({"x": [1.0]}))


def test_degraded_profile_is_explicit_and_recorded() -> None:
    executor = SandboxedExecutor(profile=SandboxProfile.DEGRADED_DEVELOPMENT)
    assert executor.provenance == {
        "sandbox_profile": "degraded_development",
        "sandbox_degraded": True,
        "sandbox_landlock_abi": None,
        "sandbox_mechanism": "process_only_ast",
    }
    result = executor.execute(_CODE, pd.DataFrame({"x": [3.0]}), source="development")
    assert result["double"].tolist() == [6.0]


def test_production_default_profile_is_strict(fake_llm: LLMClient) -> None:
    """The production default stays strict — no silent auto-downgrade (plan 23 §5.2)."""
    from feature_forge.methods.malmas.pipeline.core import CorePipeline

    assert Settings().evaluation.sandbox_profile is SandboxProfile.STRICT
    # Default wiring (no injected eval_kit/evaluator/sandbox): the pipeline
    # builds its executor from the resolved config (core.py __init__). No
    # worker is launched here, so no host containment dependency exists.
    pipeline = CorePipeline(config=Settings(), llm_client=fake_llm)
    assert pipeline.sandbox.profile is SandboxProfile.STRICT


def test_worker_containment_failure_reports_and_maps_to_typed_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Worker-side containment failure reports ``containment_unavailable`` (§7.6).

    The worker runs in-process with containment monkeypatched to fail (spawn
    children re-import the module, so parent monkeypatches cannot reach a real
    child). Process-group and environment setup are no-oped because they would
    otherwise mutate the pytest process; max_memory_mb=0 skips the RLIMIT_AS
    write for the same reason.
    """

    def _fail_containment(*_args: object, **_kwargs: object) -> sandbox_module._SandboxContainment:
        raise SandboxContainmentError("boom")

    monkeypatch.setattr(sandbox_module, "_apply_strict_containment", _fail_containment)
    monkeypatch.setattr(sandbox_module, "_establish_worker_process_group", lambda: None)
    monkeypatch.setattr(sandbox_module, "_scrub_worker_environment", lambda: None)

    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    pd.DataFrame({"x": [1.0]}).to_parquet(input_path)
    response_queue: mp.Queue[tuple[str, str, dict[str, object]]] = mp.Queue()
    try:
        sandbox_module._sandbox_worker_main(
            _CODE,
            str(input_path),
            str(output_path),
            0,
            response_queue,
            profile=SandboxProfile.STRICT.value,
        )
        status, payload, containment = response_queue.get(timeout=5)
        assert status == "containment_unavailable"
        assert payload == "boom"

        # The parent maps the worker report onto the typed error hierarchy.
        executor = SandboxedExecutor(profile=SandboxProfile.DEGRADED_DEVELOPMENT)
        handle = _WorkerHandle(
            cast("mp.Process", None), response_queue, str(input_path), str(output_path)
        )
        with pytest.raises(SandboxContainmentError, match="boom"):
            executor._consume_response(
                handle, status, payload, containment, deadline=time.monotonic() + 5
            )
    finally:
        response_queue.close()
        response_queue.cancel_join_thread()
