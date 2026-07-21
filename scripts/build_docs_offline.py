#!/usr/bin/env python3
"""Build strict MkDocs with application configuration scrubbed and sockets denied."""

from __future__ import annotations

from pathlib import Path

from feature_forge.verification.documentation import deny_network, scrubbed_environment

ROOT = Path(__file__).resolve().parent.parent


def build_docs(config_file: Path = ROOT / "mkdocs.yml") -> None:
    """Build the site while in-process MkDocs/plugins cannot open sockets."""
    from mkdocs.commands.build import build
    from mkdocs.config import load_config

    with scrubbed_environment(), deny_network():
        config = load_config(config_file=str(config_file), strict=True)
        build(config)


if __name__ == "__main__":
    build_docs()
