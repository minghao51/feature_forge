"""Run pip-audit with a repository-managed vulnerability allowlist."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ALLOWLIST_PATH = Path("security/pip_audit_allowlist.txt")


def _load_allowlist(path: Path) -> list[str]:
    if not path.exists():
        return []
    vuln_ids: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        vuln_id = line.split(maxsplit=1)[0]
        vuln_ids.append(vuln_id)
    return vuln_ids


def _build_command(requirements: Path | None) -> list[str]:
    """Build the pip-audit invocation, optionally auditing a requirements file.

    With no ``requirements`` path the active environment is audited
    (``--skip-editable``), preserving the default tooling-lane behavior. With a
    path, ``-r`` restricts the audit to the pinned packages in that file, which
    lets the tooling lane cover the optional-extras closure that no environment
    sync installs. The allowlist is applied in both modes.
    """
    cmd = [sys.executable, "-m", "pip_audit"]
    if requirements is None:
        cmd.append("--skip-editable")
    else:
        cmd.extend(["-r", str(requirements)])
    for vuln_id in _load_allowlist(ALLOWLIST_PATH):
        cmd.extend(["--ignore-vuln", vuln_id])
    return cmd


def main(argv: list[str] | None = None) -> int:
    """Run pip-audit and return its exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--requirements",
        type=Path,
        default=None,
        help=(
            "audit the packages pinned in this requirements file (pip-audit -r) "
            "instead of the active environment"
        ),
    )
    args = parser.parse_args(argv)

    proc = subprocess.run(_build_command(args.requirements), check=False)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
