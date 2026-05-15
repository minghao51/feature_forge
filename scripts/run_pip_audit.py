"""Run pip-audit with a repository-managed vulnerability allowlist."""

from __future__ import annotations

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


def main() -> int:
    cmd = [sys.executable, "-m", "pip_audit", "--skip-editable"]
    for vuln_id in _load_allowlist(ALLOWLIST_PATH):
        cmd.extend(["--ignore-vuln", vuln_id])

    proc = subprocess.run(cmd, check=False)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
