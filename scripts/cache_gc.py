"""Garbage-collect the LLM response cache (expired entries + size cap).

Reads TTL / size-cap settings from the project configuration (YAML, env,
or ``.env``) unless overridden on the command line. ``--dry-run`` reports
entry count and volume without deleting anything.

Run:
    uv run python scripts/cache_gc.py [--dry-run]
    uv run python scripts/cache_gc.py --cache-dir memory_files/llm_cache \
        --ttl-days 30 --size-limit-mb 512
"""

from __future__ import annotations

import argparse
import json
import sys

from feature_forge.llm.cache import DiskCache


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", default=None, help="Cache directory (default: from Settings)"
    )
    parser.add_argument(
        "--ttl-days", type=float, default=None, help="Entry TTL in days (default: from Settings)"
    )
    parser.add_argument(
        "--size-limit-mb", type=float, default=None, help="Size cap in MiB (default: from Settings)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Report stats only; delete nothing")
    args = parser.parse_args(argv)

    from feature_forge.config import get_settings

    settings = get_settings()
    cache_dir = args.cache_dir or settings.llm.cache_dir
    ttl_days = args.ttl_days if args.ttl_days is not None else settings.llm.cache_ttl_days
    size_mb = (
        args.size_limit_mb if args.size_limit_mb is not None else settings.llm.cache_size_limit_mb
    )

    cache = DiskCache(
        cache_dir=cache_dir,
        ttl_seconds=(ttl_days * 86400.0) if ttl_days is not None else None,
        size_limit_bytes=(int(size_mb * 1024 * 1024)) if size_mb is not None else None,
    )
    try:
        if args.dry_run:
            print(json.dumps({"cache_dir": cache_dir, "dry_run": True, **cache.stats()}))
        else:
            print(json.dumps({"cache_dir": cache_dir, **cache.maintain()}))
    finally:
        cache.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
