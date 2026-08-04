.PHONY: docs docs-check docs-generated docs-serve

docs-generated:
	uv run --extra pipeline python scripts/generate_medallion_docs.py --write

docs-check:
	uv run --extra pipeline python scripts/generate_medallion_docs.py --check
	uv run python scripts/build_docs_offline.py

docs: docs-generated
	uv run python scripts/build_docs_offline.py

docs-serve: docs-generated
	uv run mkdocs serve
