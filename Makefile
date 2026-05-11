.PHONY: notebooks notebooks-staged docs docs-serve

notebooks:
	uv run quarto render notebooks/
	uv run python scripts/extract_shared_notebook_assets.py

notebooks-staged:
	@changed=$$(git diff --cached --name-only -- 'notebooks/*.qmd'); \
	if [ -n "$$changed" ]; then \
		for f in $$changed; do uv run quarto render "$$f"; done; \
		uv run python scripts/extract_shared_notebook_assets.py; \
		git add notebooks/_freeze/ docs/notebooks/html/; \
	fi

docs: notebooks
	uv run python scripts/generate_notebook_docs.py
	uv run mkdocs build

docs-serve: notebooks
	uv run python scripts/generate_notebook_docs.py
	uv run mkdocs serve
