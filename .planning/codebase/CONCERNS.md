# Codebase Concerns & Tech Debt

_Analyzed: 2026-05-12 | ~7,400 LOC across 30+ modules, ~3,300 LOC tests_

---

## Critical

### SEC-1: Encrypted `.env` committed to git
- **File:** `.env` (tracked in git)
- **Risk:** The `.env.keys` file (which contains the decryption key) is correctly gitignored. However, if `.env.keys` is ever accidentally committed or leaked, all secrets are compromised.
- **Recommendation:** Verify `.env.keys` has never appeared in git history (`git log --all -- .env.keys`). Consider adding a pre-commit hook to prevent it.

### SEC-2: Sandbox `exec()` in worker process
- **File:** `src/feature_forge/evaluation/sandbox.py:249`
- `exec(compile(code, ...), safe_globals, local_vars)` executes LLM-generated code. The `ALLOWED_BUILTINS` set includes `__import__`, which could potentially be used to import restricted modules.
- **Risk:** `__import__` in `ALLOWED_BUILTINS` contradicts `__import__` in `FORBIDDEN_NAMES`. The AST check only catches direct `ast.Name` references, not indirect access through builtins dict.
- **Recommendation:** Remove `__import__` from `ALLOWED_BUILTINS`.

### SEC-3: API key stored as plaintext in memory via `os.environ`
- **File:** `src/feature_forge/llm/providers/litellm_provider.py:64`
- `os.environ.setdefault("API_KEY", self.api_key)` mutates global state for every LLM call.

---

## High

### PERF-1: Sequential feature evaluation (N CV runs per round)
- **File:** `src/feature_forge/methods/malmas/pipeline/core.py:200-211`
- Features are evaluated one at a time in a `for` loop. With 50 candidates × 5 folds = 250 model fits per round.
- **Recommendation:** Evaluate features in parallel using `asyncio.gather` or batch evaluation.

### PERF-2: Baseline re-evaluated every round
- **File:** `src/feature_forge/methods/malmas/pipeline/core.py:196`
- `evaluate_baseline()` called every round even though `X_train_enhanced` only changes by adding columns.
- **Recommendation:** Cache baseline when base features haven't changed.

### ARCH-2: Memoized context built but marginal utility
- **File:** `src/feature_forge/methods/malmas/pipeline/iterative.py:139-142`
- `_build_agent_context` adds `memory`, `positive_features`, `negative_features` to each agent's context, but `BaseFeatureAgent._build_user_prompt` only displays them as text — they never actually influence LLM generation beyond being appended to the prompt.

### ARCH-3: No retry logic for LLM API calls at provider level
- The base `LLMClient.complete()`/`complete_json()` methods now support retry via `RetryConfig`, but `_retry()` only activates when a `RetryConfig` is explicitly set on the client.

### BUG-1: Broad exception catch in router silently swallows errors
- **File:** `src/feature_forge/methods/malmas/agents/router.py:207-208`
- `except Exception: pass` silently swallows all LLM errors during agent selection.

### BUG-2: `NoMemoryPipeline._post_round` still updates router performance
- **File:** `src/feature_forge/methods/malmas/pipeline/ablations.py:36-40`
- The "no memory" ablation still tracks router performance, which may confound results.

---

## Medium

### TD-2: Notebook files duplicate LLM config instead of using `Settings`
- **Files:** All files in `notebooks/`
- Each notebook manually constructs `LLMConfig` instead of using the centralized `get_settings()`.

### TD-3: `CVEvaluator._preprocess` has data leakage
- **File:** `src/feature_forge/evaluation/cv.py:129-141`
- Categorical encoding uses `.cat.codes` which is not fitted — category codes may differ between train and validation splits.

### TD-5: `_AsyncBridge` singleton still global
- **File:** `src/feature_forge/utils.py`
- The `_AsyncBridge` instance is now encapsulated in a class, but the `_bridge` module-level singleton and `run_coro_sync()` global function remain. Tests must still share the process-global event loop.

### TD-6: `pyproject.toml` URLs point to placeholder org
- **File:** `pyproject.toml:94-97`
- `Homepage = "https://github.com/your-org/feature-forge"` — placeholder URLs.

### PERF-3: Large HTML files in git history
- **Files:** `docs/notebooks/html/*.html` — each file is 200KB-2MB of base64-encoded images/fonts.
- **Recommendation:** Store rendered HTML in a separate artifact store or generate in CI.

### SEC-4: Notebook API keys passed as empty string fallback
- **Files:** `notebooks/*.py`
- Pattern: `api_key=os.environ.get("FF_LLM__API_KEY", "")` — falls back to empty string instead of `None`.

### TEST-1: `conftest.py` fixtures still growing
- Basic fixtures now exist (`FakeLLM`, `sample_config`, `sample_dataframe`, `sample_series`), but still no mock Settings or mock pipeline fixtures.

### TEST-2: No tests for `pipeline/ablations.py`
- The ablation pipeline variants have no dedicated test file.

### TEST-3: Test files still define their own `FakeLLM` classes
- Several test files (e.g., `test_evaluation.py`, `test_artifact_exporter.py`) define inline `FakeLLM` classes instead of importing from `conftest`.

---

## Low

### STYLE-3: Inconsistent type annotations in ablations
- **File:** `src/feature_forge/methods/malmas/pipeline/ablations.py:65-68`
- `SingleAgentPipeline._select_agents` uses `*args, **kwargs` without type hints.

### TD-8: `wandb/` and `htmlcov/` artifacts in working tree
- Runtime artifacts present on disk (gitignored but clutter workspace).

### TD-9: `.DS_Store` in source directory
- **File:** `src/feature_forge/.DS_Store`
- macOS metadata file. `.gitignore` pattern may be root-only.

### TD-10: No `__all__` exports in package `__init__.py`
- **File:** `src/feature_forge/__init__.py`
- Missing `__all__` means `from feature_forge import *` imports everything.

### PERF-4: `MalmusBaseline` iterative mode evaluates features one at a time
- **File:** `src/feature_forge/methods/malmus.py:190-199`

### PERF-5: `_infer_column_descriptions` recomputes stats every agent call
- **File:** `src/feature_forge/methods/malmas/agents/base.py:139-159`
- Column statistics recomputed 6+ times per round on same data.
- **Recommendation:** Cache column descriptions per pipeline run.

---

## Resolved

- ~~ARCH-1: `api.py` named `MALMASFeatureEngineer` but project is `feature_forge`~~ → Renamed to `FeatureForge`
- ~~TD-1: Massively duplicated boilerplate in LLM providers~~ → Extracted to template method in `LLMClient` base
- ~~TD-4: `MemoryError` shadows Python builtin~~ → Renamed to `AgentMemoryError`
- ~~TD-7: `_version.py` exists but unused~~ → Removed, version from `importlib.metadata`
- ~~STYLE-1: Misplaced docstring in `pipeline/core.py`~~ → Moved docstring before method body
- ~~STYLE-2: `RouterAgent` imported alongside agent base classes~~ → Moved to lazy `__getattr__`
