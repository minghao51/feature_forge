# Code Simplification & Refinement — Full `src/` Pass

**Date:** May 2026
**Status:** Ready to execute
**Depends on:** `16_prompt_colocation_pydantic.md` (already executed)
**Type:** Refactor — behavior-preserving only

## Motivation

The codebase has accumulated duplicated patterns across the 5 method implementations (MALMAS, CAAFE, LLMFE, Malmus, OpenFE) and the pipeline internals. While each individual instance is correct, the aggregate creates maintenance burden:

- **Identical 5-line blocks** repeated 3× across method files
- **Near-identical async closures** duplicated within `core.py`
- **Parallel/serial branches** in feature evaluation that duplicate try/except/log logic
- **Identical property implementations** across 3 method files
- **Deeply nested parsing logic** that obscures the happy path

**Scope:** Every `.py` file under `src/feature_forge/`. Only simplification — no new features, no behavior changes.

## Guiding Principle

**Explicit over clever. Clear over compact.** Never change what code does, only how it does it.

## Tier 1 — High-Impact Deduplication

### 1.1 Extract `SandboxedExecutor.from_evaluator()` factory

**Files involved:**
- `evaluation/sandbox.py` (add method)
- `methods/caafe/method.py:57-61` (consume)
- `methods/llmfe/method.py:63-67` (consume)
- `methods/malmus/method.py:139-143` (consume)

**Problem:** Three methods repeat identical sandbox construction:

```python
# Repeated in caafe, llmfe, malmus __init__
eval_cfg = evaluator.config.evaluation if evaluator else None
self.sandbox = SandboxedExecutor(
    timeout_seconds=eval_cfg.sandbox_timeout_seconds if eval_cfg else 5.0,
    max_memory_mb=eval_cfg.sandbox_max_memory_mb if eval_cfg else 512,
)
```

**Solution:** Add a `@staticmethod` to `SandboxedExecutor`:

```python
# evaluation/sandbox.py — add to SandboxedExecutor class
@staticmethod
def from_evaluator(evaluator: CVEvaluator | None) -> SandboxedExecutor:
    """Create a SandboxedExecutor with sensible defaults from an optional evaluator."""
    if evaluator is not None:
        cfg = evaluator.config.evaluation
        return SandboxedExecutor(
            timeout_seconds=cfg.sandbox_timeout_seconds,
            max_memory_mb=cfg.sandbox_max_memory_mb,
        )
    return SandboxedExecutor(timeout_seconds=5.0, max_memory_mb=512)
```

Then consume in all three methods:

```python
# methods/caafe/method.py __init__ — replace lines 57-61
self.sandbox = SandboxedExecutor.from_evaluator(evaluator)

# methods/llmfe/method.py __init__ — replace lines 63-67
self.sandbox = SandboxedExecutor.from_evaluator(evaluator)

# methods/malmus/method.py __init__ — replace lines 139-143
self.sandbox = SandboxedExecutor.from_evaluator(evaluator)
```

**Imports to update:** Add `SandboxedExecutor` import where not already present (all three files already import it).

---

### 1.2 Consolidate parallel/serial evaluation in `CorePipeline._evaluate_and_select`

**File:** `methods/malmas/pipeline/core.py:549-661`

**Problem:** The method has two branches — parallel (`len(candidate_columns) > 1`) and serial (`else`) — that duplicate the same evaluate-handle-log-assign logic:

```python
# Parallel branch (lines 584-601)
eval_results = Parallel(n_jobs=n_jobs, backend=...)(...)
for col, result in zip(candidate_columns, eval_results, strict=True):
    if isinstance(result, Exception):
        logger.warning("feature_evaluation_failed", feature=col, error=str(result))
        if self.config.evaluation.fail_on_feature_error:
            raise PipelineError(...) from result
        gains[col] = float("-inf")
    else:
        gains[col] = result
        logger.debug("feature_evaluated", ...)

# Serial branch (lines 609-624) — SAME logic
for col in candidate_columns:
    try:
        gain = self.evaluator.evaluate_feature(...)
        gains[col] = gain
        logger.debug("feature_evaluated", ...)
    except Exception as exc:
        logger.warning("feature_evaluation_failed", ...)
        if self.config.evaluation.fail_on_feature_error:
            raise PipelineError(...) from exc
        gains[col] = float("-inf")
```

**Solution:** Use `Parallel(n_jobs=1)` for both cases, eliminating the serial branch entirely. `joblib.Parallel` with `n_jobs=1` runs sequentially with negligible overhead.

Replace lines 571-624 with:

```python
n_jobs = min(os.cpu_count() or 4, 8) if len(candidate_columns) > 1 else 1
if (
    len(candidate_columns) > 1
    and self.config.evaluation.feature_eval_backend == "loky"
    and X_train.shape[0] * X_train.shape[1] > 2_000_000
):
    logger.warning(
        "feature_eval_backend_loky_large_matrix",
        rows=X_train.shape[0],
        cols=X_train.shape[1],
        candidates=len(candidate_columns),
        hint="Consider evaluation.feature_eval_backend='threading' to reduce IPC overhead",
    )
eval_results = Parallel(
    n_jobs=n_jobs, backend=self.config.evaluation.feature_eval_backend
)(
    delayed(self._eval_single_feature)(
        self.evaluator, X_train, y_train, features_train[[col]], col, baseline_score
    )
    for col in candidate_columns
)
for col, result in zip(candidate_columns, eval_results, strict=True):
    if isinstance(result, Exception):
        logger.warning("feature_evaluation_failed", feature=col, error=str(result))
        if self.config.evaluation.fail_on_feature_error:
            raise PipelineError(
                f"Feature evaluation failed for '{col}': {result}"
            ) from result
        gains[col] = float("-inf")
    else:
        gains[col] = result
        logger.debug(
            "feature_evaluated",
            feature=col,
            gain=round(result, 6),
            effective=result > 0,
        )
```

---

### 1.3 Extract shared sandbox execution helper in `CorePipeline`

**File:** `methods/malmas/pipeline/core.py`

**Problem:** `_execute_train` (lines 406-439) and `_execute_test` (lines 486-521) define near-identical async closures:

```python
# _exec_for_agent (train, line 406-439)
async def _exec_for_agent(agent_name: str, code: str) -> tuple[str, pd.DataFrame] | None:
    sandbox_t0 = time.perf_counter()
    try:
        part = await asyncio.wait_for(
            asyncio.to_thread(self.sandbox.execute, code, X_train, source="malmas_core_train", agent_name=agent_name),
            timeout=max(sandbox_timeout * 2, 30.0),
        )
        logger.info("agent_sandbox_complete", agent=agent_name, result_shape=part.shape, ...)
        return (agent_name, part)
    except TimeoutError:
        logger.warning("agent_sandbox_timeout", ...)
        return None
    except Exception as exc:
        logger.warning("agent_code_execution_failed", ...)
        return None

# _exec_for_agent_test (test, line 486-521) — IDENTICAL except:
#   - X_train → X_test
#   - source="malmas_core_test"
#   - log suffix "_test"
```

**Solution:** Add a private method to `CorePipeline`:

```python
async def _exec_sandbox(
    self,
    agent_name: str,
    code: str,
    X: pd.DataFrame,
    source: str,
) -> tuple[str, pd.DataFrame] | None:
    """Execute code in sandbox with timeout and error handling."""
    sandbox_timeout = self.config.evaluation.sandbox_timeout_seconds
    sandbox_t0 = time.perf_counter()
    try:
        part = await asyncio.wait_for(
            asyncio.to_thread(
                self.sandbox.execute, code, X,
                source=source, agent_name=agent_name,
            ),
            timeout=max(sandbox_timeout * 2, 30.0),
        )
        logger.info(
            f"agent_sandbox_complete{'_test' if 'test' in source else ''}",
            agent=agent_name,
            result_shape=part.shape,
            latency_ms=round((time.perf_counter() - sandbox_t0) * 1000, 1),
        )
        return (agent_name, part)
    except TimeoutError:
        logger.warning(
            f"agent_sandbox_timeout{'_test' if 'test' in source else ''}",
            agent=agent_name,
            timeout=sandbox_timeout * 2,
        )
        return None
    except Exception as exc:
        logger.warning(
            f"agent_code_execution_failed{'_test' if 'test' in source else ''}",
            agent=agent_name,
            error=str(exc)[:200],
        )
        return None
```

Then refactor `_execute_train` and `_execute_test` to call it:

```python
# In _execute_train, replace _exec_for_agent closure (lines 406-439):
exec_results = await asyncio.gather(
    *[self._exec_sandbox(name, code, X_train, "malmas_core_train") for name, code in all_code_parts]
)

# In _execute_test, replace _exec_for_agent_test closure (lines 486-521):
test_exec_results = await asyncio.gather(
    *[self._exec_sandbox(name, code, X_test, "malmas_core_test") for name, code in all_code_parts],
    return_exceptions=True,
)
```

Also extract the duplicate "concat + dedup columns" pattern used in both methods into a small helper:

```python
@staticmethod
def _concat_dedup(parts: list[pd.DataFrame], index: pd.Index | None = None) -> pd.DataFrame:
    """Concatenate DataFrames column-wise and drop duplicates."""
    if not parts:
        return pd.DataFrame(index=index) if index is not None else pd.DataFrame()
    combined = pd.concat(parts, axis=1) if len(parts) > 1 else parts[0]
    dup_cols = combined.columns[combined.columns.duplicated()].tolist()
    if dup_cols:
        logger.warning("column_dedup", duplicated_columns=dup_cols)
    return combined.loc[:, ~combined.columns.duplicated()]
```

---

### 1.4 Simplify `_parse_response` in `BaseFeatureAgent`

**File:** `methods/malmas/agents/base.py:243-295`

**Problem:** Deeply nested if/else for JSON extraction — 3 levels of nesting in the string-parsing branch.

**Solution:** Split into two focused helpers with early returns:

```python
def _parse_response(self, content: str | dict[str, Any] | list[Any]) -> list[FeatureSpec]:
    if isinstance(content, (dict, list)):
        data = content
    else:
        data = self._extract_json_from_string(content)

    if isinstance(data, dict):
        data = data.get("features", data)
    if not isinstance(data, list):
        raise AgentError(f"{self.name} expected JSON list/object, got {type(data).__name__}")

    return self._build_specs_from_list(data)

@staticmethod
def _extract_json_from_string(content: str) -> dict[str, Any] | list[Any]:
    """Parse JSON from a raw string, handling markdown fences."""
    content = content.strip()
    json_str = content
    if content.startswith("```"):
        json_str = re.sub(r"^```(?:json)?\s*", "", content)
        json_str = re.sub(r"\s*```$", "", json_str).strip()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    bracket_match = re.search(r"\[.*\]", content, re.DOTALL)
    if bracket_match:
        try:
            return json.loads(bracket_match.group())
        except json.JSONDecodeError as exc:
            raise AgentError(f"Invalid JSON: {exc}") from exc

    raise AgentError("Could not extract JSON from response")

@staticmethod
def _build_specs_from_list(data: list[Any]) -> list[FeatureSpec]:
    """Build FeatureSpec list from parsed JSON data."""
    specs: list[FeatureSpec] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        base_cols = item.get("base_columns", [])
        if isinstance(base_cols, str):
            base_cols = [base_cols]
        for feat in item.get("derived_features", []):
            if not isinstance(feat, dict):
                continue
            specs.append(FeatureSpec(
                name=feat.get("name", "unknown"),
                type=feat.get("type", "numerical"),
                transform=feat.get("transform", ""),
                logic=feat.get("logic", ""),
                base_columns=base_cols,
                agent_name="",  # caller sets this
            ))
    return specs
```

**Note:** `_parse_response` currently sets `agent_name=self.name` inside the loop. After this refactor, the caller in `generate()` must set it:

```python
# In generate() method, after parsing:
specs = self._parse_response(response)
for spec in specs:
    spec.agent_name = self.name
```

---

### 1.5 Extract shared iterative helpers to `BaseMethod`

**Files involved:**
- `methods/base.py` (add helpers)
- `methods/caafe/method.py` (consume)
- `methods/llmfe/method.py` (consume)
- `methods/malmus/method.py` (consume)

**Problem:** Three method files have nearly identical `feature_metadata` and `provenance_records` properties that iterate `self._artifacts.get("iterations")` and extract gains. Only the method name string and minor keys differ.

**Solution:** Add helpers to `BaseMethod`:

```python
# methods/base.py — add to BaseMethod class

def _iterative_feature_metadata(self, method_name: str) -> list[dict[str, Any]]:
    """Build feature_metadata from iteration artifacts."""
    iterations = self._artifacts.get("iterations")
    if not iterations:
        return []
    meta: list[dict[str, Any]] = []
    for it in iterations:
        for col, gain in it.get("gains", {}).items():
            meta.append({
                "name": col,
                "method": method_name,
                "iteration": it.get("iteration"),
                "gain": gain,
                "kept": it.get("kept", False),
                "code": it.get("generated_code", ""),
            })
    return meta

def _iterative_provenance_records(self, method_name: str) -> list[dict[str, Any]]:
    """Build provenance_records from iteration artifacts."""
    iterations = self._artifacts.get("iterations")
    if not iterations:
        return []
    records: list[dict[str, Any]] = []
    for it in iterations:
        for col, gain in it.get("gains", {}).items():
            records.append({
                "feature_name": col,
                "source_method": method_name,
                "iteration_index": it.get("iteration"),
                "generated_code": it.get("generated_code", ""),
                "cv_gain": gain,
            })
    return records
```

Then simplify in consuming methods:

```python
# caafe/method.py — replace feature_metadata property
@property
def feature_metadata(self) -> list[dict[str, Any]]:
    meta = self._iterative_feature_metadata("caafe")
    if meta:
        return meta
    code = self._artifacts.get("generated_code", "")
    if code:
        return [{"name": "fidelity", "method": "caafe", "code": code}]
    return []

# caafe/method.py — replace provenance_records property
@property
def provenance_records(self) -> list[dict[str, Any]]:
    return self._iterative_provenance_records("caafe")

# llmfe/method.py — replace feature_metadata property
@property
def feature_metadata(self) -> list[dict[str, Any]]:
    meta = self._iterative_feature_metadata("llmfe")
    if meta:
        return meta
    code = self._artifacts.get("generated_code", "")
    if code:
        return [{"name": "single_shot", "method": "llmfe", "code": code}]
    return []

# llmfe/method.py — replace provenance_records property
@property
def provenance_records(self) -> list[dict[str, Any]]:
    return self._iterative_provenance_records("llmfe")

# malmus/method.py — feature_metadata is more complex (has matching defs),
# so it stays custom but can use _iterative_provenance_records:
@property
def provenance_records(self) -> list[dict[str, Any]]:
    base = self._iterative_provenance_records("malmus")
    if not base:
        return []
    iterations = self._artifacts.get("iterations", [])
    for record, it in zip(
        base,
        [it for it in iterations for _ in it.get("gains", {})],
    ):
        defs = it.get("feature_definitions", [])
        matching = next((d for d in defs if d.get("name") == record["feature_name"]), {})
        record["description"] = matching.get("description", "")
        record["libraries"] = matching.get("libraries", [])
    return base
```

**Note on malmus:** Its `feature_metadata` includes per-feature `description` and `libraries` from `FeatureDefinition`, making it more complex than the other two. It should stay custom but can be simplified to call `_iterative_feature_metadata("malmus")` as a base and then augment each entry with matching definition data. Use judgment — if the augmentation makes it less clear, keep it custom.

---

### 1.6 Extract shared transform helper to `BaseMethod`

**Files involved:**
- `methods/base.py` (add helper)
- `methods/caafe/method.py:205-220` (consume)
- `methods/llmfe/method.py:165-180` (consume)

**Problem:** CAAFE and LLMFE have nearly identical `transform()` methods that iterate `self._iteration_codes`, execute each in sandbox, and collect new columns.

**Solution:** Add to `BaseMethod`:

```python
# methods/base.py — add to BaseMethod class

def _transform_via_iteration_codes(self, X: pd.DataFrame) -> pd.DataFrame:
    """Transform by executing iteration codes sequentially in sandbox."""
    if not hasattr(self, "_iteration_codes") or not self._iteration_codes:
        raise RuntimeError(f"{self.name} not fitted yet")
    result = X.copy()
    for code in self._iteration_codes:
        try:
            features = self.sandbox.execute(code, result)
            for col in features.columns:
                if col not in result.columns:
                    result[col] = features[col].values
        except Exception as exc:
            logger.warning(f"{self.name}_transform_step_failed", error=str(exc))
            if (
                hasattr(self, "evaluator")
                and self.evaluator is not None
                and self.evaluator.config.evaluation.fail_on_feature_error
            ):
                raise
    new_cols = [c for c in result.columns if c not in X.columns]
    return result[new_cols]
```

Then simplify:

```python
# caafe/method.py — replace transform()
def transform(self, X: pd.DataFrame) -> pd.DataFrame:
    if self.variant == "fidelity":
        return self._transform_fidelity(X)
    return self._transform_via_iteration_codes(X)

# llmfe/method.py — replace transform()
def transform(self, X: pd.DataFrame) -> pd.DataFrame:
    return self._transform_via_iteration_codes(X)
```

---

## Tier 2 — Medium-Impact Clarity Improvements

### 2.1 Simplify router `_data_driven_selection`

**File:** `methods/malmas/agents/router.py:116-147`

**Problem:** Sequential `if X in excluded_if and not Y` checks — 5 separate if-blocks.

**Solution:** Use a conditions map:

```python
_EXCLUSION_CONDITIONS: ClassVar[dict[str, Callable[[dict[str, Any]], bool]]] = {
    "no_datetime_columns": lambda chars: not chars["datetime_columns"],
    "single_column_dataset": lambda chars: chars.get("single_column_dataset", False),
    "no_numerical_columns": lambda chars: not chars["numerical_columns"],
    "no_categorical_for_grouping": lambda chars: len(chars["categorical_columns"]) < 1,
}

def _data_driven_selection(self) -> list[str]:
    """Select agents based on dataset characteristics."""
    if self.dataset_characteristics is None:
        return self.agent_names[: self.max_agents]

    selected: list[str] = []
    chars = self.dataset_characteristics
    for agent_name in self.agent_names:
        capabilities = self.AGENT_CAPABILITIES.get(agent_name, {})
        excluded_if = capabilities.get("excluded_if", [])
        if capabilities.get("requires_enrich") and not chars["has_enrich_description"]:
            continue
        if any(
            cond_name in excluded_if and self._EXCLUSION_CONDITIONS.get(cond_name, lambda _: False)(chars)
            for cond_name in self._EXCLUSION_CONDITIONS
        ):
            continue
        selected.append(agent_name)

    if len(selected) < self.min_agents:
        for agent in self.agent_names:
            if agent not in selected and len(selected) < self.min_agents:
                selected.append(agent)
    return selected[: self.max_agents]
```

---

### 2.2 Clean up `update_performance` repeated lookups

**File:** `methods/malmas/agents/router.py:291-306`

**Problem:** `self.agent_performance.get(agent_name, [])` called 4× in 6 lines.

**Solution:**

```python
def update_performance(self, agent_name: str, gain: float) -> None:
    if agent_name in self.agent_performance:
        self.agent_performance[agent_name].append(gain)
        self.agent_performance[agent_name] = self.agent_performance[agent_name][-10:]
    gains = self.agent_performance.get(agent_name, [])
    avg_gain = sum(gains) / len(gains) if gains else 0.0
    logger.debug(
        "router_performance_update",
        agent=agent_name,
        gain=round(gain, 6),
        avg_gain=round(avg_gain, 6),
    )
```

---

### 2.3 Move inline JSON schema string to constant

**File:** `methods/malmas/agents/base.py:218-222`

**Problem:** Multi-line JSON string literal embedded in method call:

```python
schema_description=(
    '{"features":[{"base_columns":["col_a"],'
    '"derived_features":[{"name":"f","type":"numerical",'
    '"transform":"op","logic":"..."}]}]}'
),
```

**Solution:** Module-level constant:

```python
_FEATURE_JSON_SCHEMA_DESC = json.dumps({
    "features": [{
        "base_columns": ["col_a"],
        "derived_features": [{
            "name": "f",
            "type": "numerical",
            "transform": "op",
            "logic": "...",
        }],
    }]
})
```

Then use `schema_description=_FEATURE_JSON_SCHEMA_DESC` in the `complete_json` call at line 218.

---

### 2.4 Simplify ablation no-op

**File:** `methods/malmas/pipeline/ablations.py:56-57`

**Problem:** `del agents, core_results, round_idx; return None` is overcomplicated for a no-op override.

**Solution:**

```python
class NoMemoryStaticRouterPipeline(NoMemoryPipeline):
    """No-memory ablation with static router behavior."""

    async def _post_round(
        self,
        agents: list[Agent],
        core_results: dict[str, Any],
        round_idx: int,
    ) -> None:
        ...
```

---

### 2.5 Extract memory recording helper in `IterativePipeline._post_round`

**File:** `methods/malmas/pipeline/iterative.py:302-343`

**Problem:** Deep nesting in `_post_round` — 4 levels of indentation.

**Solution:** Extract a helper:

```python
@staticmethod
def _record_feature_in_memory(
    memory: AgentMemory,
    spec: FeatureSpec,
    gain: float,
    round_idx: int,
    metric: str,
) -> None:
    """Record a single feature's results into agent memory."""
    memory.record_procedure(
        base_columns=spec.base_columns,
        transform=spec.transform,
        feature_name=spec.name,
        ty=spec.type,
        description=spec.logic,
        round_idx=round_idx,
    )
    effective = gain > 0
    memory.record_feedback(
        feature_name=spec.name,
        metric=metric,
        value=gain,
        effective=effective,
        round_idx=round_idx,
        base=spec.base_columns,
        ty=spec.type,
    )
    if not effective:
        memory.record_unused_procedure(
            base_columns=spec.base_columns,
            transform=spec.transform,
            feature_name=spec.name,
            ty=spec.type,
            description=spec.logic,
            round_idx=round_idx,
        )
```

Then simplify `_post_round`:

```python
async def _post_round(
    self,
    agents: list[Agent],
    core_results: dict[str, Any],
    round_idx: int,
) -> None:
    for agent in agents:
        memory = self._get_memory(agent.name)
        agent_gain_df = core_results["agent_gains"].get(agent.name, pd.DataFrame())
        for _, row in agent_gain_df.iterrows():
            spec = next((s for s in core_results["specs"] if s.name == row["feature"]), None)
            if spec:
                self._record_feature_in_memory(
                    memory, spec, row["gain"], round_idx, self.config.metric,
                )
        memory.save()
        if not agent_gain_df.empty:
            self.router.update_performance(agent.name, agent_gain_df["gain"].mean())
```

---

### 2.6 Extract shared entry-point discovery

**Files involved:**
- `evaluation/metrics.py:107-134` (`MetricRegistry.discover`)
- `evaluation/model_factory.py:116-143` (`ModelRegistry.discover`)
- Optionally: `methods/base.py:94-137` (`MethodRegistry.discover`)

**Problem:** All three registries have identical entry-point discovery logic: load EP → warn on fail → warn on dup → collect.

**Solution:** Add a utility function (either in a new `registry_utils.py` or inline in a shared location):

```python
# evaluation/registry_utils.py (new file)

import importlib.metadata
import warnings
from collections.abc import Callable
from typing import Any


def discover_entry_points(
    group: str,
    builtins: dict[str, Any] | None = None,
) -> dict[str, Callable[..., Any]]:
    """Discover entry points with standardized warning behavior."""
    discovered: dict[str, Callable[..., Any]] = {}
    for ep in importlib.metadata.entry_points(group=group):
        try:
            loaded = ep.load()
        except Exception as exc:
            warnings.warn(
                f"Failed to load entry point '{ep.name}': {exc}",
                RuntimeWarning,
                stacklevel=3,
            )
            continue
        if builtins and ep.name in builtins and builtins[ep.name] is not loaded:
            warnings.warn(
                f"Entry point '{ep.name}' overrides built-in.",
                RuntimeWarning,
                stacklevel=3,
            )
        if ep.name in discovered:
            warnings.warn(
                f"Duplicate entry point name '{ep.name}'. Last registered wins.",
                RuntimeWarning,
                stacklevel=3,
            )
        discovered[ep.name] = loaded
    return discovered
```

Then simplify both registries:

```python
# evaluation/metrics.py — MetricRegistry.discover
from feature_forge.evaluation.registry_utils import discover_entry_points

@classmethod
def discover(cls) -> dict[str, Callable[..., Any]]:
    return discover_entry_points(cls.ENTRY_POINT_GROUP, builtins=cls._builtin)

# evaluation/model_factory.py — ModelRegistry.discover
from feature_forge.evaluation.registry_utils import discover_entry_points

@classmethod
def discover(cls) -> dict[str, Callable[..., Any]]:
    return discover_entry_points(cls.ENTRY_POINT_GROUP, builtins=cls.get_builtin())
```

**Note on MethodRegistry:** Its discover has additional checks (`isinstance(loaded, type)` and `required.issubset(dir(loaded))`). Those are protocol-specific and should stay. The utility helps `MetricRegistry` and `ModelRegistry` but NOT `MethodRegistry`.

---

## Execution Order

Execute in this order to minimize merge conflicts and allow incremental testing:

1. **`evaluation/sandbox.py`** — add `from_evaluator()` (#1.1)
2. **`methods/caafe/method.py`** — use factory (#1.1) + helpers (#1.5, #1.6)
3. **`methods/llmfe/method.py`** — use factory (#1.1) + helpers (#1.5, #1.6)
4. **`methods/malmus/method.py`** — use factory (#1.1) + provenance helper (#1.5)
5. **`methods/base.py`** — add iterative helpers (#1.5, #1.6, `_transform_via_iteration_codes`)
6. **`methods/malmas/pipeline/core.py`** — consolidate eval (#1.2) + extract sandbox exec (#1.3)
7. **`methods/malmas/agents/base.py`** — simplify parsing (#1.4) + constant (#2.3)
8. **`methods/malmas/agents/router.py`** — simplify selection (#2.1) + performance (#2.2)
9. **`methods/malmas/pipeline/ablations.py`** — no-op (#2.4)
10. **`methods/malmas/pipeline/iterative.py`** — extract memory recording (#2.5)
11. **`evaluation/registry_utils.py`** (new) — shared EP discovery (#2.6)
12. **`evaluation/metrics.py` + `evaluation/model_factory.py`** — consume shared utility (#2.6)

## Verification

After all changes:

```bash
uv run ruff check src/
uv run mypy src/
uv run pytest tests/ -x
```

- All existing tests must pass without modification
- No new warnings from ruff
- No new type errors from mypy
- `git diff --stat` should show only modified files, no new test files needed

## Files NOT Changed (already clean)

These files were reviewed and found to be already clear and consistent:

- `config.py` — well-structured Pydantic settings hierarchy
- `utils.py` — focused utilities with clean async bridge
- `exceptions.py` — clean exception hierarchy
- `types.py` — concise type aliases
- `_prompting.py` — focused prompt infrastructure
- `observability/structlog_config.py` — clean logging config
- `observability/langfuse_tracer.py` — clean tracing decorators
- `llm/base.py` — clean ABC with hook pattern
- `llm/factory.py` — clean provider factory
- `llm/cache.py` — clean disk cache
- `llm/retry.py` — clean tenacity config
- `llm/providers/*.py` — clean provider implementations
- `evaluation/cv.py` — clean CV evaluator
- `evaluation/sandbox.py` — clean sandboxed executor (only adding factory)
- `experiment/runner.py` — clean experiment runner
- `experiment/matrix.py` — clean Cartesian product builder
- `experiment/reporter.py` — clean reporter
- `experiment/tracker.py` — clean ABC
- `experiment/wandb_backend.py` — clean WandB backend
- `experiment/mlflow_backend.py` — clean MLflow backend
- `artifacts/base.py` — clean artifact exporter
- `artifacts/storage.py` — clean storage management
- `artifacts/schema.py` — clean Pydantic schemas
- `artifacts/diff.py` — clean diff computation
- `artifacts/comparison.py` — clean method comparison
- `artifacts/dashboard.py` — clean HTML dashboard
- `data/registry.py` — clean dataset registry
- `data/ingestion.py` — clean data ingestion
- `platform.py` — clean platform facade
- `api.py` — clean sklearn-compatible API
- Individual agent files (`unary.py`, `aggregation.py`, etc.) — minimal, clean
- `malmas/method.py` — clean adapter
- `malmas/types.py` — clean type alias
- `malmas/memory/base.py` — clean memory system
- `malmas/memory/conceptual.py` — clean conceptual memory
- `malmas/memory/persistence.py` — clean persistence
- `malmas/memory/prompts.py` — clean prompt params
- `openfe/method.py` — clean OpenFE wrapper

## Summary Stats

| Metric | Value |
|--------|-------|
| Files to modify | 12 |
| New files | 1 (`registry_utils.py`) |
| Duplicated lines eliminated | ~150 |
| Max nesting reduction | 3→1 levels (in `_parse_response`) |
| Methods extracted | 7 new helpers |
| Behavior changes | 0 |
