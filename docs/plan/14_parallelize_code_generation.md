# Plan: Parallelize Per-Agent Code Generation

**Date:** May 2026
**Status:** Draft
**Prerequisite:** `docs/plan/12_code_generation_improvements.md` (Phases 0-4 implemented)

## Problem

Per-agent code generation (introduced in Phase 3) iterates agent groups **sequentially**:

```python
# pipeline/core.py:298-315 — current sequential approach
all_code_parts: list[tuple[str, str]] = []
for agent_name, agent_specs in specs_by_agent.items():
    last_error: str | None = None
    for _ in range(2):
        try:
            code = await self.code_generator.generate_code(
                agent_specs, schema=schema, error_feedback=last_error
            )
        except Exception as exc:
            last_error = str(exc)
            continue
        all_code_parts.append((agent_name, code))
        break
```

With 6 agent groups × ~2-10 seconds per LLM call × up to 2 retries, wall-clock time is **12-60+ seconds** for code generation alone. This is a **sequential bottleneck** — the LLM calls have zero dependency on each other.

## Proposed Solution

Parallelize both **code generation** AND **sandbox execution** using `asyncio.gather` with the existing `asyncio.Semaphore` (already wired from `config.llm.max_concurrent_calls`).

### Data Flow (after)

```
specs_by_agent  (6 groups)
  │
  ├─▶ _gen_code_for_agent("unary")           ─┐
  ├─▶ _gen_code_for_agent("cross_compositional")  │  asyncio.gather
  ├─▶ _gen_code_for_agent("aggregation")    │  (semaphore-bounded)
  ├─▶ _gen_code_for_agent("temporal")       │
  ├─▶ _gen_code_for_agent("local_transform")    │
  └─▶ _gen_code_for_agent("local_pattern")  ─┘
  │
  ▼
all_code_parts  (list of (agent_name, code) tuples)
  │
  ├─▶ _exec_code_for_agent("unary", code)   ─┐
  ├─▶ ...                                        │  asyncio.gather
  └─▶ _exec_code_for_agent("local_pattern") ─┘
  │
  ▼
features_train_parts → pd.concat(axis=1) → dedup
```

## Changes

### File: `src/feature_forge/pipeline/core.py`

#### 1. Parallelize code generation (replace sequential for-loop)

```python
semaphore = asyncio.Semaphore(self.config.llm.max_concurrent_calls)

async def _gen_for_agent(agent_name: str, agent_specs: list[FeatureSpec]) -> tuple[str, str] | None:
    """Generate code for one agent group with retry. Returns (agent_name, code) or None on failure."""
    last_error: str | None = None
    for attempt in range(2):
        try:
            async with semaphore:
                code = await self.code_generator.generate_code(
                    agent_specs, schema=schema, error_feedback=last_error
                )
            return (agent_name, code)
        except Exception as exc:
            last_error = str(exc)
            logger.warning("agent_code_gen_failed", agent=agent_name, attempt=attempt, error=last_error[:200])
    return None

code_gen_tasks = [
    _gen_for_agent(name, specs)
    for name, specs in specs_by_agent.items()
]
code_gen_results = await asyncio.gather(*code_gen_tasks)

all_code_parts: list[tuple[str, str]] = [
    r for r in code_gen_results if r is not None
]
```

#### 2. Parallelize sandbox execution (replace sequential for-loop)

```python
async def _exec_for_agent(agent_name: str, code: str) -> pd.DataFrame | None:
    """Execute generated code in sandbox for one agent."""
    try:
        return self.sandbox.execute(code, X_train)
    except Exception as exc:
        logger.warning("agent_code_execution_failed", agent=agent_name, error=str(exc)[:200])
        return None

exec_tasks = [
    _exec_for_agent(name, code)
    for name, code in all_code_parts
]
exec_results = await asyncio.gather(*exec_tasks)

features_train_parts: list[pd.DataFrame] = []
combined_code_parts: list[str] = []
for (name, code), result in zip(all_code_parts, exec_results):
    if result is not None:
        features_train_parts.append(result)
        combined_code_parts.append(code)
        logger.info("agent_sandbox_complete", agent=name, result_shape=result.shape)

code = "\n\n".join(combined_code_parts)
```

**Note:** `SandboxedExecutor.execute()` is currently a synchronous subprocess call wrapped in `run_in_executor` internally. Moving it inside an async function is safe — it won't block the event loop.

#### 3. Apply same parallelization to X_test execution

```python
if X_test is not None:
    test_exec_tasks = [
        _exec_for_agent_test(name, code, X_test)
        for name, code in all_code_parts
    ]
    test_exec_results = await asyncio.gather(*test_exec_tasks)
    features_test_parts = [r for r in test_exec_results if r is not None]
    # ... concat + dedup as before
```

### Helper: Extract `_exec_for_agent_test` optional variant

Either generalize `_exec_for_agent` to accept a DataFrame parameter, or create a separate inner function for test execution.

### Files Touched

| File | Change |
|---|---|
| `src/feature_forge/pipeline/core.py` | Replace 2 sequential for-loops with `asyncio.gather` + semaphore |

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Semaphore scope** | Reuse existing `asyncio.Semaphore(self.config.llm.max_concurrent_calls)` | Already wired at top of `run()`, used for agent spec generation. Single semaphore shared across agents AND code gen ensures total LLM concurrency is capped. |
| **Retry handling** | Inline in each parallel task | Each agent's code gen can fail independently; retries happen inside the per-agent task, not at the gather level. |
| **Partial failure** | Return `None` from failed tasks, filter after gather | Same pattern as agent spec generation (`return_exceptions=True`). Using `None` sentinels instead of exceptions is cleaner for downstream filtering. |
| **Test set parallelization** | Same pattern | Symmetric to train; no reason to sequentialize. |

## Tradeoffs

| Pro | Con |
|---|---|
| Wall-clock time for 6 agents: ~2-10s instead of 12-60s | More concurrent LLM calls → higher rate limit risk |
| Better resource utilization | Log output interleaved (structlog handles this) |
| Symmetric with existing agent spec parallelization | Slightly more complex error handling |

## Verification

- **Existing tests pass** — all `test_pipeline_core.py` and `test_pipeline.py` tests must pass unchanged.
- **Wall-clock speedup** — a notebook cell timing `CorePipeline.run()` should show code generation latency drop from sum-of-agent-latencies to max-of-agent-latencies.
- **Partial failure** — a test with 1 failing agent code and 5 succeeding agents should result in 5 feature sets, not 0.
- **Semaphore respect** — if `max_concurrent_calls=1`, code gen calls should be strictly sequential despite being launched concurrently.
