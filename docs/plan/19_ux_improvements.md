# UX Improvements Plan

**Version:** 0.1.0
**Date:** 2026-05-17

---

## 1. Executive Summary

Feature Forge is currently a **library-only Python package** — users must write scripts, handle async/await, configure YAML/env vars, and parse markdown reports. This plan identifies 9 UX gaps and proposes solutions to make the project accessible to data scientists of all skill levels.

---

## 2. Current UX Assessment

### 2.1 User Personas

| Persona | Skill Level | Current Pain | Need |
|---------|------------|--------------|------|
| **Kaggle competitor** | Intermediate Python | Wants 1-2 lines to get better features | Zero-setup CLI, instant results |
| **Data scientist** | Proficient | Doesn't want to learn async or pydantic | Simple sklearn API, clear output |
| **ML researcher** | Expert | Needs to compare methods systematically | Experiment matrix, programmatic access |
| **Non-programmer** | Minimal | Wants to upload data and get features | Web UI (future) |

### 2.2 Current Gaps

| # | Gap | User Impact |
|---|-----|-------------|
| G1 | **No CLI** — must write Python scripts | Cannot quickly run from terminal; debugging requires manual script editing |
| G2 | **Config complexity** — YAML + env vars + constructor args | Unclear where to set what; users discover errors at runtime |
| G3 | **Async surface exposed** — `async_fit()`, `run_coro_sync` | Confusing for sklearn users; unexplained abstractions |
| G4 | **No progress transparency** — feature generation is a black box | Users wait without knowing what's happening or estimated time remaining |
| G5 | **Results are markdown tables** — no visualizations | Hard to compare methods at a glance; no charts, no ranking |
| G6 | **Memory is invisible** — JSON files in `memory_files/` | Cannot inspect, debug, or understand what memory recorded |
| G7 | **Error messages are opaque** — 22 broad `except Exception` blocks | Failed features vanish silently; no actionable feedback |
| G8 | **No exploration mode** — must run full pipeline to see suggestions | Users want "suggest some features for this dataset" without committing |
| G9 | **Dataset setup hassle** — must configure Kaggle/local datasets | Trying out on your own CSV requires registry ceremony |
| G10 | **API key friction** — multiple keys, complicated setup | "I just want to try it" → blocked by configuration |

---

## 3. Proposed Improvements

### 3.1 Improvement Matrix

| # | Improvement | Gap(s) | Impact | Effort | Risk | Phase |
|---|------------|--------|--------|--------|------|-------|
| U1 | **CLI tool** (`forge`) | G1,G4 | High | Med | Low | 1 |
| U2 | **Config wizard** (`forge init`) | G2,G10 | High | Low | Low | 1 |
| U3 | **Hide async from users** | G3 | Med | Low | Low | 1 |
| U4 | **Real-time progress + ETA** | G4,G7 | High | Med | Low | 1 |
| U5 | **Result dashboard (HTML)** | G5 | High | Med | Med | 2 |
| U6 | **Memory inspector** (`forge memory`) | G6 | Med | Low | Low | 1 |
| U7 | **Feature explorer** (`forge suggest`) | G8 | High | Med | Med | 2 |
| U8 | **Quick CSV mode** (`forge run data.csv`) | G9 | High | Low | Low | 1 |
| U9 | **Better error UX** | G7 | High | Low | Low | 1 |

---

## 4. Phase 1: Quick Wins (U1-U4, U6, U8, U9)

**Timeline:** 1-2 weeks
**Goal:** Users can run experiments from the terminal with zero configuration

### 4.1 U1: CLI Tool (`forge`)

**Concept:** Single `forge` CLI command for the most common workflows. Built with `typer` or `click`.

```bash
# Quick run on a CSV with sensible defaults
forge run titanic.csv

# Compare methods
forge run titanic.csv --methods malmus,caafe,openfe

# List available datasets
forge datasets list

# Inspect memory
forge memory --agent unary

# Get feature suggestions without running
forge suggest titanic.csv --agent cross_compositional

# Init a config wizards
forge init
```

**Implementation:**

```python
# New file: src/feature_forge/cli.py (or src/feature_forge/cli/main.py)

import typer

app = typer.Typer()

@app.command()
def run(
    data: str = typer.Argument(..., help="Path to CSV or dataset name"),
    target: str = typer.Option(None, "--target", "-t", help="Target column name"),
    methods: str = typer.Option("malmus", "--methods", "-m", help="Comma-separated methods"),
    model: str = typer.Option("xgboost", "--model", help="Model for evaluation"),
    output: str = typer.Option(None, "--output", "-o", help="Output directory for results"),
):
    """Run feature engineering on a dataset."""
    ...

@app.command()
def list_datasets():
    """List available datasets from the registry."""
    ...

@app.command()
def init(
    force: bool = typer.Option(False, "--force", help="Overwrite existing config"),
):
    """Initialize feature_forge configuration interactively."""
    ...

if __name__ == "__main__":
    app()
```

**Entry point** (add to `pyproject.toml`):
```toml
[project.scripts]
forge = "feature_forge.cli:app"
```

**New dependencies:** `typer` (lightweight, built on click)

### 4.2 U2: Config Wizard (`forge init`)

**Concept:** Interactive CLI wizard that:
1. Prompts for LLM provider (DeepSeek/OpenAI/Anthropic)
2. Prompts for API key (writes to `.env`, optionally encrypts with dotenvx)
3. Prompts for default model, temperature, max tokens
4. Prompts for task type (classification/regression)
5. Generates `config/settings.yaml` with defaults
6. Generates `.env` with API key
7. Verifies connectivity with a test LLM call

**Implementation:**

```python
# New file: src/feature_forge/cli/wizard.py

class ConfigWizard:
    def run(self) -> None:
        provider = self._ask_provider()
        api_key = self._ask_api_key(provider)
        model = self._ask_model(provider)
        task = self._ask_task()
        self._write_env(provider, api_key)
        self._write_yaml(model, task)
        self._verify(provider, model, api_key)
```

### 4.3 U3: Hide Async from Users

**Concept:** The `FeatureForge` API already has a `fit()` sync wrapper. But the user still needs to know about `run_coro_sync`. Improve the UX by:

1. Making `fit()` fully synchronous internally (already done)
2. Renaming `async_fit()` to `fit_async()` for clarity (more conventional)
3. Removing `run_coro_sync` from public imports
4. Adding a warning log when async is used without the configured event loop

**Changes:**

```python
# src/feature_forge/api.py

class FeatureForge:
    async def fit_async(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        """Async variant of fit() for notebook/service environments."""
        ...  # unchanged logic, renamed from async_fit

    def fit(self, X: pd.DataFrame, y: pd.Series) -> FeatureForge:
        """Synchronous fit(). Calls fit_async() internally."""
        self.pipeline_result = run_coro_sync(self.fit_async(X, y))
        return self
```

### 4.4 U4: Real-time Progress + ETA

**Concept:** Currently, the pipeline is silent during execution — users see nothing until results appear. Add:

1. **tqdm progress bars** for rounds and agents (partially exists, make richer)
2. **Rich live display** showing:
   - Current round / total rounds
   - Current agent / total agents
   - Features generated so far
   - Estimated time remaining (per round)
   - Best feature gain so far
3. **Structured log events** emitted so the CLI (U1) can display real-time status

**Implementation:**

```python
# New: src/feature_forge/cli/progress.py

class PipelineDisplay:
    """Rich live display for iterative pipeline progress."""

    def __init__(self) -> None:
        self.layout = rich.layout.Layout()
        self.progress_table = rich.table.Table(title="Feature Generation Progress")
        self._build_table()

    def update(self, round_idx: int, agent: str, num_features: int, best_gain: float) -> None:
        ...

    def finish(self) -> str:
        """Return final status summary."""
        ...
```

**Changes to `iterative.py`:**
```python
# pipeline/iterative.py
class IterativePipeline:
    async def run(self, X, y) -> dict[str, Any]:
        for round_idx in range(self.config.n_rounds):
            self._emit_progress(round=round_idx, ...)
            ...
```

### 4.5 U6: Memory Inspector

**Concept:** CLI command to inspect agent memory without reading JSON files.

```bash
# List all agent memory files
forge memory list

# View key content
forge memory show --agent unary
forge memory show --agent cross_compositional --entries top-10
forge memory stats  # Summary across all agents

# Visualize memory graph
forge memory graph --output memory_graph.png
```

**Implementation:**

```python
# New: src/feature_forge/cli/memory.py

@app.command()
def memory(
    action: str = typer.Argument("list"),
    agent: str = typer.Option(None, "--agent"),
):
    """Inspect agent memory."""
    if action == "list":
        ...
    elif action == "show":
        memory = AgentMemory(agent, f"memory_files/agent_memories/{agent}_memory.json")
        memory.load()
        # Pretty print with Rich
        console = rich.console.Console()
        console.print(memory.generate_prompt_section(use_feedback=True))
    elif action == "stats":
        ...
```

### 4.6 U8: Quick CSV Mode

**Concept:** Allow users to run feature engineering on any CSV without registering it as a dataset.

```bash
forge run data.csv --target price

# Or in Python
from feature_forge.api import FeatureForge
fe = FeatureForge()
fe.fit_from_csv("data.csv", target="price", id_col="id")
X_enhanced = fe.transform_from_csv("data.csv")
```

**Implementation:**

```python
# Add to: src/feature_forge/api.py
class FeatureForge:
    def fit_from_csv(
        self,
        path: str,
        target: str,
        id_col: str | None = None,
        **kwargs: Any,
    ) -> FeatureForge:
        """Load CSV, infer column types, run fit()."""
        df = pd.read_csv(path)
        drop_cols = [id_col] if id_col else []
        X = df.drop(columns=[target] + drop_cols)
        y = df[target]
        return self.fit(X, y)

    def transform_from_csv(self, path: str) -> pd.DataFrame:
        """Load new CSV and apply transforms."""
        df = pd.read_csv(path)
        return self.transform(df)
```

### 4.7 U9: Better Error UX

**Concept:** Replace all 22 silent `except Exception` blocks with structured, actionable error messages.

**Pattern:**

```python
# Bad (current - too many)
try:
    result = risky_operation()
except Exception as exc:
    logger.warning("operation_failed", error=str(exc))

# Good
class FeatureGenerationError(MemoryError):  # specific exception
    ...

try:
    result = risky_operation()
except FeatureValidationError as exc:
    logger.error("feature_validation_failed", reason=exc.reason, hint=exc.hint)
    raise  # propagate instead of swallowing
except TimeoutError:
    raise FeatureGenerationError("Feature generation timed out. Try increasing sandbox_timeout_seconds.")
```

**Principles:**
1. Never swallow exceptions silently — always log + hint
2. Surface actionable error messages: "X failed because Y. Try Z."
3. Categorize known errors into specific exception types
4. Add context (current round, agent, dataset) to all errors

---

## 5. Phase 2: Rich UX (U5, U7)

**Timeline:** 2-3 weeks

### 5.1 U5: Result Dashboard (HTML)

**Concept:** Replace the markdown reporter with an interactive HTML dashboard that persists locally.

**Dashboard features:**

```
┌─────────────────────────────────────────────────────────────┐
│  Feature Forge Results - titanic (2026-05-17 14:32)         │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────────┐ │
│ │  Leaderboard (sorted by CV score)                       │ │
│ │  ┌─────────┬──────────┬──────────┬──────────┬────────┐ │ │
│ │  │ Method  │ CV Score │ Gain     │ Features │ Cost   │ │ │
│ │  │ malmus  │ 0.8523   │ +0.0312  │ 12       │ $0.47  │ │ │
│ │  │ caafe   │ 0.8431   │ +0.0220  │ 8        │ $0.32  │ │ │
│ │  │ openfe  │ 0.8340   │ +0.0129  │ 15       │ $0.00  │ │ │
│ │  │ llmfe   │ 0.8211   │ -        │ 10       │ $0.55  │ │ │
│ │  └─────────┴──────────┴──────────┴──────────┴────────┘ │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                               │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │  Best Features - malmus (by gain)                       │ │
│ │  ┌──────────────────┬──────────┬─────────┬────────────┐ │ │
│ │  │ Feature          │ Gain     │ Columns │ Transform  │ │ │
│ │  │ fare_per_age     │ +0.021   │ fare,age│ ratio      │ │ │
│ │  │ family_size      │ +0.015   │ sibsp,px│ sum        │ │ │
│ │  │ title            │ +0.009   │ name    │ extract    │ │ │
│ │  └──────────────────┴──────────┴─────────┴────────────┘ │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                               │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │  Visualization                                          │ │
│ │  [Bar chart: Method comparison]                          │ │
│ │  [Scatter: Gain vs Features generated]                  │ │
│ │  [Per-round: Cumulative gain trajectory]                │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                               │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │  Feature descriptions + code                            │ │
│ │  fare_per_age = fare / age                              │ │
│ │  # Cross-column ratio; 3x effective on titanic          │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**Implementation:**

```python
# New files:
# src/feature_forge/reporting/dashboard.py
# src/feature_forge/reporting/templates/dashboard.html.j2

class ExperimentDashboard:
    """Generate interactive HTML dashboard from experiment results."""

    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = results

    def render(self) -> str:
        """Render the dashboard as a self-contained HTML string."""
        ...

    def save(self, path: str) -> None:
        """Save the dashboard to an HTML file."""
        ...

    def open(self) -> None:
        """Open the dashboard in the default browser."""
        import webbrowser
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            f.write(self.render().encode())
            webbrowser.open(f"file://{f.name}")
```

**Key design choices:**
- Self-contained HTML (no server needed)
- Uses `plotly` for interactive charts OR inline Chart.js from CDN
- Feature descriptions link back to provenance info
- Cost tracking per method (from Langfuse or LLM cache stats)

**New dependencies:** `plotly` optional (alternative: inline JS charts)

### 5.2 U7: Feature Explorer (`forge suggest`)

**Concept:** Allow users to explore feature suggestions without running the full pipeline.

```bash
# Get feature suggestions for a dataset
forge suggest titanic.csv --agent unary

# Suggest with specific columns in mind
forge suggest titanic.csv --columns fare,age --type cross_compositional

# Interactive mode
forge suggest --interactive
> Dataset: titanic.csv
> Target column: survived
> Suggested features:
>   1. fare_per_age (ratio) - estimated gain: +0.015
>   2. family_size (aggregation) - estimated gain: +0.010
>   3. ...
```

**Additional Python API:**

```python
from feature_forge import FeatureForge
fe = FeatureForge()
suggestions = fe.suggest_features(
    X_train,
    agent_type="cross_compositional",
    top_k=5,
    include_estimate=True,
)
for s in suggestions:
    print(f"{s.name}: {s.code} (estimated gain: {s.estimated_gain})")
```

**Implementation approach:**
This is essentially a "dry run" of the agent's LLM feature generation step, without sandbox execution. The exploration mode:

1. Loads or samples the dataset
2. Runs the agent's prompt with memory context
3. Returns the generated feature suggestions
4. Optionally estimates gain from similar features in memory (I2 from the memory evolution plan)

**Changes to existing code:**
- `api.py`: Add `suggest_features()` method
- `cli.py`: Add `suggest` command
- Agents already have `generate()` — just expose it without the post-processing pipeline

---

## 6. Future: Web UI

**Timeline:** Post-Phase 2
**Concept:** A lightweight web application using FastAPI + htmx that wraps the Python library.

```
┌─────────────────────────────┐
│  Upload CSV → Preview       │
│  Select target column       │
│  Choose methods             │
│  Run (server-side)          │
│  Real-time logs (SSE)       │
│  Interactive dashboard      │
│  Download enhanced dataset  │
└─────────────────────────────┘
```

This would serve **non-Python-user** persona. Not recommended for immediate implementation — the CLI covers 90% of use cases with much lower overhead.

---

## 7. Implementation Roadmap

```
Phase 1 (Weeks 1-2): CLI + Zero-Setup
├── U1: CLI tool (forge run, forge init, forge list)
├── U2: Config wizard (forge init)
├── U3: Rename async_fit → fit_async
├── U4: Progress bars + Rich live display
├── U6: Memory inspector (forge memory)
├── U8: Quick CSV mode (fit_from_csv)
└── U9: Replace broad except blocks with specific errors

Phase 2 (Weeks 3-5): Rich UX
├── U5: HTML result dashboard with charts
└── U7: Feature explorer (forge suggest)

Future:
└── Web UI (FastAPI + htmx + SSE)
```

---

## 8. File Changes Summary

### New Files

| File | Phase | Purpose |
|---|---|---|
| `src/feature_forge/cli/__init__.py` | 1 | CLI package |
| `src/feature_forge/cli/main.py` | 1 | `forge` CLI entry point |
| `src/feature_forge/cli/wizard.py` | 1 | Config wizard |
| `src/feature_forge/cli/progress.py` | 1 | Rich live display |
| `src/feature_forge/cli/memory.py` | 1 | Memory inspector |
| `src/feature_forge/reporting/dashboard.py` | 2 | HTML dashboard |
| `src/feature_forge/reporting/templates/dashboard.html.j2` | 2 | Dashboard template |

### Modified Files

| File | Phase | Changes |
|---|---|---|
| `src/feature_forge/api.py` | 1 | Rename `async_fit`→`fit_async`, add `fit_from_csv()`, `suggest_features()` |
| `src/feature_forge/methods/malmas/pipeline/iterative.py` | 1 | Emit progress events, structured logging |
| `pyproject.toml` | 1 | Add `typer`, `rich` entry points `[project.scripts]` |
| `README.md` | 1 | Update quick start with CLI examples |

---

## 9. User Journey (After)

### Journey A: Data scientist trying Feature Forge for the first time

```bash
# 1. Install
pip install feature-forge

# 2. Init (one-time)
forge init
# → "Which LLM provider? [deepseek/openai/anthropic]"
# → "Enter your API key:"
# → "Config written! Testing connection..."
# → "✓ Connected! Model: deepseek-chat"

# 3. Run on their CSV
forge run my_data.csv --target price

# → Progress bars show: Round 1/3 | Agent 2/6 | 8 features found | Best gain: +0.023
# → HTML dashboard opens in browser automatically
# → "✓ Results saved to results/experiment-2026-05-17/"

# 4. Inspect what happened
forge memory --agent unary
forge memory stats

# 5. Get feature suggestions for a new dataset
forge suggest new_data.csv --columns age,income
```

### Journey B: ML researcher comparing methods

```python
from feature_forge import ExperimentalPlatform

platform = ExperimentalPlatform()
results = platform.run(
    datasets=["titanic", "house_prices", "credit_fraud"],
    methods=["malmus", "caafe", "malmas", "openfe", "llmfe"],
    models=["xgboost", "lightgbm"],
    seeds=[42, 123, 456],
    parallel=True,
    progress=True,   # ← rich live display
)

# Results as interactive HTML
from feature_forge.reporting import ExperimentDashboard
dashboard = ExperimentDashboard(results)
dashboard.open()  # opens in browser
dashboard.save("comparison.html")
```

### Journey C: Sklearn pipeline user

```python
from feature_forge.api import FeatureForge
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

pipeline = Pipeline([
    ("fe", FeatureForge.from_csv("train.csv", target="survived")),
    ("clf", XGBClassifier()),
])
pipeline.fit(X_train, y_train)
```

---

## 10. Success Metrics

| Metric | Current | Target | Measured by |
|--------|---------|--------|-------------|
| Time to first result | 15 min (read docs + config + write script) | 30 sec (`forge init && forge run data.csv`) | Timer test |
| Number of CLI commands needed | N/A (no CLI) | 1-2 for common workflows | Count |
| Error actionability | Low (swallowed) | High (specific exception + hint) | User report |
| Result inspection time | 5 min (parse markdown) | 30 sec (browser dashboard) | Timer test |
| Lines of code for common task | ~10 (script) | 1 (CLI) | LOC count |
| Feature exploration capability | None | Available via `forge suggest` | Feature exists |

---

## 11. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| CLI adds maintenance burden | Keep CLI thin — delegates to existing Python API |
| typer dependency bloat | typer is lightweight (depends on click, no heavy deps) |
| Dashboard HTML file too large | Lazy-load charts; paginate features; compress inline assets |
| Progress overhead slows pipeline | Emit events; display is separate process (or disabled when not on TTY) |
| Config wizard exposes API keys | Write to `.env` only; never print; offer dotenvx encryption |
| Async confusion persists | Clear docstrings + rename to `fit_async`; sync `fit()` is the default |
