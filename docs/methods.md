# Methods & References

This document describes every method and pipeline implemented in Feature Forge, along with the academic papers and open-source repositories they are based on.

---

## Table of Contents

1. [Core System: MALMAS Multi-Agent Pipeline](#1-core-system-malmas-multi-agent-pipeline)
2. [Methods](#2-methods)
   - [OpenFE](#21-openfe)
   - [CAAFE](#22-caafe)
   - [LLM-FE](#23-llm-fe)
   - [Malmus (Structured JSON)](#24-malmus-structured-json)
3. [Agent Architecture](#3-agent-architecture)
4. [Memory System](#4-memory-system)
5. [Router Strategies](#5-router-strategies)
6. [Evaluation & Sandboxing](#6-evaluation-sandboxing)
7. [Prompt Registry](#7-prompt-registry)
8. [Thinking Mode](#8-thinking-mode)
9. [Full Reference List](#9-full-reference-list)

---

## 1. Core System: MALMAS Multi-Agent Pipeline

Feature Forge is a production-ready refactoring of the **MALMAS** (Memory-Augmented LLM-based Multi-Agent System) research codebase.

**Paper:**
> "Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data"
> Authors: MINE-USTC Research Group
> arXiv: [2604.20261](https://arxiv.org/abs/2604.20261) (April 2026)
> Submitted to: ACL ARR 2026 January cycle ([OpenReview](https://openreview.net/forum?id=07xc55lKR8))
> GitHub: [MINE-USTC/MALMAS](https://github.com/MINE-USTC/MALMAS)

### How MALMAS Works

MALMAS decomposes automated feature engineering into a multi-round, multi-agent loop:

1. **Router Agent** selects a subset of specialized agents for each round based on dataset characteristics and historical performance.
2. **Specialized Agents** generate feature specifications from the data, each focusing on a different type of transformation:
   - **Unary Feature Agent**: Single-column transforms (log, sqrt, binning).
   - **Cross-Compositional Agent**: Multi-column interactions (ratios, products).
   - **Aggregation Construct Agent**: GroupBy aggregations (mean, count per category).
   - **Temporal Feature Agent**: Date/time-derived features (day-of-week, elapsed time).
   - **Local Transform Agent**: Localized numerical transformations (rolling, diff).
   - **Local Pattern Agent**: Distributional pattern detection (outlier flags, quantile ranks).
3. **Code Generator** converts feature specifications into executable Python code.
4. **Sandboxed Executor** runs the code with AST validation for safety.
5. **CVEvaluator** assesses each generated feature via cross-validation gain over baseline.
6. **Memory System** records results across rounds to guide future generation.

### Feature Forge Enhancements over Original MALMAS

| Aspect | Original MALMAS | Feature Forge |
|--------|----------------|---------------|
| Packaging | Flat scripts | Modular `src/` layout with entry points |
| Config | Hardcoded dicts | pydantic-settings with YAML + env vars |
| LLM Support | OpenAI only | DeepSeek native + LiteLLM (100+ providers) |
| JSON Mode | None | Native DeepSeek JSON mode + structured output |
| Secret Management | Plaintext `.env` | dotenvx encrypted `.env` |
| Experiment Tracking | None | WandB + MLflow backends |
| Observability | Print statements | structlog + Langfuse tracing |
| Artifact Storage | Ad-hoc files | Schema-validated parquet with provenance |
| Testing | None | 42 unit + integration tests |
| CI/CD | None | GitHub Actions (ruff, mypy, pytest) |
| Sklearn API | None | `FeatureForge(BaseEstimator, TransformerMixin)` |

### Implementation Files

- Pipeline orchestration: `src/feature_forge/methods/malmas/pipeline/core.py`, `src/feature_forge/methods/malmas/pipeline/iterative.py`
- Sklearn-compatible API: `src/feature_forge/api.py`
- Agent base class: `src/feature_forge/methods/malmas/agents/base.py`
- Agent implementations: `src/feature_forge/methods/malmas/agents/unary.py`, `cross_compositional.py`, `aggregation.py`, `temporal.py`, `local_transform.py`, `local_pattern.py`

### Methods Plugin Architecture

Feature Forge uses a plugin architecture for feature engineering methods. All methods live under the `methods/` directory:

```
src/feature_forge/methods/
├── base.py            # MethodProtocol, BaseMethod, MethodRegistry
├── caafe/method.py    # CAAFEMethod
├── llmfe/method.py    # LLMFEMethod
├── malmas/            # Full MALMAS sub-package
│   ├── method.py      # MALMASMethod
│   ├── agents/        # 6 agents + router + registry
│   ├── pipeline/      # CorePipeline, IterativePipeline, ablations
│   ├── memory/        # 3-tier memory
│   └── prompts/       # YAML prompt loading (local copy)
├── malmus/method.py   # MalmusMethod
└── openfe/method.py   # OpenFEMethod
```

The architecture is built on three core abstractions in `base.py`:

- **`MethodProtocol`** — A `@runtime_checkable` Protocol that enables duck-typing. Third-party methods can satisfy the protocol without importing anything from `feature_forge`. Required attributes: `name` (str). Required methods: `fit`, `transform`, `fit_transform`, `generated_scripts` (property), `feature_metadata` (property), `get_artifacts`.

- **`BaseMethod`** — Inherits `ArtifactExporter` and provides a concrete base class with default implementations for `fit_transform` (calls `fit` then `transform`), `generated_scripts`, `get_artifacts`, and artifact management. Subclasses only need to implement `fit` and `transform`.

- **`MethodRegistry`** — Discovers methods via Python entry points in the `feature_forge.methods` group. Falls back to built-in methods if entry points aren't available. Key APIs:
  - `get_builtin_methods()` — Returns the built-in method classes
  - `get_all_methods()` — Returns built-in + entry-point-discovered methods
  - `discover()` — Explicitly scans entry points and returns discovered method classes

### Plugin Authoring Guide

You can add custom feature engineering methods to Feature Forge via two approaches:

#### Option A: Protocol-compliant (zero imports)

Implement the `MethodProtocol` interface without any `feature_forge` dependency:

```python
# my_method/method.py
from typing import Any
import pandas as pd

class MyMethod:
    name = "my_method"

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "MyMethod":
        self._code = "# feature generation code"
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"my_feature": X.iloc[:, 0] * 2}, index=X.index)

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        self.fit(X_train, y_train)
        return self.transform(X_train)

    @property
    def generated_scripts(self) -> list[str]:
        return [self._code] if hasattr(self, '_code') else []

    @property
    def feature_metadata(self) -> list[dict[str, Any]]:
        return [{"name": "my_feature", "type": "numerical"}]

    def get_artifacts(self) -> dict[str, Any]:
        return {}
```

#### Option B: BaseMethod subclass (full artifact support)

Subclass `BaseMethod` for integrated artifact export and lifecycle management:

```python
from feature_forge.methods.base import BaseMethod

class MyMethod(BaseMethod):
    def __init__(self):
        super().__init__("my_method")
        self._code = ""

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "MyMethod":
        self._code = "# generated feature code"
        self._artifacts["generated_code"] = self._code
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"my_feature": X.iloc[:, 0] * 2}, index=X.index)
```

#### Registration via entry points

Register your method so Feature Forge discovers it automatically:

```toml
[project.entry-points."feature_forge.methods"]
my_method = "my_method.method:MyMethod"
```

#### Programmatic registration

Register at runtime without modifying package metadata:

```python
from feature_forge import ExperimentalPlatform
platform = ExperimentalPlatform()
platform.register_method("my_method", MyMethod)
```

#### Verification

Confirm your method is registered:

```python
from feature_forge.methods import MethodRegistry
all_methods = MethodRegistry.get_all_methods()
assert "my_method" in all_methods

# Or via platform
"my_method" in platform.list_methods()
```

### Per-Agent Parallel Code Generation

The MALMAS pipeline uses per-agent parallel code generation for improved throughput:

1. Feature specs are grouped by agent name (`specs_by_agent`) after the agent generation phase.
2. Each agent's code is generated in parallel via `asyncio.gather` with a semaphore controlling concurrency.
3. Data schema (shape, column dtypes, nullable flags) is injected as context into the LLM prompt so generated code matches the actual data layout.
4. Each agent's code is executed independently in the sandbox with timeout guards (`asyncio.wait_for`).
5. Results are concatenated via `pd.concat(axis=1)` with duplicate column deduplication.
6. If code fails AST validation after 2 attempts, raises `PipelineError` (no longer silently skipped).

The `max_tokens` parameter for code generation is configurable via `config.llm.codegen_max_tokens` (default: 16384).

---

## 2. Methods

### 2.1 OpenFE

**Non-LLM method** using tree-based feature generation with gradient boosting evaluation.

**Paper:**
> "OpenFE: Automated Feature Generation with Expert-level Performance"
> Authors: Tianping Zhang, Zheyu Zhang, Zhiyuan Fan, Haoyan Luo, Fengyuan Liu, Qian Liu, Wei Cao, Jian Li
> Published at: **ICML 2023**
> arXiv: [2211.12507](https://arxiv.org/abs/2211.12507)
> GitHub: [IIIS-Li-Group/OpenFE](https://github.com/IIIS-Li-Group/OpenFE)
> Proceedings: [PMLR v202](https://proceedings.mlr.press/v202/zhang23ay/zhang23ay.pdf)

#### How OpenFE Works

1. **Expansion**: Applies 23 operators (arithmetic, GroupBy, Combine) to base features up to second-order.
2. **FeatureBoost**: Incremental GBDT training on top of base feature predictions instead of retraining from scratch.
3. **Two-Stage Pruning**: Successive halving followed by Mean Decrease in Impurity (MDI) to identify effective features.
4. **Benchmark**: Beat 99.3% of Kaggle teams on IEEE-CIS Fraud Detection using a simple XGBoost + OpenFE pipeline.

#### Feature Forge Method

`OpenFEMethod` wraps the `openfe` Python package with best-effort artifact extraction (selected operators, candidate features, feature importances). Since OpenFE doesn't generate code, artifacts are operator names and importance scores.

**Implementation:** `src/feature_forge/methods/openfe/method.py`

---

### 2.2 CAAFE

**Context-Aware Automated Feature Engineering** using iterative LLM prompting.

**Paper:**
> "LLMs for Semi-Automated Data Science: Introducing CAAFE for Context-Aware Automated Feature Engineering"
> Authors: Noah Hollmann, Samuel Müller, Frank Hutter
> Published at: **NeurIPS 2023**
> arXiv: [2305.03403](https://arxiv.org/abs/2305.03403)
> GitHub: [noahho/CAAFE](https://github.com/noahho/CAAFE)

#### How CAAFE Works

1. **Context Integration**: Feeds the LLM dataset descriptions + 10 random rows to convey feature scales and encodings.
2. **Iterative Prompting**: Each iteration asks the LLM to generate Python code for new features using chain-of-thought reasoning.
3. **CV-Based Filtering**: Generated features are evaluated via cross-validation; results feed back into the next prompt.
4. **Interpretable Output**: Each feature comes with both Python code and a textual explanation.

CAAFE improved performance on 11/14 benchmark datasets, boosting mean ROC AUC from 0.798 to 0.822.

#### Feature Forge Variants

| Variant | Description |
|---------|-------------|
| `unified` (default) | Reimplementation using our `CVEvaluator` and `SandboxedExecutor` for full artifact control and per-feature gain tracking |
| `fidelity` | Wraps the original `caafe` library for exact reproduction of published behavior |

**Implementation:** `src/feature_forge/methods/caafe/method.py`

---

### 2.3 LLM-FE

**LLM-based Feature Engineering with Evolutionary Optimization.**

**Paper:**
> "LLM-FE: Automated Feature Engineering for Tabular Data with LLMs as Evolutionary Optimizers"
> Authors: Nikhil Abhyankar, Parshin Shojaee, Chandan K. Reddy (Virginia Tech)
> arXiv: [2503.14434](https://arxiv.org/abs/2503.14434) (March 2025, revised May 2025)
> OpenReview: [JhJPJtau8B](https://openreview.net/forum?id=JhJPJtau8B)
> GitHub: [nikhilsab/LLMFE](https://github.com/nikhilsab/llmfe)

#### How LLM-FE Works

1. **Program Search**: Treats feature engineering as a program search problem where the LLM generates feature transformation hypotheses as Python programs.
2. **Evolutionary Optimization**: The LLM acts as a knowledge-guided evolutionary optimizer, mutating successful feature programs.
3. **Experience Buffer**: Maintains a buffer of high-scoring programs as in-context examples for future prompts.
4. **Data-Driven Feedback**: Performance scores serve as rewards informing the next iteration.

LLM-FE outperforms traditional baselines (OCTree) across 19 classification and 10 regression benchmarks.

#### Feature Forge Method

`LLMFEMethod` implements the core iterative prompting pattern with two modes:
- **`single_shot`**: One LLM call generates all features at once.
- **`iterative`**: Sequential LLM calls with CV-based keep/discard after each iteration.

**Implementation:** `src/feature_forge/methods/llmfe/method.py`

---

### 2.4 Malmus (Structured JSON)

**Feature Forge's own structured method** that enforces JSON-mode output from LLMs.

Malmus is not based on a published paper. It is a novel contribution of Feature Forge that addresses a key limitation of LLM-FE and CAAFE: **free-text code generation is unreliable**. By forcing the LLM to return structured JSON with per-feature metadata (name, code, description, libraries), Malmus achieves:

- **Reliable parsing** without regex or markdown-fence stripping
- **Per-feature provenance tracking** (name, description, libraries)
- **Safer execution** with explicit library dependency declarations
- **Provider-agnostic JSON mode** via DeepSeek native or LiteLLM

#### How Malmus Works

1. **Schema Injection**: Injects a JSON schema into the system prompt describing the expected output format: `{"features": [{"name", "code", "description", "libraries"}]}`.
2. **JSON Mode**: Uses `complete_json()` with `response_format={"type": "json_object"}` (DeepSeek) or equivalent JSON mode via LiteLLM.
3. **Pydantic Validation**: Parses the LLM response into `StructuredFeatureOutput` with strict validation.
4. **Code Synthesis**: Converts validated feature definitions into an executable `generate_features(df)` function.

#### Modes

| Mode | Description |
|------|-------------|
| `single_shot` | One LLM call generates all features at once |
| `iterative` | Sequential LLM calls with CV-based keep/discard and feedback loops |

**Implementation:** `src/feature_forge/methods/malmus/method.py`

---

## 3. Agent Architecture

The 6 specialized agents are based on the MALMAS agent taxonomy:

| Agent | Specialization | Prompt File |
|-------|---------------|-------------|
| **Unary Feature** | Single-column transforms: log, sqrt, binning, encoding | `config/prompts/unary.yaml` |
| **Cross-Compositional** | Multi-column interactions: ratios, products, differences | `config/prompts/cross_compositional.yaml` |
| **Aggregation Construct** | GroupBy aggregations: mean/count/sum per category | `config/prompts/aggregation.yaml` |
| **Temporal Feature** | Date/time features: day-of-week, elapsed, cyclical encoding | `config/prompts/temporal.yaml` |
| **Local Transform** | Local numerical transforms: rolling, diff, lag | `config/prompts/local_transform.yaml` |
| **Local Pattern** | Distributional patterns: outlier flags, quantile ranks, z-scores | `config/prompts/local_pattern.yaml` |

Each agent:
1. Receives dataset column metadata + memory context + positive/negative feature lists
2. Calls the LLM with a specialized system prompt
3. Parses structured JSON feature specifications
4. Returns `list[FeatureSpec]` for the code generator

Agents are discoverable via Python entry points (`feature_forge.methods.malmas.agents` group), allowing external packages to register custom agents.

> **Note:** Agents are MALMAS-specific components. They live under `methods/malmas/agents/`, not as shared infrastructure.

**Source:** MALMAS paper, Section 3.2 ("Specialized Feature Generation Agents")

**Implementation:** `src/feature_forge/methods/malmas/agents/`

**Import example:**
```python
from feature_forge.methods.malmas.agents import UnaryFeatureAgent, CrossCompositionalAgent
```

---

## 4. Memory System

The 3-tier memory architecture is based on the MALMAS memory module. Memory is a MALMAS-specific subsystem living under `methods/malmas/memory/`.

**Source:** MALMAS paper, Section 3.3 ("Memory Module")

### Tiers

| Tier | Purpose | Storage |
|------|---------|---------|
| **Procedural** | Records successful transform attempts (columns, transform, feature name, type) | JSON per-agent |
| **Feedback** | Records evaluation outcomes per feature (metric, gain, effective flag) | JSON per-agent |
| **Conceptual** | LLM-summarized actionable rules distilled from effective features | JSON per-agent |

### Conceptual Memory (LLM Summarization)

The conceptual memory tier uses a two-level LLM summarization process:

1. **Per-Agent Summary**: For each agent, the LLM receives effective feature examples and statistics, then generates 1-3 concise rules to guide future generation.
2. **Global Summary**: A second LLM call synthesizes all per-agent summaries into 2-5 high-level rules that inform the entire system.

This approach prevents prompt length explosion by replacing raw history with compressed heuristics.

**Implementation:** `src/feature_forge/methods/malmas/memory/base.py`, `src/feature_forge/methods/malmas/memory/conceptual.py`

---

## 5. Router Strategies

The Router Agent dynamically selects which specialized agents to activate for each round. Based on MALMAS Section 3.1 ("Router Agent").

| Strategy | Description |
|----------|-------------|
| **`data_driven`** | Selects agents based on dataset characteristics (column types, datetime presence, column count). Excludes agents whose required column types are missing. |
| **`performance_driven`** | Selects agents based on historical average gain. Agents with non-negative average gain are prioritized; at least `min_agents` are always included. |
| **`hybrid`** (default) | Union of data-driven and performance-driven selections. Ensures `min_agents` to `max_agents` are selected. |
| **`llm`** | Uses the LLM itself to decide which agents to activate, given dataset characteristics and performance history. Falls back to hybrid on failure. |

The router includes a warmup phase (1 round by default) where all agents run to collect initial performance data.

**Implementation:** `src/feature_forge/methods/malmas/agents/router.py`

Entry point group: `feature_forge.methods.malmas.agents`

---

## 6. Evaluation & Sandboxing

### Cross-Validation Evaluator

`CVEvaluator` computes a baseline score on the original feature set, then evaluates each generated feature by measuring the **gain** (improvement in metric) when the feature is added to the baseline.

Supported metrics: `auc`, `acc`, `f1`, `rmse`, `mae`, `r2`

**Implementation:** `src/feature_forge/evaluation/cv.py`

### Sandboxed Executor

`SandboxedExecutor` runs LLM-generated Python code in a restricted environment:
1. **AST Validation**: Parses code to block dangerous operations (`import os`, `subprocess`, file I/O, network calls).
2. **Execution**: Runs validated code in a namespace with only `pandas`, `numpy`, and the input DataFrame.
3. **Output Validation**: Ensures the result is a pandas DataFrame with matching index.

Threat model and trust boundary:
- Generated code is treated as potentially untrusted and is only allowed to transform in-memory tabular data.
- File-system and network operations are blocked by validation/runtime guards.
- This is incremental hardening, not full container/jail isolation; use an external sandbox boundary for high-risk multi-tenant workloads.

> **Note:** The per-agent sandbox execution uses `asyncio.wait_for` with a timeout of `max(sandbox_timeout * 2, 30)` seconds per agent, preventing a single slow agent from blocking the pipeline.

**Implementation:** `src/feature_forge/evaluation/sandbox.py`

---

## 7. Prompt Registry

Feature Forge manages LLM prompts as YAML files with a centralized registry.

### Prompt Files

Prompts are stored in `config/prompts/*.yaml`. Each YAML file has:
- `system` (required): The system prompt text
- `description` (optional): A human-readable description of the prompt's purpose

### Prompt Model

The `Prompt` Pydantic model validates prompt files:
- `system: str` — Must be non-empty (validated)
- `description: str = ""` — Optional metadata

### PromptRegistry

`PromptRegistry` provides lazy loading from the `config/prompts/` directory. It is accessed as a singleton via `get_registry()`.

### Available Prompts

| Name | Purpose |
|------|---------|
| `unary` | Single-column feature transforms |
| `cross_compositional` | Multi-column interaction features |
| `aggregation` | GroupBy aggregation features |
| `temporal` | Date/time-derived features |
| `local_transform` | Localized numerical transforms |
| `local_pattern` | Distributional pattern detection |
| `router` | Agent selection routing |
| `code_generation` | Code generation from feature specs |

### Usage

```python
from feature_forge.prompts import get_registry
registry = get_registry()
prompt = registry.get("unary")
print(prompt.system)
```

> **Note:** The top-level `feature_forge.prompts` module provides the registry. MALMAS-specific prompts are stored alongside the shared config in `config/prompts/`.

---

## 8. Thinking Mode

The DeepSeek provider supports a thinking/reasoning mode that enables extended chain-of-thought processing before generating a response.

### Configuration

Thinking mode is configured in `config/settings.yaml`:

```yaml
llm:
  thinking_enabled: true
  reasoning_effort: "medium"
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `thinking_enabled` | `bool` | `False` | Enable/disable thinking mode |
| `reasoning_effort` | `str` | `"medium"` | Reasoning depth: `low`, `medium`, `high`, or `max` |

### Implementation

When enabled, `DeepSeekProvider._call_api()` injects `reasoning_effort` and `extra_body.thinking.type="enabled"` into the API request. Reasoning content is extracted via `_extract_reasoning_content()` from `message.reasoning_content` in the API response.

> **Note:** Thinking mode is DeepSeek-specific. Other providers ignore these parameters.

---

## 9. Full Reference List

### Papers

| # | Method | Title | Authors | Venue | Year | arXiv |
|---|--------|-------|---------|-------|------|-------|
| 1 | MALMAS | Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data | MINE-USTC | ACL ARR 2026 | 2026 | [2604.20261](https://arxiv.org/abs/2604.20261) |
| 2 | OpenFE | OpenFE: Automated Feature Generation with Expert-level Performance | Zhang et al. | ICML 2023 | 2023 | [2211.12507](https://arxiv.org/abs/2211.12507) |
| 3 | CAAFE | LLMs for Semi-Automated Data Science: Introducing CAAFE for Context-Aware Automated Feature Engineering | Hollmann, Müller, Hutter | NeurIPS 2023 | 2023 | [2305.03403](https://arxiv.org/abs/2305.03403) |
| 4 | LLM-FE | LLM-FE: Automated Feature Engineering for Tabular Data with LLMs as Evolutionary Optimizers | Abhyankar, Shojaee, Reddy | Preprint | 2025 | [2503.14434](https://arxiv.org/abs/2503.14434) |

### Repositories

| # | Method | Repository |
|---|--------|-----------|
| 1 | MALMAS | [github.com/MINE-USTC/MALMAS](https://github.com/MINE-USTC/MALMAS) |
| 2 | OpenFE | [github.com/IIIS-Li-Group/OpenFE](https://github.com/IIIS-Li-Group/OpenFE) |
| 3 | CAAFE | [github.com/noahho/CAAFE](https://github.com/noahho/CAAFE) |
| 4 | LLM-FE | [github.com/nikhilsab/LLMFE](https://github.com/nikhilsab/llmfe) |
| 5 | Feature Forge | (this repository) |

### Related Work

- **AutoFeat** (arXiv:1905.04494) - Horn et al., 2019 - Automatic feature synthesis for relational data
- **ExploreKit** (KDD 2016) - Katz et al. - Feature generation via operator trees
- **OCTree** - Oblivious decision tree-based feature engineering
- **TPOT** (Olson et al., 2016) - Tree-based Pipeline Optimization Tool using evolutionary algorithms
- **Featuretools** (Kanter & Veeramachaneni, 2015) - Deep feature synthesis for relational data
