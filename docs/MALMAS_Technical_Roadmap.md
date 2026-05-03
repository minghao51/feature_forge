# MALMAS Technical Roadmap

## From Research Code to Production-Ready Python Package

**Memory-Augmented LLM-based Multi-Agent System for Automated Feature Engineering**

Version 1.0 | April 2026

Repository: https://github.com/MINE-USTC/MALMAS

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State Assessment](#2-current-state-assessment)
   - 2.1 [Repository Structure Analysis](#21-repository-structure-analysis)
   - 2.2 [Critical Technical Debt](#22-critical-technical-debt)
   - 2.3 [Comparison with Production Libraries](#23-comparison-with-production-libraries)
3. [Refactoring Recommendations](#3-refactoring-recommendations)
   - 3.1 [Package Structure Redesign](#31-package-structure-redesign)
   - 3.2 [Configuration System Overhaul](#32-configuration-system-overhaul)
   - 3.3 [Sklearn-Compatible API Design](#33-sklearn-compatible-api-design)
   - 3.4 [Error Handling and Input Validation](#34-error-handling-and-input-validation)
4. [Proposed Architecture](#4-proposed-architecture)
   - 4.1 [Directory Structure](#41-directory-structure)
   - 4.2 [Core Components](#42-core-components)
   - 4.3 [Agent Plugin Architecture](#43-agent-plugin-architecture)
5. [Implementation Roadmap](#5-implementation-roadmap)
   - 5.1 [Phase 1: Package Infrastructure (Days 1-5)](#51-phase-1-package-infrastructure-days-1-5)
   - 5.2 [Phase 2: Core API Redesign (Days 6-12)](#52-phase-2-core-api-redesign-days-6-12)
   - 5.3 [Phase 3: Quality and Documentation (Days 13-20)](#53-phase-3-quality-and-documentation-days-13-20)
   - 5.4 [Phase 4: Release Preparation (Days 21-28)](#54-phase-4-release-preparation-days-21-28)
6. [Risk Analysis and Mitigation](#6-risk-analysis-and-mitigation)
   - 6.1 [Breaking Changes Risk](#61-breaking-changes-risk)
   - 6.2 [Dependency Management Risk](#62-dependency-management-risk)
   - 6.3 [LLM API Compatibility Risk](#63-llm-api-compatibility-risk)
   - 6.4 [Code Execution Security Risk](#64-code-execution-security-risk)
7. [Expected Outcomes and Success Criteria](#7-expected-outcomes-and-success-criteria)
   - 7.1 [Technical Outcomes](#71-technical-outcomes)
   - 7.2 [User Experience Improvements](#72-user-experience-improvements)
   - 7.3 [Community Impact](#73-community-impact)
8. [Conclusion](#8-conclusion)
9. [Appendix A: Implementation Priority Matrix](#appendix-a-implementation-priority-matrix)
10. [Appendix B: Dependency Requirements](#appendix-b-dependency-requirements)

---

## 1. Executive Summary

This technical roadmap outlines a comprehensive strategy for transforming MALMAS (Memory-Augmented LLM-based Multi-Agent System) from a research codebase into a production-ready, pip-installable Python package. The current codebase demonstrates innovative approaches to automated feature engineering using multi-agent LLM systems, but requires significant refactoring to achieve production quality, maintainability, and widespread usability.

The transformation addresses four critical areas: (1) package structure and distribution infrastructure, (2) API design with sklearn compatibility, (3) code quality including configuration management and error handling, and (4) documentation and testing. The proposed changes will enable MALMAS to be installed via standard Python packaging tools, integrated seamlessly into existing ML pipelines, and maintained by the broader open-source community.

The roadmap is structured into four implementation phases spanning approximately four weeks, with clear milestones, risk mitigation strategies, and success criteria. Upon completion, MALMAS will transition from a research artifact to a professional-grade library suitable for both academic research and industrial applications.

---

## 2. Current State Assessment

### 2.1 Repository Structure Analysis

The current MALMAS repository exhibits characteristics typical of research-oriented code developed primarily for paper reproduction and experimental validation. While the codebase successfully implements the multi-agent feature engineering methodology described in the associated research paper, it lacks the structural elements necessary for production deployment.

The primary organizational issues include:

- **No Python package infrastructure** (no `setup.py`, `pyproject.toml`, or proper `__init__.py` hierarchy)
- **Reliance on global mutable state** for configuration
- **Hardcoded paths and API credentials** scattered throughout the codebase
- **Mixed concerns** within core modules that combine LLM operations, feature evaluation, and code execution in single files

**Current Directory Structure:**
```
MALMAS/
├── main_demo/           # Core pipeline code (NO __init__.py)
│   ├── main_func.py     # Main functions
│   ├── pipeline.py      # Pipeline orchestration
│   ├── model_factory.py # Model creation
│   ├── memory.py        # Agent memory system
│   ├── router.py        # Router agent
│   └── path_helper.py   # Path manipulation
├── baselines/           # Baseline comparisons (NO __init__.py)
│   ├── baseline_func.py
│   ├── utils_xg.py
│   └── LLMFE_demo/
├── data_file/           # Dataset modules
├── web_app/             # FastAPI web server
├── prompt_files/        # LLM prompt templates
├── global_config.py     # Global configuration (CRITICAL ISSUE)
└── *.ipynb              # Jupyter notebooks (20+ notebooks)
```

### 2.2 Critical Technical Debt

Several critical issues impede the codebase's readiness for production use:

#### Global Mutable State (CRITICAL)

**File: `global_config.py` (lines 3-51)**
```python
# These are module-level mutable globals - VERY BAD for production
data_pre = {"test_size":0.4, "random_state":42}
LLM = {"code_temp":0.2, "llm_model":"deepseek-chat", "api_key":"", "base_url":""}
total_tokens=0  # Mutable counter!
task="classification"
metric="auc"
```

**Issues:**
- Mutable global state causes race conditions in concurrent/parallel execution
- Makes testing difficult - state persists between tests
- Configuration cannot be isolated per-request in web server

#### Hardcoded Paths

**File: `main_demo/pipeline.py`**
```python
cache_dir = f"memory_files/{task+global_config.other_model}/{task_name}/{random_state}"
with open("prompt_files/codegeneration.txt", "r", encoding="utf-8") as f:
```

#### `sys.path` Manipulation (Anti-pattern)

**Multiple files manipulate sys.path directly:**
```python
# main_demo/path_helper.py:21-22
if base_path not in sys.path:
    sys.path.append(base_path)
```

**Impact:** Breaks import resolution, makes package non-installable, causes import conflicts.

#### Code Injection via `exec()`

**File: `main_demo/main_func.py` (line 370)**
```python
exec(code, injected_globals, local_vars)
```

**Security Risk:** Executes LLM-generated code without sandboxing.

### 2.3 Comparison with Production Libraries

When compared to production-ready feature engineering libraries like FeatureTools and AutoFeat, the gaps become apparent:

| Aspect | FeatureTools | AutoFeat | MALMAS |
|--------|-------------|----------|--------|
| **Package Structure** | Proper package, pip-installable | Proper package | No package structure |
| **API Design** | `EntitySet` → `dfs()` | fit/transform/predict | No clear API |
| **Sklearn Compatibility** | Partial | Full | None |
| **Input Validation** | Woodwork schema | sklearn validation | None |
| **Error Handling** | Custom exceptions | Proper hierarchy | `print()` + `return None` |
| **Logging** | Python logging module | Verbose parameter | `print()` statements |
| **Documentation** | Extensive docs | API docs | README only |
| **Testing** | 90%+ coverage | Unit tests | No tests |

#### FeatureTools Architecture Highlights

- **Proper package structure** with clear separation: `primitives/`, `synthesis/`, `computational_backends/`, `feature_base/`
- **Plugin architecture** via Python entry points for extensibility
- **Woodwork schema validation** for type-safe feature definitions
- **Config class pattern** (not global mutable state)

#### AutoFeat Architecture Highlights

- **Full sklearn compliance**: Inherits from `BaseEstimator`, implements `fit/transform/predict`
- **Input validation** using `sklearn.utils.validation`
- **Parameters stored as instance attributes** (sklearn convention)
- **Parallel processing** via joblib

---

## 3. Refactoring Recommendations

### 3.1 Package Structure Redesign

The first priority is establishing a proper Python package structure. The recommended approach follows modern Python packaging standards using `pyproject.toml` as the single source of truth for package metadata, dependencies, and build configuration.

**Recommended Structure:**
```
malmas/
├── pyproject.toml          # Modern Python packaging
├── README.md
├── LICENSE
├── malmas/                  # Actual package
│   ├── __init__.py          # Version, public API exports
│   ├── feature_engineer.py  # Main sklearn-compatible class
│   ├── config.py            # Dataclass-based configuration
│   ├── exceptions.py        # Custom exceptions
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py          # Abstract Agent class
│   │   ├── unary.py
│   │   ├── cross_compositional.py
│   │   ├── aggregation.py
│   │   ├── temporal.py
│   │   ├── local_transform.py
│   │   ├── local_pattern.py
│   │   └── router.py
│   ├── memory/
│   │   ├── __init__.py
│   │   └── memory.py
│   ├── llm/
│   │   ├── __init__.py
│   │   └── client.py        # LLM client abstraction
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluator.py
│   └── utils/
│       ├── __init__.py
│       ├── validation.py
│       └── logging.py
├── tests/
│   ├── __init__.py
│   ├── test_feature_engineer.py
│   ├── test_agents.py
│   └── test_config.py
└── docs/
    ├── index.md
    └── api_reference.md
```

### 3.2 Configuration System Overhaul

The global mutable state pattern must be replaced with immutable, instance-based configuration using Python dataclasses.

**Current Problem (Critical):**
```python
# global_config.py - THIS IS BAD
data_pre = {"test_size":0.4, "random_state":42}
LLM = {"code_temp":0.2, "llm_model":"deepseek-chat", "api_key":"", "base_url":""}
total_tokens=0  # Mutable global counter!
```

**Recommended Solution:**
```python
# malmas/config.py
from dataclasses import dataclass, field
from typing import Literal, Optional

@dataclass
class LLMConfig:
    """LLM configuration settings."""
    model: str = "deepseek-chat"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    code_temperature: float = 0.2
    max_tokens: int = 4096

@dataclass
class MALMASConfig:
    """Immutable configuration for MALMAS feature engineering.
    
    Attributes:
        task: ML task type ('classification' or 'regression')
        metric: Evaluation metric
        n_rounds: Number of feature engineering iterations
        min_effective: Minimum effective features per round
        llm: LLM configuration
        random_state: Random seed for reproducibility
        verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
    """
    task: Literal["classification", "regression"] = "classification"
    metric: str = "auc"
    n_rounds: int = 4
    min_effective: int = 2
    llm: LLMConfig = field(default_factory=LLMConfig)
    random_state: int = 42
    verbose: int = 0
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.task not in ("classification", "regression"):
            raise ValueError(f"Invalid task: {self.task}")
        if self.metric not in ("auc", "acc", "f1", "rmse", "mae", "r2"):
            raise ValueError(f"Invalid metric: {self.metric}")
        if self.n_rounds < 1:
            raise ValueError(f"n_rounds must be >= 1, got {self.n_rounds}")
```

### 3.3 Sklearn-Compatible API Design

To achieve sklearn compatibility, the main `MALMASFeatureEngineer` class must inherit from `BaseEstimator` and `TransformerMixin`, implementing the standard `fit/transform/fit_transform` methods.

```python
# malmas/feature_engineer.py
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np

class MALMASFeatureEngineer(BaseEstimator, TransformerMixin):
    """LLM-powered multi-agent feature engineering transformer.
    
    Compatible with sklearn pipelines:
    
    Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from xgboost import XGBClassifier
    >>> 
    >>> pipeline = Pipeline([
    ...     ('fe', MALMASFeatureEngineer(task='classification')),
    ...     ('scaler', StandardScaler()),
    ...     ('model', XGBClassifier())
    ... ])
    >>> pipeline.fit(X_train, y_train)
    
    Parameters
    ----------
    task : {'classification', 'regression'}
        The ML task type.
    metric : str
        Evaluation metric ('auc', 'acc', 'f1', 'rmse', 'mae').
    n_rounds : int
        Number of feature engineering iterations.
    llm_model : str
        LLM model identifier.
    api_key : str, optional
        API key for LLM service.
    verbose : int
        Verbosity level (0=silent, 1=progress, 2=detailed).
    
    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : list
        Names of features seen during fit.
    generated_features_ : dict
        Generated feature definitions.
    is_fitted_ : bool
        Whether the transformer has been fitted.
    """
    
    def __init__(
        self,
        task: str = "classification",
        metric: str = "auc",
        n_rounds: int = 4,
        llm_model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        verbose: int = 0
    ):
        # Store as instance attributes (sklearn convention)
        self.task = task
        self.metric = metric
        self.n_rounds = n_rounds
        self.llm_model = llm_model
        self.api_key = api_key
        self.base_url = base_url
        self.verbose = verbose
        
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            description: Optional[str] = None) -> "MALMASFeatureEngineer":
        """Generate features using LLM-powered agents.
        
        Parameters
        ----------
        X : DataFrame
            Training features.
        y : Series
            Training labels.
        description : str, optional
            Dataset description for context-aware feature generation.
        
        Returns
        -------
        self : MALMASFeatureEngineer
            Fitted transformer.
        """
        # Input validation
        X, y = check_X_y(X, y, dtype=None, force_all_finite='allow-nan')
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
            
        # Store training data info
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.tolist()
        self._description = description
        
        # Initialize configuration
        config = MALMASConfig(
            task=self.task,
            metric=self.metric,
            n_rounds=self.n_rounds,
            llm=LLMConfig(
                model=self.llm_model,
                api_key=self.api_key,
                base_url=self.base_url
            ),
            verbose=self.verbose
        )
        
        # Run feature generation
        self.generated_features_ = self._generate_features(X, y, config)
        self.is_fitted_ = True
        
        if self.verbose > 0:
            print(f"Generated {len(self.generated_features_)} features")
            
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply generated features to new data.
        
        Parameters
        ----------
        X : DataFrame
            Features to transform.
            
        Returns
        -------
        X_new : DataFrame
            Transformed features including generated ones.
        """
        check_is_fitted(self, 'is_fitted_')
        X = self._validate_data(X, reset=False)
        
        # Apply generated features
        X_new = self._apply_features(X)
        return X_new
    
    def fit_transform(self, X, y, description=None, **fit_params):
        """Fit and transform in one step."""
        return self.fit(X, y, description=description, **fit_params).transform(X)
    
    def get_feature_names_out(self) -> List[str]:
        """Get names of generated features."""
        check_is_fitted(self, 'is_fitted_')
        return list(self.generated_features_.keys())
    
    def _generate_features(self, X, y, config):
        """Core feature generation logic."""
        # ... implementation
        pass
    
    def _apply_features(self, X):
        """Apply generated features to data."""
        # ... implementation
        pass
```

### 3.4 Error Handling and Input Validation

Robust error handling requires a custom exception hierarchy:

```python
# malmas/exceptions.py
class MALMASError(Exception):
    """Base exception for MALMAS."""
    pass

class LLMError(MALMASError):
    """LLM API call failed."""
    pass

class FeatureGenerationError(MALMASError):
    """Feature generation failed."""
    pass

class ConfigurationError(MALMASError):
    """Invalid configuration."""
    pass

class CodeExecutionError(MALMASError):
    """Generated code execution failed."""
    pass

class AgentError(MALMASError):
    """Agent operation failed."""
    pass
```

**Sandboxed Code Execution:**

```python
# malmas/utils/sandbox.py
import ast

def safe_exec(code: str, allowed_names: dict) -> dict:
    """Execute code with restricted builtins.
    
    Parameters
    ----------
    code : str
        Python code to execute.
    allowed_names : dict
        Names allowed in the execution context.
    
    Returns
    -------
    dict
        Local variables after execution.
    
    Raises
    ------
    CodeExecutionError
        If code contains forbidden operations or fails to execute.
    """
    # Parse and validate AST first
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise CodeExecutionError(f"Invalid Python syntax: {e}")
    
    # Check for dangerous operations
    forbidden = {'eval', 'exec', 'compile', 'open', 'input', '__import__'}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise CodeExecutionError("Imports not allowed in generated code")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in forbidden:
                    raise CodeExecutionError(f"Forbidden function: {node.func.id}")
    
    # Execute with restricted globals
    safe_globals = {"__builtins__": {}}
    safe_globals.update(allowed_names)
    
    local_vars = {}
    try:
        exec(code, safe_globals, local_vars)
    except Exception as e:
        raise CodeExecutionError(f"Code execution failed: {e}")
    
    return local_vars
```

---

## 4. Proposed Architecture

### 4.1 Directory Structure

The proposed package structure follows Python best practices:

```
malmas/
├── pyproject.toml              # Package metadata and dependencies
├── README.md                   # Installation and quick start
├── LICENSE                     # MIT License
├── malmas/                     # Main package
│   ├── __init__.py             # Public API exports
│   ├── feature_engineer.py     # Main sklearn-compatible class
│   ├── config.py               # Configuration dataclasses
│   ├── exceptions.py           # Custom exceptions
│   ├── agents/                 # Agent implementations
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract Agent class
│   │   ├── unary.py            # UnaryFeatureAgent
│   │   ├── cross_compositional.py
│   │   ├── aggregation.py
│   │   ├── temporal.py
│   │   ├── local_transform.py
│   │   ├── local_pattern.py
│   │   └── router.py           # RouterAgent
│   ├── memory/                 # Memory system
│   │   ├── __init__.py
│   │   └── memory.py
│   ├── llm/                    # LLM client abstraction
│   │   ├── __init__.py
│   │   ├── client.py           # Base client interface
│   │   └── providers/          # Provider-specific implementations
│   │       ├── openai.py
│   │       └── deepseek.py
│   ├── evaluation/             # Feature evaluation
│   │   ├── __init__.py
│   │   └── evaluator.py
│   └── utils/                  # Utilities
│       ├── __init__.py
│       ├── validation.py
│       ├── sandbox.py          # Safe code execution
│       └── logging.py
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── conftest.py             # Pytest fixtures
│   ├── test_feature_engineer.py
│   ├── test_agents.py
│   ├── test_config.py
│   └── test_utils.py
└── docs/                       # Documentation
    ├── index.md
    ├── api_reference.md
    ├── examples.md
    └── migration_guide.md
```

### 4.2 Core Components

#### Agent System

```python
# malmas/agents/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import pandas as pd

class Agent(ABC):
    """Abstract base class for feature generation agents.
    
    All agents must implement the generate method that produces
    feature engineering code based on input data and context.
    """
    
    def __init__(self, name: str, config: 'MALMASConfig'):
        self.name = name
        self.config = config
        self.memory = []
    
    @abstractmethod
    def generate(self, X: pd.DataFrame, y: pd.Series, 
                 context: Dict[str, Any]) -> List['FeatureSpec']:
        """Generate feature specifications.
        
        Parameters
        ----------
        X : DataFrame
            Input features.
        y : Series
            Target variable.
        context : dict
            Additional context (previous features, feedback, etc.).
        
        Returns
        -------
        list of FeatureSpec
            Generated feature specifications.
        """
        pass
    
    def update_memory(self, feedback: Dict[str, Any]):
        """Update agent memory with feedback."""
        self.memory.append(feedback)
```

#### Router Agent

```python
# malmas/agents/router.py
from typing import List, Type
import pandas as pd

class RouterAgent:
    """Dynamically selects active agent subsets based on data characteristics."""
    
    def __init__(self, agents: List[Type[Agent]], config: 'MALMASConfig'):
        self.agents = agents
        self.config = config
        self.active_history = []
    
    def select_agents(self, X: pd.DataFrame, y: pd.Series, 
                      iteration: int) -> List[Agent]:
        """Select active agents for the current iteration.
        
        Parameters
        ----------
        X : DataFrame
            Input features.
        y : Series
            Target variable.
        iteration : int
            Current iteration number.
        
        Returns
        -------
        list of Agent
            Active agents for this iteration.
        """
        # Analyze data characteristics
        n_features = X.shape[1]
        n_samples = X.shape[0]
        has_temporal = self._detect_temporal(X)
        has_categorical = self._detect_categorical(X)
        
        # Select agents based on characteristics and iteration
        selected = []
        
        # Always include unary agent
        selected.append(self._get_agent('unary'))
        
        # Include cross-compositional for high-dimensional data
        if n_features > 5:
            selected.append(self._get_agent('cross_compositional'))
        
        # Include temporal agent if temporal features detected
        if has_temporal:
            selected.append(self._get_agent('temporal'))
        
        # Include aggregation for large datasets
        if n_samples > 1000:
            selected.append(self._get_agent('aggregation'))
        
        # Rotate in other agents based on iteration
        # ... selection logic
        
        self.active_history.append([a.name for a in selected])
        return selected
```

#### Memory System

```python
# malmas/memory/memory.py
from dataclasses import dataclass, field
from typing import List, Dict, Any
from collections import defaultdict

@dataclass
class MemoryEntry:
    """Single memory entry."""
    feature_name: str
    feature_code: str
    score: float
    iteration: int
    agent: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class AgentMemory:
    """Multi-level memory system for agents.
    
    Maintains three types of memory:
    - Procedural: Successful feature generation patterns
    - Feedback: Evaluation results and agent performance
    - Conceptual: High-level feature abstractions
    """
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.procedural: List[MemoryEntry] = []
        self.feedback: Dict[str, List[float]] = defaultdict(list)
        self.conceptual: List[str] = []
    
    def add_procedural(self, entry: MemoryEntry):
        """Add entry to procedural memory."""
        self.procedural.append(entry)
        if len(self.procedural) > self.max_size:
            self.procedural = self.procedural[-self.max_size:]
    
    def add_feedback(self, feature_name: str, score: float):
        """Add feedback for a feature."""
        self.feedback[feature_name].append(score)
    
    def add_conceptual(self, summary: str):
        """Add conceptual summary."""
        self.conceptual.append(summary)
    
    def get_top_features(self, k: int = 10) -> List[MemoryEntry]:
        """Get top-k features by score."""
        sorted_entries = sorted(self.procedural, key=lambda x: x.score, reverse=True)
        return sorted_entries[:k]
    
    def get_agent_performance(self, agent_name: str) -> float:
        """Get average performance for an agent."""
        agent_entries = [e for e in self.procedural if e.agent == agent_name]
        if not agent_entries:
            return 0.0
        return sum(e.score for e in agent_entries) / len(agent_entries)
```

### 4.3 Agent Plugin Architecture

Following FeatureTools' primitive system pattern, MALMAS should support custom agent registration through Python entry points:

```python
# In custom_agent_package/__init__.py
from malmas.agents.base import Agent

class MyCustomAgent(Agent):
    """Domain-specific feature generation agent."""
    
    def generate(self, X, y, context):
        # Custom feature generation logic
        pass
```

```toml
# In custom_agent_package/pyproject.toml
[project.entry-points."malmas.agents"]
my_agent = "custom_agent_package:MyCustomAgent"
```

```python
# In malmas/agents/__init__.py
import importlib.metadata

def discover_agents() -> Dict[str, Type[Agent]]:
    """Discover registered agent plugins via entry points."""
    agents = {}
    try:
        entry_points = importlib.metadata.entry_points(group="malmas.agents")
        for ep in entry_points:
            agents[ep.name] = ep.load()
    except Exception:
        pass
    return agents
```

---

## 5. Implementation Roadmap

### 5.1 Phase 1: Package Infrastructure (Days 1-5)

**Objectives:**
- Establish proper Python package structure
- Enable `pip install -e .` for development
- Remove anti-patterns blocking packaging

**Tasks:**

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Create `pyproject.toml` with complete metadata | Package installable via pip |
| 2 | Reorganize directory structure with `__init__.py` files | Proper package hierarchy |
| 3 | Create `MALMASConfig` and `LLMConfig` dataclasses | Configuration module |
| 4 | Remove `sys.path` manipulation and fix imports | Clean import system |
| 5 | Replace `global_config.py` with instance-based config | Thread-safe configuration |

**Deliverables:**
- `pyproject.toml` with dependencies, optional extras, and build configuration
- Proper `__init__.py` hierarchy exposing public API
- `malmas/config.py` with validated configuration classes
- Working `pip install -e .` for development installation

**Success Criteria:**
- `pip install -e .` completes without errors
- `import malmas` succeeds
- `malmas.__version__` returns correct version

### 5.2 Phase 2: Core API Redesign (Days 6-12)

**Objectives:**
- Implement sklearn-compatible API
- Refactor agent system to accept configuration
- Create LLM client abstraction

**Tasks:**

| Day | Task | Deliverable |
|-----|------|-------------|
| 6-7 | Create `MALMASFeatureEngineer` class with sklearn interface | Main API class |
| 8 | Implement `fit/transform` methods with validation | Sklearn compatibility |
| 9 | Create custom exception hierarchy | Error handling module |
| 10 | Refactor agents to accept config through constructor | Agent refactoring |
| 11 | Create LLM client abstraction layer | LLM module |
| 12 | Implement RouterAgent refactoring | Dynamic agent selection |

**Deliverables:**
- `MALMASFeatureEngineer` with `fit`, `transform`, `fit_transform` methods
- Input validation using `sklearn.utils.validation`
- Custom exceptions: `ConfigurationError`, `LLMError`, `FeatureGenerationError`
- `malmas/llm/client.py` with provider abstraction

**Success Criteria:**
```python
from sklearn.pipeline import Pipeline
from malmas import MALMASFeatureEngineer

# Should work without errors
fe = MALMASFeatureEngineer(task='classification')
fe.fit(X_train, y_train)
X_transformed = fe.transform(X_test)
```

### 5.3 Phase 3: Quality and Documentation (Days 13-20)

**Objectives:**
- Add comprehensive type hints
- Write unit tests for core functionality
- Add docstrings and documentation
- Implement sandboxed code execution

**Tasks:**

| Day | Task | Deliverable |
|-----|------|-------------|
| 13-14 | Add type hints to public API | Type coverage |
| 15-16 | Write unit tests for core modules | Test suite |
| 17 | Implement sandboxed `exec()` with AST validation | Security module |
| 18 | Add comprehensive docstrings | Documentation |
| 19 | Create API reference documentation | Docs |
| 20 | Set up CI/CD with GitHub Actions | Automation |

**Deliverables:**
- Type hints on all public functions and classes
- Test coverage >80% for core functionality
- `malmas/utils/sandbox.py` with safe execution
- Complete docstrings following Google/NumPy style
- `.github/workflows/test.yml` for CI

**Success Criteria:**
- `mypy malmas` passes without errors
- `pytest --cov=malmas` shows >80% coverage
- All public functions have docstrings with examples

### 5.4 Phase 4: Release Preparation (Days 21-28)

**Objectives:**
- Create comprehensive documentation
- Prepare for PyPI release
- Write migration guide

**Tasks:**

| Day | Task | Deliverable |
|-----|------|-------------|
| 21-22 | Write comprehensive README | Installation & quick start |
| 23 | Create examples gallery | Usage examples |
| 24 | Write migration guide | Migration docs |
| 25 | Finalize version and changelog | Release prep |
| 26 | Configure PyPI publishing | Publishing setup |
| 27 | Create GitHub release | Release notes |
| 28 | Publish to PyPI | Public package |

**Deliverables:**
- README.md with installation, quick start, and API overview
- `docs/examples/` with usage examples
- `docs/migration_guide.md` for existing users
- `CHANGELOG.md` documenting changes
- Published package on PyPI

**Success Criteria:**
- `pip install malmas` works from PyPI
- Documentation renders correctly on ReadTheDocs
- Examples run without errors

---

## 6. Risk Analysis and Mitigation

### 6.1 Breaking Changes Risk

**Risk:** The refactoring will introduce breaking changes to the existing API, potentially disrupting current users who have built workflows around the research codebase.

**Mitigation:**
- Maintain a detailed migration guide with before/after code examples
- Provide compatibility shims where feasible (e.g., deprecated function wrappers)
- Use semantic versioning to clearly communicate breaking changes
- Announce deprecations at least one major version before removal
- Keep the old research codebase available as a separate branch

### 6.2 Dependency Management Risk

**Risk:** The current codebase imports multiple optional baseline dependencies unconditionally, which can cause installation failures if users lack certain packages.

**Mitigation:**
- Declare baseline dependencies as optional extras in `pyproject.toml`:
  ```toml
  [project.optional-dependencies]
  baselines = ["featuretools", "autofeat", "openfe", "caafe"]
  ```
- Add conditional imports with graceful fallback:
  ```python
  try:
      import featuretools as ft
      FEATURETOOLS_AVAILABLE = True
  except ImportError:
      FEATURETOOLS_AVAILABLE = False
  ```
- Document which features require which optional dependencies
- Test installation with minimal and full dependency sets

### 6.3 LLM API Compatibility Risk

**Risk:** The codebase assumes specific LLM API formats that may change over time or vary between providers.

**Mitigation:**
- Create an abstraction layer for LLM clients that normalizes API differences
- Define a standard interface that all providers must implement:
  ```python
  class LLMClient(ABC):
      @abstractmethod
      def complete(self, messages: List[Dict], **kwargs) -> str:
          pass
  ```
- Support configuration-based provider selection
- Maintain provider-specific adapter modules
- Add integration tests for each supported provider

### 6.4 Code Execution Security Risk

**Risk:** The `exec()` calls on LLM-generated code present security vulnerabilities if malicious prompts are injected.

**Mitigation:**
- Implement sandboxed execution with restricted builtins
- Add AST validation to prevent dangerous operations:
  - No imports
  - No file operations
  - No network access
  - No eval/exec/compile
- Make security level configurable:
  - `strict`: Maximum security for production
  - `moderate`: Balanced for development
  - `permissive`: Full access for trusted environments
- Document security implications and recommended configurations
- Add security tests to CI pipeline

---

## 7. Expected Outcomes and Success Criteria

### 7.1 Technical Outcomes

Upon completion, MALMAS will achieve:

**Installation:**
```bash
# Core installation
pip install malmas

# With optional baselines
pip install malmas[baselines]

# Development installation
pip install malmas[dev]
```

**Sklearn Integration:**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from malmas import MALMASFeatureEngineer

# Seamless pipeline integration
pipeline = Pipeline([
    ('fe', MALMASFeatureEngineer(
        task='classification',
        n_rounds=4,
        verbose=1
    )),
    ('scaler', StandardScaler()),
    ('clf', XGBClassifier())
])

# Works with sklearn utilities
scores = cross_val_score(pipeline, X, y, cv=5)
```

**Quality Metrics:**
- Type hints: 100% coverage on public API
- Test coverage: >80% for core functionality
- Documentation: All public functions documented
- CI/CD: Automated testing and quality checks

### 7.2 User Experience Improvements

**Before (Research Code):**
```python
# Must clone repo and modify global config
import sys
sys.path.append('/path/to/MALMAS')
import global_config

global_config.LLM['api_key'] = 'your-key'
global_config.task = 'classification'

from main_demo.pipeline import MALMAS_random_experiments_async
# No clear API, requires understanding internal structure
```

**After (Production Package):**
```python
# Clean, intuitive API
from malmas import MALMASFeatureEngineer

fe = MALMASFeatureEngineer(
    task='classification',
    api_key='your-key',  # Or use environment variable
    n_rounds=4
)

X_transformed = fe.fit_transform(X_train, y_train)
```

**Improvements:**
- Clear installation via pip
- Intuitive API with sensible defaults
- Informative error messages
- Comprehensive documentation
- Standard sklearn patterns

### 7.3 Community Impact

**Adoption:**
- Lower barrier to entry for new users
- Integration with existing ML workflows
- Discoverability through PyPI

**Contributions:**
- Clear contribution guidelines
- Development setup documentation
- Modular architecture supporting extensions
- Entry points for custom agents

**Sustainability:**
- Maintainable codebase
- Comprehensive test suite
- CI/CD for quality assurance
- Active development infrastructure

---

## 8. Conclusion

This technical roadmap provides a comprehensive plan for transforming MALMAS from a research codebase into a production-ready Python package. The phased approach addresses critical issues including package structure, API design, code quality, and documentation while managing risks through careful planning and mitigation strategies.

The proposed changes will significantly enhance MALMAS's accessibility, maintainability, and extensibility. By adopting sklearn conventions and modern Python packaging standards, the project will reach a wider audience of ML practitioners and enable seamless integration into existing workflows. The modular architecture supports future development while maintaining backward compatibility for existing users who migrate to the new package structure.

Successful execution of this roadmap will establish MALMAS as a professional-grade tool for automated feature engineering, bridging the gap between innovative research methodology and practical, production-ready software. The investment in infrastructure, documentation, and quality assurance will pay dividends through increased adoption, community contributions, and long-term sustainability.

---

## Appendix A: Implementation Priority Matrix

| Priority | Task | Impact | Effort | Phase |
|----------|------|--------|--------|-------|
| Critical | Create pyproject.toml and package structure | High | Low | 1 |
| Critical | Replace global_config with dataclass | High | Medium | 1 |
| Critical | Implement sklearn-compatible API | High | High | 2 |
| High | Remove sys.path manipulation | Medium | Low | 1 |
| High | Add input validation | High | Medium | 2 |
| High | Create custom exception hierarchy | Medium | Low | 2 |
| High | Abstract LLM client | Medium | Medium | 2 |
| Medium | Add comprehensive type hints | Medium | Medium | 3 |
| Medium | Write unit tests | High | High | 3 |
| Medium | Create documentation | High | Medium | 4 |
| Medium | Implement sandboxed exec | Medium | Medium | 3 |
| Low | Add progress reporting | Low | Low | 3 |
| Low | Create examples gallery | Medium | Medium | 4 |

---

## Appendix B: Dependency Requirements

### Core Dependencies

| Package | Version Requirement | Purpose |
|---------|---------------------|---------|
| openai | >=1.0.0 | LLM API client |
| pandas | >=1.5.0 | Data manipulation |
| numpy | >=1.21.0 | Numerical operations |
| scikit-learn | >=1.0.0 | ML utilities and base classes |
| xgboost | >=1.6.0 | Gradient boosting models |
| tqdm | >=4.60.0 | Progress reporting |
| pydantic | >=2.0.0 | Data validation |

### Optional Dependencies (Baselines)

| Package | Purpose |
|---------|---------|
| featuretools | DFS baseline implementation |
| autofeat | AutoFeat baseline implementation |
| openfe | OpenFE baseline implementation |
| caafe | CAAFE baseline implementation |
| lightgbm | Alternative boosting model |
| catboost | Alternative boosting model |

### Development Dependencies

| Package | Purpose |
|---------|---------|
| pytest | Testing framework |
| pytest-cov | Coverage reporting |
| mypy | Type checking |
| black | Code formatting |
| isort | Import sorting |
| sphinx | Documentation generation |

### pyproject.toml Example

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "malmas"
version = "1.0.0"
description = "Memory-Augmented LLM-based Multi-Agent Feature Engineering System"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.9"
authors = [
    {name = "MINE-USTC", email = "mine@ustc.edu.cn"}
]
keywords = ["feature-engineering", "llm", "machine-learning", "automl"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]
dependencies = [
    "openai>=1.0.0",
    "pandas>=1.5.0",
    "numpy>=1.21.0",
    "scikit-learn>=1.0.0",
    "xgboost>=1.6.0",
    "tqdm>=4.60.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "mypy>=1.0.0",
    "black>=23.0.0",
    "isort>=5.0.0",
]
baselines = [
    "featuretools>=1.0.0",
    "autofeat>=0.1.0",
    "openfe>=0.1.0",
    "caafe>=0.1.0",
]

[project.urls]
Homepage = "https://github.com/MINE-USTC/MALMAS"
Documentation = "https://malmas.readthedocs.io"
Repository = "https://github.com/MINE-USTC/MALMAS"

[tool.setuptools.packages.find]
where = ["."]
include = ["malmas*"]

[tool.black]
line-length = 88
target-version = ['py39']

[tool.isort]
profile = "black"

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_ignores = true
