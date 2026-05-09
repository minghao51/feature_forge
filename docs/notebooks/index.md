# Notebooks

Interactive tutorials demonstrating Feature Forge capabilities.
All notebooks are rendered with [Quarto](https://quarto.org) and embedded below.

| Notebook | Description |
|----------|-------------|
| [Quick Start: Sklearn API](01-quick-start.md) | Get started with MALMASFeatureEngineer — fit, transform, and integrate into sklearn pipelines. |
| [Agents: The 6 Specialized Feature Generators](02-agents.md) | Explore all 6 built-in agents, the AgentRegistry, and how to build a custom agent. |
| [Router & Pipeline Modes](03-router.md) | Dynamic agent selection with data-driven, performance-driven, hybrid, and LLM-based router strategies. |
| [Iterative Pipeline & Memory](04-iterative-pipeline.md) | Multi-round feature engineering with procedural, feedback, and conceptual memory tiers. |
| [Experiment Matrix & Tracking](05-experiment-matrix.md) | Design Cartesian experiments, run them with ExperimentRunner, and report results. |
| [Baselines Comparison](06-baselines.md) | Run OpenFE, CAAFE, LLM-FE, and Malmus on the same data and compare artifacts. |
| [Evaluation & Sandboxed Execution](07-evaluation.md) | Cross-validation feature evaluation, sandboxed code execution, and model factory. |
| [Artifacts & Dashboard](08-artifacts.md) | Unified artifact access, comparison, diff, and offline HTML dashboards. |
| [Configuration & Tracking](09-configuration.md) | Pydantic settings, environment overrides, and experiment tracking with WandB and MLflow. |

## Prerequisites

```bash
export FF_LLM__API_KEY=sk-your-deepseek-key
uv sync --extra docs
```