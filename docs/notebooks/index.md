# Notebooks

Interactive tutorials demonstrating Feature Forge capabilities.

| Notebook | Description |
|----------|-------------|
| [01 — Getting Started](01_getting_started.ipynb) | Configuration, data loading, evaluation (CV, metrics, sandbox), ExperimentalPlatform, and a quick FeatureForge demo. Works **offline** without an API key. |
| [02 — Pipeline Deep Dive](02_pipeline_deep_dive.ipynb) | Built-in agents, router strategies, iterative pipeline, memory system, and ablation modes. |
| [03 — Benchmarks & Artifacts](03_benchmarks_and_artifacts.ipynb) | Baseline comparison, experiment matrix, artifact schema/diff/dashboard, and experiment tracking. |

## Run Locally

```bash
# Offline (notebook 1 works fully; notebooks 2–3 skip LLM cells gracefully)
uv run jupyter lab notebooks/

# With LLM (all cells run)
dotenvx run -- uv run jupyter lab notebooks/
```
