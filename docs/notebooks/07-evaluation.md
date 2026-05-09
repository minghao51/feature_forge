# Evaluation & Sandboxed Execution

Cross-validation feature evaluation, sandboxed code execution, and model factory.

## Notebook

<div style="margin: 0 -0.8rem">
  <iframe src="/feature-forge/notebooks/html/07_evaluation.html"
    style="width:100%; height:700px; border:1px solid var(--md-default-fg-color--lightest); border-radius:4px;"
    loading="lazy"></iframe>
</div>

## Run Locally

```bash
# Ensure your DeepSeek API key is set
export FF_LLM__API_KEY=sk-...

# Render this notebook
uv run quarto render notebooks/07_evaluation.qmd

# Or render all notebooks
uv run quarto render notebooks/
```