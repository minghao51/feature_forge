# Method Comparison

Head-to-head benchmark of MALMAS, CAAFE, LLM-FE, and Malmus on the same dataset with side-by-side metrics.

## Notebook

<div style="margin: 0 -0.8rem">
  <iframe src="/feature_forge/notebooks/html/10_method_comparison.html"
    style="width:100%; height:700px; border:1px solid var(--md-default-fg-color--lightest); border-radius:4px;"
    loading="lazy"></iframe>
</div>

## Run Locally

```bash
# Ensure your DeepSeek API key is set
export FF_LLM__API_KEY=sk-...

# Render this notebook
uv run quarto render notebooks/10_method_comparison.qmd

# Or render all notebooks
uv run quarto render notebooks/
```