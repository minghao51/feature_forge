# Setting Up Your LLM API Key

The project uses [dotenvx](https://dotenvx.com) for encrypted secrets and `pydantic-settings` for configuration.

## Current default backend: OpenCode Go (hy3)

`config/settings.yaml` points the OpenAI-compatible client at the OpenCode Go
subscription gateway:

```yaml
llm:
  provider: "openai"
  model: "hy3"                     # any id from https://opencode.ai/zen/go/v1/models
  base_url: "https://opencode.ai/zen/go/v1"
```

- **Key**: your OpenCode Go API key (also stored in opencode's own
  `~/.local/share/opencode/auth.json` under `opencode-go`). It is set in
  `.env` as `FF_LLM__API_KEY`.
- **Model selection**: override per run with `FF_LLM__MODEL=gpt-5.6-luna`
  (or edit `settings.yaml`). Browse the pool:
  `curl -s https://opencode.ai/zen/go/v1/models | jq -r '.data[].id'`.
  Note some Go models expose Anthropic Messages / OpenAI Responses instead of
  chat completions — those need `FF_LLM__PROVIDER=anthropic` (or `litellm`);
  `hy3`, `grok-4.6`, `gpt-5.6-luna` work with `provider: openai`.
- **Reasoning models**: `hy3` emits `reasoning_content` that counts toward
  `max_tokens` — keep per-call budgets generous (agent 8192, codegen 16384).
- **Usage caps**: the Go subscription is value-capped (~$12/5h, $30/week,
  $60/month). Enforced DiskCache makes repeated prompts free, but full
  experiment matrices belong on pay-as-you-go (e.g. direct DeepSeek API).

To switch back to DeepSeek direct: `provider: "deepseek"`,
`model: "deepseek-chat"`, `base_url: "https://api.deepseek.com"`, and set
`FF_LLM__API_KEY` to the DeepSeek key.

## Where the API Key is Read From

`src/feature_forge/config.py` loads `FF_LLM__API_KEY` via:
- Environment variables (highest priority)
- `.env` file (local, gitignored, plaintext — loaded by pydantic-settings)
- `config/settings.yaml` (plaintext defaults, no secrets)

Real environment variables override `.env`, which overrides `settings.yaml`.

> **History:** `.env` was previously committed with dotenvx-encrypted values.
> The `.env.keys` private key was lost in a machine migration (2026-09-05),
> so the ciphertexts were undecryptable and dotenvx was dropped. `.env` is now
> a local-only plaintext file, gitignored and never committed. The old
> ciphertext still in git history is inert (the private key no longer exists
> anywhere, so it is lost rather than leaked).

## Method 1: Local `.env` File (Recommended)

```bash
cp .env.example .env
# edit .env:
# FF_LLM__API_KEY=sk-your-key-here
```

The file is loaded automatically by `feature_forge.config.Settings`
(`env_file=".env"`). Verify:

```bash
uv run python -c "from feature_forge.config import get_settings; print(get_settings().llm.api_key)"
```

## Method 2: Quick One-Off (No File)

Set the env var directly before any command (overrides `.env`):

```bash
export FF_LLM__API_KEY="sk-your-key-here"
uv run quarto render notebooks/
```

## Method 4: CI / GitHub Actions

In CI, set the env var as a repository secret and reference it in the workflow:

```yaml
env:
  FF_LLM__API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
```

## Verifying It Works

```python
from feature_forge.config import get_settings
settings = get_settings()
print(settings.llm.model)   # "hy3" (default backend)
print(settings.llm.api_key) # SecretStr('**********')
```

## For Notebook Rendering

When rendering `.qmd` notebooks with real LLM calls, Quarto needs the env var available. Use one of:

```bash
# Option A: export first
export FF_LLM__API_KEY=sk-...
uv run quarto render notebooks/

# Option B: inline
FF_LLM__API_KEY=sk-... uv run quarto render notebooks/

# Option C: from your local .env (loaded automatically by pydantic-settings
# when the code runs; for shell-level tools, source it first)
set -a; source .env; set +a
uv run quarto render notebooks/
```

**Note:** If the API key is missing, notebooks with `#| error: true` will fail gracefully and show the error in the rendered output instead of crashing the render.
