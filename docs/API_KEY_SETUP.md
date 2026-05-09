# Setting Up Your DeepSeek API Key

The project uses [dotenvx](https://dotenvx.com) for encrypted secrets and `pydantic-settings` for configuration.

## Where the API Key is Read From

`src/feature_forge/config.py` loads `FF_LLM__API_KEY` via:
- Environment variables (highest priority)
- `.env` file (managed by dotenvx)
- `config/settings.yaml` (plaintext defaults, no secrets)

## Method 1: Quick One-Off (No Encryption)

Set the env var directly before any command:

```bash
export FF_LLM__API_KEY="sk-your-deepseek-key-here"
uv run quarto render notebooks/
uv run python -c "from feature_forge.config import get_settings; print(get_settings().llm.api_key)"
```

## Method 2: Update the Encrypted .env File

1. **Decrypt** (requires the private key from `.env.keys`):
   ```bash
   dotenvx decrypt
   ```

2. **Edit** `.env` — replace the encrypted value with your plaintext key:
   ```bash
   # Replace this line:
   # FF_LLM__API_KEY=encrypted:...
   # With:
   FF_LLM__API_KEY=sk-your-deepseek-key-here
   ```

3. **Re-encrypt**:
   ```bash
   dotenvx encrypt
   ```

4. **Run with decrypted vars**:
   ```bash
   dotenvx run -- uv run quarto render notebooks/
   ```

## Method 3: dotenvx Set (One-Liner)

```bash
# Set + encrypt in one step
dotenvx set FF_LLM__API_KEY sk-your-deepseek-key-here
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
print(settings.llm.model)   # "deepseek-chat"
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

# Option C: via dotenvx
dotenvx run -- uv run quarto render notebooks/
```

**Note:** If the API key is missing, notebooks with `#| error: true` will fail gracefully and show the error in the rendered output instead of crashing the render.
