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
dotenvx run -- uv run jupyter lab notebooks/
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
   dotenvx run -- uv run jupyter lab notebooks/
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

## For notebooks

The committed MkDocs notebook pages are not executed during documentation builds. For an
interactive notebook that makes real provider calls, start Jupyter with the secret available:

```bash
# Option A: export first
export FF_LLM__API_KEY=sk-...
uv run jupyter lab notebooks/

# Option B: inline
FF_LLM__API_KEY=sk-... uv run jupyter lab notebooks/

# Option C: via dotenvx
dotenvx run -- uv run jupyter lab notebooks/
```

Required documentation, replay, verification, catalog, and CI commands do not need an API key.
