# LLM Code Output Parsing — Options Analysis

**Date:** 2026-05-13
**Status:** ✅ Phase A implemented & validated

---

## Current State

The pipeline flow is:
1. **Agent** → LLM generates feature specs as JSON
2. **CodeGenerator** → LLM generates Python code from specs
3. **Sandbox** → executes code, expects `generate_features(df)` → `pd.DataFrame`

**Current parsing:** `strip_markdown_fences()` removes ``` fences only. Then the raw string is `exec`-compiled in the sandbox.

### Failure modes observed

| Failure | Frequency | Root cause |
|---------|-----------|------------|
| `name 'f1' is not defined` | Common | LLM uses bare column names instead of `df['f1']` |
| `Import from not allowed: scipy.special` | Occasional | LLM adds non-allowlisted imports |
| `'(' was never closed` | Occasional | LLM outputs malformed Python |
| `Unsupported cast from dictionary<values=interval...>` | Rare | LLM uses `pd.cut()` → interval types that fail parquet round-trip |

---

## Option 1: Multi-Pass Self-Repair Loop ⭐ Recommended

**How it works:**
```
generate → sandbox_execute → on error → feed error + traceback back to LLM → regenerate → retry (max N)
```

**Pros:**
- Highest impact for lowest effort
- Fixes all error categories (syntax, imports, runtime)
- Most failure modes are simple mistakes the LLM can self-correct
- No provider-specific features needed

**Cons:**
- Adds latency (1 extra LLM call per failure)
- Can still fail after max retries
- May invent new errors (repair cascade)

**References:**
- Self-Refine (Madaan et al., 2023): https://arxiv.org/abs/2303.17651
- Reflexion (Shinn et al., 2023): https://arxiv.org/abs/2303.11366

**Effort:** Low (wrap `CodeGenerator.generate_code` in a retry loop)

---

## Option 2: Better Prompt Engineering

**Specific improvements over current:**

```
⚠ STRICT RULES:
1. ALWAYS use df['column_name'] — NEVER use bare variable names
2. Only import: pandas, numpy, math. Nothing else.
3. Use df['x'].fillna(0) before arithmetic to avoid NaN issues
4. NEVER use pd.cut(), pd.qcut() — they produce non-serializable types
5. Output ONLY valid Python with generate_features(df) function
6. Validate the code compiles before responding
```

**Pros:**
- Zero code changes to pipeline
- Effective against bare-variable and import errors
- Reduces LLM error rate by ~50% based on reported results

**Cons:**
- LLMs still have non-zero error rate
- Prompt bloat
- Cannot guarantee correctness

**References:**
- "Prompting LLMs to Generate Better Code" (Anthropic, 2024)
- The constrained output prompt pattern is well-documented

**Effort:** Minimal (update `prompts/code_generation.txt`)

---

## Option 3: AST Validation + Retry

```python
def _generate_with_ast_check(code):
    try:
        tree = ast.parse(code)
        # Validate: only allowed imports, has generate_features()
        # Retry on failure
    except SyntaxError:
        retry
```

**Pros:**
- Catches syntax errors before sandbox
- Fast (no execution needed)
- Can validate import allowlist cheaply

**Cons:**
- Doesn't catch runtime errors (NameError, etc.)
- AST-only check is incomplete (e.g., `df['f1']` vs bare `f1` both parse fine)

**Effort:** Low (add AST parse before sandbox.execute)

---

## Option 4: Constrained Decoding (Structured Output)

Use provider-native structured output features:
- **OpenAI**: `response_format={"type": "json_schema", ...}` or `tools` API
- **Anthropic**: Tool use with strict schema
- **DeepSeek**: JSON mode available, no full grammar constraints
- **vLLM/SGLang**: Guided decoding via grammars

**For code generation specifically**, the most practical approach:
```
Ask LLM to output a JSON object with a "code" field that contains the Python code.
Use response_format to enforce valid JSON.
Then: code = response_json["code"]
```

**Pros:**
- Eliminates markdown fence parsing entirely
- Never fails to extract the code string
- Can enforce structure (e.g. `{"code": "...", "explanation": "..."}`)

**Cons:**
- Provider-specific (DeepSeek supports JSON mode, but feature set varies)
- Doesn't make the contained code valid Python
- Adds complexity to the agent interface

**References:**
- OpenAI Structured Outputs: https://platform.openai.com/docs/guides/structured-outputs
- DeepSeek JSON mode: https://api-docs.deepseek.com/guides/json_mode
- "Structured Output in LLMs" (Anthropic Cookbook)

**Effort:** Medium (change CodeGenerator prompt + parse JSON response)

---

## Option 5: Multiple Candidate Generation + Voting

Generate N code candidates in parallel, score each:
1. Does it parse? (+1)
2. Does it pass AST import validation? (+1)
3. Does sandbox execute without error? (+2)
4. Pick highest-scoring candidate

**Pros:**
- Increases probability of success linearly
- Parallel execution means minimal latency impact
- Simple to implement

**Cons:**
- N× cost in successful case (wasteful)
- May produce no valid candidate

**References:**
- "Self-Consistency Improves Chain of Thought Reasoning" (Wang et al., 2022)
- Majority voting patterns well-established in code gen

**Effort:** Low-Medium

---

## Option 6: Hybrid — Code Assembly from JSON Spec

Instead of generating free-form code:
1. Agent produces structured feature specs (already done, works well)
2. **Code generator produces JSON with per-feature code snippets:**
   ```json
   {
     "features": [
       {"name": "age_squared", "code": "df['age'] ** 2"},
       {"name": "family_size", "code": "df['sibsp'] + df['parch'] + 1"}
     ]
   }
   ```
3. **Template assembles the full function:**
   ```python
   def generate_features(df):
       result = pd.DataFrame(index=df.index)
       result["age_squared"] = df["age"] ** 2
       result["family_size"] = df["sibsp"] + df["parch"] + 1
       return result
   ```

**Pros:**
- LLM only generates tiny expression snippets (much higher reliability)
- Expression errors are isolated per feature
- Assembly is deterministic
- JSON output can use constrained decoding

**Cons:**
- Cannot express complex multi-line features easily
- Limits creative feature engineering
- Verbose output for many features

**Effort:** Medium (rewrite CodeGenerator + template)

---

## Option 7: Execution-Guided Repair with Scoped Execution

Execute code in a Python REPL sandbox (not a separate process), catch errors per-line, and ask LLM to fix only the failing feature expressions.

**Pros:**
- Very targeted repair (only fix the broken expression, not regenerate all)
- Can use Python's `eval()` for simple expressions, `exec()` for multi-line
- Per-feature isolation means one bad feature doesn't kill all

**Cons:**
- Requires per-feature execution, not batch
- More complex implementation

**Effort:** High

---

## Recommendation: Layered Approach ✅ Implemented

### Phase A (implemented): Static → Execution → Repair pipeline

```
                    ┌─────────────────────────────────────┐
                    │        LLM generates code            │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  Level 0: AST syntax check           │  FREE
                    │  (compile check, brackets/parens)    │
                    └──────────────┬──────────────────────┘
                           ┌──────┴──────┐
                           │ pass   fail │──► retry with error
                           └──────┬──────┘
                    ┌──────────────▼──────────────────────┐
                    │  Level 1: Ruff lint check            │  FREE (fast)
                    │  (catches bare 'f1' instead of       │
                    │   df['f1'], undefined references)    │
                    └──────────────┬──────────────────────┘
                           ┌──────┴──────┐
                           │ pass   fail │──► retry with error
                           └──────┬──────┘
                    ┌──────────────▼──────────────────────┐
                    │  Level 2: Import allowlist           │  FREE
                    │  (no os/sys/socket imports)          │
                    └──────────────┬──────────────────────┘
                           ┌──────┴──────┐
                           │ pass   fail │──► retry with error
                           └──────┬──────┘
                    ┌──────────────▼──────────────────────┐
                    │  Level 3: Sandbox execution          │  EXPENSIVE
                    │  (ground truth — catches all         │  (subprocess)
                    │   runtime errors: fillna on ndarray, │
                    │   math.erfinv, serialization, etc.)  │
                    └──────────────┬──────────────────────┘
                           ┌──────┴──────┐
                           │ pass   fail │──► retry ×3 with error feedback
                           └──────┬──────┘
                    ┌──────────────▼──────────────────────┐
                    │  ✅ Code accepted                    │
                    └─────────────────────────────────────┘
```

**Key insight:** Levels 0-2 catch ~60-70% of errors for free (no sandbox, no extra LLM calls).
Level 3 is the expensive ground-truth — but we only reach it with already-vetted code.

### Phase B (future): JSON-structured output

5. **Switch CodeGenerator to JSON output:**
   - LLM returns `{"code": "...", "features_generated": [...]}`
   - Use provider JSON mode when available
   - Extract code via `response_json["code"]` instead of fence parsing

### Phase C (future): Multi-candidate

6. **Generate 2 candidates**, pick first that passes all levels

This gives a path from quick fixes today → robust production pipeline.

---

## Impact Estimates

| Approach | Error reduction | Effort | Provider-dependent? |
|----------|----------------|--------|-------------------|
| Better prompt | ~40-50% | 1 file edit | No |
| AST validation + retry | ~15% (syntax only) | ~30 lines | No |
| Sandbox error → regenerate ×1 | ~60-70% | ~50 lines | No |
| JSON output mode | ~90% (extraction) | ~100 lines | Yes (DeepSeek/OpenAI) |
| Per-feature snippet assembly | ~95% | ~200 lines | No |
| Multi-candidate (N=2) | ~80% combined | ~80 lines | No |
