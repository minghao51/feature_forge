# Baseline Selection Rationale

## Performance Ranking (2026)

Based on the MALMAS paper and comparative studies [1][2][3]:

| Rank | Method | Type | Avg AUC Gain | Why Selected |
|------|--------|------|-------------|--------------|
| 1 | **MALMAS** | LLM Multi-Agent | +0.05 to +0.08 | Our core implementation |
| 2 | **LLM-FE** | LLM Evolutionary | +0.04 to +0.06 | Strongest non-MALMAS LLM method |
| 3 | **CAAFE** | Context-Aware LLM | +0.03 to +0.05 | Generates features from descriptions |
| 4 | **OpenFE** | Traditional (Non-LLM) | +0.02 to +0.04 | Strongest non-LLM, "expert-level" |
| 5 | OCTree | Tree-based | +0.01 to +0.03 | Mid-tier, often outperformed |
| 6 | AutoFeat | Classical | +0.01 to +0.02 | Foundational but limited |
| 7 | DFS | Classical | ~0.00 to +0.01 | Basic relational synthesis |

## Selected Baselines for Phase 1

We implement **MALMAS + 3 baselines**:

### 1. OpenFE (Highest Priority Non-LLM Baseline)

**Why:** 
- Strongest non-LLM method, often beats 99% of Kaggle teams
- "Feature boosting" algorithm with two-stage pruning
- Fast, deterministic, no API costs
- Serves as the "floor" for LLM methods to beat

**Implementation:**
```python
class OpenFEBaseline(Baseline):
    def fit(self, X_train, y_train):
        self.openfe = OpenFE()
        self.openfe.fit(data=X_train, label=y_train, n_jobs=4)
        return self
    
    def transform(self, X):
        return self.openfe.transform(X)
```

### 2. CAAFE (Context-Aware LLM Baseline)

**Why:**
- Generates features directly from dataset descriptions
- Uses LLM but with simpler single-agent architecture
- Good comparison for "multi-agent vs single-agent"

**Implementation:**
```python
class CAAFEBaseline(Baseline):
    def fit(self, X_train, y_train, description: str):
        self.caafe = CAAFEClassifier(
            iterations=10,
            dataset_description=description,
        )
        self.caafe.fit(X_train, y_train)
        return self
    
    def transform(self, X):
        return self.caafe.transform(X)
```

### 3. LLM-FE (Evolutionary LLM Baseline)

**Why:**
- Evolutionary search + data-driven feedback
- Consistently outperforms OCTree and classical methods
- Good comparison for "memory-augmented vs evolutionary"

**Implementation:**
```python
class LLMFEBaseline(Baseline):
    def fit(self, X_train, y_train):
        self.llmfe = LLMFE()
        self.llmfe.fit(X_train, y_train)
        return self
    
    def transform(self, X):
        return self.llmfe.transform(X)
```

## Excluded Baselines (Future Work)

| Baseline | Reason for Exclusion | Future Path |
|----------|---------------------|-------------|
| **OCTree** | Outperformed by LLM-FE, adds complexity for marginal gain | Phase 2 if needed |
| **AutoFeat** | Foundational but weak; easily beaten by OpenFE | Optional extra |
| **DFS** | Basic relational synthesis; not competitive on single-table | Phase 2 (multi-table) |

## Comparison Dimensions

Each baseline will be compared across:

| Dimension | How Measured |
|-----------|-------------|
| **Predictive Gain** | ΔAUC (classification) or Δ-NRMSE (regression) |
| **Feature Diversity** | Number of unique feature types generated |
| **Cost Efficiency** | LLM API cost per unit gain |
| **Latency** | Wall-clock time to generate features |
| **Robustness** | Standard deviation across random seeds |
| **Scalability** | Performance on datasets of varying size |

## Experiment Design

```python
# Compare all methods on same dataset/seed
matrix = (
    ExperimentMatrix()
    .datasets(["titanic", "house-prices", "porto-seguro"])
    .methods([
        "malmas_full",      # All agents, all memory
        "malmas_no_memory", # Ablated
        "openfe",
        "caafe",
        "llmfe",
        "baseline",         # No feature engineering
    ])
    .seeds([0, 1, 2, 3, 4])
    .models(["xgboost"])
)
```

## Expected Outcomes

Based on 2026 rankings:
- MALMAS should outperform all baselines on most datasets
- OpenFE should be the strongest non-LLM competitor
- LLM-FE should be the strongest single-agent LLM method
- CAAFE should show strong gains on datasets with rich descriptions

## References

[1] MALMAS paper (April 2026). Memory-Augmented LLM-based Multi-Agent System for Automated Feature Engineering. arXiv:2604.20261.

[2] OpenFE paper. Zhang et al. (2023). OpenFE: Automated Feature Generation with Expert-level Performance. ICML.

[3] CAAFE paper. Hollmann et al. (2023). Large Language Models for Automated Feature Engineering.
