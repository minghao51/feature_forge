# Data Strategy: Kaggle-First

## Philosophy

Start with **simple, single-table tabular datasets** from Kaggle to reproduce and optimize MALMAS results. Graduate to **multi-table relational datasets** in a future phase once the core system is stable.

## Why Kaggle?

| Advantage | Explanation |
|-----------|-------------|
| **Real-world complexity** | Actual competition data with realistic noise, missing values, categorical features |
| **Clear evaluation** | Public leaderboards provide ground-truth performance targets |
| **Rich metadata** | Dataset descriptions, discussion insights, proven solutions |
| **Multi-table path** | Many competitions have relational data (e.g., Home Credit, Porto Seguro) |
| **API access** | `kagglehub` library enables programmatic download |

## Phase 1 Datasets (Single-Table, Lower Risk)

| Dataset | Kaggle Competition | Rows | Features | Task | Why Include |
|---------|-------------------|------|----------|------|-------------|
| **Titanic** | `titanic` | 891 | 12 | Binary Classification | Hello-world, fast iteration |
| **House Prices** | `house-prices-advanced-regression-techniques` | 1,460 | 81 | Regression | Mixed types, missing values |
| **Porto Seguro** | `porto-seguro-safe-driver-prediction` | 595K | 57 | Binary Classification | Large scale, anonymized features |
| **Santander** | `santander-customer-transaction-prediction` | 200K | 200 | Binary Classification | High dimensionality |
| **California Housing** | N/A (sklearn) | 20K | 8 | Regression | Baseline sanity check |

## Phase 2 Datasets (Multi-Table, Higher Complexity) — Future Work

| Dataset | Competition | Tables | Challenge |
|---------|------------|--------|-----------|
| **Home Credit** | `home-credit-default-risk` | 7 | Relational joins, feature aggregation |
| **IEEE-CIS Fraud** | `ieee-fraud-detection` | 4 | Transaction + identity tables |
| **Recruit Restaurant** | `recruit-restaurant-visitor-forecasting` | 5 | Time-series + relational |

## Data Ingestion Architecture

```python
# src/feature_forge/data/ingestion.py
from abc import ABC, abstractmethod
import pandas as pd

class DatasetFetcher(ABC):
    """Abstract base for dataset fetchers."""

    @abstractmethod
    def fetch(self, name: str, save_dir: str = "data/raw") -> dict:
        """Download dataset and return paths + metadata.

        Returns:
            dict with keys: train_path, test_path, target_column,
                           description, task_type
        """
        pass

class KaggleFetcher(DatasetFetcher):
    """Fetch datasets from Kaggle using kagglehub."""

    def fetch(self, name: str, save_dir: str = "data/raw") -> dict:
        import kagglehub
        path = kagglehub.dataset_download(name)
        # Parse competition structure
        # Return standardized metadata
        return {...}

class OpenMLFetcher(DatasetFetcher):
    """Fetch datasets from OpenML."""

    def fetch(self, name: str, save_dir: str = "data/raw") -> dict:
        from sklearn.datasets import fetch_openml
        # Download and save locally
        return {...}

class LocalFetcher(DatasetFetcher):
    """Load from local files."""

    def fetch(self, name: str, save_dir: str = "data/raw") -> dict:
        # Load from data/raw/{name}/
        return {...}
```

## Dataset Registry

```python
# src/feature_forge/data/registry.py
class DatasetRegistry:
    """Built-in registry of known datasets."""

    DATASETS = {
        "titanic": {
            "source": "kaggle",
            "competition": "titanic",
            "task": "classification",
            "target": "Survived",
            "description": "Predict survival on the Titanic",
        },
        "house-prices": {
            "source": "kaggle",
            "competition": "house-prices-advanced-regression-techniques",
            "task": "regression",
            "target": "SalePrice",
            "description": "Predict house sale prices",
        },
        # ... etc
    }

    @classmethod
    def list(cls) -> list[str]:
        return list(cls.DATASETS.keys())

    @classmethod
    def get(cls, name: str) -> dict:
        return cls.DATASETS[name]
```

## Sample Datasets

Small samples (<1MB) committed to repo for quick testing:

```
data/samples/
├── titanic_sample.csv          # 100 rows
├── house_prices_sample.csv     # 100 rows
└── california_housing_sample.csv # 100 rows
```

**Usage:**
```python
from feature_forge.data.registry import DatasetRegistry
from feature_forge.data.loader import DatasetLoader

# Quick test with sample (no internet)
loader = DatasetLoader(use_sample=True)
df = loader.load("titanic")

# Full dataset (downloads if needed)
loader = DatasetLoader(use_sample=False)
df = loader.load("titanic")  # Fetches from Kaggle
```

## Data Metadata Format

Each dataset has a `metadata.json`:

```json
{
  "name": "titanic",
  "source": "kaggle",
  "competition": "titanic",
  "task": "classification",
  "target_column": "Survived",
  "description": "Predict survival on the Titanic",
  "columns": [
    {"name": "PassengerId", "type": "numeric", "role": "id"},
    {"name": "Survived", "type": "categorical", "role": "target"},
    {"name": "Pclass", "type": "categorical", "role": "feature"},
    {"name": "Name", "type": "text", "role": "feature"},
    {"name": "Sex", "type": "categorical", "role": "feature"},
    {"name": "Age", "type": "numeric", "role": "feature", "missing": true},
    {"name": "SibSp", "type": "numeric", "role": "feature"},
    {"name": "Parch", "type": "numeric", "role": "feature"},
    {"name": "Ticket", "type": "text", "role": "feature"},
    {"name": "Fare", "type": "numeric", "role": "feature"},
    {"name": "Cabin", "type": "categorical", "role": "feature", "missing": true},
    {"name": "Embarked", "type": "categorical", "role": "feature", "missing": true}
  ],
  "num_samples": 891,
  "num_features": 11,
  "missing_values": true,
  "has_text": true
}
```

## Kaggle API Setup

Users need to configure Kaggle credentials:

```bash
# Install Kaggle CLI
pip install kaggle

# Download API token from kaggle.com/account
# Save to ~/.kaggle/kaggle.json

# Or use environment variables
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key
```

**In feature_forge:**
```python
# Kaggle credentials loaded from env vars or ~/.kaggle/kaggle.json
# Never committed to repo
```

## Data Flow

```
User requests dataset "titanic"
    │
    ▼
DatasetRegistry.lookup("titanic") → metadata
    │
    ▼
DatasetLoader.load("titanic")
    │
    ├─→ Check data/raw/titanic/ exists?
    │   ├─→ Yes → Load from disk
    │   └─→ No → Fetch from Kaggle
    │       ├─→ kagglehub.dataset_download()
    │       ├─→ Parse train.csv, test.csv
    │       ├─→ Generate metadata.json
    │       └─→ Save to data/raw/titanic/
    │
    ▼
Return: df_train, df_test, target, metadata
```

## Future: Multi-Table Support

```python
# Future extension for Phase 2
class RelationalDataset:
    """Multi-table dataset with relationships."""

    def __init__(self, tables: dict[str, pd.DataFrame], relationships: list[dict]):
        self.tables = tables
        self.relationships = relationships

    def join(self, table1: str, table2: str, on: str) -> pd.DataFrame:
        """Join two tables on a key."""
        return pd.merge(self.tables[table1], self.tables[table2], on=on)
```

This will enable experiments on relational feature engineering (e.g., aggregation across joined tables).
