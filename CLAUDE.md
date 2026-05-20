# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`mltemplate` é uma biblioteca Python instalável para pipelines de Machine Learning (foco em competições Kaggle). Cada competição usa a biblioteca como dependência — não como template clonado.

## Installation

```bash
pip install -e .                              # núcleo
pip install -e ".[kaggle,xgboost,lightgbm,catboost]"  # com extras
```

## Usage per Competition

```python
from mltemplate.config import ProjectConfig
from mltemplate.storage import StorageManager
from mltemplate.data import KaggleSource, LocalSource, DataManager
from mltemplate.features import FeatureEngineer
from mltemplate.tuning import OptunaTuner, XGBoostAdapter, TuningResult
from mltemplate.ensemble import EnsembleCreator
from mltemplate import eda
from pathlib import Path

config = ProjectConfig(
    target="Survived",
    numerical_features=["Age", "Fare"],
    categorical_features=["Sex", "Embarked"],
    ignore_features=["PassengerId", "Name", "Ticket"],
)
storage = StorageManager(root=Path("."))   # raiz = diretório do notebook
dm = DataManager(storage, config)
```

## Architecture

### Core Design Principles

**Sem globais:** nenhum módulo faz `from config import *`. Todo módulo recebe `config` e `storage` via `__init__`. Dois `FeatureEngineer` com configs distintas podem coexistir na mesma sessão.

**Tuners não salvam arquivos:** `OptunaTuner.tune()` retorna `TuningResult(model, params, score, trials)`. O notebook decide se e onde salvar via `storage.save_model()` e `storage.save_metrics()`.

**Adaptadores substituem dispatch por string:** adicionar suporte a um novo framework = criar uma classe que implemente `ModelAdapter` (Protocol), sem tocar em `OptunaTuner`.

### Module Map

| Módulo | Classe/Funções | Responsabilidade |
|--------|---------------|-----------------|
| `config.py` | `ProjectConfig` | Dataclass com target, features, random_state, etc. |
| `storage.py` | `StorageManager` | I/O centralizado; paths derivados de `root` |
| `data/sources.py` | `KaggleSource`, `LocalSource` | Abstração de fonte (Protocol `DataSource`) |
| `data/manager.py` | `DataManager` | load_raw, split, save/load feature sets versionados |
| `features/engineer.py` | `FeatureEngineer` | drop, impute, encode, scale, expand_poly (fit só no treino) |
| `tuning/adapters.py` | `XGBoostAdapter`, `LightGBMAdapter`, `CatBoostAdapter`, `SklearnAdapter` | Encapsulam early stopping, verbosity, cat_features por framework |
| `tuning/tuner.py` | `OptunaTuner`, `GridTuner`, `RandomTuner` | Otimização; retornam `TuningResult` |
| `ensemble.py` | `EnsembleCreator` | Pesos por SLSQP com CV; `fit()` → `predict_proba()` / `predict()` |
| `eda.py` | funções | Retornam DataFrames ou `plt.Figure` — sem `display()`, sem `print()` |

### Data Flow

```
KaggleSource/LocalSource
        ↓ download()
DataManager.load_raw()      → train_df, test_df
        ↓
DataManager.split()         → X_train, X_val, y_train, y_val
        ↓
FeatureEngineer             → fit no treino, transform em treino e teste
        ↓
DataManager.save_feature_set("v1_basico")
        ↓
OptunaTuner.tune(adapter)   → TuningResult
        ↓
EnsembleCreator.fit()       → pesos otimizados
        ↓
StorageManager.save_model() / save_submission()
```

### Adding a New Framework Adapter

```python
# src/mltemplate/tuning/adapters.py
class MyCustomAdapter:
    def __init__(self, model_class): ...
    def build(self, params: dict, random_state: int): ...
    def fit(self, model, X_train, y_train, X_val=None, y_val=None): ...
    def needs_cat_features(self) -> bool: return False
    def cat_features(self) -> list[str]: return []
```

### StorageManager Paths (derivados de `root`)

- Raw data: `root/data/raw/`
- Processed: `root/data/processed/<nome>/`
- Models: `root/models/`
- Metrics: `root/reports/metrics/`
- Submissions: `root/submissions/`

### Scoring Suportado

`OptunaTuner` e `GridTuner`/`RandomTuner`: `"roc_auc"`, `"accuracy"`, `"rmse"`, `"mae"`

`EnsembleCreator.fit()`: `"roc_auc"`, `"accuracy"`
