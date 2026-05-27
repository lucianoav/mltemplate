# mltemplate

Biblioteca Python instalável para pipelines de Machine Learning, com foco em competições Kaggle.

Cada competição usa a biblioteca como dependência — não como template clonado.

## Instalação

```bash
pip install -e .                                              # núcleo
pip install -e ".[kaggle,xgboost,lightgbm,catboost]"         # com extras
pip install -e ".[all]"                                       # tudo
```

**Extras disponíveis:** `kaggle`, `xgboost`, `lightgbm`, `catboost`

## Uso rápido

```python
from mltemplate.config import ProjectConfig
from mltemplate.storage import StorageManager
from mltemplate.data import KaggleSource, LocalSource, DataManager
from mltemplate.features import FeatureEngineer
from mltemplate.tuning import OptunaTuner, XGBoostAdapter
from mltemplate.ensemble import EnsembleCreator
from mltemplate import eda
from pathlib import Path

config = ProjectConfig(
    target="Survived",
    numerical_features=["Age", "Fare"],
    categorical_features=["Sex", "Embarked"],
    ignore_features=["PassengerId", "Name", "Ticket"],
)

storage = StorageManager(root=Path("."))  # raiz = diretório do notebook
dm = DataManager(storage, config)
```

## Fluxo de dados

```
KaggleSource / LocalSource
        ↓ download()
DataManager.load_raw()          → train_df, test_df
        ↓
DataManager.split()             → X_train, X_val, y_train, y_val
        ↓
FeatureEngineer                 → fit no treino, transform em treino e teste
        ↓
DataManager.save_feature_set()
        ↓
OptunaTuner.tune(adapter)       → TuningResult(model, params, score, trials)
        ↓
EnsembleCreator.fit()           → pesos otimizados
        ↓
StorageManager.save_submission()
```

## Módulos

| Módulo | Responsabilidade |
|--------|-----------------|
| `config.py` | `ProjectConfig` — dataclass com target, features, random_state |
| `storage.py` | `StorageManager` — I/O centralizado com paths derivados de `root` |
| `data/sources.py` | `KaggleSource`, `LocalSource` — abstração de fonte de dados |
| `data/manager.py` | `DataManager` — load, split, versionamento de feature sets |
| `features/engineer.py` | `FeatureEngineer` — impute, encode, scale, expand_poly |
| `tuning/adapters.py` | `XGBoostAdapter`, `LightGBMAdapter`, `CatBoostAdapter`, `SklearnAdapter` |
| `tuning/tuner.py` | `OptunaTuner` — retorna `TuningResult` |
| `ensemble.py` | `EnsembleCreator` — pesos por SLSQP com CV |
| `eda.py` | Funções de análise exploratória — retornam DataFrames ou `Figure` |

## Adicionando um novo framework

```python
# src/mltemplate/tuning/adapters.py
class MyCustomAdapter:
    def build(self, params: dict, random_state: int): ...
    def fit(self, model, X_train, y_train, X_val=None, y_val=None): ...
    def needs_cat_features(self) -> bool: return False
    def cat_features(self) -> list[str]: return []
```

## Requisitos

- Python >= 3.10
- pandas, numpy, scikit-learn, optuna, scipy, matplotlib, seaborn, joblib, tqdm
