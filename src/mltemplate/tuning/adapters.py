from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class ModelAdapter(Protocol):
    def build(self, params: dict, random_state: int) -> Any: ...
    def fit(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None,
        y_val: pd.Series | None,
    ) -> Any: ...
    def needs_cat_features(self) -> bool: ...
    def cat_features(self) -> list[str]: ...


class XGBoostAdapter:
    def __init__(self, model_class, categorical_features: list[str] | None = None) -> None:
        self.model_class = model_class
        self._cat_features = categorical_features or []

    def build(self, params: dict, random_state: int) -> Any:
        p = {**params, "verbosity": 0, "random_state": random_state}
        # remove chaves de early stopping para o fit final (sem eval_set)
        p.pop("early_stopping_rounds", None)
        p.pop("eval_metric", None)
        return self.model_class(**p)

    def fit(self, model, X_train, y_train, X_val=None, y_val=None) -> Any:
        if X_val is not None:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train)
        return model

    def needs_cat_features(self) -> bool:
        return False

    def cat_features(self) -> list[str]:
        return []


class LightGBMAdapter:
    def __init__(self, model_class, categorical_features: list[str] | None = None) -> None:
        self.model_class = model_class
        self._cat_features = categorical_features or []

    def build(self, params: dict, random_state: int) -> Any:
        p = dict(params)
        p.pop("early_stopping_rounds", None)
        p.pop("metric", None)
        return self.model_class(**p, random_state=random_state, verbose=-1)

    def fit(self, model, X_train, y_train, X_val=None, y_val=None) -> Any:
        if X_val is not None:
            from lightgbm import early_stopping, log_evaluation
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[early_stopping(50, verbose=False), log_evaluation(period=-1)],
            )
        else:
            model.fit(X_train, y_train)
        return model

    def needs_cat_features(self) -> bool:
        return False

    def cat_features(self) -> list[str]:
        return []


class CatBoostAdapter:
    def __init__(self, model_class, categorical_features: list[str]) -> None:
        self.model_class = model_class
        self._cat_features = categorical_features

    def build(self, params: dict, random_state: int) -> Any:
        p = dict(params)
        p.pop("early_stopping_rounds", None)
        return self.model_class(
            **p,
            random_seed=random_state,
            logging_level="Silent",
            cat_features=self._cat_features,
        )

    def fit(self, model, X_train, y_train, X_val=None, y_val=None) -> Any:
        if X_val is not None:
            from catboost import Pool
            train_pool = Pool(X_train, y_train, cat_features=self._cat_features)
            val_pool = Pool(X_val, y_val, cat_features=self._cat_features)
            model.fit(train_pool, eval_set=val_pool)
        else:
            model.fit(X_train, y_train)
        return model

    def needs_cat_features(self) -> bool:
        return True

    def cat_features(self) -> list[str]:
        return self._cat_features


class SklearnAdapter:
    """Fallback genérico para qualquer modelo sklearn-compatível (RandomForest, etc.)."""

    def __init__(self, model_class, categorical_features: list[str] | None = None) -> None:
        self.model_class = model_class
        self._cat_features = categorical_features or []

    def build(self, params: dict, random_state: int) -> Any:
        return self.model_class(**params, random_state=random_state)

    def fit(self, model, X_train, y_train, X_val=None, y_val=None) -> Any:
        model.fit(X_train, y_train)
        return model

    def needs_cat_features(self) -> bool:
        return False

    def cat_features(self) -> list[str]:
        return []
