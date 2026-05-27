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


class _KerasWrapper:
    """Container lazy: guarda params em build(); constrói e treina a rede em fit()."""

    def __init__(self, problem_type: str, params: dict, random_state: int) -> None:
        self._problem_type = problem_type
        self._params       = params
        self._random_state = random_state
        self._model        = None

    def _build_and_train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val:   pd.DataFrame | None = None,
        y_val:   pd.Series    | None = None,
    ) -> None:
        import tensorflow as tf

        tf.random.set_seed(self._random_state)
        p          = self._params
        input_dim  = X_train.shape[1]
        n_layers   = p["n_layers"]
        hidden_dim = p["hidden_dim"]
        dropout    = p["dropout"]
        activation = p["activation"]
        lr         = p["learning_rate"]
        epochs     = p["epochs"]
        batch_size = p["batch_size"]
        patience   = p["patience"]

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(input_dim,)))
        for _ in range(n_layers):
            model.add(tf.keras.layers.Dense(hidden_dim, activation=activation))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(dropout))

        if self._problem_type == "classification":
            model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
            model.compile(
                optimizer=tf.keras.optimizers.Adam(lr),
                loss="binary_crossentropy",
            )
        else:
            model.add(tf.keras.layers.Dense(1))
            model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")

        monitor  = "val_loss" if X_val is not None else "loss"
        val_data = (self._to_numpy(X_val), self._to_numpy(y_val)) if X_val is not None else None

        model.fit(
            self._to_numpy(X_train), self._to_numpy(y_train),
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor=monitor, patience=patience, restore_best_weights=True,
                )
            ],
            verbose=0,
        )
        self._model = model

    def predict(self, X: pd.DataFrame):
        import numpy as np
        preds = self._model.predict(self._to_numpy(X), verbose=0).squeeze()
        if self._problem_type == "classification":
            return (preds >= 0.5).astype(int)
        return preds

    def predict_proba(self, X: pd.DataFrame):
        import numpy as np
        p = self._model.predict(self._to_numpy(X), verbose=0).squeeze()
        return np.column_stack([1 - p, p])

    @staticmethod
    def _to_numpy(arr):
        return arr.values if hasattr(arr, "values") else arr

    def __getstate__(self) -> dict:
        import numpy as np
        state = {k: v for k, v in self.__dict__.items() if k != "_model"}
        if self._model is not None:
            state["_model_json"]    = self._model.to_json()
            state["_model_weights"] = [w.tolist() for w in self._model.get_weights()]
        return state

    def __setstate__(self, state: dict) -> None:
        import numpy as np, tensorflow as tf
        model_json    = state.pop("_model_json", None)
        model_weights = state.pop("_model_weights", None)
        self.__dict__.update(state)
        if model_json is not None:
            self._model = tf.keras.models.model_from_json(model_json)
            self._model.set_weights([np.array(w) for w in model_weights])
        else:
            self._model = None


class KerasAdapter:
    """Adapter Keras/TensorFlow para OptunaTuner."""

    def __init__(self, config) -> None:
        self._config = config

    def build(self, params: dict, random_state: int) -> _KerasWrapper:
        return _KerasWrapper(self._config.problem_type, params, random_state)

    def fit(self, model: _KerasWrapper, X_train, y_train, X_val=None, y_val=None) -> _KerasWrapper:
        model._build_and_train(X_train, y_train, X_val, y_val)
        return model

    def needs_cat_features(self) -> bool:
        return False

    def cat_features(self) -> list[str]:
        return []
