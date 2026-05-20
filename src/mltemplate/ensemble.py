from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, minimize
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from mltemplate.config import ProjectConfig

logger = logging.getLogger(__name__)

_METRICS = {
    "roc_auc": roc_auc_score,
    "accuracy": accuracy_score,
}


class EnsembleCreator:
    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self._weights: np.ndarray | None = None

    @property
    def weights(self) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("Chame fit() antes de acessar os pesos.")
        return self._weights

    def fit(
        self,
        models: list,
        X_list: list[pd.DataFrame],
        y: pd.Series,
        metric: str = "roc_auc",
        min_weight: float = 0.1,
    ) -> EnsembleCreator:
        if metric not in _METRICS:
            raise ValueError(f"metric '{metric}' não suportado. Use: {list(_METRICS)}")

        metric_fn = _METRICS[metric]
        skf = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        fold_weights: list[np.ndarray] = []
        fold_scores: list[float] = []

        for train_idx, val_idx in skf.split(X_list[0], y):
            val_preds = np.array([
                self._get_proba(model, X.iloc[val_idx])
                for model, X in zip(models, X_list)
            ])
            y_val = y.iloc[val_idx]

            w = self._optimize_fold(val_preds, y_val, metric, metric_fn, min_weight)
            fold_weights.append(w)

            ensemble = np.average(val_preds, axis=0, weights=w)
            score = self._eval(metric, metric_fn, y_val, ensemble)
            fold_scores.append(score)

        raw_weights = np.mean(fold_weights, axis=0)
        raw_weights = np.clip(raw_weights, min_weight, 1.0)
        self._weights = raw_weights / raw_weights.sum()

        logger.info(
            "Ensemble pesos: %s | CV %.4f ± %.4f",
            np.round(self._weights, 3),
            np.mean(fold_scores),
            np.std(fold_scores),
        )
        return self

    def predict_proba(self, models: list, X_list: list[pd.DataFrame]) -> np.ndarray:
        preds = np.array([self._get_proba(m, X) for m, X in zip(models, X_list)])
        return np.average(preds, axis=0, weights=self.weights)

    def predict(self, models: list, X_list: list[pd.DataFrame], threshold: float = 0.5) -> np.ndarray:
        preds = np.array([self._get_proba(m, X) for m, X in zip(models, X_list)])
        ensemble = np.average(preds, axis=0, weights=self.weights)
        if ensemble.ndim == 1:
            return (ensemble > threshold).astype(int)
        return np.argmax(ensemble, axis=1)

    # --- privados ---

    def _get_proba(self, model, X: pd.DataFrame) -> np.ndarray:
        proba = model.predict_proba(X)
        return proba[:, 1] if proba.shape[1] == 2 else proba

    def _optimize_fold(self, preds, y_true, metric, metric_fn, min_weight) -> np.ndarray:
        n = len(preds)
        init = np.ones(n) / n

        def objective(w):
            w = np.clip(w, min_weight, 1.0)
            w = w / w.sum()
            ensemble = np.average(preds, axis=0, weights=w)
            return -self._eval(metric, metric_fn, y_true, ensemble)

        result = minimize(
            objective, init,
            method="SLSQP",
            bounds=Bounds([min_weight] * n, [1.0] * n),
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        )
        return result.x

    def _eval(self, metric, metric_fn, y_true, ensemble) -> float:
        if metric == "roc_auc":
            return metric_fn(y_true, ensemble)
        labels = (ensemble > 0.5).astype(int) if ensemble.ndim == 1 else np.argmax(ensemble, axis=1)
        return metric_fn(y_true, labels)
