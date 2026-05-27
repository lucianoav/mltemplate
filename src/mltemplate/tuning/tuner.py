from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold
from tqdm.auto import tqdm

import optuna

from mltemplate.config import ProjectConfig
from mltemplate.tuning.adapters import ModelAdapter

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)

_METRICS = {
    "roc_auc": roc_auc_score,
    "accuracy": accuracy_score,
    "rmse": mean_squared_error,
    "mae": mean_absolute_error,
}

_DIRECTION = {
    "roc_auc": "maximize",
    "accuracy": "maximize",
    "rmse": "minimize",
    "mae": "minimize",
}


def _score(model, X, y, scoring: str) -> float:
    if scoring == "roc_auc":
        return roc_auc_score(y, model.predict_proba(X)[:, 1])
    if scoring == "accuracy":
        return accuracy_score(y, model.predict(X))
    if scoring == "rmse":
        return -mean_squared_error(y, model.predict(X), squared=False)
    if scoring == "mae":
        return -mean_absolute_error(y, model.predict(X))
    raise ValueError(f"scoring '{scoring}' não suportado. Use: {list(_METRICS)}")


@dataclass
class TuningResult:
    model:      Any
    params:     dict
    score:      float
    trials:     pd.DataFrame = field(default_factory=pd.DataFrame)
    test_score: float | None = None


class OptunaTuner:
    def __init__(self, config: ProjectConfig) -> None:
        self.config = config

    def tune(
        self,
        adapter:          ModelAdapter,
        X_train:          pd.DataFrame,
        y_train:          pd.Series,
        param_space_func: Callable[[optuna.Trial], dict],
        scoring:          str,
        X_val:            pd.DataFrame | None = None,
        y_val:            pd.Series    | None = None,
        X_test:           pd.DataFrame | None = None,
        y_test:           pd.Series    | None = None,
        n_trials:         int  = 100,
        timeout:          int  = 3600,
        show_progress:    bool = True,
    ) -> TuningResult:
        if scoring not in _DIRECTION:
            raise ValueError(f"scoring '{scoring}' não suportado. Use: {list(_DIRECTION)}")

        holdout = X_val is not None and y_val is not None
        mode = "holdout" if holdout else f"CV ({self.config.cv_folds} folds)"
        logger.info("OptunaTuner — modo %s, scoring=%s, n_trials=%d", mode, scoring, n_trials)

        best_value = _DIRECTION[scoring] == "maximize" and -np.inf or np.inf
        pbar = tqdm(total=n_trials, desc="Trials") if show_progress else None

        if holdout:
            def objective(trial: optuna.Trial) -> float:
                params = param_space_func(trial)
                trial.set_user_attr("full_params", params)
                model  = adapter.build(params, self.config.random_state)
                adapter.fit(model, X_train, y_train, X_val, y_val)
                return _score(model, X_val, y_val, scoring)
        else:
            cv = self._make_cv()

            def objective(trial: optuna.Trial) -> float:
                params = param_space_func(trial)
                trial.set_user_attr("full_params", params)
                scores = []
                for i, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
                    X_tr, X_vl = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_vl = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    model = adapter.build(params, self.config.random_state)
                    adapter.fit(model, X_tr, y_tr, X_vl, y_vl)
                    scores.append(_score(model, X_vl, y_vl, scoring))
                    trial.report(float(np.mean(scores)), i)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                return float(np.mean(scores))

        def on_trial_end(study: optuna.Study, trial: optuna.FrozenTrial) -> None:
            nonlocal best_value
            if pbar:
                pbar.update(1)
                if study.best_value != best_value:
                    best_value = study.best_value
                    pbar.set_description(f"Melhor: {best_value:.4f}")

        study = optuna.create_study(
            direction=_DIRECTION[scoring],
            pruner=optuna.pruners.HyperbandPruner(),
            sampler=optuna.samplers.TPESampler(seed=self.config.random_state),
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[on_trial_end] if show_progress else [],
        )
        if pbar:
            pbar.close()

        best_params = study.best_trial.user_attrs.get("full_params", study.best_params)
        best_model  = adapter.build(best_params, self.config.random_state)
        adapter.fit(best_model, X_train, y_train, X_val, y_val)

        test_score = _score(best_model, X_test, y_test, scoring) if X_test is not None and y_test is not None else None

        return TuningResult(
            model=best_model,
            params=best_params,
            score=study.best_value,
            trials=study.trials_dataframe(),
            test_score=test_score,
        )

    def _make_cv(self):
        if self.config.problem_type == "classification":
            return StratifiedKFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_state,
            )
        return KFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )
