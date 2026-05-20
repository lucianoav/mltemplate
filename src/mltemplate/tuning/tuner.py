from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, KFold
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
    model: Any
    params: dict
    score: float
    trials: pd.DataFrame = field(default_factory=pd.DataFrame)


class OptunaTuner:
    def __init__(self, config: ProjectConfig) -> None:
        self.config = config

    def tune(
        self,
        adapter: ModelAdapter,
        X: pd.DataFrame,
        y: pd.Series,
        param_space_func: Callable[[optuna.Trial], dict],
        scoring: str,
        n_trials: int = 100,
        timeout: int = 3600,
        show_progress: bool = True,
    ) -> TuningResult:
        if scoring not in _DIRECTION:
            raise ValueError(f"scoring '{scoring}' não suportado. Use: {list(_DIRECTION)}")

        cv = self._make_cv()
        best_value = _DIRECTION[scoring] == "maximize" and -np.inf or np.inf
        pbar = tqdm(total=n_trials, desc="Trials") if show_progress else None

        def objective(trial: optuna.Trial) -> float:
            params = param_space_func(trial)
            scores = []

            for i, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model = adapter.build(params, self.config.random_state)
                adapter.fit(model, X_tr, y_tr, X_val, y_val)
                scores.append(_score(model, X_val, y_val, scoring))

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

        best_params = study.best_params
        best_model = adapter.build(best_params, self.config.random_state)
        adapter.fit(best_model, X, y, None, None)

        return TuningResult(
            model=best_model,
            params=best_params,
            score=study.best_value,
            trials=study.trials_dataframe(),
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


class GridTuner:
    def __init__(self, config: ProjectConfig) -> None:
        self.config = config

    def tune(
        self,
        adapter: ModelAdapter,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: dict,
        scoring: str,
        verbose: int = 0,
    ) -> TuningResult:
        n_jobs = 1 if adapter.needs_cat_features() else -1
        model = adapter.build({}, self.config.random_state)
        gs = GridSearchCV(
            model, param_grid,
            cv=self.config.cv_folds,
            scoring=scoring,
            verbose=verbose,
            n_jobs=n_jobs,
            error_score="raise",
        )
        fit_kwargs = {"cat_features": adapter.cat_features()} if adapter.needs_cat_features() else {}
        gs.fit(X, y, **fit_kwargs)
        return TuningResult(model=gs.best_estimator_, params=gs.best_params_, score=gs.best_score_)


class RandomTuner:
    def __init__(self, config: ProjectConfig) -> None:
        self.config = config

    def tune(
        self,
        adapter: ModelAdapter,
        X: pd.DataFrame,
        y: pd.Series,
        param_distributions: dict,
        scoring: str,
        n_iter: int = 10,
        verbose: int = 0,
    ) -> TuningResult:
        model = adapter.build({}, self.config.random_state)
        rs = RandomizedSearchCV(
            model, param_distributions,
            n_iter=n_iter,
            cv=self.config.cv_folds,
            scoring=scoring,
            verbose=verbose,
            n_jobs=-1,
            random_state=self.config.random_state,
        )
        rs.fit(X, y)
        return TuningResult(model=rs.best_estimator_, params=rs.best_params_, score=rs.best_score_)
