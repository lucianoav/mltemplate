from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from mltemplate.config import ProjectConfig
from mltemplate.data.sources import DataSource
from mltemplate.storage import StorageManager

logger = logging.getLogger(__name__)


class DataManager:
    def __init__(self, storage: StorageManager, config: ProjectConfig) -> None:
        self.storage = storage
        self.config = config

    # --- dados brutos ---

    def load_raw(
        self,
        source: DataSource,
        train_file: str = "train.csv",
        test_file: str = "test.csv",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        source.download(self.storage.raw_path)
        train = self.storage.load_data(self.storage.raw_path / train_file)
        test = self.storage.load_data(self.storage.raw_path / test_file)
        logger.info("Dados brutos carregados — treino %s, teste %s", train.shape, test.shape)
        return train, test

    # --- split ---

    def split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        target = self.config.target
        if target not in df.columns:
            raise ValueError(f"Coluna alvo '{target}' não encontrada no DataFrame.")

        X = df.drop(columns=[target])
        y = df[target]

        stratify = y if self.config.problem_type == "classification" else None
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=stratify,
        )
        logger.info(
            "Split — treino %s, validação %s", X_train.shape, X_val.shape
        )
        return X_train, X_val, y_train, y_val

    # --- feature sets versionados ---

    def save_feature_set(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        y_test: pd.Series | None = None,
        name: str = "v1",
    ) -> None:
        base = self.storage.processed_path / name
        base.mkdir(parents=True, exist_ok=True)

        self.storage.save_data(x_train, base / "x_train.pkl")
        self.storage.save_data(x_test,  base / "x_test.pkl")
        self.storage.save_data(y_train.to_frame(), base / "y_train.pkl")
        if x_val is not None:
            self.storage.save_data(x_val, base / "x_val.pkl")
        if y_val is not None:
            self.storage.save_data(y_val.to_frame(), base / "y_val.pkl")
        if y_test is not None:
            self.storage.save_data(y_test.to_frame(), base / "y_test.pkl")

        if self.config.problem_type == "classification":
            y_stats = y_train.value_counts().sort_index().to_dict()
            y_stats = {str(k): int(v) for k, v in y_stats.items()}
        else:
            y_stats = {
                "mean": round(float(y_train.mean()), 6),
                "std":  round(float(y_train.std()),  6),
                "min":  round(float(y_train.min()),  6),
                "max":  round(float(y_train.max()),  6),
            }

        metadata = {
            "feature_set":   name,
            "created_at":    datetime.now().isoformat(),
            "target":        self.config.target,
            "problem_type":  self.config.problem_type,
            "columns":       list(x_train.columns),
            "dtypes":        {col: str(dt) for col, dt in x_train.dtypes.items()},
            "x_train_shape": list(x_train.shape),
            "x_test_shape":  list(x_test.shape),
            "x_val_shape":   list(x_val.shape) if x_val is not None else None,
            "has_y_val":     y_val  is not None,
            "has_y_test":    y_test is not None,
            "y_train_stats": y_stats,
        }
        with open(base / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        logger.info("Feature set '%s' salvo em %s", name, base)

    def load_feature_set(
        self, name: str = "latest"
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame | None, pd.Series | None, pd.Series | None]:
        base = self._resolve_feature_set(name)
        x_train = self.storage.load_data(base / "x_train.pkl")
        x_test  = self.storage.load_data(base / "x_test.pkl")
        y_train = self.storage.load_data(base / "y_train.pkl").iloc[:, 0]

        def _load_df(path: Path) -> pd.DataFrame | None:
            return self.storage.load_data(path) if path.exists() else None

        def _load_y(path: Path) -> pd.Series | None:
            return self.storage.load_data(path).iloc[:, 0] if path.exists() else None

        x_val  = _load_df(base / "x_val.pkl")
        y_val  = _load_y(base / "y_val.pkl")
        y_test = _load_y(base / "y_test.pkl")

        logger.info("Feature set '%s' carregado — treino %s, teste %s", name, x_train.shape, x_test.shape)
        return x_train, x_test, y_train, x_val, y_val, y_test

    def _resolve_feature_set(self, name: str) -> Path:
        if name != "latest":
            return self.storage.processed_path / name
        dirs = sorted(self.storage.processed_path.glob("*"))
        if not dirs:
            raise FileNotFoundError("Nenhum feature set encontrado em processed/")
        return dirs[-1]
