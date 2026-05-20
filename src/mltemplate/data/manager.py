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
        self, train: pd.DataFrame, test: pd.DataFrame, name: str
    ) -> None:
        base = self.storage.processed_path / name
        base.mkdir(parents=True, exist_ok=True)

        self.storage.save_data(train, base / "train.pkl")
        self.storage.save_data(test, base / "test.pkl")

        metadata = {
            "feature_set": name,
            "created_at": datetime.now().isoformat(),
            "train_shape": list(train.shape),
            "test_shape": list(test.shape),
        }
        with open(base / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        logger.info("Feature set '%s' salvo em %s", name, base)

    def load_feature_set(
        self, name: str = "latest"
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        base = self._resolve_feature_set(name)
        train = self.storage.load_data(base / "train.pkl")
        test = self.storage.load_data(base / "test.pkl")
        logger.info("Feature set '%s' carregado — treino %s, teste %s", name, train.shape, test.shape)
        return train, test

    def _resolve_feature_set(self, name: str) -> Path:
        if name != "latest":
            return self.storage.processed_path / name
        dirs = sorted(self.storage.processed_path.glob("*"))
        if not dirs:
            raise FileNotFoundError("Nenhum feature set encontrado em processed/")
        return dirs[-1]
