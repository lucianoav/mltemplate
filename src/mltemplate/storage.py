from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StorageManager:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    @property
    def raw_path(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed_path(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def models_path(self) -> Path:
        return self.root / "models"

    @property
    def metrics_path(self) -> Path:
        return self.root / "reports" / "metrics"

    @property
    def submissions_path(self) -> Path:
        return self.root / "submissions"

    # --- dados genéricos ---

    def load_data(self, path: Path) -> pd.DataFrame:
        path = Path(path)
        if path.suffix == ".csv":
            return pd.read_csv(path)
        if path.suffix == ".pkl":
            return joblib.load(path)
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        raise ValueError(f"Formato não suportado: {path.suffix}")

    def save_data(self, df: pd.DataFrame, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".csv":
            df.to_csv(path, index=False)
        elif path.suffix == ".pkl":
            joblib.dump(df, path)
        elif path.suffix == ".parquet":
            df.to_parquet(path, index=False)
        else:
            raise ValueError(f"Formato não suportado: {path.suffix}")
        return path

    # --- modelos ---

    def save_model(self, model: Any, name: str) -> Path:
        path = self.models_path / f"{name}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        logger.info("Modelo salvo em %s", path)
        return path

    def load_model(self, name: str) -> Any:
        path = self.models_path / f"{name}.pkl"
        return joblib.load(path)

    # --- métricas ---

    def save_metrics(self, metrics: dict, name: str) -> Path:
        path = self.metrics_path / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        # converte tipos numpy para Python nativo
        serializable = {k: float(v) if hasattr(v, "item") else v for k, v in metrics.items()}
        with open(path, "w") as f:
            json.dump(serializable, f, indent=4)
        logger.info("Métricas salvas em %s", path)
        return path

    # --- submissão ---

    def save_submission(self, df: pd.DataFrame, name: str) -> Path:
        path = self.submissions_path / f"{name}.csv"
        return self.save_data(df, path)
