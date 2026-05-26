from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from mltemplate.config import ProjectConfig

logger = logging.getLogger(__name__)


class FeatureEngineer:
    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self._encoder = OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)
        self._scaler = StandardScaler()
        self._poly: PolynomialFeatures | None = None
        self._poly_features: list[str] = []
        self._imputer_medians: dict[str, float] = {}
        self._imputer_modes: dict[str, str] = {}

    # --- limpeza ---

    def drop_ignored(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.config.ignore_features if c in df.columns]
        result = df.drop(columns=cols)
        if cols:
            print(f"drop_ignored: {len(cols)} coluna(s) removida(s) — {cols}")
        else:
            print("drop_ignored: nenhuma coluna removida")
        return result

    def fit_imputer(self, df: pd.DataFrame) -> FeatureEngineer:
        for col in self.config.numerical_features:
            if col in df.columns:
                self._imputer_medians[col] = df[col].median()
        for col in self.config.categorical_features:
            if col in df.columns:
                mode = df[col].mode()
                self._imputer_modes[col] = mode[0] if not mode.empty else "Unknown"
        num_summary = {col: f"{val:.4g}" for col, val in self._imputer_medians.items()}
        cat_summary = self._imputer_modes
        print(f"fit_imputer: {len(self._imputer_medians)} numérica(s) — medianas {num_summary}")
        print(f"fit_imputer: {len(self._imputer_modes)} categórica(s) — modas {cat_summary}")
        return self

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._imputer_medians and not self._imputer_modes:
            raise RuntimeError("Chame fit_imputer antes de impute.")
        df = df.copy()
        total = 0
        filled: dict[str, int] = {}
        for col, median in self._imputer_medians.items():
            if col in df.columns:
                n = int(df[col].isna().sum())
                if n:
                    filled[col] = n
                    total += n
                df[col] = df[col].fillna(median)
        for col, mode in self._imputer_modes.items():
            if col in df.columns:
                n = int(df[col].isna().sum())
                if n:
                    filled[col] = n
                    total += n
                df[col] = df[col].fillna(mode)
        if filled:
            print(f"impute: {total} valor(es) preenchido(s) — {filled}")
        else:
            print("impute: nenhum valor ausente encontrado")
        return df

    # --- encoding ---

    def fit_encoder(self, df: pd.DataFrame) -> FeatureEngineer:
        cats = self.config.categorical_features
        self._encoder.fit(df[cats])
        n_cols = len(self._encoder.get_feature_names_out(cats))
        print(f"fit_encoder: {len(cats)} coluna(s) categórica(s) — {cats} → {n_cols} coluna(s) após encoding")
        return self

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self._encoder, "categories_"):
            raise RuntimeError("Chame fit_encoder antes de encode.")
        cats = self.config.categorical_features
        encoded = self._encoder.transform(df[cats])
        encoded_cols = list(self._encoder.get_feature_names_out(cats))
        encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
        other_cols = [c for c in df.columns if c not in cats and c not in self.config.ignore_features]
        result = pd.concat([df[other_cols], encoded_df], axis=1)
        print(f"encode: {len(cats)} coluna(s) → {len(encoded_cols)} coluna(s) gerada(s) — {encoded_cols}")
        return result

    # --- scaling ---

    def fit_scaler(self, df: pd.DataFrame) -> FeatureEngineer:
        nums = self.config.numerical_features
        self._scaler.fit(df[nums])
        means = {col: f"{m:.4g}" for col, m in zip(nums, self._scaler.mean_)}
        stds  = {col: f"{s:.4g}" for col, s in zip(nums, self._scaler.scale_)}
        print(f"fit_scaler: {len(nums)} coluna(s) numérica(s) — {nums}")
        print(f"fit_scaler: médias  {means}")
        print(f"fit_scaler: desvios {stds}")
        return self

    def scale(self, df: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self._scaler, "mean_"):
            raise RuntimeError("Chame fit_scaler antes de scale.")
        nums = self.config.numerical_features
        scaled = self._scaler.transform(df[nums])
        scaled_cols = list(self._scaler.get_feature_names_out(nums))
        scaled_df = pd.DataFrame(scaled, columns=scaled_cols, index=df.index)
        other_cols = [c for c in df.columns if c not in nums and c not in self.config.ignore_features]
        result = pd.concat([df[other_cols], scaled_df], axis=1)
        print(f"scale: {len(nums)} coluna(s) normalizada(s) — {scaled_cols}")
        return result

    # --- features polinomiais (só numéricas por padrão) ---

    def fit_poly(
        self,
        df: pd.DataFrame,
        degree: int = 2,
        features: list[str] | None = None,
    ) -> FeatureEngineer:
        self._poly_features = features if features is not None else self.config.numerical_features
        self._poly = PolynomialFeatures(degree=degree, include_bias=False)
        self._poly.fit(df[self._poly_features])
        return self

    def expand_poly(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._poly is None:
            raise RuntimeError("Chame fit_poly antes de expand_poly.")
        poly_arr = self._poly.transform(df[self._poly_features])
        poly_cols = self._poly.get_feature_names_out(self._poly_features)
        poly_df = pd.DataFrame(poly_arr, columns=poly_cols, index=df.index)
        other_cols = [c for c in df.columns if c not in self._poly_features]
        return pd.concat([df[other_cols], poly_df], axis=1)
