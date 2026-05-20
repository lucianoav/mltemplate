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

    # --- limpeza ---

    def drop_ignored(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.config.ignore_features if c in df.columns]
        if cols:
            logger.debug("Removendo colunas ignoradas: %s", cols)
        return df.drop(columns=cols)

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.config.numerical_features:
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        for col in self.config.categorical_features:
            if col in df.columns and df[col].isna().any():
                mode = df[col].mode()
                df[col] = df[col].fillna(mode[0] if not mode.empty else "Unknown")
        return df

    # --- encoding ---

    def fit_encoder(self, df: pd.DataFrame) -> FeatureEngineer:
        self._encoder.fit(df[self.config.categorical_features])
        return self

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        cats = self.config.categorical_features
        encoded = self._encoder.transform(df[cats])
        encoded_cols = self._encoder.get_feature_names_out(cats)
        encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
        other_cols = [c for c in df.columns if c not in cats and c not in self.config.ignore_features]
        return pd.concat([df[other_cols], encoded_df], axis=1)

    # --- scaling ---

    def fit_scaler(self, df: pd.DataFrame) -> FeatureEngineer:
        self._scaler.fit(df[self.config.numerical_features])
        return self

    def scale(self, df: pd.DataFrame) -> pd.DataFrame:
        nums = self.config.numerical_features
        scaled = self._scaler.transform(df[nums])
        scaled_cols = self._scaler.get_feature_names_out(nums)
        scaled_df = pd.DataFrame(scaled, columns=scaled_cols, index=df.index)
        other_cols = [c for c in df.columns if c not in nums and c not in self.config.ignore_features]
        return pd.concat([df[other_cols], scaled_df], axis=1)

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
