from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mltemplate.config import ProjectConfig


def dataset_info(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna DataFrame com tipo, nulos e % nulos por coluna."""
    info = pd.DataFrame({
        "dtype": df.dtypes,
        "nulls": df.isnull().sum(),
        "null_pct": (df.isnull().sum() / len(df) * 100).round(2),
        "nunique": df.nunique(),
    })
    return info


def summary_statistics(
    df: pd.DataFrame,
    config: ProjectConfig,
) -> dict[str, pd.DataFrame]:
    """
    Retorna dicionário com estatísticas descritivas por grupo de features.
    Chaves: "target", "numerical", "categorical" (apenas as presentes no df).
    """
    result: dict[str, pd.DataFrame] = {}

    if config.target in df.columns:
        t = df[config.target]
        if config.problem_type == "classification":
            result["target"] = pd.DataFrame({
                "count": [t.count()],
                "nunique": [t.nunique()],
                "mode": [t.mode()[0] if not t.mode().empty else None],
                "value_counts": [t.value_counts().to_dict()],
                "frequencies": [t.value_counts(normalize=True).round(4).to_dict()],
            }, index=[f"{config.target}({t.dtype})"])
        else:
            stats = t.agg(["count", "mean", "std", "min", "max"])
            quantiles = t.quantile([0.25, 0.5, 0.75])
            quantiles.index = ["p25", "p50", "p75"]
            result["target"] = pd.concat([stats, quantiles]).to_frame(name=config.target)

    num_cols = [c for c in config.numerical_features if c in df.columns]
    if num_cols:
        stats = df[num_cols].agg(["count", "mean", "std", "min", "max"])
        quantiles = df[num_cols].quantile([0.25, 0.5, 0.75])
        quantiles.index = ["p25", "p50", "p75"]
        result["numerical"] = pd.concat([stats, quantiles])

    cat_cols = [c for c in config.categorical_features if c in df.columns]
    if cat_cols:
        base = df[cat_cols].agg(["count", "nunique"])
        modes = df[cat_cols].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
        modes.name = "mode"
        result["categorical"] = pd.concat([base, modes.to_frame().T])

    return result


# --- funções de plot ---

def plot_boxplot_by_target(
    df: pd.DataFrame,
    config: ProjectConfig,
    name: str = "Dataset",
) -> plt.Figure:
    features = [c for c in config.numerical_features if c in df.columns]
    fig = _make_grid_figure(len(features), name, "Boxplots")
    axes = fig.axes
    for i, col in enumerate(features):
        sns.boxplot(data=df, x=config.target, y=col, hue=config.target, ax=axes[i])
        axes[i].set_title(col)
    _remove_empty_axes(fig, len(features))
    fig.tight_layout()
    return fig


def plot_histogram_by_target(
    df: pd.DataFrame,
    config: ProjectConfig,
    name: str = "Dataset",
) -> plt.Figure:
    features = [c for c in config.numerical_features if c in df.columns]
    fig = _make_grid_figure(len(features), name, "Histogramas")
    axes = fig.axes
    for i, col in enumerate(features):
        bins = min(df[col].nunique(), 30)
        sns.histplot(data=df, x=col, hue=config.target, bins=bins, ax=axes[i])
        axes[i].set_title(col)
    _remove_empty_axes(fig, len(features))
    fig.tight_layout()
    return fig


def plot_violin_by_target(
    df: pd.DataFrame,
    config: ProjectConfig,
    name: str = "Dataset",
) -> plt.Figure:
    features = [c for c in config.numerical_features if c in df.columns]
    fig = _make_grid_figure(len(features), name, "Violinos")
    axes = fig.axes
    for i, col in enumerate(features):
        sns.violinplot(data=df, x=config.target, y=col, hue=config.target, fill=False, ax=axes[i])
        axes[i].set_title(col)
    _remove_empty_axes(fig, len(features))
    fig.tight_layout()
    return fig


def plot_barplot_by_target(
    df: pd.DataFrame,
    config: ProjectConfig,
    name: str = "Dataset",
) -> plt.Figure:
    features = [c for c in config.categorical_features if c in df.columns]
    fig = _make_grid_figure(len(features), name, "Barras por Target")
    axes = fig.axes
    for i, col in enumerate(features):
        order = sorted(df[col].unique())
        sns.countplot(data=df, x=col, order=order, hue=config.target, ax=axes[i])
        axes[i].set_title(col)
    _remove_empty_axes(fig, len(features))
    fig.tight_layout()
    return fig


def plot_boxplot_comparative(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    config: ProjectConfig,
    name1: str = "Dataset 1",
    name2: str = "Dataset 2",
) -> plt.Figure:
    features = [c for c in config.numerical_features if c in df1.columns and c in df2.columns]
    n = len(features)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3 * n))
    fig.suptitle(f"Boxplots — {name1} vs {name2}", fontsize=14)
    if n == 1:
        axes = [axes]
    for i, col in enumerate(features):
        sns.boxplot(data=df1, y=col, ax=axes[i][0])
        axes[i][0].set_title(f"{col} — {name1}")
        sns.boxplot(data=df2, y=col, ax=axes[i][1])
        axes[i][1].set_title(f"{col} — {name2}")
    fig.tight_layout()
    return fig


# --- helpers privados ---

def _make_grid_figure(n_features: int, name: str, kind: str) -> plt.Figure:
    n_cols = 2
    n_rows = max(1, (n_features + n_cols - 1) // n_cols)
    fig, _ = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    fig.suptitle(f"{kind} — {name}", fontsize=14)
    return fig


def _remove_empty_axes(fig: plt.Figure, n_used: int) -> None:
    axes = fig.axes
    for j in range(n_used, len(axes)):
        fig.delaxes(axes[j])
