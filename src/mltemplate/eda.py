from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mltemplate.config import ProjectConfig

_N_COLS = 5
_ROW_H = 2.8
_COL_W = 3.0
_BASE_COLORS = ["#4878cf", "#e05a4e", "#6acc65", "#b47cc7", "#d35f5f"]
_QUARTILE_PALETTE = {"Q1": "#92c5de", "Q2": "#4878cf", "Q3": "#e05a4e", "Q4": "#a50026"}


def _make_palette(values: list) -> dict:
    return {v: _BASE_COLORS[i % len(_BASE_COLORS)] for i, v in enumerate(sorted(values))}


def _get_hue(df: pd.DataFrame, config: ProjectConfig) -> tuple[pd.DataFrame, str, dict]:
    """Para classificação devolve o target como hue. Para regressão cria quartis do target."""
    if config.problem_type == "classification":
        return df, config.target, _make_palette(df[config.target].dropna().unique().tolist())
    out = df.copy()
    out["_target_bin"] = pd.qcut(
        out[config.target].rank(method="first"), q=4, labels=["Q1", "Q2", "Q3", "Q4"]
    )
    return out, "_target_bin", _QUARTILE_PALETTE


def _legend_first_only(axes: list, i: int, hue_title: str) -> None:
    if i > 0:
        leg = axes[i].get_legend()
        if leg is not None:
            leg.remove()
    else:
        leg = axes[i].get_legend()
        if leg is not None:
            leg.set_title(hue_title)
            plt.setp(leg.get_texts(), fontsize=7)
            plt.setp(leg.get_title(), fontsize=7)


def _seg_label(config: ProjectConfig) -> str:
    return "classe do target" if config.problem_type == "classification" else "quartil do target"


def _finalize_figure(fig: plt.Figure) -> None:
    """tight_layout reservando espaço proporcional para o suptitle, evitando sobreposição."""
    top = 1 - 0.5 / fig.get_figheight()
    fig.tight_layout(rect=(0, 0, 1, top))


# ── Funções tabulares ────────────────────────────────────────────────────────

def dataset_info(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna DataFrame com tipo, nulos, % nulos, valores únicos e % únicos por coluna."""
    return pd.DataFrame({
        "dtype": df.dtypes,
        "nulls": df.isnull().sum(),
        "null_pct": (df.isnull().sum() / len(df) * 100).round(2),
        "nunique": df.nunique(),
        "unique_pct": (df.nunique() / len(df) * 100).round(2),
    })


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


def outlier_summary(df: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    """Retorna contagem e % de outliers por feature numérica (critério IQR 1.5×)."""
    num_cols = [c for c in config.numerical_features if c in df.columns]
    rows = []
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
        rows.append({
            "q1": round(q1, 4), "q3": round(q3, 4), "iqr": round(iqr, 4),
            "lower_fence": round(lo, 4), "upper_fence": round(hi, 4),
            "n_outliers": n_out, "pct_outliers": round(n_out / len(df) * 100, 2),
        })
    return pd.DataFrame(rows, index=num_cols)


def correlation_matrix(
    df: pd.DataFrame, config: ProjectConfig, method: str = "pearson",
) -> pd.DataFrame:
    """Retorna a matriz de correlação entre as features numéricas."""
    num_cols = [c for c in config.numerical_features if c in df.columns]
    return df[num_cols].corr(method=method)


def category_target_rate(df: pd.DataFrame, config: ProjectConfig) -> dict[str, pd.DataFrame]:
    """
    Retorna dicionário com contagem e taxa do target por nível de cada feature categórica.
    Classificação: colunas count_<v> e rate_<v> para cada valor da target (suporta multiclasse).
    Regressão: colunas 'mean' e 'std'.
    """
    cat_cols = [c for c in config.categorical_features if c in df.columns]
    result = {}
    for col in cat_cols:
        if config.problem_type == "classification":
            counts = pd.crosstab(df[col], df[config.target])
            rates  = pd.crosstab(df[col], df[config.target], normalize="index").round(4)
            counts.columns = [f"count_{v}" for v in counts.columns]
            rates.columns  = [f"rate_{v}"  for v in rates.columns]
            grp = pd.concat([counts.sum(axis=1).rename("count"), counts, rates], axis=1)
            grp = grp.sort_values(rates.columns[-1], ascending=False)
        else:
            grp = (
                df.groupby(col, observed=True)[config.target]
                .agg(count="count", mean="mean", std="std")
                .round(4)
                .sort_values("mean", ascending=False)
            )
        result[col] = grp
    return result


def mutual_information_scores(
    df: pd.DataFrame, config: ProjectConfig, sample_size: int | None = 50_000,
) -> pd.DataFrame:
    """
    Retorna Mutual Information de cada feature em relação ao target, ordenado decrescente.
    Usa amostragem (sample_size) para desempenho em datasets grandes.
    """
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    all_features = [
        c for c in config.numerical_features + config.categorical_features
        if c in df.columns
    ]
    subset = df[all_features + [config.target]].dropna()
    if sample_size is not None and len(subset) > sample_size:
        subset = subset.sample(sample_size, random_state=config.random_state)

    X = subset[all_features].copy()
    y = subset[config.target]
    cat_cols = [c for c in config.categorical_features if c in X.columns]
    discrete_mask = [col in cat_cols for col in all_features]
    for col in cat_cols:
        X[col] = X[col].astype("category").cat.codes

    fn = mutual_info_classif if config.problem_type == "classification" else mutual_info_regression
    scores = fn(X, y, discrete_features=discrete_mask, random_state=config.random_state)

    return (
        pd.DataFrame({"feature": all_features, "mi_score": scores})
        .set_index("feature")
        .sort_values("mi_score", ascending=False)
        .round(4)
    )


# ── Plots de correlação ──────────────────────────────────────────────────────

def plot_correlation_heatmap(
    df: pd.DataFrame, config: ProjectConfig,
    method: str = "pearson", name: str = "Dataset",
) -> plt.Figure:
    """Heatmap da matriz de correlação entre as features numéricas."""
    corr = correlation_matrix(df, config, method)
    n = len(corr)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.8), max(5, n * 0.7)))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
        annot_kws={"size": 8}, linewidths=0.4, ax=ax,
    )
    ax.set_title(f"Correlação ({method}) — {name}", fontsize=12)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.tight_layout()
    return fig


# ── Plots do target ──────────────────────────────────────────────────────────

def plot_target_distribution(
    df: pd.DataFrame, config: ProjectConfig, name: str = "Dataset",
) -> plt.Figure:
    """Distribuição do target: barras com contagem e % (classificação) ou histograma+KDE (regressão)."""
    t = df[config.target].dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    if config.problem_type == "classification":
        counts = t.value_counts().sort_index()
        colors = [_BASE_COLORS[i % len(_BASE_COLORS)] for i in range(len(counts))]
        bars = ax.bar(range(len(counts)), counts.values, color=colors)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels([str(v) for v in counts.index])
        ax.set_ylim(0, counts.max() * 1.18)
        total = counts.sum()
        for bar, val in zip(bars, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.005,
                f"{val:,}\n({val / total:.1%})",
                ha="center", va="bottom", fontsize=9,
            )
        ax.set_ylabel("Contagem", fontsize=10)
    else:
        sns.histplot(t, kde=True, color=_BASE_COLORS[0], ax=ax)
        ax.set_ylabel("Contagem", fontsize=10)
    ax.set_xlabel(config.target, fontsize=10)
    ax.set_title(f"Distribuição do Target — {name}", fontsize=12)
    fig.tight_layout()
    return fig


# ── Plots feature × target (classificação e regressão via quartis) ───────────

def plot_boxplot_by_target(
    df: pd.DataFrame, config: ProjectConfig, name: str = "Dataset",
    low_percentile: float = 1.0, high_percentile: float = 99.0,
) -> plt.Figure:
    features = [c for c in config.numerical_features if c in df.columns]
    hue_df, hue_col, palette = _get_hue(df, config)
    hue_title = config.target if config.problem_type == "classification" else "Quartil do target"
    fig = _make_grid_figure(len(features), name, f"Boxplots das features por {_seg_label(config)}")
    axes = fig.axes
    range_label = f"[P{low_percentile:.0f}–P{high_percentile:.0f}]"
    for i, col in enumerate(features):
        lo = df[col].quantile(low_percentile / 100.0)
        hi = df[col].quantile(high_percentile / 100.0)
        mask = hue_df[col].between(lo, hi)
        clipped_pct = (~mask).mean() * 100
        sns.boxplot(
            data=hue_df[mask], x=hue_col, y=col, hue=hue_col,
            palette=palette, legend=False, ax=axes[i],
        )
        suffix = f" (−{clipped_pct:.1f}%)" if clipped_pct > 0 else ""
        axes[i].set_title(f"{col} {range_label}{suffix}", fontsize=9)
        axes[i].set_xlabel("")
    _remove_empty_axes(fig, len(features))
    _finalize_figure(fig)
    return fig


def plot_histogram_by_target(
    df: pd.DataFrame, config: ProjectConfig, name: str = "Dataset",
    low_percentile: float = 1.0, high_percentile: float = 99.0,
) -> plt.Figure:
    features = [c for c in config.numerical_features if c in df.columns]
    hue_df, hue_col, palette = _get_hue(df, config)
    hue_title = config.target if config.problem_type == "classification" else "Quartil do target"
    fig = _make_grid_figure(len(features), name, f"Histogramas das features por {_seg_label(config)}")
    axes = fig.axes
    range_label = f"[P{low_percentile:.0f}–P{high_percentile:.0f}]"
    for i, col in enumerate(features):
        lo = df[col].quantile(low_percentile / 100.0)
        hi = df[col].quantile(high_percentile / 100.0)
        mask = hue_df[col].between(lo, hi)
        clipped_pct = (~mask).mean() * 100
        bins = min(hue_df[mask][col].nunique(), 30)
        sns.histplot(
            data=hue_df[mask], x=col, hue=hue_col, bins=bins,
            palette=palette, ax=axes[i],
        )
        suffix = f" (−{clipped_pct:.1f}%)" if clipped_pct > 0 else ""
        axes[i].set_title(f"{col} {range_label}{suffix}", fontsize=9)
        axes[i].set_xlabel("")
        _legend_first_only(axes, i, hue_title)
    _remove_empty_axes(fig, len(features))
    _finalize_figure(fig)
    return fig


def plot_violin_by_target(
    df: pd.DataFrame, config: ProjectConfig, name: str = "Dataset",
    low_percentile: float = 1.0, high_percentile: float = 99.0,
) -> plt.Figure:
    features = [c for c in config.numerical_features if c in df.columns]
    hue_df, hue_col, palette = _get_hue(df, config)
    fig = _make_grid_figure(len(features), name, f"Distribuições (violino) das features por {_seg_label(config)}")
    axes = fig.axes
    range_label = f"[P{low_percentile:.0f}–P{high_percentile:.0f}]"
    for i, col in enumerate(features):
        lo = df[col].quantile(low_percentile / 100.0)
        hi = df[col].quantile(high_percentile / 100.0)
        mask = hue_df[col].between(lo, hi)
        clipped_pct = (~mask).mean() * 100
        sns.violinplot(
            data=hue_df[mask], x=hue_col, y=col, hue=hue_col,
            fill=False, palette=palette, legend=False, ax=axes[i],
        )
        suffix = f" (−{clipped_pct:.1f}%)" if clipped_pct > 0 else ""
        axes[i].set_title(f"{col} {range_label}{suffix}", fontsize=9)
        axes[i].set_xlabel("")
    _remove_empty_axes(fig, len(features))
    _finalize_figure(fig)
    return fig


def plot_kde_by_target(
    df: pd.DataFrame, config: ProjectConfig, name: str = "Dataset",
    low_percentile: float = 1.0, high_percentile: float = 99.0,
) -> plt.Figure:
    features = [c for c in config.numerical_features if c in df.columns]
    hue_df, hue_col, palette = _get_hue(df, config)
    hue_title = config.target if config.problem_type == "classification" else "Quartil do target"
    fig = _make_grid_figure(len(features), name, f"Densidade (KDE) das features por {_seg_label(config)}")
    axes = fig.axes
    range_label = f"[P{low_percentile:.0f}–P{high_percentile:.0f}]"
    for i, col in enumerate(features):
        lo = df[col].quantile(low_percentile / 100.0)
        hi = df[col].quantile(high_percentile / 100.0)
        mask = hue_df[col].between(lo, hi)
        clipped_pct = (~mask).mean() * 100
        sns.kdeplot(
            data=hue_df[mask], x=col, hue=hue_col,
            palette=palette, common_norm=False, ax=axes[i],
        )
        suffix = f" (−{clipped_pct:.1f}%)" if clipped_pct > 0 else ""
        axes[i].set_title(f"{col} {range_label}{suffix}", fontsize=9)
        axes[i].set_xlabel("")
        _legend_first_only(axes, i, hue_title)
    _remove_empty_axes(fig, len(features))
    _finalize_figure(fig)
    return fig


# ── Plots de features categóricas ────────────────────────────────────────────

def plot_barplot_by_target(
    df: pd.DataFrame, config: ProjectConfig, name: str = "Dataset",
) -> plt.Figure:
    """
    Classificação: countplot de cada feature categórica colorido pelo target.
    Regressão: boxplot do target por nível de cada feature categórica.
    """
    features = [c for c in config.categorical_features if c in df.columns]
    kind = (
        "Contagem por categoria, segmentada pela classe do target"
        if config.problem_type == "classification"
        else "Distribuição do target por categoria"
    )
    fig = _make_grid_figure(len(features), name, kind)
    axes = fig.axes
    if config.problem_type == "classification":
        palette = _make_palette(df[config.target].dropna().unique().tolist())
        for i, col in enumerate(features):
            order = sorted(df[col].dropna().unique())
            sns.countplot(
                data=df, x=col, order=order, hue=config.target,
                palette=palette, ax=axes[i],
            )
            axes[i].set_title(col, fontsize=9)
            axes[i].set_xlabel("")
            axes[i].tick_params(axis="x", rotation=45, labelsize=7)
            _legend_first_only(axes, i, config.target)
    else:
        for i, col in enumerate(features):
            order = sorted(df[col].dropna().unique())
            sns.boxplot(
                data=df, x=col, y=config.target, hue=col,
                order=order, palette="husl", legend=False, ax=axes[i],
            )
            axes[i].set_title(col, fontsize=9)
            axes[i].set_xlabel("")
            axes[i].tick_params(axis="x", rotation=45, labelsize=7)
    _remove_empty_axes(fig, len(features))
    _finalize_figure(fig)
    return fig


# ── Plots comparativos (train vs test) ───────────────────────────────────────

def plot_boxplot_comparative(
    df1: pd.DataFrame, df2: pd.DataFrame, config: ProjectConfig,
    name1: str = "Dataset 1", name2: str = "Dataset 2",
    low_percentile: float = 1.0, high_percentile: float = 99.0,
) -> plt.Figure:
    features = [c for c in config.numerical_features if c in df1.columns and c in df2.columns]
    palette = {name1: _BASE_COLORS[0], name2: _BASE_COLORS[1]}
    d1 = df1[features].assign(_source=name1)
    d2 = df2[features].assign(_source=name2)
    combined = pd.concat([d1, d2], ignore_index=True)
    fig = _make_grid_figure(len(features), f"{name1} vs {name2}", "Boxplots das features por dataset")
    axes = fig.axes
    range_label = f"[P{low_percentile:.0f}–P{high_percentile:.0f}]"
    for i, col in enumerate(features):
        lo = combined[col].quantile(low_percentile / 100.0)
        hi = combined[col].quantile(high_percentile / 100.0)
        mask = combined[col].between(lo, hi)
        clipped_pct = (~mask).mean() * 100
        sns.boxplot(
            data=combined[mask], x="_source", y=col, hue="_source",
            palette=palette, legend=False, ax=axes[i],
        )
        suffix = f" (−{clipped_pct:.1f}%)" if clipped_pct > 0 else ""
        axes[i].set_title(f"{col} {range_label}{suffix}", fontsize=9)
        axes[i].set_xlabel("")
    _remove_empty_axes(fig, len(features))
    _finalize_figure(fig)
    return fig


def plot_kde_comparative(
    df1: pd.DataFrame, df2: pd.DataFrame, config: ProjectConfig,
    name1: str = "Dataset 1", name2: str = "Dataset 2",
    low_percentile: float = 1.0, high_percentile: float = 99.0,
) -> plt.Figure:
    features = [c for c in config.numerical_features if c in df1.columns and c in df2.columns]
    palette = {name1: _BASE_COLORS[0], name2: _BASE_COLORS[1]}
    d1 = df1[features].assign(_source=name1)
    d2 = df2[features].assign(_source=name2)
    combined = pd.concat([d1, d2], ignore_index=True)
    fig = _make_grid_figure(len(features), f"{name1} vs {name2}", "Densidade (KDE) das features por dataset")
    axes = fig.axes
    range_label = f"[P{low_percentile:.0f}–P{high_percentile:.0f}]"
    for i, col in enumerate(features):
        lo = combined[col].quantile(low_percentile / 100.0)
        hi = combined[col].quantile(high_percentile / 100.0)
        mask = combined[col].between(lo, hi)
        clipped_pct = (~mask).mean() * 100
        sns.kdeplot(
            data=combined[mask], x=col, hue="_source",
            palette=palette, common_norm=False, ax=axes[i],
        )
        suffix = f" (−{clipped_pct:.1f}%)" if clipped_pct > 0 else ""
        axes[i].set_title(f"{col} {range_label}{suffix}", fontsize=9)
        axes[i].set_xlabel("")
        _legend_first_only(axes, i, "")
    _remove_empty_axes(fig, len(features))
    _finalize_figure(fig)
    return fig


# ── Plots sequenciais ────────────────────────────────────────────────────────

def plot_feature_over_sequence(
    df: pd.DataFrame, config: ProjectConfig, sequence_col: str, name: str = "Dataset",
) -> plt.Figure:
    """Média de cada feature numérica ao longo de sequence_col, segmentada por target (ou quartis)."""
    features = [c for c in config.numerical_features if c in df.columns and c != sequence_col]
    hue_df, hue_col, palette = _get_hue(df, config)
    hue_title = config.target if config.problem_type == "classification" else "Quartil do target"
    fig = _make_grid_figure(len(features), name, f"Evolução média das features ao longo de {sequence_col}, por {_seg_label(config)}")
    axes = fig.axes
    for i, col in enumerate(features):
        means = (
            hue_df.groupby([sequence_col, hue_col], observed=True)[col]
            .mean()
            .reset_index()
        )
        sns.lineplot(
            data=means, x=sequence_col, y=col, hue=hue_col,
            palette=palette, ax=axes[i],
        )
        axes[i].set_title(col, fontsize=9)
        axes[i].set_xlabel("")
        _legend_first_only(axes, i, hue_title)
    _remove_empty_axes(fig, len(features))
    _finalize_figure(fig)
    return fig


# ── Plots exclusivos de regressão ────────────────────────────────────────────

def plot_scatter_feature_vs_target(
    df: pd.DataFrame, config: ProjectConfig, name: str = "Dataset",
    sample_size: int = 5_000,
    low_percentile: float = 1.0, high_percentile: float = 99.0,
) -> plt.Figure:
    """
    Scatter de cada feature numérica vs target com linha de tendência LOWESS.
    Exclusivo de regressão.
    """
    features = [c for c in config.numerical_features if c in df.columns]
    plot_df = df.sample(min(sample_size, len(df)), random_state=config.random_state)
    t_lo = plot_df[config.target].quantile(low_percentile / 100.0)
    t_hi = plot_df[config.target].quantile(high_percentile / 100.0)
    plot_df = plot_df[plot_df[config.target].between(t_lo, t_hi)]
    fig = _make_grid_figure(len(features), name, "Dispersão de cada feature vs target (tendência LOWESS)")
    axes = fig.axes
    range_label = f"[P{low_percentile:.0f}–P{high_percentile:.0f}]"
    for i, col in enumerate(features):
        lo = plot_df[col].quantile(low_percentile / 100.0)
        hi = plot_df[col].quantile(high_percentile / 100.0)
        mask = plot_df[col].between(lo, hi)
        clipped_pct = (~mask).mean() * 100
        sns.regplot(
            data=plot_df[mask], x=col, y=config.target,
            scatter_kws={"alpha": 0.25, "s": 8, "color": _BASE_COLORS[0]},
            line_kws={"color": _BASE_COLORS[1], "linewidth": 1.5},
            lowess=True, ax=axes[i],
        )
        suffix = f" (−{clipped_pct:.1f}%)" if clipped_pct > 0 else ""
        axes[i].set_title(f"{col} {range_label}{suffix}", fontsize=9)
        axes[i].set_xlabel("")
        axes[i].set_ylabel(config.target, fontsize=7)
    _remove_empty_axes(fig, len(features))
    _finalize_figure(fig)
    return fig


def plot_feature_target_correlation(
    df: pd.DataFrame, config: ProjectConfig,
    method: str = "pearson", name: str = "Dataset",
) -> plt.Figure:
    """
    Barplot horizontal das correlações de cada feature numérica com o target, ordenado por magnitude.
    Exclusivo de regressão.
    """
    num_cols = [c for c in config.numerical_features if c in df.columns]
    corrs = pd.Series(
        {col: df[col].corr(df[config.target], method=method) for col in num_cols}
    ).sort_values(key=abs, ascending=True)
    colors = [_BASE_COLORS[0] if v >= 0 else _BASE_COLORS[1] for v in corrs.values]
    fig, ax = plt.subplots(figsize=(7, max(4, len(num_cols) * 0.5)))
    ax.barh(corrs.index, corrs.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Correlação ({method}) com {config.target}", fontsize=10)
    ax.set_title(f"Correlação Feature-Target — {name}", fontsize=12)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    return fig


def plot_target_by_categorical(
    df: pd.DataFrame, config: ProjectConfig, name: str = "Dataset",
    low_percentile: float = 1.0, high_percentile: float = 99.0,
) -> plt.Figure:
    """
    Boxplot da distribuição do target por nível de cada feature categórica.
    Exclusivo de regressão.
    """
    features = [c for c in config.categorical_features if c in df.columns]
    t_lo = df[config.target].quantile(low_percentile / 100.0)
    t_hi = df[config.target].quantile(high_percentile / 100.0)
    plot_df = df[df[config.target].between(t_lo, t_hi)]
    clipped_pct = (1 - len(plot_df) / len(df)) * 100
    range_label = f"[P{low_percentile:.0f}–P{high_percentile:.0f}]"
    fig = _make_grid_figure(len(features), name, "Distribuição do target por categoria")
    axes = fig.axes
    for i, col in enumerate(features):
        order = sorted(plot_df[col].dropna().unique())
        sns.boxplot(
            data=plot_df, x=col, y=config.target, hue=col,
            order=order, palette="husl", legend=False, ax=axes[i],
        )
        suffix = f" (−{clipped_pct:.1f}%)" if clipped_pct > 0 else ""
        axes[i].set_title(f"{col} {range_label}{suffix}", fontsize=9)
        axes[i].set_xlabel("")
        axes[i].tick_params(axis="x", rotation=45, labelsize=7)
    _remove_empty_axes(fig, len(features))
    _finalize_figure(fig)
    return fig


# ── Helpers privados ─────────────────────────────────────────────────────────

def _make_grid_figure(n_features: int, name: str, kind: str) -> plt.Figure:
    n_cols = _N_COLS
    n_rows = max(1, (n_features + n_cols - 1) // n_cols)
    fig, _ = plt.subplots(n_rows, n_cols, figsize=(n_cols * _COL_W, n_rows * _ROW_H))
    fig.suptitle(f"{kind} — {name}", fontsize=13)
    return fig


def _remove_empty_axes(fig: plt.Figure, n_used: int) -> None:
    axes = fig.axes
    for j in range(n_used, len(axes)):
        fig.delaxes(axes[j])
