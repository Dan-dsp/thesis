"""Save evaluation figures from existing results, without GUI windows or model fitting."""

from pathlib import Path

import numpy as np
import pandas as pd


def _figure(width=10, height=6):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(width, height), layout="constrained")
    FigureCanvasAgg(fig)
    return fig, fig.subplots()


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=160)
    finally:
        fig.clear()


def plot_feature_scores(table, score, path, title, top=20):
    values = table.loc[:, ["feature", score]].copy()
    values[score] = pd.to_numeric(values[score], errors="coerce")
    values = values.loc[np.isfinite(values[score])].head(top).iloc[::-1]
    if values.empty:
        return
    fig, ax = _figure(11, max(4, 0.3 * len(values) + 1.5))
    ax.barh(np.arange(len(values)), values[score], color="#168575")
    ax.set_yticks(np.arange(len(values)), values.feature.astype(str))
    ax.set(xlabel=score.replace("_", " "), title=title)
    ax.axvline(0, color="0.6", linewidth=0.6)
    _save(fig, path)


def plot_grid_search(table, path, title, metric="f1_macro"):
    mean_col = f"mean_test_{metric}"
    std_col = f"std_test_{metric}"
    if mean_col not in table:
        mean_col, std_col = "mean_test_score", "std_test_score"
    if mean_col not in table:
        return
    means = pd.to_numeric(table[mean_col], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(means)
    if not finite.any():
        return
    std = (pd.to_numeric(table[std_col], errors="coerce").fillna(0).to_numpy(dtype=float)
           if std_col in table else np.zeros(len(table)))
    fig, ax = _figure()
    candidates = np.arange(1, len(table) + 1)
    ax.errorbar(candidates[finite], means[finite], yerr=std[finite], fmt="o",
                capsize=3, color="#168575", label="CV mean +/- fold SD")
    best = int(np.nanargmax(np.where(finite, means, np.nan)))
    ax.scatter([candidates[best]], [means[best]], marker="*", s=140,
               color="#c13f57", zorder=3, label="Best mean score")
    ax.set(xlabel="Candidate number (CSV row + 1; not training epochs)",
           ylabel=metric, title=f"{title}\nTuning diagnostic, not independent evaluation")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_rfecv(table, selected_count, path, title, metric="f1_macro"):
    values = table.sort_values("n_features")
    x = values.n_features.to_numpy(dtype=int)
    mean = values.mean_test_score.to_numpy(dtype=float)
    std = values.std_test_score.to_numpy(dtype=float)
    fig, ax = _figure()
    ax.plot(x, mean, "o-", color="#168575", label="CV mean")
    ax.fill_between(x, mean - std, mean + std, alpha=0.18,
                    color="#168575", label="+/- fold SD (not a confidence interval)")
    ax.axvline(selected_count, linestyle="--", color="#c13f57",
               label=f"RFECV selected {selected_count} features")
    ax.set(xlabel="Number of retained features", ylabel=metric,
           title=f"{title}\nFeature-selection diagnostic")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_matrix(matrix, labels, path, title, *, fractional=False, xlabel="", ylabel=""):
    matrix = np.asarray(matrix)
    if not matrix.size:
        return
    fig, ax = _figure(11, 9)
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0,
                      vmax=1 if fractional else None)
    fig.colorbar(image, ax=ax, shrink=0.8)
    positions = np.arange(len(labels))
    ax.set_xticks(positions, labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(positions, labels, fontsize=8)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    midpoint = (1 if fractional else np.nanmax(matrix)) / 2
    for i, j in np.ndindex(matrix.shape):
        value = matrix[i, j]
        if np.isfinite(value):
            ax.text(j, i, f"{value:.2f}" if fractional else f"{value:g}",
                    ha="center", va="center", fontsize=8,
                    color="white" if value > midpoint else "black")
    _save(fig, path)


def plot_feature_overlap(feature_lists, path, title):
    names = list(feature_lists)
    sets = [set(feature_lists[name]) for name in names]
    matrix = np.array([[len(a & b) / len(a | b) if a | b else np.nan
                        for b in sets] for a in sets])
    plot_matrix(matrix, names, path, title + " (Jaccard overlap)", fractional=True)


def plot_feature_counts(feature_lists, path):
    names = list(feature_lists)[::-1]
    fig, ax = _figure()
    ax.barh(names, [len(set(feature_lists[name])) for name in names], color="#168575")
    ax.set(xlabel="Number of unique selected features", title="Selection-stage subset sizes")
    _save(fig, path)


def plot_model_metrics(table, columns, path, title):
    if table.empty:
        return
    fig, ax = _figure(11, 6)
    x = np.arange(len(table))
    width = 0.8 / len(columns)
    colors = ["#168575", "#c13f57", "#c99b22", "#5265ac"]
    for i, (column, label) in enumerate(columns.items()):
        ax.bar(x - 0.4 + width * (i + 0.5), table[column], width,
               label=label, color=colors[i % len(colors)])
    ax.set_xticks(x, table.model.astype(str), rotation=30, ha="right", fontsize=8)
    ax.set(ylabel="Score", ylim=(0, 1.06), title=title)
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_outer_scores(scores, path):
    if scores.empty:
        return
    if scores.family.nunique() != 1:
        raise ValueError("Plot independent CV scores separately for SL and DL.")
    fig, ax = _figure(12, 6)
    for name, values in scores.groupby("model", sort=True):
        values = values.sort_values(["repeat", "fold"])
        labels = [f"R{r}/F{f}" for r, f in zip(values.repeat, values.fold)]
        ax.plot(labels, values.score, "o-", label=name)
    ax.set(xlabel="Held-out outer fold", ylabel="Macro-F1", ylim=(0, 1.06),
           title=f"{scores.family.iloc[0]} independent outer-CV scores")
    ax.legend(fontsize=8)
    _save(fig, path)
