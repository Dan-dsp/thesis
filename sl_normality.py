"""Within-species training diagnostics; never transforms classifier inputs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from result_plots import _figure, _save

FAMILIES = ("hu", "zernike", "fourier", "affine", "logpolarfft", "hsvhist",
            "colormoment", "lbp", "hog", "glcm", "gabor", "edgehist",
            "sift_bovw", "orb_bovw")


def finite_values(series):
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def representative_features(data, features, label_col):
    """Maximize nonconstant-class coverage; break ties by schema order, not p."""
    rows = []
    for family in FAMILIES:
        candidates = [f for f in features if f.startswith(family + "_")]
        if not candidates:
            continue
        coverage = {f: sum(len(np.unique(finite_values(g[f]))) > 1
                           for _, g in data.groupby(label_col)) for f in candidates}
        feature = max(candidates, key=coverage.get)
        rows.append({"family": family, "feature": feature,
                     "nonconstant_species": coverage[feature],
                     "selection_rule": "maximum nonconstant-class coverage; ties use schema order"})
    return pd.DataFrame(rows, columns=["family", "feature", "nonconstant_species", "selection_rule"])


def plot_distributions(data, representatives, tests, path, label_col, alpha):
    species = sorted(data[label_col].unique())
    for representative in representatives.itertuples():
        feature = representative.feature
        values = finite_values(data[feature])
        bins = np.histogram_bin_edges(values, bins=15) if values.size else np.linspace(0, 1, 16)
        nrows = int(np.ceil(len(species) / 3))
        fig, unused = _figure(15, 3.3 * nrows)
        fig.delaxes(unused)
        axes = fig.subplots(nrows, 3, squeeze=False, sharex=True)
        for ax, label in zip(axes.flat, species):
            x = finite_values(data.loc[data[label_col].eq(label), feature])
            record = tests.loc[tests.species.eq(str(label)) & tests.feature.eq(feature)].iloc[0]
            if x.size:
                ax.hist(x, bins=bins, density=True, color="#168575", alpha=0.65)
            if x.size >= 3 and np.ptp(x) > 0:
                grid = np.linspace(bins[0], bins[-1], 400)
                ax.plot(grid, norm.pdf(grid, x.mean(), x.std(ddof=1)), color="#c13f57")
            p_label = f"p={record.p_value:.3g}" if pd.notna(record.p_value) else "p=unavailable"
            ax.set_title(f"{str(label).replace('_', ' ')}\nn={len(x)}; {p_label}; {record.test_status}", fontsize=9)
            ax.set_xlabel(feature + " (raw descriptor value)", fontsize=8)
            ax.set_ylabel("Density", fontsize=8)
            ax.tick_params(labelsize=8)
        for ax in list(axes.flat)[len(species):]:
            ax.set_visible(False)
        fig.suptitle(f"{representative.family}: {feature} within species\nOriginal training images; Gaussian reference in red; raw p-values, alpha={alpha}")
        _save(fig, path / "distributions" / f"{representative.family}_within_species.png")


def analyze_within_species(df, features, save_dir=None, alpha=0.05,
                           max_samples=5000, random_state=42, label_col="species"):
    from sl_feature_comparison_tools import _run_pooled_shapiro_tests

    if not 0 < alpha < 1 or not 3 <= max_samples <= 5000:
        raise ValueError("Require 0 < alpha < 1 and 3 <= max_samples <= 5000.")
    if label_col not in df or not features:
        raise ValueError("Normality diagnostics require species labels and features.")
    data = df.loc[df["split"].eq("train")].copy() if "split" in df else df.copy()
    if data.empty or data[label_col].isna().any():
        raise ValueError("Require nonempty, labeled original training rows.")
    tables = []
    for species, group in data.groupby(label_col, sort=True):
        table = _run_pooled_shapiro_tests(group, features, alpha=alpha,
                                         max_samples=max_samples, random_state=random_state)
        table.insert(0, "species", str(species))
        table["test_status"] = np.where(table.p_value.isna(), "not_testable",
                                       np.where(table.p_value < alpha, "rejected", "not_rejected"))
        tables.append(table)
    result = pd.concat(tables, ignore_index=True)
    if save_dir is not None:
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        result.to_csv(path / "shapiro_normality.csv", index=False)
        summary = result.groupby(["species", "test_status"]).size().unstack(fill_value=0)
        summary.reindex(columns=["rejected", "not_rejected", "not_testable"], fill_value=0).to_csv(
            path / "shapiro_by_species_summary.csv")
        (path / "shapiro_summary.json").write_text(json.dumps({
            "scope": "within_species_original_train", "alpha": alpha,
            "n_features": len(features), "n_species": len(tables), "n_tests": len(result),
            "status_counts": result.test_status.value_counts().to_dict(),
            "p_value_adjustment": "none; exploratory multiple tests",
            "max_samples_per_species_feature": max_samples, "seed": random_state,
            "interpretation": "Not rejected does not prove Gaussianity; no feature removal or normalization.",
        }, indent=2), encoding="utf-8")
        representatives = representative_features(data, features, label_col)
        representatives.to_csv(path / "representative_features.csv", index=False)
        plot_distributions(data, representatives, result, path, label_col, alpha)
        fig, ax = _figure()
        for species, group in result.groupby("species", sort=True):
            ax.hist(group.p_value.dropna(), bins=np.linspace(0, 1, 21), histtype="step", label=species)
        ax.axvline(alpha, color="#c13f57", linestyle="--")
        ax.set(xlabel="Within-species Shapiro p-value (unadjusted)", ylabel="Feature count",
               title="Within-species normality: exploratory diagnostics")
        ax.legend(fontsize=7)
        _save(fig, path / "shapiro_p_values.png")
    return result


if __name__ == "__main__":
    from sl_feature_comparison_tools import load_features_and_labels
    from shared_training_pool import DEFAULT_POOL_MANIFEST

    root = Path(__file__).resolve().parent.parent
    print("[Normality] Loading descriptors and validating the shared training pool...", flush=True)
    data, _, _, features = load_features_and_labels(
        root / "sl_results" / "shallow_learning_normality_test_within_class_birds.csv", selection_split="train",
        pool_manifest_path=DEFAULT_POOL_MANIFEST, seed=42)
    analyze_within_species(data, features, root / "sl_results" / "exploratory" / "shapiro")
    print("[Normality] Saved within-species tables and distribution PNGs.", flush=True)
