"""
Reusable feature-comparison tools for shallow-learning datasets.

This module contains the lower-level building blocks used by
`sl_feature_comparison.py`:
- data loading and preprocessing
- exploratory analysis helpers
- filter methods
- wrapper-model helpers
- embedded-model helpers
- plotting utilities
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import parallel
from scipy.stats import shapiro
from tqdm import tqdm

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, make_scorer, matthews_corrcoef
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


METADATA_COLUMNS = {"sample_id", "sample_name", "orig_filename", "species", "split"}
METADATA_COLUMN_ORDER = ("sample_id", "sample_name", "orig_filename", "species", "split")
EXCLUDED_FEATURE_PREFIXES: tuple[str, ...] = ()
DUPLICATE_DESCRIPTIVE_FEATURES = {"affine_6"}
DUPLICATE_LEGACY_FEATURES = {"f39"}
DEFAULT_TOP_K = 100
DERIVED_TOP_COUNTS = (80, 50)
_ACTIVE_WRAPPER_FIT_PROGRESS = None


def _import_pyplot():
    import matplotlib.pyplot as plt

    return plt


@contextmanager
def tqdm_joblib(total: int, desc: str):
    """
    Show a tqdm progress bar for joblib-driven tasks such as GridSearchCV.
    """
    progress_bar = tqdm(total=total, desc=desc, unit="fit", dynamic_ncols=True)
    original_callback = parallel.BatchCompletionCallBack

    class TqdmBatchCompletionCallback(original_callback):
        def __call__(self, *args, **kwargs):
            progress_bar.update(self.batch_size)
            return super().__call__(*args, **kwargs)

    parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield progress_bar
    finally:
        parallel.BatchCompletionCallBack = original_callback
        progress_bar.close()


@contextmanager
def wrapper_fit_progress(total: int, desc: str):
    """
    Track RFECV estimator fits, which happen outside joblib progress hooks.
    """
    global _ACTIVE_WRAPPER_FIT_PROGRESS

    progress_bar = tqdm(total=total, desc=desc, unit="fit", dynamic_ncols=True)
    _ACTIVE_WRAPPER_FIT_PROGRESS = progress_bar
    try:
        yield progress_bar
    finally:
        _ACTIVE_WRAPPER_FIT_PROGRESS = None
        progress_bar.close()


def _update_wrapper_fit_progress(step: int = 1) -> None:
    global _ACTIVE_WRAPPER_FIT_PROGRESS

    if _ACTIVE_WRAPPER_FIT_PROGRESS is not None:
        _ACTIVE_WRAPPER_FIT_PROGRESS.update(step)


def ensure_dir(save_dir: str | Path) -> Path:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    return save_path


def save_json(data: dict[str, Any], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def get_feature_columns(df: pd.DataFrame, label_col: str = "species") -> list[str]:
    """
    Detect valid numeric feature columns while excluding metadata.
    """
    excluded = set(METADATA_COLUMNS)
    excluded.add(label_col)

    numeric_cols = [
        col for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]

    legacy_f_cols = [
        col for col in numeric_cols
        if col.startswith("f") and col not in DUPLICATE_LEGACY_FEATURES
    ]
    descriptive_cols = [
        col for col in numeric_cols
        if not col.startswith("f") and col not in DUPLICATE_DESCRIPTIVE_FEATURES
    ]

    selected = descriptive_cols if descriptive_cols else legacy_f_cols
    selected = [
        col for col in selected
        if not any(col.startswith(prefix) for prefix in EXCLUDED_FEATURE_PREFIXES)
    ]
    return selected


def get_metadata_columns(df: pd.DataFrame, label_col: str = "species") -> list[str]:
    metadata_cols = [col for col in METADATA_COLUMN_ORDER if col in df.columns]
    if label_col in df.columns and label_col not in metadata_cols:
        metadata_cols.append(label_col)
    return metadata_cols


def build_training_ready_dataset(
    df: pd.DataFrame,
    selected_features: list[str],
    label_col: str = "species",
) -> pd.DataFrame:
    """
    Build a reduced dataset that stays compatible with `sl_training_pipeline.py`.
    """
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset.")
    if "split" not in df.columns:
        raise ValueError("The dataset must include a 'split' column for training export.")
    if not selected_features:
        raise ValueError("No selected features were provided for export.")

    missing_features = [feature for feature in selected_features if feature not in df.columns]
    if missing_features:
        preview = ", ".join(missing_features[:10])
        raise ValueError(
            "Cannot export training-ready dataset because some selected features are missing: "
            f"{preview}"
        )

    seen: set[str] = set()
    ordered_features: list[str] = []
    for feature in selected_features:
        if feature in seen:
            continue
        seen.add(feature)
        ordered_features.append(feature)

    metadata_cols = get_metadata_columns(df, label_col=label_col)
    export_cols = metadata_cols + ordered_features
    export_df = df.loc[:, export_cols].copy()

    for feature in ordered_features:
        export_df[feature] = pd.to_numeric(export_df[feature], errors="coerce")

    return export_df


def export_training_ready_dataset(
    df: pd.DataFrame,
    selected_features: list[str],
    out_path: str | Path,
    label_col: str = "species",
) -> dict[str, Any]:
    export_df = build_training_ready_dataset(df, selected_features, label_col=label_col)
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(output_path, index=False)

    feature_cols = get_feature_columns(export_df, label_col=label_col)
    metadata_cols = get_metadata_columns(export_df, label_col=label_col)
    return {
        "csv_path": str(output_path),
        "n_rows": int(export_df.shape[0]),
        "n_features": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "metadata_columns": metadata_cols,
    }


def load_features_and_labels(
    csv_path: str,
    label_col: str = "species",
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """
    Load the dataset and return the cleaned numeric matrix plus labels.
    """
    df = pd.read_csv(csv_path)
    feature_names = get_feature_columns(df, label_col=label_col)

    if not feature_names:
        raise ValueError(
            "No feature columns were detected. "
            "Expected descriptive feature columns or legacy f-columns."
        )
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset.")

    feature_df = df[feature_names].apply(pd.to_numeric, errors="coerce")
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

    missing_total = int(feature_df.isna().sum().sum())
    if missing_total > 0:
        missing_by_column = feature_df.isna().sum()
        missing_by_column = missing_by_column[missing_by_column > 0].sort_values(ascending=False)

        print(
            f"[WARN] Detected {missing_total} missing or non-finite values across "
            f"{len(missing_by_column)} feature columns."
        )
        print("[WARN] Top columns with missing values:")
        print(missing_by_column.head(10))

        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(feature_df)
    else:
        X = feature_df.to_numpy(dtype=np.float64, copy=True)

    y = df[label_col].to_numpy()
    return df, np.asarray(X, dtype=np.float64), y, feature_names


def encode_labels(y: np.ndarray) -> tuple[np.ndarray, LabelEncoder]:
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(np.asarray(y))
    return y_encoded, label_encoder


def build_scoring_dict() -> dict[str, Any]:
    return {
        "accuracy": "accuracy",
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
        "mcc": make_scorer(matthews_corrcoef),
    }


def save_top_ranked_feature_slices(
    ranking_df: pd.DataFrame,
    save_dir: str | Path,
    file_stem: str,
    top_k: int = DEFAULT_TOP_K,
    derived_top_counts: tuple[int, ...] = DERIVED_TOP_COUNTS,
) -> dict[int, pd.DataFrame]:
    """
    Save a top-100 list and derive top-80/top-50 by taking the first rows of it.
    """
    save_path = ensure_dir(save_dir)
    top_reference = ranking_df.head(top_k).copy().reset_index(drop=True)

    requested_sizes = [top_k]
    requested_sizes.extend(size for size in derived_top_counts if size < top_k)

    seen: set[int] = set()
    slices: dict[int, pd.DataFrame] = {}
    for size in requested_sizes:
        if size in seen:
            continue
        seen.add(size)

        slice_df = top_reference.head(size).copy().reset_index(drop=True)
        slice_df.to_csv(save_path / f"{file_stem}_top_{size}.csv", index=False)
        slices[size] = slice_df

    return slices


def run_shapiro_tests(
    df: pd.DataFrame,
    feature_names: list[str],
    save_dir: str | Path | None = None,
    alpha: float = 0.05,
    max_samples: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run a Shapiro-Wilk normality test for each feature.

    Shapiro is common and appropriate here, but its p-value becomes less precise
    for very large samples. When a feature has more than `max_samples` values,
    a deterministic subsample is used.
    """
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, Any]] = []

    for feature in tqdm(feature_names, desc="Shapiro per feature", leave=False):
        series = pd.to_numeric(df[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        values = series.dropna().to_numpy(dtype=np.float64, copy=True)

        note = ""
        used_values = values
        was_subsampled = False

        if values.size < 3:
            w_stat = np.nan
            p_value = np.nan
            note = "Not enough finite samples for Shapiro."
        else:
            if values.size > max_samples:
                sample_idx = rng.choice(values.size, size=max_samples, replace=False)
                used_values = values[np.sort(sample_idx)]
                was_subsampled = True
                note = f"Subsampled to {max_samples} values for Shapiro."

            try:
                w_stat, p_value = shapiro(used_values)
            except Exception as exc:  # pragma: no cover - defensive safeguard
                w_stat = np.nan
                p_value = np.nan
                note = f"Shapiro failed: {exc}"

        rows.append({
            "feature": feature,
            "n_original": int(values.size),
            "n_used": int(used_values.size),
            "was_subsampled": bool(was_subsampled),
            "shapiro_W": w_stat,
            "p_value": p_value,
            "gaussian_at_alpha": bool(pd.notna(p_value) and p_value >= alpha),
            "notes": note,
        })

    result = pd.DataFrame(rows).sort_values(
        ["gaussian_at_alpha", "p_value", "shapiro_W"],
        ascending=[False, False, False],
    )

    if save_dir is not None:
        save_path = ensure_dir(save_dir)
        result.to_csv(save_path / "shapiro_normality.csv", index=False)
        summary = {
            "alpha": float(alpha),
            "n_features": int(len(result)),
            "gaussian_features": int(result["gaussian_at_alpha"].sum()),
            "non_gaussian_features": int((~result["gaussian_at_alpha"]).sum()),
        }
        save_json(summary, save_path / "shapiro_summary.json")

    return result


def compute_anova_scores(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    save_dir: str | Path | None = None,
) -> pd.DataFrame:
    f_scores, p_values = f_classif(X, y)
    result = pd.DataFrame({
        "feature": feature_names,
        "F_score": f_scores,
        "p_value": p_values,
    }).sort_values("F_score", ascending=False)

    if save_dir is not None:
        save_path = ensure_dir(save_dir)
        result.to_csv(save_path / "anova_scores.csv", index=False)

        top_n = min(20, len(result))
        plt = _import_pyplot()
        plt.figure(figsize=(12, 6))
        plt.bar(result["feature"].head(top_n), result["F_score"].head(top_n))
        plt.xticks(rotation=90)
        plt.title("ANOVA F-Scores (Top Features)")
        plt.tight_layout()
        plt.savefig(save_path / "anova_top20.png", dpi=300)
        plt.close()

    return result


def compute_fisher_scores(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    save_dir: str | Path | None = None,
) -> pd.DataFrame:
    X = np.asarray(X)
    y = np.asarray(y)
    classes = np.unique(y)
    _, n_features = X.shape

    fisher_scores = np.zeros(n_features, dtype=np.float64)
    for j in tqdm(range(n_features), desc="Fisher per feature", leave=False):
        xj = X[:, j]
        mu = xj.mean()

        num = 0.0
        den = 0.0
        for class_name in classes:
            idx = y == class_name
            x_c = xj[idx]
            if x_c.size <= 1:
                continue
            mu_c = x_c.mean()
            sigma_c = x_c.std(ddof=1)
            num += x_c.size * (mu_c - mu) ** 2
            den += x_c.size * sigma_c ** 2

        fisher_scores[j] = num / (den + 1e-8)

    result = pd.DataFrame({
        "feature": feature_names,
        "fisher_score": fisher_scores,
    }).sort_values("fisher_score", ascending=False)

    if save_dir is not None:
        save_path = ensure_dir(save_dir)
        result.to_csv(save_path / "fisher_scores.csv", index=False)

        top_n = min(20, len(result))
        plt = _import_pyplot()
        plt.figure(figsize=(12, 6))
        plt.bar(result["feature"].head(top_n), result["fisher_score"].head(top_n))
        plt.xticks(rotation=90)
        plt.title("Fisher Scores (Top Features)")
        plt.tight_layout()
        plt.savefig(save_path / "fisher_top20.png", dpi=300)
        plt.close()

    return result


def compute_mutual_information(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    save_dir: str | Path | None = None,
) -> pd.DataFrame:
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=42)
    result = pd.DataFrame({
        "feature": feature_names,
        "mutual_info": mi,
    }).sort_values("mutual_info", ascending=False)

    if save_dir is not None:
        save_path = ensure_dir(save_dir)
        result.to_csv(save_path / "mutual_information_scores.csv", index=False)

        top_n = min(20, len(result))
        plt = _import_pyplot()
        plt.figure(figsize=(12, 6))
        plt.bar(result["feature"].head(top_n), result["mutual_info"].head(top_n))
        plt.xticks(rotation=90)
        plt.title("Mutual Information (Top Features)")
        plt.tight_layout()
        plt.savefig(save_path / "mutual_information_top20.png", dpi=300)
        plt.close()

    return result


def run_pca_analysis(
    X: np.ndarray,
    feature_names: list[str],
    n_components: int = 3,
    standardize: bool = True,
    save_dir: str | Path | None = None,
) -> tuple[PCA, pd.DataFrame]:
    X_proc = np.array(X, dtype=np.float64, copy=True)
    if standardize:
        scaler = StandardScaler()
        X_proc = scaler.fit_transform(X_proc)

    n_components = min(n_components, X_proc.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(X_proc)

    loadings = pca.components_.T
    cols = [f"PC{i + 1}" for i in range(n_components)]
    loadings_df = pd.DataFrame(loadings, index=feature_names, columns=cols)

    if save_dir is not None:
        save_path = ensure_dir(save_dir)
        explained_df = pd.DataFrame({
            "component": cols,
            "explained_variance_ratio": pca.explained_variance_ratio_,
        })
        explained_df.to_csv(save_path / "pca_explained_variance.csv", index=False)
        loadings_df.to_csv(save_path / "pca_loadings.csv")

        plt = _import_pyplot()
        plt.figure(figsize=(8, 5))
        plt.bar(cols, pca.explained_variance_ratio_)
        plt.ylabel("Explained Variance Ratio")
        plt.title("PCA Explained Variance")
        plt.tight_layout()
        plt.savefig(save_path / "pca_explained_variance.png", dpi=300)
        plt.close()

    return pca, loadings_df


def rank_pca_features(
    loadings_df: pd.DataFrame,
    explained_variance_ratio: np.ndarray,
    save_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Rank features using weighted absolute PCA loadings.
    """
    abs_loadings = loadings_df.abs()
    weights = np.asarray(explained_variance_ratio, dtype=np.float64)
    weights = weights / weights.sum()

    ranked = loadings_df.copy()
    ranked["weighted_abs_loading"] = abs_loadings.mul(weights, axis=1).sum(axis=1)
    ranked["abs_mean_loading"] = abs_loadings.mean(axis=1)
    ranked["abs_max_loading"] = abs_loadings.max(axis=1)
    ranked = (
        ranked
        .reset_index()
        .rename(columns={"index": "feature"})
        .sort_values(
            ["weighted_abs_loading", "abs_mean_loading", "abs_max_loading"],
            ascending=False,
        )
    )

    if save_dir is not None:
        save_path = ensure_dir(save_dir)
        ranked.to_csv(save_path / "pca_feature_ranking.csv", index=False)

    return ranked


def plot_violin_for_features(
    df: pd.DataFrame,
    label_col: str,
    features: list[str],
    save_dir: str | Path | None = None,
    n_cols: int = 3,
    fig_width: int = 16,
    fig_height: int = 10,
    file_name: str = "violin_features.png",
) -> None:
    if not features:
        return

    classes = sorted(df[label_col].unique())
    n_classes = len(classes)
    n_feats = len(features)
    n_rows = int(np.ceil(n_feats / n_cols))

    plt = _import_pyplot()
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )
    axes = axes.ravel()

    for idx, feature in enumerate(features):
        ax = axes[idx]
        for pos, class_name in enumerate(classes):
            values = pd.to_numeric(
                df.loc[df[label_col] == class_name, feature],
                errors="coerce",
            ).dropna().to_numpy(dtype=np.float64)
            if values.size > 0:
                ax.violinplot(values, positions=[pos], showmedians=True)

        ax.set_title(feature)
        ax.set_xlabel("Class")
        ax.set_ylabel("Value")
        ax.set_xticks(range(n_classes))
        ax.set_xticklabels(classes, rotation=45, ha="right")

    for idx in range(n_feats, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    if save_dir is not None:
        save_path = ensure_dir(save_dir)
        plt.savefig(save_path / file_name, dpi=300)

    plt.close(fig)


def plot_correlation_matrix(
    df: pd.DataFrame,
    feature_names: list[str],
    save_dir: str | Path | None = None,
    max_features: int = 40,
    fig_width: int = 12,
    fig_height: int = 10,
    file_stem: str = "correlation_matrix",
) -> pd.DataFrame:
    corr = df[feature_names].apply(pd.to_numeric, errors="coerce").corr()

    if save_dir is not None:
        save_path = ensure_dir(save_dir)
        corr.to_csv(save_path / f"{file_stem}.csv")

        corr_plot = corr.iloc[:max_features, :max_features]
        plt = _import_pyplot()
        plt.figure(figsize=(fig_width, fig_height))
        im = plt.imshow(
            corr_plot,
            interpolation="nearest",
            aspect="auto",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
        )
        plt.colorbar(im)
        plt.xticks(range(len(corr_plot.columns)), corr_plot.columns, rotation=90)
        plt.yticks(range(len(corr_plot.index)), corr_plot.index)
        plt.title(f"Correlation Matrix (First {len(corr_plot.columns)} Features)")
        plt.tight_layout()
        plt.savefig(save_path / f"{file_stem}.png", dpi=300)
        plt.close()

    return corr


def build_wrapper_model_spaces(seed: int = 42) -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    svm_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=seed)),
        ]
    )
    svm_grid = {
        "clf__C": [0.1, 1.0, 10.0, 30.0],
        "clf__gamma": ["scale", 0.1, 0.01, 0.001],
    }

    knn_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier()),
        ]
    )
    knn_grid = {
        "clf__n_neighbors": [3, 5, 7, 11],
        "clf__weights": ["uniform", "distance"],
        "clf__metric": ["minkowski"],
        "clf__p": [1, 2],
    }

    return {
        "svm_rbf": (svm_pipe, svm_grid),
        "knn": (knn_pipe, knn_grid),
    }


def build_embedded_model_spaces(
    seed: int = 42,
    tree_model_n_jobs: int = 1,
    xgb_n_jobs: int = 1,
) -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    spaces: dict[str, tuple[Pipeline, dict[str, list[Any]]]] = {}

    rf_pipe = Pipeline(
        [
            ("clf", RandomForestClassifier(random_state=seed, n_jobs=tree_model_n_jobs)),
        ]
    )
    rf_grid = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 15, 30],
        "clf__min_samples_split": [2, 5],
        "clf__min_samples_leaf": [1, 2],
    }
    spaces["random_forest"] = (rf_pipe, rf_grid)

    try:
        from xgboost import XGBClassifier

        xgb_pipe = Pipeline(
            [
                (
                    "clf",
                    XGBClassifier(
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        random_state=seed,
                        tree_method="hist",
                        verbosity=0,
                        n_jobs=xgb_n_jobs,
                    ),
                )
            ]
        )
        xgb_grid = {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [3, 4],
            "clf__learning_rate": [0.1],
            "clf__subsample": [0.8],
            "clf__colsample_bytree": [0.8, 1.0],
        }
        spaces["xgboost"] = (xgb_pipe, xgb_grid)
    except ImportError:
        print("[WARN] xgboost is not installed. Skipping XGBoost embedded selection.")

    return spaces


class PermutationImportanceWrapper(BaseEstimator, ClassifierMixin):
    """
    Fit an estimator and expose permutation importances for RFECV.

    This makes RFECV usable with tuned non-linear models such as SVM-RBF and
    k-NN, which do not expose native coefficients or tree importances.
    """

    def __init__(
        self,
        estimator: Any,
        scoring: str = "f1_macro",
        n_repeats: int = 5,
        random_state: int = 42,
        n_jobs: int | None = 1,
    ) -> None:
        self.estimator = estimator
        self.scoring = scoring
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PermutationImportanceWrapper":
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = int(X.shape[1])

        perm = permutation_importance(
            self.estimator_,
            X,
            y,
            scoring=self.scoring,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.feature_importances_ = np.nan_to_num(perm.importances_mean, nan=0.0)
        _update_wrapper_fit_progress(1)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self.estimator_, "predict_proba"):
            raise AttributeError("Wrapped estimator does not expose predict_proba.")
        return self.estimator_.predict_proba(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self.estimator_, "decision_function"):
            raise AttributeError("Wrapped estimator does not expose decision_function.")
        return self.estimator_.decision_function(X)


def summarize_grid_search(search: GridSearchCV) -> tuple[dict[str, Any], pd.DataFrame]:
    cv_results_df = pd.DataFrame(search.cv_results_)
    best_idx = int(search.best_index_)

    summary = {
        "refit_metric": str(search.refit),
        "best_score": float(search.best_score_),
        "best_params": search.best_params_,
        "best_accuracy": float(cv_results_df.loc[best_idx, "mean_test_accuracy"]),
        "best_f1_macro": float(cv_results_df.loc[best_idx, "mean_test_f1_macro"]),
        "best_mcc": float(cv_results_df.loc[best_idx, "mean_test_mcc"]),
    }
    return summary, cv_results_df


def _extract_final_estimator(model: Any) -> Any:
    if hasattr(model, "best_estimator_"):
        model = model.best_estimator_
    if hasattr(model, "named_steps"):
        return model.named_steps[next(reversed(model.named_steps))]
    return model


def run_wrapper_rfecv(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    model_name: str,
    pipeline: Pipeline,
    param_grid: dict[str, list[Any]],
    save_dir: str | Path | None = None,
    refit_metric: str = "f1_macro",
    cv_splits: int = 5,
    min_features_to_select: int = 5,
    rfecv_step: int = 10,
    grid_n_jobs: int = -1,
    rfecv_n_jobs: int = 1,
    seed: int = 42,
) -> dict[str, Any]:
    scoring = build_scoring_dict()
    tuning_cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    rfecv_cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)

    initial_grid = GridSearchCV(
        estimator=clone(pipeline),
        param_grid=param_grid,
        scoring=scoring,
        refit=refit_metric,
        cv=tuning_cv,
        n_jobs=grid_n_jobs,
        return_train_score=False,
    )

    n_features = int(X.shape[1])
    n_candidates = len(ParameterGrid(param_grid))
    rfecv_step = max(1, int(rfecv_step))
    estimated_elimination_rounds = 1 + max(0, int(np.ceil((n_features - min_features_to_select) / rfecv_step)))
    estimated_total_model_fits = (
        (n_candidates * cv_splits)
        + (estimated_elimination_rounds * cv_splits)
        + (n_candidates * cv_splits)
    )
    grid_fit_total = n_candidates * cv_splits
    rfecv_fit_total = estimated_elimination_rounds * cv_splits
    print(
        f"[Wrapper:{model_name}] Wrapper selection starting with {n_features} features, "
        f"step={rfecv_step}, {n_candidates} grid candidates, {cv_splits}-fold CV. "
        f"Estimated total model fits: ~{estimated_total_model_fits}."
    )
    print(f"[Wrapper:{model_name}] Initial tuning on the full feature space...")
    with tqdm_joblib(
        total=grid_fit_total,
        desc=f"[Wrapper:{model_name}] Initial grid",
    ):
        initial_grid.fit(X, y)
    print(f"[Wrapper:{model_name}] Initial tuning completed.")

    rfecv_estimator = PermutationImportanceWrapper(
        estimator=clone(initial_grid.best_estimator_),
        scoring=refit_metric,
        n_repeats=3,
        random_state=seed,
        n_jobs=1,
    )

    rfecv = RFECV(
        estimator=rfecv_estimator,
        step=rfecv_step,
        cv=rfecv_cv,
        scoring=refit_metric,
        min_features_to_select=min_features_to_select,
        n_jobs=rfecv_n_jobs,
    )
    with wrapper_fit_progress(
        total=rfecv_fit_total,
        desc=f"[Wrapper:{model_name}] RFECV",
    ):
        rfecv.fit(X, y)
    print(f"[Wrapper:{model_name}] RFECV completed.")

    selected_mask = rfecv.support_
    selected_features = [feature for feature, keep in zip(feature_names, selected_mask) if keep]

    final_grid = GridSearchCV(
        estimator=clone(pipeline),
        param_grid=param_grid,
        scoring=scoring,
        refit=refit_metric,
        cv=tuning_cv,
        n_jobs=grid_n_jobs,
        return_train_score=False,
    )
    print(f"[Wrapper:{model_name}] Final tuning on the selected subset...")
    with tqdm_joblib(
        total=grid_fit_total,
        desc=f"[Wrapper:{model_name}] Final grid",
    ):
        final_grid.fit(X[:, selected_mask], y)
    print(
        f"[Wrapper:{model_name}] Final grid completed on {int(selected_mask.sum())} selected features."
    )

    final_perm = permutation_importance(
        final_grid.best_estimator_,
        X[:, selected_mask],
        y,
        scoring=refit_metric,
        n_repeats=10,
        random_state=seed,
        n_jobs=1,
    )
    selected_importance_map = {
        feature: importance
        for feature, importance in zip(selected_features, final_perm.importances_mean)
    }

    ranking_df = pd.DataFrame({
        "feature": feature_names,
        "selected": selected_mask,
        "rfecv_rank": rfecv.ranking_,
    })
    ranking_df["selected_subset_importance"] = ranking_df["feature"].map(selected_importance_map)
    sort_key = ranking_df["selected_subset_importance"].fillna(-np.inf)
    ranking_df = (
        ranking_df.assign(_sort_key=sort_key)
        .sort_values(
            ["selected", "_sort_key", "rfecv_rank", "feature"],
            ascending=[False, False, True, True],
        )
        .drop(columns="_sort_key")
        .reset_index(drop=True)
    )
    ranking_df["wrapper_rank_order"] = np.arange(1, len(ranking_df) + 1)

    initial_best_summary, initial_cv_results_df = summarize_grid_search(initial_grid)
    best_summary, cv_results_df = summarize_grid_search(final_grid)
    top_feature_slices: dict[int, pd.DataFrame] = {}

    if save_dir is not None:
        save_path = ensure_dir(save_dir)
        ranking_df.to_csv(save_path / f"{model_name}_feature_ranking.csv", index=False)
        pd.DataFrame({"selected_feature": selected_features}).to_csv(
            save_path / f"{model_name}_selected_features.csv",
            index=False,
        )

        if hasattr(rfecv, "cv_results_"):
            rfecv_curve_df = pd.DataFrame(rfecv.cv_results_)
            rfecv_curve_df.to_csv(save_path / f"{model_name}_rfecv_curve.csv", index=False)

        initial_cv_results_df.to_csv(
            save_path / f"{model_name}_initial_grid_search_results.csv",
            index=False,
        )
        cv_results_df.to_csv(save_path / f"{model_name}_grid_search_results.csv", index=False)
        save_json(initial_best_summary, save_path / f"{model_name}_initial_best_summary.json")
        save_json(best_summary, save_path / f"{model_name}_best_summary.json")

        top_feature_slices = save_top_ranked_feature_slices(
            ranking_df,
            save_path,
            f"{model_name}_ranked_features",
            top_k=DEFAULT_TOP_K,
            derived_top_counts=DERIVED_TOP_COUNTS,
        )

    return {
        "rfecv": rfecv,
        "initial_grid": initial_grid,
        "final_grid": final_grid,
        "ranking_df": ranking_df,
        "selected_mask": selected_mask,
        "selected_features": selected_features,
        "initial_best_summary": initial_best_summary,
        "best_summary": best_summary,
        "top_feature_slices": top_feature_slices,
    }


def run_embedded_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    model_name: str,
    pipeline: Pipeline,
    param_grid: dict[str, list[Any]],
    save_dir: str | Path | None = None,
    refit_metric: str = "f1_macro",
    cv_splits: int = 5,
    grid_n_jobs: int = -1,
    seed: int = 42,
) -> dict[str, Any]:
    scoring = build_scoring_dict()
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)

    grid = GridSearchCV(
        estimator=clone(pipeline),
        param_grid=param_grid,
        scoring=scoring,
        refit=refit_metric,
        cv=cv,
        n_jobs=grid_n_jobs,
        return_train_score=False,
    )
    grid.fit(X, y)

    final_estimator = _extract_final_estimator(grid)
    if not hasattr(final_estimator, "feature_importances_"):
        raise ValueError(f"{model_name} does not expose feature_importances_.")

    importances = np.asarray(final_estimator.feature_importances_, dtype=np.float64)
    ranking_df = pd.DataFrame({
        "feature": feature_names,
        "feature_importance": importances,
    }).sort_values("feature_importance", ascending=False).reset_index(drop=True)
    ranking_df["embedded_rank_order"] = np.arange(1, len(ranking_df) + 1)

    best_summary, cv_results_df = summarize_grid_search(grid)
    top_feature_slices: dict[int, pd.DataFrame] = {}

    if save_dir is not None:
        save_path = ensure_dir(save_dir)
        ranking_df.to_csv(save_path / f"{model_name}_feature_importance.csv", index=False)
        cv_results_df.to_csv(save_path / f"{model_name}_grid_search_results.csv", index=False)
        save_json(best_summary, save_path / f"{model_name}_best_summary.json")

        top_feature_slices = save_top_ranked_feature_slices(
            ranking_df,
            save_path,
            f"{model_name}_ranked_features",
            top_k=DEFAULT_TOP_K,
            derived_top_counts=DERIVED_TOP_COUNTS,
        )

    return {
        "grid_search": grid,
        "ranking_df": ranking_df,
        "best_summary": best_summary,
        "top_feature_slices": top_feature_slices,
    }
