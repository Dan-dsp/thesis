"""
sl_training_pipeline.py

End-to-end shallow-learning pipeline for:
- SVM (RBF)
- Random Forest
- XGBoost
- k-NN

Workflow:
1) Load feature CSV with split column (train/test)
2) Build an internal validation split from the TRAIN rows only
3) Hyperparameter search on the internal TRAIN subset (Stratified K-Fold CV)
4) Evaluate best model on the internal VAL subset
5) Refit best model on the full original TRAIN split, evaluate once on TEST
6) Save models, metrics, confusion matrices, and run metadata
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from joblib import parallel

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


DEFAULT_DATA_CSV_PATH = r"F:/01_Univalle/01_TG/dataset_features/shallow_learning_birds.csv"
DEFAULT_OUT_DIR = Path(r"F:/01_Univalle/01_TG/sl_outputs")
DEFAULT_TRAINING_READY_MANIFEST_PATH = Path(
    r"F:/01_Univalle/01_TG/sl_results/training_ready_datasets/training_ready_dataset_manifest.csv"
)
DEFAULT_BATCH_OUT_DIR = Path(r"F:/01_Univalle/01_TG/sl_outputs_batch")

METADATA_COLUMNS = {"sample_id", "sample_name", "orig_filename", "species", "split"}
DUPLICATE_DESCRIPTIVE_FEATURES = {"affine_6"}
DUPLICATE_LEGACY_FEATURES = {"f39"}


def _import_pyplot():
    import matplotlib.pyplot as plt

    return plt


def set_seed(seed: int = 42) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)


def format_seconds(seconds: float) -> str:
    total_seconds = max(0.0, float(seconds))
    minutes, secs = divmod(total_seconds, 60.0)
    hours, minutes = divmod(minutes, 60.0)

    if hours >= 1:
        return f"{int(hours)}h {int(minutes)}m {secs:.1f}s"
    if minutes >= 1:
        return f"{int(minutes)}m {secs:.1f}s"
    return f"{secs:.1f}s"


@contextmanager
def tqdm_joblib(total: int, desc: str):
    if tqdm is None:
        yield None
        return

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


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols = [
        c for c in df.columns
        if c not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    ]

    legacy_f_cols = [
        c for c in numeric_cols
        if c.startswith("f") and c not in DUPLICATE_LEGACY_FEATURES
    ]
    descriptive_cols = [
        c for c in numeric_cols
        if not c.startswith("f") and c not in DUPLICATE_DESCRIPTIVE_FEATURES
    ]

    if descriptive_cols:
        return descriptive_cols
    return legacy_f_cols


def assert_all_finite(name: str, X: np.ndarray) -> None:
    finite_mask = np.isfinite(X)
    if finite_mask.all():
        return

    invalid_total = int((~finite_mask).sum())
    bad_rows, bad_cols = np.where(~finite_mask)
    preview = list(zip(bad_rows[:5].tolist(), bad_cols[:5].tolist()))
    raise ValueError(
        f"{name} contains {invalid_total} non-finite values (NaN or inf). "
        f"First invalid positions: {preview}"
    )


def make_balanced_subset(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """
    Downsample every class to the smallest class count so evaluation uses the
    same support for each class.
    """
    classes, counts = np.unique(y, return_counts=True)
    min_count = int(counts.min())
    rng = np.random.default_rng(seed)

    selected_indices = []
    for cls in classes:
        cls_indices = np.flatnonzero(y == cls)
        chosen = rng.choice(cls_indices, size=min_count, replace=False)
        selected_indices.append(np.sort(chosen))

    selected_indices = np.sort(np.concatenate(selected_indices))
    support_summary = {str(cls): min_count for cls in classes}
    return X[selected_indices], y[selected_indices], support_summary


def prepare_run_output_dirs(output_root: Path) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    runs_dir = output_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    models_dir = run_dir / "models"
    reports_dir = run_dir / "reports"
    cm_dir = run_dir / "cm"
    metadata_dir = run_dir / "metadata"

    for path in (models_dir, reports_dir, cm_dir, metadata_dir):
        path.mkdir(parents=True, exist_ok=True)

    (output_root / "latest_run.txt").write_text(str(run_dir), encoding="utf-8")

    return {
        "output_root": output_root,
        "runs_dir": runs_dir,
        "run_dir": run_dir,
        "models_dir": models_dir,
        "reports_dir": reports_dir,
        "cm_dir": cm_dir,
        "metadata_dir": metadata_dir,
    }


def load_training_ready_manifest(manifest_csv_path: str | Path) -> pd.DataFrame:
    manifest_path = Path(manifest_csv_path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Training-ready manifest not found: {manifest_path}"
        )

    manifest_df = pd.read_csv(manifest_path)
    required_columns = {"family", "method_name", "variant", "csv_path"}
    missing_columns = required_columns - set(manifest_df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Manifest is missing required columns: {missing_text}"
        )
    if manifest_df.empty:
        raise ValueError(f"Manifest has no dataset rows: {manifest_path}")

    return manifest_df


def collect_batch_training_datasets(
    manifest_csv_path: str | Path,
) -> list[dict[str, Any]]:
    manifest_df = load_training_ready_manifest(manifest_csv_path)
    datasets: list[dict[str, Any]] = []

    def optional_int(row: dict[str, Any], key: str) -> int | None:
        value = row.get(key)
        if value is None or pd.isna(value):
            return None
        return int(value)

    for row in manifest_df.to_dict(orient="records"):
        csv_path = Path(str(row["csv_path"]))
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Exported training CSV listed in manifest does not exist: {csv_path}"
            )

        datasets.append({
            "family": str(row["family"]),
            "method_name": str(row["method_name"]),
            "variant": str(row["variant"]),
            "csv_path": str(csv_path),
            "n_features": optional_int(row, "n_features"),
            "n_rows": optional_int(row, "n_rows"),
        })

    return datasets


def annotate_comparison_table(
    comparison_df: pd.DataFrame,
    dataset_info: dict[str, Any],
    run_dir: Path,
) -> pd.DataFrame:
    annotated_df = comparison_df.copy()
    annotated_df.insert(0, "dataset_variant", dataset_info["variant"])
    annotated_df.insert(0, "dataset_method_name", dataset_info["method_name"])
    annotated_df.insert(0, "dataset_family", dataset_info["family"])
    annotated_df.insert(3, "dataset_csv_path", dataset_info["csv_path"])
    annotated_df.insert(4, "dataset_n_features", dataset_info["n_features"])
    annotated_df.insert(5, "dataset_n_rows", dataset_info["n_rows"])
    annotated_df.insert(6, "run_dir", str(run_dir))
    return annotated_df


def load_split_dataset(csv_path: str, val_fraction: float = 0.2, seed: int = 42) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    feature_cols = get_feature_columns(df)

    if not feature_cols:
        raise ValueError(
            "No feature columns found. Expected descriptive feature names or legacy f-columns."
        )
    if "species" not in df.columns:
        raise ValueError("Missing required 'species' column.")
    if "split" not in df.columns:
        raise ValueError("Missing required 'split' column.")

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y_raw = df["species"].to_numpy()
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = label_encoder.classes_
    splits = df["split"].to_numpy()

    train_mask = splits == "train"
    test_mask = splits == "test"

    if not train_mask.any():
        raise ValueError("No rows with split='train'.")
    if not test_mask.any():
        raise ValueError("No rows with split='test'.")

    if np.any(splits == "val"):
        print("[INFO] Existing 'val' rows detected, but they will be ignored.")
        print("[INFO] Validation is now created internally from the TRAIN split only.")

    X_train_full = X[train_mask]
    y_train_full = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    try:
        train_idx, val_idx = next(splitter.split(X_train_full, y_train_full))
    except ValueError as exc:
        raise ValueError(
            "Could not create an internal stratified validation split from the TRAIN rows. "
            "Check whether every class has enough samples in the train split."
        ) from exc

    X_train = X_train_full[train_idx]
    y_train = y_train_full[train_idx]
    X_val = X_train_full[val_idx]
    y_val = y_train_full[val_idx]
    X_val_balanced, y_val_balanced, val_balanced_support = make_balanced_subset(
        X_val, y_val, seed=seed
    )
    X_test_balanced, y_test_balanced, test_balanced_support = make_balanced_subset(
        X_test, y_test, seed=seed
    )

    assert_all_finite("X_train", X_train)
    assert_all_finite("X_val", X_val)
    assert_all_finite("X_val_balanced", X_val_balanced)
    assert_all_finite("X_test", X_test)
    assert_all_finite("X_test_balanced", X_test_balanced)

    feature_schema = (
        "descriptive"
        if feature_cols and any(not c.startswith("f") for c in feature_cols)
        else "legacy_f"
    )

    metadata_cols_present = [
        c for c in ["sample_id", "sample_name", "orig_filename", "species", "split"]
        if c in df.columns
    ]
    absolute_train_indices = np.flatnonzero(train_mask)
    absolute_test_indices = np.flatnonzero(test_mask)
    absolute_internal_train = absolute_train_indices[train_idx]
    absolute_internal_val = absolute_train_indices[val_idx]

    assignment = pd.Series("ignored", index=df.index, dtype=object)
    assignment.iloc[absolute_internal_train] = "train_internal"
    assignment.iloc[absolute_internal_val] = "val_internal"
    assignment.iloc[absolute_test_indices] = "test"

    selected_indices = np.concatenate(
        [absolute_internal_train, absolute_internal_val, absolute_test_indices]
    )
    split_membership_df = df.loc[selected_indices, metadata_cols_present].copy()
    split_membership_df["pipeline_split"] = assignment.loc[selected_indices].values
    split_membership_df = split_membership_df.reset_index(names="row_index")

    split_summary = {
        "train_internal_samples": int(X_train.shape[0]),
        "val_internal_samples": int(X_val.shape[0]),
        "val_balanced_samples": int(X_val_balanced.shape[0]),
        "train_full_samples": int(X_train_full.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "test_balanced_samples": int(X_test_balanced.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_schema": feature_schema,
        "internal_val_fraction": float(val_fraction),
        "n_classes": int(len(class_names)),
    }

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_train_full": X_train_full,
        "y_train_full": y_train_full,
        "X_test": X_test,
        "y_test": y_test,
        "X_val_balanced": X_val_balanced,
        "y_val_balanced": y_val_balanced,
        "X_test_balanced": X_test_balanced,
        "y_test_balanced": y_test_balanced,
        "classes": class_names,
        "label_encoder": label_encoder,
        "n_features": X.shape[1],
        "feature_cols": feature_cols,
        "val_fraction": val_fraction,
        "feature_schema": feature_schema,
        "split_membership_df": split_membership_df,
        "split_summary": split_summary,
        "val_balanced_support": val_balanced_support,
        "test_balanced_support": test_balanced_support,
    }


def make_model_spaces(
    seed: int = 42,
    tree_model_n_jobs: int = 1,
    xgb_n_jobs: int = 1,
) -> Dict[str, Tuple[Pipeline, Dict[str, list]]]:
    spaces: Dict[str, Tuple[Pipeline, Dict[str, list]]] = {}

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
    spaces["svm_rbf"] = (svm_pipe, svm_grid)

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
    spaces["knn"] = (knn_pipe, knn_grid)

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

        # Active XGBoost grid: FAST
        # Purpose:
        # - quick benchmarking
        # - debugging the full pipeline without letting XGBoost dominate runtime
        # Grid size:
        # - 2 * 2 * 1 * 1 * 2 = 8 combinations
        # - with 5 CV folds = 40 fits
        xgb_grid = {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [3, 4],
            "clf__learning_rate": [0.1],
            "clf__subsample": [0.8],
            "clf__colsample_bytree": [0.8, 1.0],
        }

        # Alternative XGBoost grid: BALANCED
        # Purpose:
        # - regular experiments
        # - stronger search than FAST without the full runtime cost of THOROUGH
        # Grid size:
        # - 3 * 2 * 2 * 2 * 2 = 48 combinations
        # - with 5 CV folds = 240 fits
        #
        # xgb_grid = {
        #     "clf__n_estimators": [100, 200, 300],
        #     "clf__max_depth": [4, 6],
        #     "clf__learning_rate": [0.05, 0.1],
        #     "clf__subsample": [0.8, 1.0],
        #     "clf__colsample_bytree": [0.8, 1.0],
        # }

        # Alternative XGBoost grid: THOROUGH
        # Purpose:
        # - final tuning when XGBoost already looks promising
        # - much slower; not ideal for routine runs
        # Grid size:
        # - 3 * 4 * 3 * 2 * 2 = 144 combinations
        # - with 5 CV folds = 720 fits
        #
        # xgb_grid = {
        #     "clf__n_estimators": [100, 200, 400],
        #     "clf__max_depth": [3, 4, 6, 8],
        #     "clf__learning_rate": [0.03, 0.05, 0.1],
        #     "clf__subsample": [0.8, 1.0],
        #     "clf__colsample_bytree": [0.8, 1.0],
        # }
        spaces["xgboost"] = (xgb_pipe, xgb_grid)
    except ImportError:
        print("[WARN] xgboost is not installed. Skipping XGBoost model.")

    return spaces


def evaluate(
    model,
    X: np.ndarray,
    y_encoded: np.ndarray,
    label_encoder: LabelEncoder,
    labels: np.ndarray,
) -> Dict[str, Any]:
    y_pred_encoded = np.asarray(model.predict(X), dtype=np.int64)
    y_true = label_encoder.inverse_transform(np.asarray(y_encoded, dtype=np.int64))
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "cm": cm,
        "report_text": classification_report(y_true, y_pred, zero_division=0),
    }


def save_confusion_matrix(cm: np.ndarray, labels: np.ndarray, out_path: Path, title: str) -> None:
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)

    max_val = cm.max() if cm.size > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > (max_val / 2) else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8, color=color)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_training_pipeline(
    data_csv_path: str | Path = DEFAULT_DATA_CSV_PATH,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    seed: int = 42,
    cv_folds: int = 5,
    scoring: str = "f1_macro",
    val_fraction: float = 0.2,
    gridsearch_n_jobs: int = -1,
    gridsearch_pre_dispatch: str = "2*n_jobs",
    gridsearch_verbose: int = 0,
    tree_model_n_jobs: int = 1,
    xgb_n_jobs: int = 1,
) -> Path:
    data_csv_path = str(data_csv_path)
    out_dir = Path(out_dir)
    total_start = time.perf_counter()
    stage_timings = []
    model_timings = []

    set_seed(seed)

    stage_start = time.perf_counter()
    paths = prepare_run_output_dirs(out_dir)
    run_dir = paths["run_dir"]
    models_dir = paths["models_dir"]
    reports_dir = paths["reports_dir"]
    cm_dir = paths["cm_dir"]
    metadata_dir = paths["metadata_dir"]
    stage_elapsed = time.perf_counter() - stage_start
    stage_timings.append({"stage": "prepare_run_output_dirs", "seconds": stage_elapsed})
    print(f"[TIME] prepare_run_output_dirs: {format_seconds(stage_elapsed)}")

    stage_start = time.perf_counter()
    data = load_split_dataset(data_csv_path, val_fraction=val_fraction, seed=seed)
    stage_elapsed = time.perf_counter() - stage_start
    stage_timings.append({"stage": "load_split_dataset", "seconds": stage_elapsed})
    print(f"[TIME] load_split_dataset: {format_seconds(stage_elapsed)}")

    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_val_balanced, y_val_balanced = data["X_val_balanced"], data["y_val_balanced"]
    X_train_full, y_train_full = data["X_train_full"], data["y_train_full"]
    X_test, y_test = data["X_test"], data["y_test"]
    X_test_balanced, y_test_balanced = data["X_test_balanced"], data["y_test_balanced"]
    classes = data["classes"]
    label_encoder = data["label_encoder"]
    feature_cols = data["feature_cols"]
    feature_schema = data["feature_schema"]
    split_membership_df = data["split_membership_df"]
    split_summary = data["split_summary"]
    val_balanced_support = data["val_balanced_support"]
    test_balanced_support = data["test_balanced_support"]

    print(f"[INFO] Train shape:        {X_train.shape}")
    print(f"[INFO] Internal val shape: {X_val.shape}")
    print(f"[INFO] Balanced val shape: {X_val_balanced.shape}")
    print(f"[INFO] Full train shape:   {X_train_full.shape}")
    print(f"[INFO] Test shape:         {X_test.shape}")
    print(f"[INFO] Balanced test shape:{X_test_balanced.shape}")
    print(f"[INFO] Classes:            {len(classes)}")
    print(f"[INFO] Feature columns:    {len(feature_cols)}")
    print(f"[INFO] Feature schema:     {feature_schema}")
    print(f"[INFO] Run directory:       {run_dir}")

    stage_start = time.perf_counter()
    with (metadata_dir / "feature_columns.json").open("w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)
    (metadata_dir / "feature_columns.txt").write_text("\n".join(feature_cols), encoding="utf-8")
    with (metadata_dir / "class_names.json").open("w", encoding="utf-8") as f:
        json.dump(classes.tolist(), f, indent=2)
    joblib.dump(label_encoder, metadata_dir / "label_encoder.pkl")
    with (metadata_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "data_csv_path": data_csv_path,
                "seed": seed,
                "cv_folds": cv_folds,
                "scoring": scoring,
                "internal_val_fraction": val_fraction,
                "gridsearch_n_jobs": gridsearch_n_jobs,
                "gridsearch_pre_dispatch": gridsearch_pre_dispatch,
                "tree_model_n_jobs": tree_model_n_jobs,
                "xgb_n_jobs": xgb_n_jobs,
            },
            f,
            indent=2,
        )
    with (metadata_dir / "split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)
    with (metadata_dir / "balanced_eval_support.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "internal_validation": val_balanced_support,
                "test": test_balanced_support,
            },
            f,
            indent=2,
        )
    split_membership_df.to_csv(metadata_dir / "split_membership.csv", index=False)
    stage_elapsed = time.perf_counter() - stage_start
    stage_timings.append({"stage": "save_initial_metadata", "seconds": stage_elapsed})
    print(f"[TIME] save_initial_metadata: {format_seconds(stage_elapsed)}")

    stage_start = time.perf_counter()
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    spaces = make_model_spaces(
        seed=seed,
        tree_model_n_jobs=tree_model_n_jobs,
        xgb_n_jobs=xgb_n_jobs,
    )
    stage_elapsed = time.perf_counter() - stage_start
    stage_timings.append({"stage": "build_model_spaces", "seconds": stage_elapsed})
    print(f"[TIME] build_model_spaces: {format_seconds(stage_elapsed)}")

    if not spaces:
        raise RuntimeError("No model spaces available.")

    val_rows = []
    test_rows = []
    best_models_train = {}
    best_params = {}

    for model_name, (pipe, grid) in spaces.items():
        print("\n" + "=" * 70)
        print(f"[INFO] Tuning model: {model_name}")
        model_stage_start = time.perf_counter()
        n_candidates = len(list(ParameterGrid(grid)))
        total_cv_fits = n_candidates * cv_folds
        print(f"[INFO] Grid candidates:    {n_candidates}")
        print(f"[INFO] Expected CV fits:  {total_cv_fits}")

        gs = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring=scoring,
            cv=cv,
            n_jobs=gridsearch_n_jobs,
            pre_dispatch=gridsearch_pre_dispatch,
            refit=True,
            error_score="raise",
            verbose=gridsearch_verbose,
        )
        with tqdm_joblib(total=total_cv_fits, desc=f"{model_name} CV") as progress_bar:
            gs.fit(X_train, y_train)
            if progress_bar is not None:
                progress_bar.set_postfix_str("refit")
        tuning_elapsed = time.perf_counter() - model_stage_start

        best_model = gs.best_estimator_
        best_models_train[model_name] = best_model
        best_params[model_name] = gs.best_params_

        print(f"[INFO] Best {model_name} params: {gs.best_params_}")
        print(f"[INFO] Best {model_name} CV {scoring}: {gs.best_score_:.4f}")
        print(f"[TIME] {model_name} tuning: {format_seconds(tuning_elapsed)}")

        eval_stage_start = time.perf_counter()
        val_metrics = evaluate(
            best_model,
            X_val_balanced,
            y_val_balanced,
            label_encoder,
            classes,
        )
        val_rows.append(
            {
                "model": model_name,
                "best_cv_score": float(gs.best_score_),
                "val_samples_balanced": int(X_val_balanced.shape[0]),
                "val_accuracy": val_metrics["accuracy"],
                "val_precision_macro": val_metrics["precision_macro"],
                "val_recall_macro": val_metrics["recall_macro"],
                "val_f1_macro": val_metrics["f1_macro"],
            }
        )

        report_path = reports_dir / f"{model_name}_internal_val_report.txt"
        report_path.write_text(val_metrics["report_text"], encoding="utf-8")
        save_confusion_matrix(
            val_metrics["cm"],
            classes,
            cm_dir / f"{model_name}_internal_val_cm.png",
            f"{model_name} - Internal VAL (Balanced)",
        )

        joblib.dump(best_model, models_dir / f"{model_name}_train_only.pkl")
        validation_elapsed = time.perf_counter() - eval_stage_start
        total_model_elapsed = time.perf_counter() - model_stage_start
        print(f"[TIME] {model_name} internal validation + save: {format_seconds(validation_elapsed)}")
        print(f"[TIME] {model_name} total train-stage time: {format_seconds(total_model_elapsed)}")
        model_timings.append(
            {
                "model": model_name,
                "phase": "internal_train_and_validation",
                "seconds": total_model_elapsed,
                "tuning_seconds": tuning_elapsed,
                "post_tuning_seconds": validation_elapsed,
            }
        )

    for model_name, train_best_model in best_models_train.items():
        model_stage_start = time.perf_counter()
        final_model = clone(train_best_model)
        final_model.fit(X_train_full, y_train_full)
        refit_elapsed = time.perf_counter() - model_stage_start

        eval_stage_start = time.perf_counter()
        test_metrics = evaluate(
            final_model,
            X_test_balanced,
            y_test_balanced,
            label_encoder,
            classes,
        )
        test_rows.append(
            {
                "model": model_name,
                "test_samples_balanced": int(X_test_balanced.shape[0]),
                "test_accuracy": test_metrics["accuracy"],
                "test_precision_macro": test_metrics["precision_macro"],
                "test_recall_macro": test_metrics["recall_macro"],
                "test_f1_macro": test_metrics["f1_macro"],
            }
        )

        report_path = reports_dir / f"{model_name}_test_report.txt"
        report_path.write_text(test_metrics["report_text"], encoding="utf-8")
        save_confusion_matrix(
            test_metrics["cm"],
            classes,
            cm_dir / f"{model_name}_test_cm.png",
            f"{model_name} - TEST (Balanced)",
        )

        joblib.dump(final_model, models_dir / f"{model_name}_final.pkl")
        test_eval_elapsed = time.perf_counter() - eval_stage_start
        total_model_elapsed = time.perf_counter() - model_stage_start
        print(f"[TIME] {model_name} refit_full_train: {format_seconds(refit_elapsed)}")
        print(f"[TIME] {model_name} test_evaluation + save: {format_seconds(test_eval_elapsed)}")
        print(f"[TIME] {model_name} total test-stage time: {format_seconds(total_model_elapsed)}")
        model_timings.append(
            {
                "model": model_name,
                "phase": "final_refit_and_test",
                "seconds": total_model_elapsed,
                "refit_seconds": refit_elapsed,
                "test_eval_seconds": test_eval_elapsed,
            }
        )

    stage_start = time.perf_counter()
    val_df = pd.DataFrame(val_rows).sort_values("val_f1_macro", ascending=False)
    test_df = pd.DataFrame(test_rows).sort_values("test_f1_macro", ascending=False)

    val_df.to_csv(run_dir / "internal_validation_comparison.csv", index=False)
    test_df.to_csv(run_dir / "test_comparison.csv", index=False)
    with (run_dir / "best_params.json").open("w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    stage_elapsed = time.perf_counter() - stage_start
    stage_timings.append({"stage": "save_final_artifacts", "seconds": stage_elapsed})
    pd.DataFrame(stage_timings).to_csv(run_dir / "stage_timings.csv", index=False)
    pd.DataFrame(model_timings).to_csv(run_dir / "model_timings.csv", index=False)
    total_elapsed = time.perf_counter() - total_start
    with (run_dir / "runtime_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "total_seconds": total_elapsed,
                "total_formatted": format_seconds(total_elapsed),
                "stage_timings": stage_timings,
                "model_timings": model_timings,
            },
            f,
            indent=2,
        )
    print(f"[TIME] save_final_artifacts: {format_seconds(stage_elapsed)}")

    print("\n" + "=" * 70)
    print("[DONE] Internal validation ranking:")
    print(val_df)
    print("\n[DONE] Test ranking:")
    print(test_df)
    print(f"[TIME] Total pipeline runtime: {format_seconds(total_elapsed)}")
    print(f"\n[INFO] Artifacts saved in: {run_dir.resolve()}")
    return run_dir


def run_training_pipeline_batch(
    manifest_csv_path: str | Path = DEFAULT_TRAINING_READY_MANIFEST_PATH,
    batch_out_dir: str | Path = DEFAULT_BATCH_OUT_DIR,
    seed: int = 42,
    cv_folds: int = 5,
    scoring: str = "f1_macro",
    val_fraction: float = 0.2,
    gridsearch_n_jobs: int = -1,
    gridsearch_pre_dispatch: str = "2*n_jobs",
    gridsearch_verbose: int = 0,
    tree_model_n_jobs: int = 1,
    xgb_n_jobs: int = 1,
) -> Path:
    datasets = collect_batch_training_datasets(manifest_csv_path)

    batch_root = Path(batch_out_dir)
    batch_root.mkdir(parents=True, exist_ok=True)
    batch_run_dir = batch_root / datetime.now().strftime("batch_%Y%m%d_%H%M%S")
    batch_run_dir.mkdir(parents=True, exist_ok=False)
    (batch_root / "latest_batch_run.txt").write_text(str(batch_run_dir), encoding="utf-8")

    per_dataset_root = batch_run_dir / "per_dataset"
    per_dataset_root.mkdir(parents=True, exist_ok=True)

    print("[BATCH] Starting batch shallow-learning training.")
    print(f"[BATCH] Manifest: {Path(manifest_csv_path)}")
    print(f"[BATCH] Datasets to run: {len(datasets)}")

    batch_start = time.perf_counter()
    batch_rows: list[dict[str, Any]] = []
    validation_tables: list[pd.DataFrame] = []
    test_tables: list[pd.DataFrame] = []

    for idx, dataset_info in enumerate(datasets, start=1):
        dataset_label = (
            f"{dataset_info['family']}/{dataset_info['method_name']}/{dataset_info['variant']}"
        )
        print("\n" + "#" * 80)
        print(f"[BATCH] Dataset {idx}/{len(datasets)}: {dataset_label}")
        print(f"[BATCH] CSV path: {dataset_info['csv_path']}")

        dataset_out_dir = (
            per_dataset_root
            / dataset_info["family"]
            / dataset_info["method_name"]
            / dataset_info["variant"]
        )

        dataset_start = time.perf_counter()
        run_dir = run_training_pipeline(
            data_csv_path=dataset_info["csv_path"],
            out_dir=dataset_out_dir,
            seed=seed,
            cv_folds=cv_folds,
            scoring=scoring,
            val_fraction=val_fraction,
            gridsearch_n_jobs=gridsearch_n_jobs,
            gridsearch_pre_dispatch=gridsearch_pre_dispatch,
            gridsearch_verbose=gridsearch_verbose,
            tree_model_n_jobs=tree_model_n_jobs,
            xgb_n_jobs=xgb_n_jobs,
        )
        dataset_elapsed = time.perf_counter() - dataset_start

        val_df = pd.read_csv(run_dir / "internal_validation_comparison.csv")
        test_df = pd.read_csv(run_dir / "test_comparison.csv")
        validation_tables.append(annotate_comparison_table(val_df, dataset_info, run_dir))
        test_tables.append(annotate_comparison_table(test_df, dataset_info, run_dir))

        top_val_row = val_df.iloc[0].to_dict()
        top_test_row = test_df.iloc[0].to_dict()
        batch_rows.append({
            "family": dataset_info["family"],
            "method_name": dataset_info["method_name"],
            "variant": dataset_info["variant"],
            "csv_path": dataset_info["csv_path"],
            "n_features": dataset_info["n_features"],
            "n_rows": dataset_info["n_rows"],
            "run_dir": str(run_dir),
            "runtime_seconds": dataset_elapsed,
            "best_val_model": top_val_row["model"],
            "best_val_f1_macro": top_val_row["val_f1_macro"],
            "best_test_model": top_test_row["model"],
            "best_test_f1_macro": top_test_row["test_f1_macro"],
        })

    batch_manifest_df = pd.DataFrame(batch_rows).sort_values(
        ["family", "method_name", "variant"],
        ignore_index=True,
    )
    batch_manifest_df.to_csv(batch_run_dir / "batch_run_manifest.csv", index=False)

    if validation_tables:
        validation_all_df = pd.concat(validation_tables, ignore_index=True)
        validation_all_df.to_csv(
            batch_run_dir / "batch_internal_validation_comparison.csv",
            index=False,
        )
    else:
        validation_all_df = pd.DataFrame()

    if test_tables:
        test_all_df = pd.concat(test_tables, ignore_index=True)
        test_all_df.to_csv(batch_run_dir / "batch_test_comparison.csv", index=False)
    else:
        test_all_df = pd.DataFrame()

    batch_elapsed = time.perf_counter() - batch_start
    summary = {
        "manifest_csv_path": str(Path(manifest_csv_path)),
        "batch_run_dir": str(batch_run_dir),
        "dataset_count": int(len(datasets)),
        "model_evaluation_rows_validation": int(len(validation_all_df)),
        "model_evaluation_rows_test": int(len(test_all_df)),
        "total_seconds": float(batch_elapsed),
        "total_formatted": format_seconds(batch_elapsed),
    }
    with (batch_run_dir / "batch_runtime_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "#" * 80)
    print("[BATCH] Completed batch shallow-learning training.")
    print(f"[BATCH] Batch run directory: {batch_run_dir}")
    print(f"[BATCH] Total runtime: {format_seconds(batch_elapsed)}")
    return batch_run_dir


def main(
    data_csv_path: str | Path = DEFAULT_DATA_CSV_PATH,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    batch_manifest_csv_path: str | Path | None = None,
    batch_out_dir: str | Path = DEFAULT_BATCH_OUT_DIR,
) -> None:
    if batch_manifest_csv_path is not None:
        run_training_pipeline_batch(
            manifest_csv_path=batch_manifest_csv_path,
            batch_out_dir=batch_out_dir,
        )
        return

    run_training_pipeline(data_csv_path=data_csv_path, out_dir=out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train shallow-learning models on a selected feature CSV."
    )
    parser.add_argument(
        "--data-csv",
        default=DEFAULT_DATA_CSV_PATH,
        help="Path to the feature CSV that includes species, split, and selected feature columns.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Directory where training artifacts will be saved.",
    )
    parser.add_argument(
        "--batch-manifest-csv",
        default=None,
        help=(
            "Path to training_ready_dataset_manifest.csv. "
            "When provided, the script runs all exported feature-set CSVs in batch mode."
        ),
    )
    parser.add_argument(
        "--run-exported-batch",
        action="store_true",
        help=(
            "Run all exported feature-set CSVs listed in the default training-ready manifest at "
            f"{DEFAULT_TRAINING_READY_MANIFEST_PATH}."
        ),
    )
    parser.add_argument(
        "--batch-out-dir",
        default=str(DEFAULT_BATCH_OUT_DIR),
        help="Root directory where batch-mode training outputs will be saved.",
    )
    args = parser.parse_args()
    batch_manifest_csv_path = args.batch_manifest_csv
    if args.run_exported_batch and batch_manifest_csv_path is None:
        batch_manifest_csv_path = DEFAULT_TRAINING_READY_MANIFEST_PATH

    main(
        data_csv_path=args.data_csv,
        out_dir=args.out_dir,
        batch_manifest_csv_path=batch_manifest_csv_path,
        batch_out_dir=args.batch_out_dir,
    )
