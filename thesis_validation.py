"""Nested SL evaluation and statistical tests within one model family."""

from __future__ import annotations

import argparse
from functools import partial
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from classification_analysis import partition_id, save_json
from result_plots import plot_matrix, plot_outer_scores
from shared_training_pool import DEFAULT_POOL_MANIFEST, save_pool_record, select_training_rows
from shared_augmentation import TrainingOnlyAugmentationCV, validate_feature_augmentation


def compare_outer_scores(paths, out_dir):
    scores = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    required = {"family", "model", "repeat", "fold", "partition_id", "score",
                "evaluation_role", "n_train", "n_validation"}
    if not required.issubset(scores):
        raise ValueError(f"Required columns: {sorted(required)}")
    if scores.family.nunique() != 1 or scores.family.iloc[0] not in {"SL", "DL"}:
        raise ValueError("Statistical comparisons must contain exactly one family: SL or DL.")
    if not scores.evaluation_role.eq("independent_outer").all():
        raise ValueError("Use outer-fold scores with selection/tuning confined to inner training.")
    if scores.duplicated(["model", "repeat", "fold"]).any():
        raise ValueError("Duplicate model/repeat/fold records.")
    if not np.isfinite(scores.score).all() or not scores.score.between(0, 1).all():
        raise ValueError("Expected finite macro-F1 scores in [0, 1].")
    if "training_pool_id" in scores and scores.training_pool_id.nunique(dropna=False) != 1:
        raise ValueError("Models must use the same shared training pool.")
    if "augmentation_id" in scores and scores.augmentation_id.nunique(dropna=False) != 1:
        raise ValueError("Models must use the same shared augmentation protocol.")
    results = []
    for left, right in combinations(sorted(scores.model.unique()), 2):
        a = scores.loc[scores.model.eq(left)].set_index(["repeat", "fold"]).sort_index()
        b = scores.loc[scores.model.eq(right)].set_index(["repeat", "fold"]).sort_index()
        if not a.index.equals(b.index) or len(a) < 3:
            raise ValueError("Models need the same repeat/fold pairs (at least three).")
        for col in ["partition_id", "n_train", "n_validation"]:
            if not a[col].equals(b[col]):
                raise ValueError(f"Unmatched outer folds: {col}.")
        if (a.n_train <= 0).any() or (a.n_validation <= 0).any():
            raise ValueError("Fold sizes must be positive.")
        differences = a.score.to_numpy() - b.score.to_numpy()
        variance = float(differences.var(ddof=1))
        mean = float(differences.mean())
        # Correct for overlapping training samples in repeated k-fold CV.
        correction = 1 / len(a) + float((a.n_validation / a.n_train).mean())
        if np.all(differences == 0):
            statistic, p_value, status = 0.0, 1.0, "identical_scores"
        elif variance == 0:
            statistic, p_value, status = np.nan, np.nan, "insufficient_difference_variance"
        else:
            statistic = mean / np.sqrt(correction * variance)
            p_value = float(2 * student_t.sf(abs(statistic), df=len(a) - 1))
            status = "approximate_corrected_paired_t"
        results.append({"family": scores.family.iloc[0], "left_model": left,
                        "right_model": right, "pairs": len(a), "mean_f1_difference": mean,
                        "statistic": statistic, "p_value": p_value, "status": status})
    if not results:
        raise ValueError("At least two models required within the same family.")
    result = pd.DataFrame(results)
    result["p_holm"] = np.nan
    valid = result.p_value.dropna().sort_values()
    running = 0.0
    for rank, (idx, p_value) in enumerate(valid.items()):
        running = max(running, min(1.0, (len(valid) - rank) * p_value))
        result.loc[idx, "p_holm"] = running
    result["significant_at_0_05"] = result.p_holm < 0.05
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_dir / "outer_scores.csv", index=False)
    scores.groupby(["family", "model"]).score.agg(["mean", "std", "count"]).to_csv(
        out_dir / "family_cv_summary.csv")
    result.to_csv(out_dir / "within_family_tests.csv", index=False)
    plot_outer_scores(scores, out_dir / "outer_cv_scores.png")
    save_json(out_dir / "statistical_protocol.json", {
        "family": scores.family.iloc[0], "metric": "macro_f1",
        "test": "Nadeau-Bengio corrected paired t-test", "multiplicity": "Holm",
        "limitations": "Approximate inference; few folds have low power. Models must use the "
        "same sampling, partitions and outer evaluation protocol. Never use test scores, "
        "epoch-selection validation scores, or winning GridSearch folds as outer scores.",
    })
    return result


def run_sl_nested_cv(data_csv, out_dir, method="mi", n_features=50, folds=5, seed=42,
                     pool_manifest_path=DEFAULT_POOL_MANIFEST):
    from sklearn.base import clone
    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
    from sklearn.metrics import confusion_matrix, f1_score
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline
    import joblib
    from sl_training_pipeline import get_feature_columns, make_model_spaces

    df = pd.read_csv(data_csv)
    if "split" not in df or "species" not in df:
        raise ValueError("Input must have split and species columns.")
    df = df.loc[df.split.isin(["train", "train_augmented"])].copy()
    df, pool_info = select_training_rows(df, seed=seed, manifest_path=pool_manifest_path)
    validate_feature_augmentation(df, pool_manifest_path)
    features = get_feature_columns(df)
    if not 0 < n_features < len(features):
        raise ValueError("Select fewer features than the original feature-vector size.")
    X = df[features].to_numpy(dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError("Nested evaluation requires finite features.")
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder().fit(df.species)
    y = encoder.transform(df.species)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_pool_record(pool_info, out_dir)
    keys = df.species.astype(str) + "/" + df.orig_filename.astype(str)
    if keys.duplicated().any():
        raise ValueError("Each original image must have a unique species/filename key.")
    outer = TrainingOnlyAugmentationCV(df, folds=folds, seed=seed)
    score_func = f_classif if method == "anova" else partial(mutual_info_classif, random_state=seed)
    rows = []
    for fold, (train_idx, val_idx) in enumerate(outer.split(X, y), 1):
        for name, (pipe, grid) in make_model_spaces(seed=seed).items():
            pipeline = Pipeline([("selection", SelectKBest(score_func, k=n_features))] +
                                clone(pipe).steps)
            # Classification scoring does not require SVM probability calibration.
            if name == "svm_rbf":
                pipeline.set_params(clf__probability=False)
            inner = TrainingOnlyAugmentationCV(df.iloc[train_idx], folds=folds, seed=seed)
            search = GridSearchCV(pipeline, grid, scoring="f1_macro", cv=inner,
                                  n_jobs=-1, pre_dispatch="n_jobs", error_score="raise")
            with joblib.parallel_backend("threading"):
                search.fit(X[train_idx], y[train_idx])
            pred = search.predict(X[val_idx])
            model_name = f"{name}_{method}_top_{n_features}"
            plot_matrix(confusion_matrix(y[val_idx], pred, labels=np.arange(len(encoder.classes_))),
                        encoder.classes_, out_dir / f"{model_name}_fold_{fold}_cm.png",
                        f"{model_name}: independent outer fold {fold}",
                        xlabel="Predicted class", ylabel="True class")
            rows.append({
                "family": "SL", "model": model_name, "repeat": 0, "fold": fold,
                "partition_id": partition_id(keys.iloc[val_idx].tolist()),
                "evaluation_role": "independent_outer",
                "n_train": int(df.iloc[train_idx].split.eq("train").sum()),
                "n_validation": len(val_idx),
                "score": float(f1_score(y[val_idx], pred, average="macro", zero_division=0)),
                "training_pool_id": pool_info["training_pool_id"] if pool_info else None,
                "augmentation_id": (str(df.augmentation_id.dropna().iloc[0])
                                     if "augmentation_id" in df and df.augmentation_id.notna().any() else None),
            })
            pd.DataFrame(rows).to_csv(out_dir / "outer_cv_scores.csv", index=False)
            save_json(out_dir / f"{model_name}_fold_{fold}_params.json", search.best_params_)
    plot_outer_scores(pd.DataFrame(rows), out_dir / "outer_cv_scores.png")
    save_json(out_dir / "protocol.json", {
        "family": "SL", "selection": method, "n_features": n_features,
        "folds": folds, "seed": seed, "selection_and_tuning": "inside_inner_CV",
        "input": str(data_csv), "test_rows_used": False,
        "training_pool": pool_info,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="command", required=True)
    compare = modes.add_parser("compare")
    compare.add_argument("scores", type=Path, nargs="+")
    compare.add_argument("--out-dir", type=Path, required=True)
    cv = modes.add_parser("sl-cv")
    cv.add_argument("--data-csv", type=Path, default=Path(
        "F:/01_Univalle/01_TG/dataset_features/shallow_learning_birds.csv"))
    cv.add_argument("--out-dir", type=Path, required=True)
    cv.add_argument("--method", choices=["mi", "anova"], default="mi")
    cv.add_argument("--features", type=int, default=50)
    cv.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    if args.command == "compare":
        compare_outer_scores(args.scores, args.out_dir)
    else:
        run_sl_nested_cv(args.data_csv, args.out_dir, args.method, args.features, args.folds)
