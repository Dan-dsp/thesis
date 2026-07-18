"""
High-level feature comparison workflow for shallow-learning datasets.

This file orchestrates the full selection procedure:
1) exploratory analysis
2) filters
3) wrappers
4) embedded methods
5) final comparisons and diagnostics

The lower-level methods live in `sl_feature_comparison_tools.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from sl_feature_comparison_tools import (
    DEFAULT_TOP_K,
    DERIVED_TOP_COUNTS,
    build_embedded_model_spaces,
    build_wrapper_model_spaces,
    compute_anova_scores,
    compute_fisher_scores,
    compute_mutual_information,
    encode_labels,
    ensure_dir,
    export_training_ready_dataset,
    load_features_and_labels,
    plot_correlation_matrix,
    plot_violin_for_features,
    rank_pca_features,
    run_embedded_feature_importance,
    run_pca_analysis,
    run_shapiro_tests,
    run_wrapper_rfecv,
    save_json,
    save_top_ranked_feature_slices,
)


SEED = 42
PCA_COMPONENTS = 3
CV_FOLDS = 5
REFIT_METRIC = "f1_macro"
MAJORITY_THRESHOLD = 3
GRIDSEARCH_N_JOBS = -1
RFECV_N_JOBS = 1
WRAPPER_RFECV_STEP = 10
TREE_MODEL_N_JOBS = 1
XGB_N_JOBS = 1


def compare_filter_method_rankings(
    anova_df: pd.DataFrame,
    fisher_df: pd.DataFrame,
    mi_df: pd.DataFrame,
    pca_ranked_df: pd.DataFrame,
    save_dir: str | Path | None = None,
    top_k: int = DEFAULT_TOP_K,
    majority_threshold: int = MAJORITY_THRESHOLD,
) -> dict[str, Any]:
    """
    Compare filter rankings using top-100 lists, then derive top-80 and top-50
    by slicing the ordered top-100 consensus list.
    """
    method_frames = {
        "anova": anova_df[["feature", "F_score"]].rename(columns={"F_score": "score"}),
        "fisher": fisher_df[["feature", "fisher_score"]].rename(columns={"fisher_score": "score"}),
        "mi": mi_df[["feature", "mutual_info"]].rename(columns={"mutual_info": "score"}),
        "pca": pca_ranked_df[["feature", "weighted_abs_loading"]].rename(
            columns={"weighted_abs_loading": "score"}
        ),
    }

    method_top_slices: dict[str, dict[int, pd.DataFrame]] = {}
    rank_frames: list[pd.DataFrame] = []
    score_frames: list[pd.DataFrame] = []

    comparison_dir = ensure_dir(save_dir) if save_dir is not None else None

    for method_name, ranking_df in method_frames.items():
        ordered_df = ranking_df.copy().reset_index(drop=True)
        ordered_df[f"rank_{method_name}"] = np.arange(1, len(ordered_df) + 1)
        top_reference = ordered_df.head(top_k).copy().reset_index(drop=True)

        if comparison_dir is not None:
            method_top_slices[method_name] = save_top_ranked_feature_slices(
                top_reference,
                comparison_dir / method_name,
                f"{method_name}_features",
                top_k=top_k,
                derived_top_counts=DERIVED_TOP_COUNTS,
            )
        else:
            method_top_slices[method_name] = {top_k: top_reference}
            for derived_top in DERIVED_TOP_COUNTS:
                method_top_slices[method_name][derived_top] = top_reference.head(derived_top).copy()

        rank_frames.append(top_reference[["feature", f"rank_{method_name}"]])
        score_frames.append(
            top_reference[["feature", "score"]].rename(columns={"score": f"score_{method_name}"})
        )

    consensus_df = rank_frames[0]
    for frame in rank_frames[1:]:
        consensus_df = consensus_df.merge(frame, on="feature", how="outer")
    for frame in score_frames:
        consensus_df = consensus_df.merge(frame, on="feature", how="left")

    rank_cols = [f"rank_{method_name}" for method_name in method_frames]
    consensus_df["vote_count"] = consensus_df[rank_cols].notna().sum(axis=1)
    consensus_df["majority_vote"] = consensus_df["vote_count"] >= majority_threshold
    consensus_df["rank_mean_present"] = consensus_df[rank_cols].mean(axis=1, skipna=True)
    consensus_df["borda_score"] = (
        consensus_df[rank_cols]
        .apply(lambda col: np.where(col.notna(), top_k + 1 - col, 0))
        .sum(axis=1)
    )
    consensus_df = consensus_df.sort_values(
        ["vote_count", "borda_score", "rank_mean_present", "feature"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    overlap_rows: list[dict[str, Any]] = []
    method_names = list(method_frames.keys())
    top_sets = {
        method_name: set(method_top_slices[method_name][top_k]["feature"])
        for method_name in method_names
    }
    for idx, left_name in enumerate(method_names):
        for right_name in method_names[idx + 1:]:
            left_set = top_sets[left_name]
            right_set = top_sets[right_name]
            overlap_rows.append({
                "left_method": left_name,
                "right_method": right_name,
                "top_k": top_k,
                "overlap_count": len(left_set & right_set),
                "jaccard_index": len(left_set & right_set) / len(left_set | right_set),
            })
    pairwise_overlap_df = pd.DataFrame(overlap_rows)

    consensus_slices: dict[int, pd.DataFrame]
    if comparison_dir is not None:
        consensus_slices = save_top_ranked_feature_slices(
            consensus_df,
            comparison_dir,
            "filter_consensus",
            top_k=top_k,
            derived_top_counts=DERIVED_TOP_COUNTS,
        )
        consensus_df.to_csv(comparison_dir / "filter_consensus_full.csv", index=False)
        pairwise_overlap_df.to_csv(comparison_dir / "filter_pairwise_overlap.csv", index=False)
        consensus_df[consensus_df["majority_vote"]].to_csv(
            comparison_dir / "filter_strict_majority_features.csv",
            index=False,
        )
    else:
        consensus_slices = {top_k: consensus_df.head(top_k).copy()}
        for derived_top in DERIVED_TOP_COUNTS:
            consensus_slices[derived_top] = consensus_df.head(derived_top).copy()

    return {
        "method_top_slices": method_top_slices,
        "consensus_df": consensus_df,
        "consensus_slices": consensus_slices,
        "pairwise_overlap_df": pairwise_overlap_df,
    }


def compare_stage_feature_lists(
    stage_feature_lists: dict[str, list[str]],
    save_dir: str | Path | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    stage_names = list(stage_feature_lists.keys())

    for idx, left_name in enumerate(stage_names):
        left_set = set(stage_feature_lists[left_name])
        for right_name in stage_names[idx + 1:]:
            right_set = set(stage_feature_lists[right_name])
            union = left_set | right_set
            rows.append({
                "left_stage": left_name,
                "right_stage": right_name,
                "left_count": len(left_set),
                "right_count": len(right_set),
                "overlap_count": len(left_set & right_set),
                "jaccard_index": (len(left_set & right_set) / len(union)) if union else np.nan,
            })

    overlap_df = pd.DataFrame(rows)

    if save_dir is not None:
        save_path = ensure_dir(save_dir)
        overlap_df.to_csv(save_path / "stage_feature_overlap.csv", index=False)
        pd.DataFrame(
            [
                {"stage_name": stage_name, "n_features": len(features)}
                for stage_name, features in stage_feature_lists.items()
            ]
        ).to_csv(save_path / "stage_feature_counts.csv", index=False)

    return overlap_df


def run_final_selection_diagnostics(
    df: pd.DataFrame,
    label_col: str,
    feature_lists: dict[str, list[str]],
    save_dir: str | Path | None = None,
) -> None:
    if save_dir is None:
        return

    root_dir = ensure_dir(save_dir)

    for list_name, features in feature_lists.items():
        valid_features = [feature for feature in features if feature in df.columns]
        if not valid_features:
            continue

        diag_dir = ensure_dir(root_dir / list_name)
        pd.DataFrame({"feature": valid_features}).to_csv(
            diag_dir / f"{list_name}_features.csv",
            index=False,
        )

        plot_correlation_matrix(
            df,
            valid_features,
            save_dir=diag_dir,
            max_features=min(40, len(valid_features)),
            file_stem=f"{list_name}_correlation",
        )
        plot_violin_for_features(
            df,
            label_col,
            valid_features[: min(12, len(valid_features))],
            save_dir=diag_dir,
            file_name=f"{list_name}_violin.png",
        )


def export_training_ready_feature_datasets(
    df: pd.DataFrame,
    label_col: str,
    filter_comparison: dict[str, Any],
    wrapper_results: dict[str, dict[str, Any]],
    embedded_results: dict[str, dict[str, Any]],
    save_dir: str | Path,
) -> pd.DataFrame:
    export_root = ensure_dir(save_dir)
    manifest_rows: list[dict[str, Any]] = []

    def extract_features(feature_df: pd.DataFrame) -> list[str]:
        if "feature" not in feature_df.columns:
            raise ValueError("Expected a 'feature' column when exporting ranked feature datasets.")
        return feature_df["feature"].dropna().astype(str).tolist()

    def export_feature_list(
        family: str,
        method_name: str,
        variant: str,
        selected_features: list[str],
    ) -> None:
        if not selected_features:
            return

        output_path = export_root / family / method_name / f"{method_name}_{variant}_training.csv"
        export_info = export_training_ready_dataset(
            df,
            selected_features,
            output_path,
            label_col=label_col,
        )
        manifest_rows.append({
            "family": family,
            "method_name": method_name,
            "variant": variant,
            "n_features": export_info["n_features"],
            "n_rows": export_info["n_rows"],
            "csv_path": export_info["csv_path"],
        })

    for method_name, slices in filter_comparison["method_top_slices"].items():
        for top_k, feature_df in sorted(slices.items(), reverse=True):
            export_feature_list(
                "filters",
                method_name,
                f"top_{top_k}",
                extract_features(feature_df),
            )

    for top_k, feature_df in sorted(filter_comparison["consensus_slices"].items(), reverse=True):
        export_feature_list(
            "filters",
            "consensus",
            f"top_{top_k}",
            extract_features(feature_df),
        )

    for model_name, result in wrapper_results.items():
        export_feature_list(
            "wrappers",
            model_name,
            "selected",
            list(result["selected_features"]),
        )

        for top_k, feature_df in sorted(result["top_feature_slices"].items(), reverse=True):
            export_feature_list(
                "wrappers",
                model_name,
                f"top_{top_k}",
                extract_features(feature_df),
            )

    for model_name, result in embedded_results.items():
        for top_k, feature_df in sorted(result["top_feature_slices"].items(), reverse=True):
            export_feature_list(
                "embedded",
                model_name,
                f"top_{top_k}",
                extract_features(feature_df),
            )

    manifest_df = pd.DataFrame(manifest_rows).sort_values(
        ["family", "method_name", "variant"],
        ignore_index=True,
    )
    manifest_df.to_csv(export_root / "training_ready_dataset_manifest.csv", index=False)
    return manifest_df


def run_feature_comparison_workflow(
    csv_path: str,
    save_dir: str,
    label_col: str = "species",
) -> None:
    save_path = ensure_dir(save_dir)
    output_dirs = {
        "exploratory": save_path / "exploratory",
        "filters": save_path / "filters",
        "wrappers": save_path / "wrappers",
        "embedded": save_path / "embedded",
        "stage_comparison": save_path / "stage_comparison",
        "diagnostics": save_path / "diagnostics",
        "training_ready_datasets": save_path / "training_ready_datasets",
    }
    for path in output_dirs.values():
        ensure_dir(path)

    df, X, y_raw, feature_names = load_features_and_labels(csv_path, label_col=label_col)
    y_encoded, _ = encode_labels(y_raw)

    wrapper_spaces = build_wrapper_model_spaces(seed=SEED)
    embedded_spaces = build_embedded_model_spaces(
        seed=SEED,
        tree_model_n_jobs=TREE_MODEL_N_JOBS,
        xgb_n_jobs=XGB_N_JOBS,
    )

    task_total = 2 + len(wrapper_spaces) + len(embedded_spaces) + 3
    progress = tqdm(total=task_total, desc="Feature comparison workflow", unit="stage")

    shapiro_df = run_shapiro_tests(
        df,
        feature_names,
        save_dir=output_dirs["exploratory"] / "shapiro",
    )
    correlation_df = plot_correlation_matrix(
        df,
        feature_names,
        save_dir=output_dirs["exploratory"] / "correlation",
        file_stem="global_feature_correlation",
    )
    print(
        f"\n[Exploratory] Shapiro marked {(shapiro_df['gaussian_at_alpha']).sum()} "
        f"features as Gaussian at alpha=0.05."
    )
    print(f"[Exploratory] Global correlation matrix shape: {correlation_df.shape}")
    progress.update(1)

    filter_dirs = {
        "anova": output_dirs["filters"] / "anova",
        "fisher": output_dirs["filters"] / "fisher",
        "mutual_information": output_dirs["filters"] / "mutual_information",
        "pca": output_dirs["filters"] / "pca",
        "comparison": output_dirs["filters"] / "comparison",
    }

    anova_df = compute_anova_scores(X, y_raw, feature_names, save_dir=filter_dirs["anova"])
    fisher_df = compute_fisher_scores(X, y_raw, feature_names, save_dir=filter_dirs["fisher"])
    mi_df = compute_mutual_information(
        X,
        y_raw,
        feature_names,
        save_dir=filter_dirs["mutual_information"],
    )
    pca, loadings_df = run_pca_analysis(
        X,
        feature_names,
        n_components=PCA_COMPONENTS,
        save_dir=filter_dirs["pca"],
    )
    pca_ranked_df = rank_pca_features(
        loadings_df,
        pca.explained_variance_ratio_,
        save_dir=filter_dirs["pca"],
    )

    filter_comparison = compare_filter_method_rankings(
        anova_df,
        fisher_df,
        mi_df,
        pca_ranked_df,
        save_dir=filter_dirs["comparison"],
        top_k=DEFAULT_TOP_K,
        majority_threshold=MAJORITY_THRESHOLD,
    )
    print(
        f"\n[Filters] Consensus top-{DEFAULT_TOP_K} created. "
        f"Strict majority features: "
        f"{int(filter_comparison['consensus_df']['majority_vote'].sum())}"
    )
    print(
        "[Filters] Top 5 consensus features:\n",
        filter_comparison["consensus_slices"][DEFAULT_TOP_K].head()[
            ["feature", "vote_count", "borda_score"]
        ],
    )
    progress.update(1)

    wrapper_results: dict[str, dict[str, Any]] = {}
    for model_name, (pipeline, param_grid) in wrapper_spaces.items():
        print(f"\n[Wrapper:{model_name}] Starting wrapper selection.")
        wrapper_result = run_wrapper_rfecv(
            X,
            y_encoded,
            feature_names,
            model_name=model_name,
            pipeline=pipeline,
            param_grid=param_grid,
            save_dir=output_dirs["wrappers"] / model_name,
            refit_metric=REFIT_METRIC,
            cv_splits=CV_FOLDS,
            min_features_to_select=5,
            rfecv_step=WRAPPER_RFECV_STEP,
            grid_n_jobs=GRIDSEARCH_N_JOBS,
            rfecv_n_jobs=RFECV_N_JOBS,
            seed=SEED,
        )
        wrapper_results[model_name] = wrapper_result
        print(
            f"\n[Wrapper:{model_name}] Selected {len(wrapper_result['selected_features'])} features. "
            f"Best {REFIT_METRIC}: {wrapper_result['best_summary']['best_f1_macro']:.4f}"
        )
        progress.update(1)

    embedded_results: dict[str, dict[str, Any]] = {}
    for model_name, (pipeline, param_grid) in embedded_spaces.items():
        embedded_result = run_embedded_feature_importance(
            X,
            y_encoded,
            feature_names,
            model_name=model_name,
            pipeline=pipeline,
            param_grid=param_grid,
            save_dir=output_dirs["embedded"] / model_name,
            refit_metric=REFIT_METRIC,
            cv_splits=CV_FOLDS,
            grid_n_jobs=GRIDSEARCH_N_JOBS,
            seed=SEED,
        )
        embedded_results[model_name] = embedded_result
        print(
            f"\n[Embedded:{model_name}] Top feature: "
            f"{embedded_result['ranking_df'].iloc[0]['feature']}"
        )
        progress.update(1)

    primary_stage_lists = {
        "filters_consensus_top_100": filter_comparison["consensus_slices"][DEFAULT_TOP_K][
            "feature"
        ].tolist(),
        "svm_rbf_selected": wrapper_results["svm_rbf"]["selected_features"],
        "knn_selected": wrapper_results["knn"]["selected_features"],
        "random_forest_top_100": embedded_results["random_forest"]["top_feature_slices"][
            DEFAULT_TOP_K
        ]["feature"].tolist(),
    }
    if "xgboost" in embedded_results:
        primary_stage_lists["xgboost_top_100"] = embedded_results["xgboost"][
            "top_feature_slices"
        ][DEFAULT_TOP_K]["feature"].tolist()

    stage_overlap_df = compare_stage_feature_lists(
        primary_stage_lists,
        save_dir=output_dirs["stage_comparison"],
    )
    print(f"\n[Stage Comparison] Saved {len(stage_overlap_df)} pairwise overlap rows.")
    progress.update(1)

    diagnostic_feature_lists = {
        "filters_consensus_top_50": filter_comparison["consensus_slices"][50]["feature"].tolist(),
        "svm_rbf_selected": wrapper_results["svm_rbf"]["selected_features"],
        "knn_selected": wrapper_results["knn"]["selected_features"],
        "random_forest_top_50": embedded_results["random_forest"]["top_feature_slices"][50][
            "feature"
        ].tolist(),
    }
    if "xgboost" in embedded_results:
        diagnostic_feature_lists["xgboost_top_50"] = embedded_results["xgboost"][
            "top_feature_slices"
        ][50]["feature"].tolist()

    run_final_selection_diagnostics(
        df,
        label_col,
        diagnostic_feature_lists,
        save_dir=output_dirs["diagnostics"],
    )
    print("[Diagnostics] Saved final correlation matrices and violin plots.")
    progress.update(1)

    exported_training_datasets_df = export_training_ready_feature_datasets(
        df,
        label_col,
        filter_comparison,
        wrapper_results,
        embedded_results,
        save_dir=output_dirs["training_ready_datasets"],
    )
    print(
        "[Training CSVs] Saved "
        f"{len(exported_training_datasets_df)} training-ready datasets in "
        f"{output_dirs['training_ready_datasets']}"
    )
    progress.update(1)

    workflow_summary = {
        "csv_path": csv_path,
        "label_col": label_col,
        "n_samples": int(df.shape[0]),
        "n_features": int(len(feature_names)),
        "filter_majority_count": int(filter_comparison["consensus_df"]["majority_vote"].sum()),
        "svm_selected_count": int(len(wrapper_results["svm_rbf"]["selected_features"])),
        "knn_selected_count": int(len(wrapper_results["knn"]["selected_features"])),
        "random_forest_top_100_count": int(
            len(embedded_results["random_forest"]["top_feature_slices"][DEFAULT_TOP_K])
        ),
    }
    if "xgboost" in embedded_results:
        workflow_summary["xgboost_top_100_count"] = int(
            len(embedded_results["xgboost"]["top_feature_slices"][DEFAULT_TOP_K])
        )
    workflow_summary["training_ready_dataset_count"] = int(len(exported_training_datasets_df))
    save_json(workflow_summary, save_path / "feature_comparison_summary.json")

    progress.close()


def main() -> None:
    csv_path = r"F:/01_Univalle/01_TG/dataset_features/shallow_learning_birds.csv"
    save_dir = r"F:/01_Univalle/01_TG/sl_results"
    run_feature_comparison_workflow(csv_path, save_dir, label_col="species")


if __name__ == "__main__":
    main()
