from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from result_plots import (
    plot_feature_counts, plot_feature_overlap, plot_feature_scores, plot_grid_search,
    plot_matrix, plot_model_metrics, plot_outer_scores, plot_rfecv,
)


class ResultPlotTests(unittest.TestCase):
    def assert_image(self, path):
        self.assertTrue(path.exists(), path)
        with Image.open(path) as im:
            pixels = np.asarray(im.convert("RGB"))
            self.assertGreater(im.width, 500)
            self.assertGreater(im.height, 300)
            self.assertGreater(float(pixels.std()), 5)

    def test_result_tables_create_nonblank_pngs_without_mutation(self):
        scores = pd.DataFrame({"n_features": [2, 4, 6], "mean_test_score": [0.5, 0.8, 0.7],
                               "std_test_score": [0.02, 0.03, 0.04]})
        before = scores.copy(deep=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plot_rfecv(scores, 4, root / "rfecv.png", "k-NN")
            plot_grid_search(scores, root / "grid.png", "k-NN")
            plot_feature_scores(pd.DataFrame({"feature": ["a", "b"], "importance": [0.5, -0.1]}),
                                "importance", root / "importance.png", "Permutation importance")
            lists = {"filters": ["a", "b"], "knn": ["b", "c"]}
            plot_feature_overlap(lists, root / "overlap.png", "Agreement")
            plot_feature_counts(lists, root / "counts.png")
            plot_matrix(np.array([[3, 1], [2, 4]]), ["a", "b"], root / "cm.png", "Test",
                        xlabel="Predicted", ylabel="True")
            plot_model_metrics(pd.DataFrame({"model": ["knn", "svm"], "f1": [0.4, 0.7]}),
                               {"f1": "Macro-F1"}, root / "metrics.png", "SL")
            folds = pd.DataFrame({"family": ["SL"] * 3, "model": ["knn"] * 3,
                                  "repeat": [0] * 3, "fold": [1, 2, 3], "score": [0.5, 0.7, 0.6]})
            plot_outer_scores(folds, root / "outer.png")
            for name in ["rfecv", "grid", "importance", "overlap", "counts", "cm", "metrics", "outer"]:
                self.assert_image(root / f"{name}.png")
            mixed = pd.concat([folds, folds.assign(family="DL")])
            with self.assertRaisesRegex(ValueError, "separately"):
                plot_outer_scores(mixed, root / "mixed.png")
        pd.testing.assert_frame_equal(scores, before)

    def test_tiny_wrapper_and_embedded_runs_save_their_plots(self):
        from sl_feature_comparison_tools import run_embedded_feature_importance, run_wrapper_rfecv

        X, y = make_classification(n_samples=30, n_features=6, n_informative=3,
                                   n_redundant=0, random_state=42)
        names = [f"fourier_{i}" for i in range(1, 7)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = run_wrapper_rfecv(
                X, y, names, "knn", Pipeline([("scaler", StandardScaler()),
                                               ("clf", KNeighborsClassifier(n_neighbors=3))]),
                {"clf__weights": ["uniform"]}, save_dir=root / "knn",
                cv_splits=2, min_features_to_select=2, rfecv_step=2, grid_n_jobs=1)
            curve = pd.read_csv(root / "knn" / "knn_rfecv_curve.csv")
            self.assertEqual(curve.n_features.tolist(), [2, 4, 6])
            self.assertIn(len(result["selected_features"]), curve.n_features.tolist())
            for name in ["rfecv_curve", "initial_grid_search", "grid_search", "selected_importance"]:
                self.assert_image(root / "knn" / f"knn_{name}.png")
            run_embedded_feature_importance(
                X, y, names, "random_forest", Pipeline([("clf", RandomForestClassifier(
                    n_estimators=3, random_state=42))]), {"clf__max_depth": [2]},
                save_dir=root / "rf", cv_splits=2, grid_n_jobs=1)
            self.assert_image(root / "rf" / "random_forest_feature_importance.png")
            self.assert_image(root / "rf" / "random_forest_grid_search.png")

    def test_filter_and_stage_diagnostics_save_plots(self):
        from sl_feature_comparison import compare_filter_method_rankings, compare_stage_feature_lists
        from sl_feature_comparison_tools import rank_pca_features, run_shapiro_tests

        names = ["fourier_1", "lbp_1", "lbp_2"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = [pd.DataFrame({"feature": names, column: [3.0, 2.0, 1.0]})
                      for column in ["F_score", "fisher_score", "mutual_info", "weighted_abs_loading"]]
            compare_filter_method_rankings(*frames, save_dir=root / "filters", top_k=3)
            compare_stage_feature_lists({"filters": names[:2], "knn": names[1:]}, root / "stages")
            run_shapiro_tests(pd.DataFrame({name: np.arange(20) for name in names}),
                             names, root / "normality")
            rank_pca_features(pd.DataFrame({"PC1": [0.5, 0.2, 0.1]}, index=names),
                              np.array([1.0]), root / "pca")
            for name in ["filters/filter_consensus_votes.png", "filters/filter_pairwise_overlap.png",
                         "stages/stage_feature_overlap.png", "stages/stage_feature_counts.png",
                         "normality/shapiro_p_values.png", "pca/pca_feature_ranking.png"]:
                self.assert_image(root / name)


if __name__ == "__main__":
    unittest.main()
