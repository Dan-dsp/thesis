from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import shapiro

from sl_feature_comparison_tools import run_shapiro_tests
from sl_normality import representative_features


class NormalityTests(unittest.TestCase):
    def test_species_are_separate_and_nontraining_rows_excluded(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"species": ["a"] * 20 + ["b"] * 20,
                           "hu_1": np.r_[rng.normal(size=20), rng.normal(50, 1, 20)],
                           "hu_2": 1.0, "split": "train"})
        extra = df.assign(split="test", hu_1=999)
        data = pd.concat([df, extra, extra.assign(split="train_augmented")], ignore_index=True)
        before = data.copy(deep=True)
        result = run_shapiro_tests(data, ["hu_1", "hu_2"])
        self.assertEqual(len(result), 4)
        for species in ["a", "b"]:
            row = result.loc[result.species.eq(species) & result.feature.eq("hu_1")].iloc[0]
            self.assertEqual(row.n_used, 20)
            self.assertAlmostEqual(row.p_value, shapiro(df.loc[df.species.eq(species), "hu_1"]).pvalue)
        self.assertTrue(result.loc[result.feature.eq("hu_2"), "test_status"].eq("not_testable").all())
        pd.testing.assert_frame_equal(data, before)

    def test_plots_representatives_and_missing_values(self):
        df = pd.DataFrame({"species": ["a"] * 8 + ["b"] * 8,
                           "hu_1": 1.0, "hu_2": np.arange(16),
                           "lbp_1": [np.nan, np.inf] + [1.] * 14})
        names = ["hu_1", "hu_2", "lbp_1"]
        reps = representative_features(run_shapiro_tests(df, names), names)
        self.assertEqual(reps.loc[reps.family.eq("hu"), "feature"].tolist(), ["hu_2", "hu_2"])
        self.assertTrue(reps.loc[reps.family.eq("lbp"), "selection_status"].eq("no_fully_testable_feature").all())
        with tempfile.TemporaryDirectory() as tmp:
            result = run_shapiro_tests(df, names, tmp)
            self.assertEqual(len(result), 6)
            for name in ["distributions/normality_extremes/hu_highest_normality.png",
                         "distributions/normality_extremes/hu_lowest_normality.png",
                         "shapiro_p_values.png"]:
                with Image.open(Path(tmp) / name) as image:
                    self.assertGreater(np.asarray(image).std(), 5)
                    self.assertGreater(image.width, 500)
            self.assertTrue((Path(tmp) / "shapiro_by_species_summary.csv").exists())

    def test_extremes_rank_mean_w_not_p_and_require_all_species(self):
        features = ["hu_1", "hu_2", "hu_3", "hu_4", "hu_5"]
        tests = pd.DataFrame({
            "feature": np.repeat(features, 2), "species": ["a", "b"] * 5,
            "shapiro_W": [0.95, 0.97, 0.6, 0.7, 0.99, np.nan, 0.95, 0.97, 0.8, 0.8],
            "p_value": [0.1, 0.1, 0.9, 0.9, 0.99, np.nan, 0.1, 0.1, 0.2, 0.2],
            "test_status": ["not_rejected"] * 5 + ["not_testable"] + ["not_rejected"] * 4,
        })
        before = tests.copy(deep=True)
        reps = representative_features(tests, features)
        self.assertEqual(reps.feature.tolist(), ["hu_1", "hu_2"])
        self.assertAlmostEqual(reps.iloc[0].mean_shapiro_W, 0.96)
        self.assertEqual(reps.eligible_features.tolist(), [4, 4])
        pd.testing.assert_frame_equal(tests, before)

    def test_subsampling_and_invalid_inputs(self):
        df = pd.DataFrame({"species": "a", "hu_1": np.arange(30)})
        a = run_shapiro_tests(df, ["hu_1"], max_samples=10)
        b = run_shapiro_tests(df, ["hu_1"], max_samples=10)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(a.iloc[0].n_used, 10)
        small = run_shapiro_tests(df.iloc[:2], ["hu_1"])
        self.assertEqual(small.iloc[0].test_status, "not_testable")
        with self.assertRaises(ValueError):
            run_shapiro_tests(df.drop(columns="species"), ["hu_1"])


if __name__ == "__main__":
    unittest.main()
