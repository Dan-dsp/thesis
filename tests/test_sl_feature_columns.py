from pathlib import Path
import tempfile
import unittest

import pandas as pd

from sl_feature_comparison import export_training_ready_feature_datasets
from sl_feature_comparison_tools import get_feature_columns as selection_columns
from sl_training_pipeline import get_feature_columns as training_columns
from sl_training_pipeline import load_training_ready_manifest


class SLFeatureColumnTests(unittest.TestCase):
    def test_fourier_is_descriptive_and_legacy_aliases_are_not_duplicated(self):
        fourier = [f"fourier_{i}" for i in range(1, 21)]
        df = pd.DataFrame({name: [0.1, 0.2] for name in
                           ["f0", "f20", "f39", "affine_6", "lbp_1", *fourier]})
        df["species"] = ["a", "b"]
        df["split"] = ["train", "test"]
        df["sample_id"] = [0, 1]
        for detect in (selection_columns, training_columns):
            with self.subTest(detect=detect.__module__):
                self.assertEqual(detect(df), ["lbp_1", *fourier])

    def test_fourier_only_dataset_is_not_mistaken_for_legacy(self):
        df = pd.DataFrame({"f0": [1.0], "fourier_1": [2.0], "fourier_20": [3.0]})
        for detect in (selection_columns, training_columns):
            self.assertEqual(detect(df), ["fourier_1", "fourier_20"])

    def test_legacy_only_dataset_still_excludes_duplicate_f39(self):
        df = pd.DataFrame({"f0": [1.0], "f20": [2.0], "f39": [3.0]})
        for detect in (selection_columns, training_columns):
            self.assertEqual(detect(df), ["f0", "f20"])

    def test_new_exports_keep_fourier_and_old_manifest_is_refused(self):
        df = pd.DataFrame({"fourier_1": [1.0, 2.0], "lbp_1": [3.0, 4.0],
                           "species": ["a", "b"], "split": ["train", "test"],
                           "orig_filename": ["a.jpg", "b.jpg"]})
        rankings = {"method_top_slices": {"mi": {1: pd.DataFrame(
            {"feature": ["fourier_1"]})}}, "consensus_slices": {}}
        with tempfile.TemporaryDirectory() as tmp:
            export_training_ready_feature_datasets(df, "species", rankings, {}, {}, tmp,
                                                    training_pool_id="a" * 64)
            path = Path(tmp) / "training_ready_dataset_manifest.csv"
            manifest = load_training_ready_manifest(path)
            exported = pd.read_csv(manifest.iloc[0].csv_path)
            self.assertEqual(training_columns(exported), ["fourier_1"])
            self.assertEqual(exported.split.tolist(), ["train", "test"])
            manifest["selection_protocol"] = "train_only_v1"
            manifest.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Regenerate"):
                load_training_ready_manifest(path)


if __name__ == "__main__":
    unittest.main()
