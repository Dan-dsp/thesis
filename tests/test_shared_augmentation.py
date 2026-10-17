from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

from shared_augmentation import (
    SharedAugmentedDataset, TrainingOnlyAugmentationCV, build_augmentation,
    expanded_training_indices, load_augmentation,
)


def feature_rows():
    rows = []
    for species in ("a", "b"):
        for i in range(10):
            key = f"{species}/{i}.png"
            rows.append(dict(species=species, orig_filename=f"{i}.png", split="train",
                             fourier_1=float(i), lbp_1=float(i % 2)))
            for view in (1, 2):
                rows.append(dict(species=species, orig_filename=f"{i}_{view}.png",
                                 split="train_augmented", augmentation_parent=key,
                                 augmentation_id="abc", fourier_1=float(i + 100), lbp_1=float(view)))
        rows.append(dict(species=species, orig_filename="test.png", split="test",
                         fourier_1=9999., lbp_1=9999.))
    return pd.DataFrame(rows)


class SharedAugmentationTests(unittest.TestCase):
    def test_fixed_views_are_reproducible_and_dl_reads_exact_saved_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for species in ("a", "b"):
                folder = root / "train" / species
                folder.mkdir(parents=True)
                for i in range(2):
                    pixels = np.zeros((20, 30, 3), dtype=np.uint8)
                    pixels[3:17, 2:12] = (255, 50 + i, 0)
                    Image.fromarray(pixels).save(folder / f"{i}.png")
            kwargs = dict(dataset_root=root / "train", pool_manifest=root / "pool.csv",
                          out_dir=root / "views")
            first = build_augmentation(**kwargs)
            np.random.seed(987)
            second = build_augmentation(**kwargs)
            pd.testing.assert_frame_equal(first, second)
            table, info = load_augmentation(root / "views" / "augmentation_manifest.csv")
            self.assertEqual(len(table), 8)
            self.assertEqual(info["split_assignment"].split(";")[0], "none")
            dataset = ImageFolder(str(root / "train"), transform=ToTensor())
            selected = [0, 2]
            augmented = SharedAugmentedDataset(dataset, selected, table)
            self.assertEqual(len(augmented), 6)
            selected_keys = {"a/0.png", "b/0.png"}
            expected = table.loc[table.augmentation_parent.isin(selected_keys)].sort_values(
                ["augmentation_parent", "orig_filename"])
            for index, row in enumerate(expected.itertuples(), len(selected)):
                with Image.open(row.image_path) as image:
                    np.testing.assert_array_equal(augmented[index][0].numpy(), ToTensor()(image).numpy())
            path = root / "views" / "augmentation_manifest.csv"
            path.write_text(path.read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid"):
                load_augmentation(path)

    def test_runtime_cv_never_scores_views_or_fits_validation_relatives(self):
        frame = feature_rows()
        frame = frame.loc[~frame.split.eq("test")].reset_index(drop=True)
        cv = TrainingOnlyAugmentationCV(frame, folds=5, seed=42)
        evaluated = []
        for train, val in cv.split(np.zeros((len(frame), 2)), frame.species):
            self.assertTrue(frame.iloc[val].split.eq("train").all())
            originals = frame.iloc[train].loc[lambda f: f.split.eq("train")]
            allowed = set(originals.species + "/" + originals.orig_filename)
            views = frame.iloc[train].loc[lambda f: f.split.eq("train_augmented")]
            self.assertEqual(set(views.augmentation_parent), allowed)
            self.assertEqual(len(train), 48)
            evaluated.extend(val)
        self.assertEqual(sorted(evaluated), np.flatnonzero(frame.split.eq("train")).tolist())

    def test_sl_runtime_split_and_final_pool_keep_test_unchanged(self):
        from sl_training_pipeline import load_split_dataset
        from sl_feature_comparison_tools import build_training_ready_dataset

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.csv"
            frame = feature_rows()
            exported = build_training_ready_dataset(frame, ["fourier_1", "lbp_1"])
            self.assertIn("augmentation_parent", exported)
            exported.to_csv(path, index=False)
            data = load_split_dataset(path, pool_manifest_path=None, val_fraction=0.2)
            self.assertEqual(len(data["X_train"]), 48)
            self.assertEqual(len(data["X_val"]), 4)
            self.assertEqual(len(data["X_train_full"]), 60)
            self.assertEqual(len(data["X_test"]), 2)
            self.assertTrue((data["X_val"][:, 0] < 100).all())
            self.assertTrue((data["X_test"][:, 0] == 9999).all())
            self.assertEqual(data["n_features"], 2)
            cv = TrainingOnlyAugmentationCV(data["training_cv_frame"], folds=2)
            for train, val in cv.split(data["X_train"], data["y_train"]):
                self.assertTrue((data["X_train"][val, 0] < 100).all())

    def test_seeded_pca_is_independent_of_global_numpy_rng(self):
        from sl_feature_comparison_tools import run_pca_analysis

        X = np.random.default_rng(123).normal(size=(600, 80))
        features = [f"x{i}" for i in range(80)]
        first, _ = run_pca_analysis(X, features, seed=42)
        np.random.seed(199)
        second, _ = run_pca_analysis(X, features, seed=42)
        self.assertEqual(first.random_state, 42)
        np.testing.assert_array_equal(first.components_, second.components_)

    def test_augmented_sl_training_and_nested_cv_finish_with_original_only_audits(self):
        from unittest.mock import patch
        import joblib
        import sl_training_pipeline as training
        import thesis_validation
        from sklearn.pipeline import Pipeline
        from sklearn.tree import DecisionTreeClassifier

        spaces = {"tree": (Pipeline([("clf", DecisionTreeClassifier(random_state=42))]),
                            {"clf__max_depth": [1]})}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv = root / "features.csv"
            feature_rows().to_csv(csv, index=False)
            with patch.object(training, "make_model_spaces", return_value=spaces), \
                    patch.object(training, "save_confusion_matrix"), patch.object(training, "tqdm", None):
                run = training.run_training_pipeline(csv, root / "training", cv_folds=2,
                                                     gridsearch_n_jobs=1, pool_manifest_path=None)
                thesis_validation.run_sl_nested_cv(csv, root / "nested", method="anova",
                                                   n_features=1, folds=2, pool_manifest_path=None)
            final = joblib.load(run / "models" / "tree_final.pkl")
            self.assertEqual(final.named_steps["clf"].tree_.n_node_samples[0], 60)
            audit = pd.read_csv(run / "predictions" / "tree_test_predictions.csv")
            self.assertEqual(len(audit), 2)
            scores = pd.read_csv(root / "nested" / "outer_cv_scores.csv")
            self.assertTrue(scores.n_train.eq(10).all())
            self.assertTrue(scores.n_validation.eq(10).all())


if __name__ == "__main__":
    unittest.main()
