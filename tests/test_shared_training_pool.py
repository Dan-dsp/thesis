import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from shared_training_pool import (
    SELECTION_PROTOCOL, balanced_indices, get_training_pool, select_training_rows,
)


class Images:
    classes = ["a", "b"]
    targets = [0] * 3 + [1] * 5
    samples = [(f"a/a{i}.jpg", 0) for i in range(3)] + [(f"b/b{i}.jpg", 1) for i in range(5)]


def descriptors():
    rows = [{"species": Images.classes[target], "orig_filename": Path(path).name,
             "split": "train", "fourier_1": float(i), "lbp_1": float(i % 2)}
            for i, (path, target) in enumerate(Images.samples)]
    rows += [{"species": species, "orig_filename": f"test_{species}.jpg", "split": "test",
              "fourier_1": 1e6, "lbp_1": 1e6} for species in ["a", "b"]]
    return pd.DataFrame(rows).sample(frac=1, random_state=12).reset_index(drop=True)


class SharedTrainingPoolTests(unittest.TestCase):
    def test_sampler_preserves_original_dl_rng_sequence(self):
        for mode in ["undersample", "oversample"]:
            rng = np.random.default_rng(42)
            targets = np.asarray(Images.targets)
            count = 3 if mode == "undersample" else 5
            chunks = []
            for cls in [0, 1]:
                positions = np.where(targets == cls)[0]
                chunks.append(rng.choice(positions, size=count,
                                         replace=mode == "oversample" and len(positions) < count))
            expected = np.concatenate(chunks)
            rng.shuffle(expected)
            np.testing.assert_array_equal(balanced_indices(targets, 42, mode), expected)

    def test_manifest_persists_and_shuffled_csv_uses_same_ids_and_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "pool.csv"
            selected, pool, info = get_training_pool(Images(), seed=42, manifest_path=path,
                                                      dataset_root=root / "train")
            before = path.read_bytes()
            _, again, other_info = get_training_pool(Images(), seed=42, manifest_path=path,
                                                      dataset_root=root / "train")
            pd.testing.assert_frame_equal(pool, again)
            self.assertEqual(info["training_pool_id"], other_info["training_pool_id"])
            source = descriptors()
            filtered, _ = select_training_rows(source, manifest_path=path, dataset=Images(),
                                                dataset_root=root / "train")
            train = filtered.loc[filtered.split.eq("train")]
            self.assertEqual((train.species + "/" + train.orig_filename).tolist(), pool.sample_key.tolist())
            self.assertEqual(train.species.value_counts().to_dict(), {"a": 3, "b": 3})
            pd.testing.assert_frame_equal(filtered.loc[filtered.split.eq("test")].reset_index(drop=True),
                                          source.loc[source.split.eq("test")].reset_index(drop=True))
            self.assertEqual(before, path.read_bytes())
            self.assertEqual(len(selected), 6)

    def test_changed_seed_order_or_manifest_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pool.csv"
            get_training_pool(Images(), manifest_path=path)
            with self.assertRaisesRegex(ValueError, "order/seed"):
                get_training_pool(Images(), seed=43, manifest_path=path)
            changed = Images()
            changed.samples = list(reversed(changed.samples))
            changed.targets = [target for _, target in changed.samples]
            with self.assertRaisesRegex(ValueError, "order/seed"):
                get_training_pool(changed, manifest_path=path)
            table = pd.read_csv(path).iloc[::-1]
            table.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "changed or reordered"):
                get_training_pool(Images(), manifest_path=path)

    def test_missing_and_duplicate_train_descriptors_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pool.csv"
            _, pool, _ = get_training_pool(Images(), manifest_path=path)
            frame = descriptors()
            key = frame.species + "/" + frame.orig_filename
            with self.assertRaisesRegex(ValueError, "descriptors missing"):
                select_training_rows(frame.loc[key.ne(pool.sample_key.iloc[0])],
                                     manifest_path=path, dataset=Images())
            duplicate = frame.loc[frame.split.eq("train")].iloc[:1]
            with self.assertRaisesRegex(ValueError, "Duplicate image"):
                select_training_rows(pd.concat([frame, duplicate]), manifest_path=path, dataset=Images())

    def test_selection_and_training_use_balanced_pool_and_no_second_val_sampling(self):
        from sl_feature_comparison_tools import load_features_and_labels
        import sl_training_pipeline as training

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "pool.csv"
            pool_result = get_training_pool(Images(), manifest_path=path)
            frame = descriptors()
            csv = root / "features.csv"
            frame.to_csv(csv, index=False)
            with patch("shared_training_pool.get_training_pool", return_value=pool_result), \
                    patch.object(training, "make_balanced_subset", wraps=training.make_balanced_subset) as balance:
                df, X, y, _ = load_features_and_labels(str(csv), selection_split="train", pool_manifest_path=path)
                data = training.load_split_dataset(str(csv), val_fraction=1 / 3, pool_manifest_path=path)
            self.assertEqual(len(X), 6)
            self.assertEqual(pd.Series(y).value_counts().to_dict(), {"a": 3, "b": 3})
            self.assertEqual(balance.call_count, 1)  # Test evaluation only, not validation.
            self.assertEqual(len(data["X_train_full"]), 6)
            np.testing.assert_array_equal(data["X_train_full"], X.astype(np.float32))
            self.assertEqual(len(data["X_val"]), 2)
            self.assertEqual(len(data["X_val_balanced"]), 2)
            self.assertEqual(df.loc[df.split.eq("test"), "orig_filename"].tolist(),
                             frame.loc[frame.split.eq("test"), "orig_filename"].tolist())

    def test_final_sl_model_and_metadata_use_balanced_pool(self):
        import sl_training_pipeline as training

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "pool.csv"
            pool_result = get_training_pool(Images(), manifest_path=path)
            csv = root / "features.csv"
            descriptors().to_csv(csv, index=False)
            spaces = {"tree": (Pipeline([("clf", DecisionTreeClassifier(random_state=42))]),
                                {"clf__max_depth": [1]})}
            with patch("shared_training_pool.get_training_pool", return_value=pool_result), \
                    patch.object(training, "get_training_pool", return_value=pool_result), \
                    patch.object(training, "make_model_spaces", return_value=spaces), \
                    patch.object(training, "save_confusion_matrix"), patch.object(training, "tqdm", None):
                run = training.run_training_pipeline(csv, root / "out", val_fraction=1 / 3,
                                                     cv_folds=2, gridsearch_n_jobs=1,
                                                     pool_manifest_path=path)
            model = joblib.load(run / "models" / "tree_final.pkl")
            self.assertEqual(model.named_steps["clf"].tree_.n_node_samples[0], 6)
            metadata = json.loads((run / "metadata" / "training_pool.json").read_text())
            self.assertEqual(metadata["training_pool_id"], pool_result[2]["training_pool_id"])
            self.assertEqual(len(pd.read_csv(run / "predictions" / "tree_test_predictions.csv")), 2)

    def test_training_manifest_checks_pool_identity_and_protocol(self):
        from sl_training_pipeline import load_training_ready_manifest

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "exports.csv"
            row = {"family": "filters", "method_name": "mi", "variant": "top_50",
                   "csv_path": "reduced.csv", "selection_split": "train",
                   "selection_protocol": SELECTION_PROTOCOL, "training_pool_id": "a" * 64}
            pd.DataFrame([row]).to_csv(path, index=False)
            load_training_ready_manifest(path, expected_pool_id="a" * 64)
            with self.assertRaisesRegex(ValueError, "differs"):
                load_training_ready_manifest(path, expected_pool_id="b" * 64)
            row["selection_protocol"] = "train_only_v2_fourier"
            pd.DataFrame([row]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Regenerate"):
                load_training_ready_manifest(path)


if __name__ == "__main__":
    unittest.main()
