import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from image_error_analysis import save_image_predictions, summarize_image_errors


class ImageErrorAnalysisTests(unittest.TestCase):
    def predictions(self, root, family, model, predictions, keys=None, filename=None):
        keys = keys or [f"a/image{i}.jpg" for i in range(len(predictions))]
        metadata = pd.DataFrame({"sample_key": keys})
        return save_image_predictions(root / (filename or f"{family}_{model}_test_predictions.csv"),
                                      family, model, metadata, ["a"] * len(keys), predictions)

    def test_shared_images_and_gallery_ignore_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images" / "a"
            image_dir.mkdir(parents=True)
            for i in range(4):
                Image.new("RGB", (10, 10), color="green").save(image_dir / f"image{i}.jpg")
            sl = self.predictions(root, "SL", "knn", ["b", "b", "a", "a"])
            dl = self.predictions(root, "DL", "resnet", ["b", "a", "b", "a"])
            # Rejection flags and decisions must not change raw-error analysis.
            table = pd.read_csv(sl).assign(rejected=True, decision="REJECT")
            table.to_csv(sl, index=False)
            out = root / "report"
            result = summarize_image_errors([sl, dl], out, image_root=root / "images")
            self.assertEqual(result.set_index("sample_key").error_group.to_dict(), {
                "a/image0.jpg": "both_families_error", "a/image1.jpg": "sl_only_error",
                "a/image2.jpg": "dl_only_error", "a/image3.jpg": "no_errors"})
            overlap = pd.read_csv(out / "descriptive_error_overlap.csv").iloc[0]
            self.assertEqual(overlap.same_image_errors, 1)
            self.assertEqual(overlap.same_wrong_species, 1)
            self.assertEqual(overlap.left_wrong_right_correct, 1)
            self.assertEqual(overlap.left_correct_right_wrong, 1)
            summary = json.loads((out / "error_summary.json").read_text())
            self.assertFalse(summary["rejection_used"])
            self.assertEqual(summary["images_missing_files"], 0)
            gallery = (out / "error_gallery.html").read_text()
            self.assertIn("a/image0.jpg", gallery)
            self.assertNotIn("<h2>a/image3.jpg</h2>", gallery)
            self.assertIn("file:///", gallery)
            with Image.open(out / "image_errors.png") as im:
                self.assertGreater(float(np.asarray(im).std()), 5)

    def test_missing_predictions_are_not_counted_as_correct(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sl = self.predictions(root, "SL", "knn", ["b", "a"])
            dl = self.predictions(root, "DL", "resnet", ["a"], keys=["a/image1.jpg"])
            out = root / "report"
            result = summarize_image_errors([sl, dl], out)
            self.assertEqual(result.set_index("sample_key").loc["a/image0.jpg", "error_group"],
                             "not_evaluated_by_both_families")
            matrix = pd.read_csv(out / "image_error_matrix.csv").set_index("sample_key")
            self.assertEqual(matrix.loc["a/image0.jpg"].isna().sum(), 1)
            self.assertEqual(len(pd.read_csv(out / "unmatched_images.csv")), 1)

    def test_multiple_models_in_one_source_and_repeated_run_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            one = self.predictions(root, "SL", "one", ["b", "a"])
            two = self.predictions(root, "SL", "two", ["a", "b"])
            combined = root / "combined.csv"
            pd.concat([pd.read_csv(one), pd.read_csv(two)]).to_csv(combined, index=False)
            again = self.predictions(root, "SL", "one", ["a", "a"], filename="another_run.csv")
            out = root / "report"
            summarize_image_errors([combined, again], out)
            self.assertEqual(len(pd.read_csv(out / "evaluator_sources.csv")), 3)
            self.assertEqual(len(pd.read_csv(out / "descriptive_error_overlap.csv")), 3)

    def test_annotations_keep_correct_images_and_do_not_get_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.predictions(root, "SL", "knn", ["b", "a"])
            out = root / "report"
            summarize_image_errors([path], out)
            annotation_path = out / "annotation_template.csv"
            annotation = pd.read_csv(annotation_path)
            self.assertNotIn("bird_id", annotation)
            annotation["conditions"] = "blur;blur"
            annotation.to_csv(annotation_path, index=False)
            summarize_image_errors([path], out, annotations=annotation_path)
            self.assertTrue(pd.read_csv(annotation_path).conditions.eq("blur;blur").all())
            summary = pd.read_csv(out / "condition_summary.csv").iloc[0]
            self.assertEqual(summary.samples, 2)
            self.assertEqual(summary.errors, 1)
            self.assertEqual(summary.error_rate, 0.5)

    def test_duplicate_sources_and_conflicting_labels_fail_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            one = self.predictions(root, "SL", "one", ["a"])
            two = self.predictions(root, "DL", "two", ["a"])
            with self.assertRaisesRegex(ValueError, "Duplicate prediction source"):
                summarize_image_errors([one, one], root / "report")
            table = pd.read_csv(two).assign(true_label="different")
            table.to_csv(two, index=False)
            with self.assertRaisesRegex(ValueError, "Conflicting true labels"):
                summarize_image_errors([one, two], root / "report")

    def test_zero_errors_and_missing_images_still_generate_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            one = self.predictions(root, "SL", "one", ["a"])
            summarize_image_errors([one], root / "report")
            self.assertTrue((root / "report" / "error_gallery.html").exists())
            self.assertTrue(pd.read_csv(root / "report" / "error_review.csv").empty)

    def test_sl_balanced_metrics_preserve_full_image_audit(self):
        from sklearn.preprocessing import LabelEncoder
        from sklearn.tree import DecisionTreeClassifier
        from compare_dl_sl_models import evaluate_sl_model

        X = np.arange(4, dtype=np.float32).reshape(-1, 1)
        truth = np.array(["a", "a", "a", "b"])
        model = DecisionTreeClassifier().fit(X, [0, 0, 0, 1])
        metadata = pd.DataFrame({"species": truth, "orig_filename": [f"{i}.jpg" for i in range(4)]})
        with tempfile.TemporaryDirectory() as tmp, patch("compare_dl_sl_models.joblib.load", return_value=model):
            result = evaluate_sl_model("tree", Path("unused"), X, truth, ["a", "b"],
                                       LabelEncoder().fit(truth), metadata=metadata,
                                       audit_dir=Path(tmp), evaluation_indices=np.array([0, 3]))
            self.assertEqual(result.samples, 2)
            self.assertEqual(len(pd.read_csv(Path(tmp) / "tree_test_predictions.csv")), 4)

    def test_dl_balanced_metrics_preserve_full_image_audit(self):
        import torch
        from torch.utils.data import Dataset
        from compare_dl_sl_models import evaluate_dl_model

        class Images(Dataset):
            classes = ["a", "b"]
            targets = [0, 0, 0, 1]
            samples = [(f"a/{i}.jpg" if i < 3 else "b/3.jpg", target)
                       for i, target in enumerate(targets)]

            def __len__(self):
                return 4

            def __getitem__(self, index):
                return torch.tensor([1.0, 0.0]), self.targets[index]

        with tempfile.TemporaryDirectory() as tmp, \
                patch("compare_dl_sl_models.torch_load_checkpoint", return_value={
                    "class_names": ["a", "b"], "model_state_dict": {}}), \
                patch("compare_dl_sl_models.datasets.ImageFolder", return_value=Images()), \
                patch("compare_dl_sl_models.build_resnet18", return_value=torch.nn.Identity()):
            result = evaluate_dl_model("resnet", Path("unused"), Path("unused"), "cpu", 2, 0,
                                       True, 42, audit_dir=Path(tmp))
            self.assertEqual(result.samples, 2)
            self.assertEqual(len(pd.read_csv(Path(tmp) / "resnet_test_predictions.csv")), 4)


if __name__ == "__main__":
    unittest.main()
