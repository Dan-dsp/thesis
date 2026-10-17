import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from classification_analysis import (
    choose_rejection_policy, save_prediction_audit, sl_predictions, summarize_errors,
)
from thesis_validation import compare_outer_scores, run_sl_nested_cv


class ThesisAnalysisTests(unittest.TestCase):
    def test_rejection_target_and_failed_target(self):
        confidence = np.r_[np.full(10, 0.9), np.full(10, 0.2)]
        policy = choose_rejection_policy(np.zeros(20), np.r_[np.zeros(10), np.ones(10)],
                                        confidence)
        self.assertEqual(policy["threshold"], 0.9)
        self.assertEqual(policy["validation_accepted"], 10)
        failed = choose_rejection_policy(np.zeros(20), np.ones(20), confidence)
        self.assertGreater(failed["threshold"], 1)

    def test_svm_confidence_tracks_predict_not_probability_argmax(self):
        class Model:
            classes_ = np.array([0, 1])

            def predict(self, X):
                return np.array([0])

            def predict_proba(self, X):
                return np.array([[0.4, 0.6]])
        pred, confidence = sl_predictions(Model(), np.zeros((1, 1)))
        self.assertEqual(pred[0], 0)
        self.assertEqual(confidence[0], 0.4)

    def test_error_audit_retains_rejected_mistakes_and_all_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = save_prediction_audit(tmp, "SL", "m", pd.DataFrame(
                {"sample_key": ["a/x", "a/y", "b/z"]}),
                ["a", "a", "b"], ["a", "b", "b"], [0.9, 0.3, 0.8], {"threshold": 0.5})
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows.rejected.sum(), 1)
            self.assertEqual(len(pd.read_csv(Path(tmp) / "m_test_errors.csv")), 1)
            report = json.loads((Path(tmp) / "m_rejection.json").read_text())
            self.assertAlmostEqual(report["coverage"], 2 / 3)

    def score_table(self, family="SL"):
        return pd.DataFrame([
            {"family": family, "model": model, "repeat": 0, "fold": fold,
             "partition_id": f"split{fold}", "score": score,
             "evaluation_role": "independent_outer", "n_train": 80, "n_validation": 20}
            for model in ["a", "b"] for fold, score in enumerate([0.6, 0.7, 0.8], 1)
        ])

    def test_statistics_identical_and_cross_family_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.csv"
            self.score_table().to_csv(path, index=False)
            result = compare_outer_scores([path], Path(tmp) / "reports")
            self.assertEqual(result.iloc[0].p_holm, 1)
            data = self.score_table()
            data.loc[data.model.eq("b"), "family"] = "DL"
            data.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "exactly one family"):
                compare_outer_scores([path], Path(tmp) / "reports")

    def test_statistics_rejects_search_folds_and_mismatched_partitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.csv"
            data = self.score_table()
            data["evaluation_role"] = "epoch_selection"
            data.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                compare_outer_scores([path], tmp)
            data = self.score_table()
            data.loc[0, "partition_id"] = "other"
            data.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Unmatched"):
                compare_outer_scores([path], tmp)

    def test_family_report_has_no_joint_ranking(self):
        from compare_dl_sl_models import EvalResult, save_outputs
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = [EvalResult(family, "m", 2, 0.5, 0.5, 0.5, 0.5,
                                  "report", np.eye(2)) for family in ["SL", "DL"]]
            save_outputs(results, root)
            self.assertFalse((root / "summary_metrics.csv").exists())
            self.assertTrue((root / "sl" / "summary_metrics.csv").exists())
            self.assertTrue((root / "dl" / "summary_metrics.csv").exists())
            for family in ["sl", "dl"]:
                self.assertTrue((root / family / "summary_metrics.png").exists())
                self.assertTrue((root / family / "m_confusion_matrix.png").exists())

    def test_legacy_selection_manifest_requires_regeneration(self):
        from sl_training_pipeline import load_training_ready_manifest
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.csv"
            pd.DataFrame([{"family": "filters", "method_name": "mi", "variant": "top_50",
                           "csv_path": "old.csv"}]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Regenerate"):
                load_training_ready_manifest(path)

    def test_nested_sl_results_are_invariant_to_test_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            rng = np.random.default_rng(2)
            data = pd.DataFrame(rng.normal(size=(72, 3)), columns=["g1", "g2", "g3"])
            data["species"] = ["a", "b"] * 36
            data["split"] = ["train"] * 60 + ["test"] * 12
            data["orig_filename"] = [f"image{i}.jpg" for i in range(72)]
            csv = Path(tmp) / "features.csv"
            spaces = {"tree": (Pipeline([("clf", DecisionTreeClassifier(random_state=42))]),
                               {"clf__max_depth": [2]})}
            with patch("sl_training_pipeline.make_model_spaces", return_value=spaces):
                data.to_csv(csv, index=False)
                run_sl_nested_cv(csv, Path(tmp) / "one", "anova", 2, 3, pool_manifest_path=None)
                data.loc[data.split.eq("test"), ["g1", "g2", "g3"]] = 1e20
                data.loc[data.split.eq("test"), "species"] = "test_only_label"
                data.to_csv(csv, index=False)
                run_sl_nested_cv(csv, Path(tmp) / "two", "anova", 2, 3, pool_manifest_path=None)
            pd.testing.assert_frame_equal(
                pd.read_csv(Path(tmp) / "one" / "outer_cv_scores.csv"),
                pd.read_csv(Path(tmp) / "two" / "outer_cv_scores.csv"))

    def test_overlap_and_manual_bird_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for model in ["one", "two"]:
                save_prediction_audit(root, "SL", model, pd.DataFrame(
                    {"sample_key": ["a/x", "b/y"]}), ["a", "b"], ["b", "b"],
                    [0.5, 0.9], {"threshold": 0.6})
                paths.append(root / f"{model}_test_predictions.csv")
            annotation = root / "labels.csv"
            pd.DataFrame({"sample_key": ["a/x", "b/y"], "bird_id": ["bird1", "bird2"],
                          "conditions": ["blur;occlusion", "clear"]}).to_csv(annotation, index=False)
            summarize_errors(paths, root / "review", annotation)
            overlap = pd.read_csv(root / "review" / "descriptive_error_overlap.csv")
            self.assertEqual(overlap.iloc[0].same_image_errors, 1)
            self.assertTrue((root / "review" / "bird_error_summary.csv").exists())
            self.assertTrue((root / "review" / "condition_summary.csv").exists())

    def test_small_training_run_writes_validation_selected_model_and_full_audit(self):
        import sl_training_pipeline as training
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = pd.DataFrame({
                "g1": [float(i % 2) for i in range(60)],
                "g2": [float(i % 2) * 2 for i in range(60)],
                "species": ["a", "b"] * 30, "split": ["train"] * 48 + ["test"] * 12,
                "orig_filename": [f"{i}.jpg" for i in range(60)],
            })
            csv = root / "reduced.csv"
            data.to_csv(csv, index=False)
            spaces = {"svm_rbf": (Pipeline([("scaler", StandardScaler()),
                                           ("clf", SVC(probability=True, random_state=42))]),
                                  {"clf__C": [1.0]})}
            with patch.object(training, "make_model_spaces", return_value=spaces), \
                    patch.object(training, "save_confusion_matrix"), patch.object(training, "tqdm", None):
                run = training.run_training_pipeline(csv, root / "out", cv_folds=2,
                                                     gridsearch_n_jobs=1, pool_manifest_path=None)
            selected = json.loads((run / "selected_model.json").read_text())
            self.assertEqual(selected["selected_by"], "internal_validation_f1_macro")
            predictions = pd.read_csv(run / "predictions" / "svm_rbf_test_predictions.csv")
            self.assertEqual(len(predictions), 12)
            self.assertTrue((run / "svm_rbf_model_analysis.json").exists())
            for name in ["svm_rbf_grid_search.png", "internal_validation_metrics.png",
                         "test_metrics.png"]:
                self.assertTrue((run / "cm" / name).exists())

    def test_dl_policy_uses_best_checkpoint(self):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from classification_analysis import finish_dl_training
        with tempfile.TemporaryDirectory() as tmp:
            model = torch.nn.Linear(2, 2)
            with torch.no_grad():
                model.weight.copy_(torch.tensor([[4.0, 0.0], [0.0, 4.0]]))
                model.bias.zero_()
            path = Path(tmp) / "best.pth"
            torch.save({"class_names": ["a", "b"], "model_state_dict": model.state_dict()}, path)
            inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0]] * 10)
            targets = torch.tensor([0, 1] * 10)
            loader = DataLoader(TensorDataset(inputs, targets), batch_size=4)
            with torch.no_grad():
                model.weight.zero_()
            finish_dl_training(model, path, loader, "cpu", {"seed": 42})
            result = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(result["rejection_policy"]["validation_accuracy_accepted"], 1.0)
            self.assertEqual(result["training_analysis"]["total_parameters"], 6)

    def test_both_dl_cv_runners_reserve_outer_fold(self):
        import importlib
        import torch
        from torch.utils.data import Dataset

        class TinyImages(Dataset):
            classes = ["a", "b"]
            targets = [0, 1] * 24
            samples = [(f"image{i}.jpg", i % 2) for i in range(48)]

            def __len__(self):
                return len(self.targets)

            def __getitem__(self, idx):
                return torch.tensor([float(idx % 2), float(1 - idx % 2)]), self.targets[idx]

        with tempfile.TemporaryDirectory() as tmp:
            for name in ["dl_transfer_learning_bird_classifier",
                         "dl_transfer_learning_bird_classifier_partial_finetune"]:
                module = importlib.import_module(name)
                out = Path(tmp) / name
                out.mkdir()
                dataset = TinyImages()
                with patch.object(module.datasets, "ImageFolder", return_value=dataset), \
                        patch.object(module, "build_model", side_effect=lambda n: torch.nn.Linear(2, n)), \
                        patch.object(module, "freeze_backbone"), patch.object(module, "unfreeze_backbone"), \
                        patch.object(module, "plot_curves"), \
                        patch.object(module, "train_one_epoch", return_value=(0.5, 0.5)), \
                        patch.object(module, "eval_one_epoch", return_value=(0.5, 0.5)):
                    if "partial" in name:
                        with patch.object(module, "fine_tune_last_internal_layers", return_value=[]):
                            module.run_kfold_on_train("unused", out, k=3, warmup_epochs=1,
                                                      finetune_epochs=1, balance_train=True,
                                                      pool_manifest_path=Path(tmp) / "shared_pool.csv")
                    else:
                        module.run_kfold_on_train("unused", out, k=3, warmup_epochs=1,
                                                  finetune_epochs=1, balance_train=True,
                                                  pool_manifest_path=Path(tmp) / "shared_pool.csv")
                scores = pd.read_csv(out / "outer_cv_scores.csv")
                self.assertEqual(len(scores), 3)
                self.assertTrue(scores.training_pool_id.notna().all())
                pd.testing.assert_frame_equal(pd.read_csv(out / "training_pool.csv"),
                                              pd.read_csv(Path(tmp) / "shared_pool.csv"))
                self.assertTrue((out / "outer_cv_scores.png").exists())
                for i in range(1, 4):
                    self.assertTrue((out / f"outer_fold_{i}_cm.png").exists())
                self.assertTrue(scores.evaluation_role.eq("independent_outer").all())
                keys = pd.concat([pd.read_csv(out / f"outer_predictions_fold_{i}.csv")
                                  for i in range(1, 4)]).sample_key
                self.assertEqual(len(keys), 48)
                self.assertFalse(keys.duplicated().any())


if __name__ == "__main__":
    unittest.main()
