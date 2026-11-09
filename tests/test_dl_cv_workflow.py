import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import dl_cv_workflow as cv
import rerun_dl_workflow as runner
from dl_training_protocol import fit_with_f1_checkpoints


class MixedCVTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_mean_not_median_and_rounding(self):
        schedules = [{"warmup_epochs": w, "finetune_epochs": f}
                     for w, f in zip([1, 1, 3, 3, 3], [1, 1, 1, 5, 5])]
        means, schedule = cv.mean_epoch_schedule(schedules)
        self.assertAlmostEqual(means["warmup_epochs"], 2.2)
        self.assertAlmostEqual(means["finetune_epochs"], 2.6)
        self.assertEqual(schedule, {"warmup_epochs": 2, "finetune_epochs": 3})
        for bad in ([], [{"warmup_epochs": 0, "finetune_epochs": 1}]):
            with self.assertRaises(ValueError):
                cv.mean_epoch_schedule(bad)

    def test_warmup_restored_and_phases_have_independent_winners(self):
        model = torch.nn.Linear(1, 2)
        torch.nn.init.zeros_(model.weight)
        loader = DataLoader(TensorDataset(torch.zeros(4, 1), torch.tensor([0, 1, 0, 1])))
        metrics = [dict(loss=1., accuracy=.5, f1_macro=v, f1_weighted=v)
                   for v in [.8, .9, .7, .4, .6, .6, .5, .3]]
        starts = []
        def train(model, *args):
            starts.append(float(model.weight[0, 0].detach()))
            with torch.no_grad():
                model.weight.add_(1)
            return 1., .5
        with tempfile.TemporaryDirectory() as tmp, patch("dl_training_protocol.validation_metrics", side_effect=metrics):
            path = Path(tmp) / "best_finetune_macro_f1.pth"
            checkpoint, history = fit_with_f1_checkpoints(model, loader, loader, ["a", "b"],
                path, 3, 5, .01, .001, "cpu", lambda m: None, lambda m: None, train,
                phase_specific=True)
            self.assertEqual(starts, [0., 1., 2., 2., 3., 4., 5., 6.])
            self.assertEqual(checkpoint["selected_schedule"], {"warmup_epochs": 2, "finetune_epochs": 2})
            self.assertEqual(checkpoint["selection_metrics"]["f1_macro"], .6)
            warmup = torch.load(Path(tmp) / "best_warmup_macro_f1.pth", weights_only=True)
            self.assertEqual(warmup["selection_metrics"]["f1_macro"], .9)
            table = pd.read_csv(path.with_suffix(".history.csv"))
            self.assertEqual(len(table), 8)
            self.assertEqual(table.selected_phase_checkpoint.sum(), 2)
            self.assertEqual(table.loc[table.selected_phase_checkpoint, "phase_epoch"].tolist(), [2, 2])

    def test_five_runs_then_fresh_refit_every_row(self):
        data = TensorDataset(torch.zeros(20, 2), torch.tensor([0, 1] * 10))
        data.classes = ["a", "b"]
        data.targets = [0, 1] * 10
        data.samples = [(f"image_{i}.png", i % 2) for i in range(20)]
        data.augmentation_id = "synthetic_mixed_views"
        sizes, factories = [], []
        def build(n):
            model = torch.nn.Linear(2, n)
            factories.append(id(model))
            return model
        def train(model, loader, *args):
            sizes.append(len(loader.dataset))
            # Exercise real optimization on tiny tensors, never load images/pretrained weights.
            optimizer = args[1]
            for x, y in loader:
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(model(x), y)
                loss.backward()
                optimizer.step()
            return 1., .5
        values = [.5, .8, .6, .4, .5, .6, .9, .7] * 5
        metrics = [dict(loss=1., accuracy=.5, f1_macro=v, f1_weighted=v) for v in values]
        with tempfile.TemporaryDirectory() as tmp, patch("dl_training_protocol.validation_metrics", side_effect=metrics) as validate:
            final = cv.select_cv_and_refit(data, Path(tmp), model_name="full", seed=42,
                batch_size=4, num_workers=0, device="cpu", build_model=build,
                set_seed=torch.manual_seed, freeze=lambda m: None, unfreeze=lambda m: None,
                train_epoch=train)
            self.assertEqual(validate.call_count, 40)  # Not called during final refit.
            self.assertEqual(len(factories), 6)
            self.assertEqual(sizes, [16] * 40 + [20] * 6)
            checkpoint = torch.load(final, weights_only=True)
            self.assertEqual(checkpoint["selected_schedule"], {"warmup_epochs": 2, "finetune_epochs": 4})
            self.assertEqual(checkpoint["training_analysis"]["refit_samples"], 20)
            self.assertEqual(checkpoint["training_analysis"]["hyperparameters"]["early_stopping"], False)
            validations = []
            for fold in range(1, 6):
                membership = pd.read_csv(Path(tmp) / "cv_selection" / f"fold_{fold}" / "split_membership.csv")
                validations.extend(membership.loc[membership.role.eq("validation"), "row_index"].tolist())
            self.assertEqual(sorted(validations), list(range(20)))
            table = pd.read_csv(Path(tmp) / "cv_selection" / "fold_scores.csv")
            self.assertTrue(table.evaluation_role.eq(cv.ROLE).all())
            self.assertFalse((Path(tmp) / "kfold_train_only" / "outer_cv_scores.csv").exists())
            other = table.copy()
            other["model"] = "partial"
            other_path = Path(tmp) / "partial_scores.csv"
            other.to_csv(other_path, index=False)
            cv.compare_selection_scores([Path(tmp) / "cv_selection" / "fold_scores.csv", other_path],
                                        Path(tmp) / "comparison")
            self.assertFalse((Path(tmp) / "comparison" / "within_family_tests.csv").exists())
            other.loc[0, "partition_id"] = "wrong"
            other.to_csv(other_path, index=False)
            with self.assertRaisesRegex(ValueError, "partition_id"):
                cv.compare_selection_scores([Path(tmp) / "cv_selection" / "fold_scores.csv", other_path],
                                            Path(tmp) / "invalid")

    def test_runner_uses_exact_new_checkpoints_and_both_test_populations(self):
        with tempfile.TemporaryDirectory() as tmp:
            stages = runner.build_stages(Path(tmp) / "run", Path(tmp) / "sl")
            self.assertEqual(len(stages), 5)
            self.assertEqual(stages[0].arguments[0], "dl_transfer_learning_bird_classifier.py")
            self.assertEqual(stages[2].arguments[0], "dl_cv_workflow.py")
            self.assertIn("--balance-test", stages[3].arguments)
            self.assertIn("--no-balance-test", stages[4].arguments)
            self.assertEqual(Path(stages[3].arguments[2]), Path(tmp) / "run" / "full" / "final_train" / runner.CHECKPOINT)
            self.assertNotIn("thesis_validation.py", [stage.arguments[0] for stage in stages])
            self.assertEqual(len(runner.build_stages(Path(tmp), Path(tmp), reports_only=True)), 3)

    def test_dry_run_does_not_execute(self):
        with patch.object(runner, "resolve_sl_run", return_value=Path("saved_sl")), \
             patch.object(runner, "execute") as execute, patch.object(runner, "preflight") as preflight:
            self.assertEqual(runner.main(["--dry-run"]), 0)
            execute.assert_not_called()
            preflight.assert_not_called()


if __name__ == "__main__":
    unittest.main()
