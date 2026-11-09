from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from dl_training_protocol import fit_with_f1_checkpoints, refit_full_pool, validation_metrics
from shared_test_pool import keyed_test_cohort, sl_test_indices


class ProtocolTests(unittest.TestCase):
    def test_macro_selects_checkpoint_not_accuracy_and_weighted_is_separate(self):
        model = torch.nn.Linear(2, 2)
        loader = DataLoader(TensorDataset(torch.zeros(4, 2), torch.tensor([0, 1, 0, 1])))
        metrics = [dict(loss=1., accuracy=a, f1_macro=m, f1_weighted=w)
                   for a, m, w in [(0.9, .5, .7), (.95, .6, .65), (.7, .9, .6), (.99, .8, .95)]]
        with tempfile.TemporaryDirectory() as tmp, patch("dl_training_protocol.validation_metrics", side_effect=metrics):
            path = Path(tmp) / "best.pth"
            best, history = fit_with_f1_checkpoints(
                model, loader, loader, ["a", "b"], path, 2, 2, .01, .001, "cpu",
                lambda m: None, lambda m: None, lambda *args: (1., .5))
            self.assertEqual(best["selected_schedule"], {"warmup_epochs": 2, "finetune_epochs": 1})
            self.assertEqual(best["selection_metrics"]["accuracy"], .7)
            self.assertEqual(best["selection_metric"], "f1_macro")
            weighted = torch.load(path.with_name("best_weighted_f1.pth"), weights_only=True)
            self.assertEqual(weighted["selected_schedule"]["finetune_epochs"], 2)
            self.assertEqual(weighted["selection_metric"], "f1_weighted")
            self.assertEqual(len(history["val_f1_macro"]), 4)

    def test_refit_starts_fresh_and_uses_every_row_without_validation(self):
        data = TensorDataset(torch.arange(20).reshape(10, 2).float(), torch.tensor([0, 1] * 5))
        data.classes = ["a", "b"]
        data.samples = [(f"image_{i}.png", i % 2) for i in range(10)]
        seen = []
        def factory(n):
            model = torch.nn.Linear(2, n)
            torch.nn.init.zeros_(model.weight)
            return model
        def epoch(model, loader, *args):
            seen.append(sorted(int(x[0]) for batch, _ in loader for x in batch))
            self.assertTrue(torch.all(model.weight == 0))
            return 1., .5
        selection = {"selected_schedule": {"warmup_epochs": 1, "finetune_epochs": 1},
                     "selection_metrics": {"f1_macro": .9},
                     "model_state_dict": {"weight": torch.full((2, 2), 99.)}}
        with tempfile.TemporaryDirectory() as tmp, patch("dl_training_protocol.validation_metrics") as validate:
            path = refit_full_pool(data, data.classes, selection, Path(tmp) / "final.pth",
                seed=42, batch_size=3, num_workers=0, device="cpu", lr_head=.01, lr_finetune=.001,
                build_model=factory, set_seed=torch.manual_seed, freeze=lambda m: None,
                unfreeze=lambda m: ["last"], train_epoch=epoch, settings={})
            validate.assert_not_called()
            ckpt = torch.load(path, weights_only=True)
            self.assertEqual(ckpt["checkpoint_role"], "full_training_pool_refit")
            self.assertEqual(ckpt["training_analysis"]["refit_samples"], 10)
            self.assertEqual(len(pd.read_csv(Path(tmp) / "refit_membership.csv")), 10)
        self.assertEqual(seen, [list(range(0, 20, 2))] * 2)

    def test_f1_uses_all_classes_and_whole_epoch(self):
        data = TensorDataset(torch.tensor([[5., 0.], [5., 0.], [0., 5.]]), torch.tensor([0, 1, 1]))
        metrics = validation_metrics(torch.nn.Identity(), DataLoader(data, batch_size=2), "cpu", 2)
        self.assertAlmostEqual(metrics["f1_macro"], 2 / 3)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)

    def test_identity_matching_is_order_independent_and_matches_sl(self):
        from sl_training_pipeline import make_balanced_subset
        reference = pd.DataFrame({"species": ["a"] * 5 + ["b"] * 3,
                                  "orig_filename": [f"x{i}.png" for i in range(8)]})
        old, _, _ = make_balanced_subset(np.arange(8)[:, None], reference.species.to_numpy())
        np.testing.assert_array_equal(old.ravel(), sl_test_indices(reference.species))
        order = reference.sample(frac=1, random_state=7).reset_index(drop=True)
        idx, cohort = keyed_test_cohort(order, reference)
        self.assertEqual((order.iloc[idx].species + "/" + order.iloc[idx].orig_filename).tolist(), cohort.sample_key.tolist())
        self.assertEqual(len(cohort), 6)
        with self.assertRaisesRegex(ValueError, "identities"):
            keyed_test_cohort(order.iloc[:-1], reference)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            keyed_test_cohort(pd.concat([order, order.iloc[:1]]), reference)

    def test_both_trainers_default_to_twenty_percent(self):
        import inspect
        import dl_transfer_learning_bird_classifier as full
        import dl_transfer_learning_bird_classifier_partial_finetune as partial
        for module in (full, partial):
            self.assertEqual(inspect.signature(module.train_final_model_with_internal_val).parameters["val_ratio"].default, .2)

    def test_both_final_trainers_select_then_refit_all_rows(self):
        import importlib
        from contextlib import ExitStack
        data = TensorDataset(torch.zeros(20, 2), torch.tensor([0, 1] * 10))
        data.classes = ["a", "b"]
        data.targets = [0, 1] * 10
        data.samples = [(f"image{i}.jpg", i % 2) for i in range(20)]
        for name in ("dl_transfer_learning_bird_classifier",
                     "dl_transfer_learning_bird_classifier_partial_finetune"):
            module = importlib.import_module(name)
            sizes = []
            def epoch(model, loader, *args):
                sizes.append(len(loader.dataset))
                return .5, .5
            with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                stack.enter_context(patch.object(module.datasets, "ImageFolder", return_value=data))
                factory = stack.enter_context(patch.object(module, "build_model", side_effect=lambda n: torch.nn.Linear(2, n)))
                for method in ("freeze_backbone", "unfreeze_backbone", "plot_curves"):
                    stack.enter_context(patch.object(module, method))
                if "partial" in name:
                    stack.enter_context(patch.object(module, "fine_tune_last_internal_layers", return_value=["last"]))
                stack.enter_context(patch.object(module, "train_one_epoch", side_effect=epoch))
                final = module.train_final_model_with_internal_val(
                    "unused", Path(tmp), warmup_epochs=1, finetune_epochs=1,
                    pool_manifest_path=Path(tmp) / "pool.csv")
                checkpoint = torch.load(final, weights_only=True)
                self.assertEqual(factory.call_count, 2)
                self.assertEqual(sizes[:2], [16, 16])
                self.assertTrue(all(size == 20 for size in sizes[2:]))
                self.assertGreater(len(sizes), 2)
                self.assertEqual(checkpoint["checkpoint_role"], "full_training_pool_refit")
                self.assertEqual(checkpoint["training_analysis"]["hyperparameters"]["internal_validation_samples"], 4)


if __name__ == "__main__":
    unittest.main()
