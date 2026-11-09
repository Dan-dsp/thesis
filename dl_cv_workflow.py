"""Five mixed-row selection folds, mean epoch counts, and one fresh full-pool refit.

These are checkpoint-selection scores, NOT independent outer-CV estimates.
Originals and augmented views are deliberately allowed to cross fold boundaries.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

from classification_analysis import partition_id, save_json
from dl_training_protocol import fit_with_f1_checkpoints, refit_full_pool
from shared_augmentation import dl_development_pool
from shared_training_pool import DEFAULT_POOL_MANIFEST


PROJECT_ROOT = Path(DEFAULT_POOL_MANIFEST).parent.parent
PROTOCOL = "pooled_rows_five_fold_mean_epochs_v1"
ROLE = "checkpoint_selection_not_independent"
CHECKPOINT_NAME = "bird_species_resnet18_FINAL_best.pth"
LIMITATION = ("Original and augmented rows are mixed before splitting. Related views may "
              "cross fitting/validation boundaries. Scores also select checkpoints and are "
              "not independent outer evaluation; do not use them for outer-CV significance tests.")


def mean_epoch_schedule(schedules):
    """Arithmetic mean per phase, rounded to nearest integer (.5 rounds up)."""
    if not schedules:
        raise ValueError("At least one completed fold schedule is required.")
    means, selected = {}, {}
    for phase in ("warmup_epochs", "finetune_epochs"):
        values = np.asarray([row[phase] for row in schedules], dtype=float)
        if not np.isfinite(values).all() or (values < 1).any() or (values != np.floor(values)).any():
            raise ValueError("Selected phase epochs must be positive integers.")
        means[phase] = float(values.mean())
        selected[phase] = int(np.floor(means[phase] + 0.5))
    return means, selected


def select_cv_and_refit(development, out_dir, *, model_name, seed, batch_size,
                        num_workers, device, build_model, set_seed, freeze, unfreeze,
                        train_epoch, pool_info=None, folds=5, warmup_epochs=3,
                        finetune_epochs=5, lr_head=1e-3, lr_finetune=1e-4):
    """No inner split or test access: one fixed validation fold per 3+5 run."""
    if (folds, warmup_epochs, finetune_epochs) != (5, 3, 5):
        raise ValueError("This protocol requires five folds, three warmup and five fine-tuning epochs.")
    out_dir = Path(out_dir)
    cv_dir = out_dir / "cv_selection"
    cv_dir.mkdir(parents=True, exist_ok=True)
    targets = np.asarray(development.targets)
    names = list(development.classes)
    keys = [f"{names[int(y)]}/{Path(p).name}" for p, y in development.samples]
    if len(keys) != len(set(keys)) or len(keys) != len(development):
        raise ValueError("Development rows need unique species/filename identities.")
    labels, counts = np.unique(targets, return_counts=True)
    if len(labels) != len(names) or counts.min() < folds:
        raise ValueError("Each species must have at least five development rows.")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows, schedules = [], []
    pool_id = pool_info.get("training_pool_id") if pool_info else None
    aug_id = getattr(development, "augmentation_id", None)
    protocol = {"protocol": PROTOCOL, "evaluation_role": ROLE, "seed": seed,
                "folds": folds, "warmup_budget": warmup_epochs,
                "finetune_budget": finetune_epochs, "early_stopping": False,
                "selection_metric": "f1_macro", "tie_rule": "earliest_epoch",
                "schedule_rule": "arithmetic_mean_then_nearest_integer_half_up",
                "development_samples": len(development), "training_pool_id": pool_id,
                "augmentation_id": aug_id, "limitations": LIMITATION}
    save_json(cv_dir / "selection_protocol.json", protocol)
    print(f"[CV SELECTION] {LIMITATION}", flush=True)
    started = time.perf_counter()
    for fold, (train_idx, val_idx) in enumerate(cv.split(np.zeros(len(targets)), targets), 1):
        fold_dir = cv_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        membership = pd.DataFrame({"row_index": np.arange(len(targets)), "sample_key": keys,
                                   "species": [names[int(y)] for y in targets],
                                   "role": "fitting"})
        membership.loc[val_idx, "role"] = "validation"
        membership.to_csv(fold_dir / "split_membership.csv", index=False)
        set_seed(seed + fold)
        train_loader = DataLoader(Subset(development, train_idx.tolist()), batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers, pin_memory=device != "cpu")
        val_loader = DataLoader(Subset(development, val_idx.tolist()), batch_size=batch_size,
                                shuffle=False, num_workers=num_workers, pin_memory=device != "cpu")
        model = build_model(len(names)).to(device)
        best_path = fold_dir / "best_finetune_macro_f1.pth"
        print(f"[FOLD {fold}/5] fitting={len(train_idx)}, validation={len(val_idx)}", flush=True)
        checkpoint, history = fit_with_f1_checkpoints(
            model, train_loader, val_loader, names, best_path, warmup_epochs, finetune_epochs,
            lr_head, lr_finetune, device, freeze, unfreeze, train_epoch, phase_specific=True)
        warmup_path = fold_dir / "best_warmup_macro_f1.pth"
        warmup = torch.load(warmup_path, map_location="cpu", weights_only=True)
        schedule = dict(checkpoint["selected_schedule"])
        schedules.append(schedule)
        row = {"family": "DL", "model": model_name, "repeat": 0, "fold": fold,
               "partition_id": partition_id([keys[i] for i in val_idx]),
               "evaluation_role": ROLE, "n_train": len(train_idx), "n_validation": len(val_idx),
               "training_pool_id": pool_id, "augmentation_id": aug_id,
               "best_warmup_epoch": schedule["warmup_epochs"],
               "best_finetune_epoch": schedule["finetune_epochs"],
               "warmup_checkpoint": str(warmup_path.resolve()),
               "finetune_checkpoint": str(best_path.resolve())}
        for phase, state in (("warmup", warmup), ("finetune", checkpoint)):
            for metric, value in state["selection_metrics"].items():
                row[f"{phase}_{metric}"] = float(value)
        rows.append(row)
        pd.DataFrame(rows).to_csv(cv_dir / "fold_scores.csv", index=False)
        save_json(fold_dir / "selected_epochs.json", schedule)
        # Do not retain five GPU models/checkpoints while training the next fold.
        del model, checkpoint, warmup, train_loader, val_loader, history
    means, schedule = mean_epoch_schedule(schedules)
    table = pd.DataFrame(rows)
    summaries = []
    for phase in ("warmup", "finetune"):
        for metric in ("f1_macro", "f1_weighted", "accuracy", "loss"):
            values = table[f"{phase}_{metric}"]
            summaries.append({"model": model_name, "phase": phase, "metric": metric,
                              "mean": float(values.mean()), "std": float(values.std(ddof=1)),
                              "folds": folds, "evaluation_role": ROLE})
    pd.DataFrame(summaries).to_csv(cv_dir / "cv_summary.csv", index=False)
    selection = dict(protocol, per_fold_schedules=schedules, mean_epochs=means,
                     selected_schedule=schedule, selection_seconds=time.perf_counter() - started)
    save_json(cv_dir / "epoch_selection.json", selection)
    print(f"[FINAL SCHEDULE] means={means}; rounded={schedule}", flush=True)
    # Only the schedule/summary travels to refit: no selected fold weights.
    descriptor = {"selected_schedule": schedule,
                  "selection_metrics": {"evaluation_role": ROLE,
                      "mean_finetune_f1_macro": float(table.finetune_f1_macro.mean()),
                      "std_finetune_f1_macro": float(table.finetune_f1_macro.std(ddof=1))}}
    settings = dict(selection, fine_tuning_mode=model_name,
                    validation_protocol=PROTOCOL, shared_training_pool_id=pool_id,
                    schedule_source=str((cv_dir / "epoch_selection.json").resolve()))
    return refit_full_pool(development, names, descriptor,
        out_dir / "final_train" / CHECKPOINT_NAME, seed=seed, batch_size=batch_size,
        num_workers=num_workers, device=device, lr_head=lr_head, lr_finetune=lr_finetune,
        build_model=build_model, set_seed=set_seed, freeze=freeze, unfreeze=unfreeze,
        train_epoch=train_epoch, settings=settings)


def run_training_cli(trainer, mode, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    default_name = "outputs_resnet18_thesis" + ("_partial_finetune" if mode == "partial" else "")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / default_name)
    parser.add_argument("--train-dir", type=Path, default=PROJECT_ROOT / "dataset_bbox" / "train")
    parser.add_argument("--test-dir", type=Path, default=PROJECT_ROOT / "dataset_bbox" / "test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch-size must be positive and num-workers nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    transform, _ = trainer.get_transforms()
    source = trainer.datasets.ImageFolder(str(args.train_dir), transform=transform)
    indices, _, pool_info = trainer.select_dl_training_pool(
        source, seed=42, enabled=True, mode="undersample", manifest_path=DEFAULT_POOL_MANIFEST,
        out_dir=args.output_dir, dataset_root=str(args.train_dir))
    development = dl_development_pool(source, indices, DEFAULT_POOL_MANIFEST, args.output_dir)
    unfreeze = (trainer.unfreeze_backbone if mode == "full" else
                lambda model: trainer.fine_tune_last_internal_layers(model, 4))
    path = select_cv_and_refit(development, args.output_dir, model_name=f"resnet18_{mode}_finetune",
        seed=42, batch_size=args.batch_size, num_workers=args.num_workers, device=args.device,
        build_model=trainer.build_model, set_seed=trainer.set_seed, freeze=trainer.freeze_backbone,
        unfreeze=unfreeze, train_epoch=trainer.train_one_epoch, pool_info=pool_info)
    trainer.evaluate_on_test(path, str(args.test_dir), args.output_dir / "test_eval",
        batch_size=args.batch_size, num_workers=args.num_workers, device=args.device,
        balance_test=True, balance_mode="undersample", seed=42)
    print(f"[COMPLETE] Fresh full-pool model: {path}", flush=True)
    return path


def compare_selection_scores(paths, out_dir):
    """Descriptive paired selection comparison, deliberately without p-values."""
    tables = [pd.read_csv(path) for path in paths]
    if len(tables) != 2:
        raise ValueError("Supply exactly the full and partial selection score files.")
    for table in tables:
        if len(table) != 5 or set(table.fold) != set(range(1, 6)) or table.model.nunique() != 1:
            raise ValueError("Each variant must have exactly five unique fold rows.")
        if not table.evaluation_role.eq(ROLE).all():
            raise ValueError("Expected mixed-row checkpoint-selection scores, not outer scores.")
    left, right = [table.sort_values("fold").reset_index(drop=True) for table in tables]
    for column in ("fold", "partition_id", "n_train", "n_validation", "training_pool_id", "augmentation_id"):
        if not left[column].equals(right[column]):
            raise ValueError(f"Full and partial folds differ: {column}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(out_dir / "selection_scores.csv", index=False)
    summary = combined.groupby("model")[["warmup_f1_macro", "finetune_f1_macro",
                                         "warmup_accuracy", "finetune_accuracy"]].agg(["mean", "std"])
    summary.to_csv(out_dir / "selection_summary.csv")
    pd.DataFrame({"fold": left.fold, "left_model": left.model, "right_model": right.model,
                  "warmup_f1_difference": left.warmup_f1_macro - right.warmup_f1_macro,
                  "finetune_f1_difference": left.finetune_f1_macro - right.finetune_f1_macro}).to_csv(
                      out_dir / "paired_selection_differences.csv", index=False)
    save_json(out_dir / "comparison_protocol.json", {"evaluation_role": ROLE,
        "sources": [str(Path(path).resolve()) for path in paths], "limitations": LIMITATION,
        "statistical_significance_tests": "not_applicable_to_checkpoint_selection_scores"})
    from result_plots import _figure, _save
    fig, ax = _figure()
    for table in tables:
        ax.plot(table.fold, table.finetune_f1_macro, marker="o", label=table.model.iloc[0])
    ax.set(xlabel="Selection fold", ylabel="Selected validation macro F1",
           title="DL selection scores (not independent evaluation)")
    ax.legend()
    _save(fig, out_dir / "selection_scores.png")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descriptive DL selection-fold comparison")
    parser.add_argument("scores", nargs=2, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    arguments = parser.parse_args()
    print(compare_selection_scores(arguments.scores, arguments.out_dir))
