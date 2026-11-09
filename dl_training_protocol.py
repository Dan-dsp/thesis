"""Shared validation-selected F1 checkpoints and fresh full-pool DL refits."""

from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from classification_analysis import save_json
from result_plots import _figure, _save


def validation_metrics(model, loader, device, num_classes):
    model.eval()
    truth, predicted = [], []
    loss_sum = 0.0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for images, labels in loader:
            output = model(images.to(device))
            loss_sum += float(criterion(output, labels.to(device)).item())
            truth.extend(labels.cpu().tolist())
            predicted.extend(output.argmax(1).cpu().tolist())
    if not truth:
        raise ValueError("Checkpoint selection requires nonempty validation data.")
    return {"loss": loss_sum / len(truth), "accuracy": float(accuracy_score(truth, predicted)),
            "f1_macro": float(f1_score(truth, predicted, labels=np.arange(num_classes),
                                        average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(truth, predicted, labels=np.arange(num_classes),
                                           average="weighted", zero_division=0))}


def fit_with_f1_checkpoints(model, train_loader, val_loader, class_names, best_path,
                            warmup_epochs, finetune_epochs, lr_head, lr_finetune,
                            device, freeze, unfreeze, train_epoch, *, phase_specific=False):
    """Select macro-F1 checkpoints; phase_specific keeps a winner for each phase.

    The historical default retains its global winner. The current CV workflow
    sets phase_specific=True and always selects a fine-tuning-phase winner.
    """
    if warmup_epochs < 1 or finetune_epochs < 0:
        raise ValueError("Require at least one warmup epoch and nonnegative fine-tuning epochs.")
    best_path = Path(best_path)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    weighted_path = best_path.with_name(best_path.stem + "_weighted_f1.pth")
    history = {key: [] for key in ("train_loss", "train_acc", "val_loss", "val_acc",
                                   "val_f1_macro", "val_f1_weighted", "phase", "phase_epoch",
                                   "improved_macro_f1")}
    best = {"f1_macro": -1.0, "f1_weighted": -1.0}
    selected_warmup = 0
    layers = []
    for phase, epochs, lr in (("warmup", warmup_epochs, lr_head),
                               ("finetune", finetune_epochs, lr_finetune)):
        if phase == "warmup":
            freeze(model)
        else:
            checkpoint = torch.load(best_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            selected_warmup = checkpoint["selected_schedule"]["warmup_epochs"]
            if phase_specific:
                # Preserve the warmup winner; select fine-tuning independently even
                # if every fine-tuning score is below the warmup score.
                torch.save(checkpoint, best_path.with_name("best_warmup_macro_f1.pth"))
                best = {"f1_macro": -1.0, "f1_weighted": -1.0}
            layers = unfreeze(model)
            layers = "all" if layers is None else layers
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        for epoch in range(1, epochs + 1):
            loss, acc = train_epoch(model, train_loader, torch.nn.CrossEntropyLoss(), optimizer, device, "train")
            metrics = validation_metrics(model, val_loader, device, len(class_names))
            for key, value in {"train_loss": loss, "train_acc": acc, "val_loss": metrics["loss"],
                               "val_acc": metrics["accuracy"], "val_f1_macro": metrics["f1_macro"],
                               "val_f1_weighted": metrics["f1_weighted"], "phase": phase,
                               "phase_epoch": epoch,
                               "improved_macro_f1": metrics["f1_macro"] > best["f1_macro"]}.items():
                history[key].append(value)
            schedule = {"warmup_epochs": epoch if phase == "warmup" else selected_warmup,
                        "finetune_epochs": 0 if phase == "warmup" else epoch}
            print(f"[{phase} {epoch}/{epochs}] Val macro-F1={metrics['f1_macro']:.4f}; "
                  f"weighted-F1={metrics['f1_weighted']:.4f}; accuracy={metrics['accuracy']:.4f}")
            for metric, destination in (("f1_macro", best_path), ("f1_weighted", weighted_path)):
                if metrics[metric] > best[metric]:
                    best[metric] = metrics[metric]
                    torch.save({"model_state_dict": model.state_dict(), "class_names": list(class_names),
                                "checkpoint_role": ("pooled_cv_selection" if phase_specific
                                                    else "internal_validation_selection"),
                                "selection_metric": metric, "selection_metrics": metrics,
                                "best_val_acc": metrics["accuracy"],
                                "selected_schedule": schedule, "phase": phase,
                                "fine_tuned_internal_layers": layers,
                                "history": history}, destination)
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    table = pd.DataFrame(history)
    schedule = checkpoint["selected_schedule"]
    table["selected_phase_checkpoint"] = (
        (table.phase.eq("warmup") & table.phase_epoch.eq(schedule["warmup_epochs"])) |
        (table.phase.eq("finetune") & table.phase_epoch.eq(schedule["finetune_epochs"])))
    table.to_csv(best_path.with_suffix(".history.csv"), index=False)
    fig, ax = _figure()
    for metric in ("val_f1_macro", "val_f1_weighted"):
        ax.plot(np.arange(1, len(history[metric]) + 1), history[metric], label=metric)
    ax.set(xlabel="Selection epoch", ylabel="Validation F1", title="DL checkpoint selection")
    ax.legend()
    _save(fig, best_path.with_suffix(".f1.png"))
    return checkpoint, history


def refit_full_pool(development, class_names, selection_checkpoint, final_path, *,
                    seed, batch_size, num_workers, device, lr_head, lr_finetune,
                    build_model, set_seed, freeze, unfreeze, train_epoch, settings):
    """Fresh pretrained initialization; fixed validation-selected schedule, no evaluation."""
    start = time.perf_counter()
    set_seed(seed)
    model = build_model(len(class_names)).to(device)
    loader = DataLoader(development, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=device != "cpu")
    schedule = dict(selection_checkpoint["selected_schedule"])
    rows, layers = [], []
    for phase, count, lr in (("warmup", schedule["warmup_epochs"], lr_head),
                              ("finetune", schedule["finetune_epochs"], lr_finetune)):
        if not count:
            continue
        if phase == "warmup":
            freeze(model)
        else:
            layers = unfreeze(model)
            layers = "all" if layers is None else layers
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        for epoch in range(1, count + 1):
            loss, accuracy = train_epoch(model, loader, torch.nn.CrossEntropyLoss(), optimizer, device, "refit")
            rows.append({"phase": phase, "epoch": epoch, "train_loss": loss, "train_acc": accuracy})
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    policy = {"threshold": 0.0, "status": "disabled_full_pool_refit_no_independent_calibration"}
    analysis = {"family": "DL", "architecture": "ResNet18",
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters_final_stage": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "training_algorithm": "Adam, cross-entropy; fresh full-pool refit",
                "hyperparameters": dict(settings, seed=seed, batch_size=batch_size,
                                         lr_head=lr_head, lr_finetune=lr_finetune),
                "refit_samples": len(development), "selected_schedule": schedule,
                "refit_seconds": time.perf_counter() - start}
    torch.save({"model_state_dict": model.state_dict(), "class_names": list(class_names),
                "checkpoint_role": "full_training_pool_refit", "selection_metric": "f1_macro",
                "selection_metrics": selection_checkpoint["selection_metrics"],
                "selected_schedule": schedule, "fine_tuned_internal_layers": layers,
                "rejection_policy": policy, "training_analysis": analysis}, final_path)
    pd.DataFrame(rows).to_csv(final_path.parent / "refit_history.csv", index=False)
    save_json(final_path.parent / "model_analysis.json", analysis)
    save_json(final_path.parent / "rejection_policy.json", policy)
    save_json(final_path.parent / "refit_protocol.json", {"selection_metric": "f1_macro", "selected_schedule": schedule,
               "selection_metrics_not_refit_validation": selection_checkpoint["selection_metrics"],
               "schedule_source": settings.get("schedule_source", "single_internal_validation_split"),
               "validation_protocol": settings.get("validation_protocol"),
               "refit_samples": len(development)})
    pd.DataFrame({"sample_key": [f"{development.classes[int(y)]}/{Path(p).name}"
                                 for p, y in development.samples]}).to_csv(
                                     final_path.parent / "refit_membership.csv", index=False)
    return final_path
