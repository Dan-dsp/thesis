# import torch
# from torch import nn, optim
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms, models
# from pathlib import Path
# import time
# from tqdm import tqdm  # <--- NEW: progress bar


# def set_seed(seed: int = 42):
#     import random, numpy as np
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


# def get_dataloaders(train_dir, val_dir, batch_size=32, num_workers=0):
#     train_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.RandomHorizontalFlip(0.5),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2),
#         transforms.RandomRotation(10),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225]),
#     ])

#     val_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225]),
#     ])

#     train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
#     val_dataset   = datasets.ImageFolder(val_dir,   transform=val_transform)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size,
#                               shuffle=True, num_workers=num_workers, pin_memory=True)
#     val_loader   = DataLoader(val_dataset, batch_size=batch_size,
#                               shuffle=False, num_workers=num_workers, pin_memory=True)
#     return train_dataset, val_dataset, train_loader, val_loader


# def build_model(num_classes: int):
#     model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
#     in_features = model.fc.in_features
#     model.fc = nn.Linear(in_features, num_classes)
#     return model


# def train_one_epoch(model, dataloader, criterion, optimizer, device, phase_name="train"):
#     model.train()
#     running_loss, running_correct, running_total = 0.0, 0, 0

#     # tqdm progress bar
#     pbar = tqdm(dataloader, desc=f"{phase_name.capitalize()} batch progress", leave=False)

#     for images, labels in pbar:
#         images, labels = images.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         loss.backward()
#         optimizer.step()

#         preds = outputs.argmax(dim=1)
#         running_loss += loss.item() * images.size(0)
#         running_correct += (preds == labels).sum().item()
#         running_total += labels.size(0)

#         avg_loss = running_loss / running_total
#         avg_acc = running_correct / running_total
#         pbar.set_postfix(loss=f"{avg_loss:.3f}", acc=f"{avg_acc*100:.1f}%")

#     return running_loss / running_total, running_correct / running_total


# def eval_one_epoch(model, dataloader, criterion, device, phase_name="val"):
#     model.eval()
#     running_loss, running_correct, running_total = 0.0, 0, 0
#     pbar = tqdm(dataloader, desc=f"{phase_name.capitalize()} batch progress", leave=False)

#     with torch.no_grad():
#         for images, labels in pbar:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             loss = criterion(outputs, labels)

#             preds = outputs.argmax(dim=1)
#             running_loss += loss.item() * images.size(0)
#             running_correct += (preds == labels).sum().item()
#             running_total += labels.size(0)

#             avg_loss = running_loss / running_total
#             avg_acc = running_correct / running_total
#             pbar.set_postfix(loss=f"{avg_loss:.3f}", acc=f"{avg_acc*100:.1f}%")

#     return running_loss / running_total, running_correct / running_total


# def freeze_backbone(model):
#     for name, param in model.named_parameters():
#         if not name.startswith("fc."):
#             param.requires_grad = False


# def unfreeze_backbone(model):
#     for param in model.parameters():
#         param.requires_grad = True


# def main():
#     # -------------------------------------------------
#     # Paths (update as needed)
#     # -------------------------------------------------
#     train_dir = r"F:/Univalle/01_TG/nuevo_dataset_split/train"
#     val_dir   = r"F:/Univalle/01_TG/nuevo_dataset_split/val"

#     # -------------------------------------------------
#     # Hyperparameters
#     # -------------------------------------------------
#     set_seed(42)
#     batch_size = 32
#     warmup_epochs = 3
#     finetune_epochs = 5
#     lr_head = 1e-3
#     lr_finetune = 1e-4
#     num_workers = 0
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"[INFO] Using device: {device}")

#     # -------------------------------------------------
#     # Data
#     # -------------------------------------------------
#     train_dataset, val_dataset, train_loader, val_loader = get_dataloaders(
#         train_dir, val_dir, batch_size, num_workers
#     )
#     num_classes = len(train_dataset.classes)
#     print(f"[INFO] Classes: {train_dataset.classes}")
#     print(f"[INFO] Train images: {len(train_dataset)} | Val images: {len(val_dataset)}")

#     # -------------------------------------------------
#     # Model
#     # -------------------------------------------------
#     model = build_model(num_classes).to(device)
#     criterion = nn.CrossEntropyLoss()

#     # -------------------------------------------------
#     # Warmup phase
#     # -------------------------------------------------
#     print("[INFO] Warmup: training classifier head only...")
#     freeze_backbone(model)
#     optimizer_head = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_head)

#     for epoch in range(warmup_epochs):
#         print(f"\n[WARMUP EPOCH {epoch+1}/{warmup_epochs}]")
#         start = time.time()
#         train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_head, device, "train")
#         val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device, "val")
#         elapsed = time.time() - start
#         print(f"Epoch {epoch+1}: TrainLoss={train_loss:.4f} TrainAcc={train_acc*100:.2f}% "
#               f"| ValLoss={val_loss:.4f} ValAcc={val_acc*100:.2f}% | Time={elapsed:.1f}s")

#     # -------------------------------------------------
#     # Fine-tune phase
#     # -------------------------------------------------
#     print("\n[INFO] Fine-tune: unfreezing entire network...")
#     unfreeze_backbone(model)
#     optimizer_full = optim.Adam(model.parameters(), lr=lr_finetune)

#     for epoch in range(finetune_epochs):
#         print(f"\n[FINETUNE EPOCH {epoch+1}/{finetune_epochs}]")
#         start = time.time()
#         train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_full, device, "train")
#         val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device, "val")
#         elapsed = time.time() - start
#         print(f"Epoch {epoch+1}: TrainLoss={train_loss:.4f} TrainAcc={train_acc*100:.2f}% "
#               f"| ValLoss={val_loss:.4f} ValAcc={val_acc*100:.2f}% | Time={elapsed:.1f}s")

#     # -------------------------------------------------
#     # Save checkpoint
#     # -------------------------------------------------
#     out_path = Path("bird_species_resnet18.pth")
#     torch.save({
#         "model_state_dict": model.state_dict(),
#         "class_names": train_dataset.classes,
#     }, out_path)
#     print(f"\n[INFO] Training complete. Model saved to {out_path.resolve()}")


# if __name__ == "__main__":
#     main()


"""
birds_resnet18_train_kfold_final_test.py

What this script does (end-to-end, thesis-friendly):
1) Loads a dataset organized as:
      TRAIN_DIR/
        classA/ img1.jpg ...
        classB/ ...
      TEST_DIR/
        classA/ ...
        classB/ ...
2) (Optional) Runs Stratified K-Fold Cross-Validation ONLY on the TRAIN set:
      - trains K separate runs, reports mean±std validation accuracy
      - saves per-fold curves + best-per-fold checkpoints
   This is for robust evaluation and hyperparameter justification.
3) Trains ONE final deployable model on TRAIN:
      - internally splits TRAIN into train_sub + val_sub (stratified)
      - warmup head-only then finetune full net
      - saves BEST final checkpoint (by val accuracy)
      - saves training curves (loss/acc)
4) Evaluates the final checkpoint on TEST once:
      - accuracy + classification report
      - confusion matrix figure (PNG) + report text file

Notes on augmentation:
- Train augmentation is applied ON-THE-FLY to the training subset only.
- Validation and test use deterministic transforms (no random aug).
- K-Fold splitting is done on ORIGINAL samples; aug does not add samples to folds.

Requirements:
- pip install torch torchvision tqdm matplotlib scikit-learn pandas
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility (best-effort)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# Transforms
# ----------------------------
def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns:
    - train_transform: includes random augmentation (used only for training subset)
    - eval_transform: deterministic (used for validation and test)
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, eval_transform


# ----------------------------
# Model
# ----------------------------
def build_model(num_classes: int) -> torch.nn.Module:
    """Build a ResNet-18 with a new final layer for num_classes."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    # The last layer of the model is called model.fc
    model.fc = nn.Linear(in_features, num_classes)
    return model

# Head only training (warmup)
def freeze_backbone(model: torch.nn.Module) -> None:
    """Freeze all layers except the classifier head (fc)."""
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = False

# Fine-tuning
def unfreeze_backbone(model: torch.nn.Module) -> None:
    """Unfreeze all layers."""
    for param in model.parameters():
        param.requires_grad = True


# ----------------------------
# Train / Eval loops
# ----------------------------
def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module, # The loss function
    optimizer: optim.Optimizer,
    device: str,
    phase_name: str = "train" # For visualization purposes in the progress bar
) -> Tuple[float, float]:
    """Train for one epoch. Returns (epoch loss, epoch accuracy)."""
    model.train()
    running_loss, running_correct, running_total = 0.0, 0, 0

    """
    Initializes counters to accumulate metrics across all batches in the epoch:

    running_loss: sum of batch losses weighted by batch size.

    running_correct: total number of correct predictions so far.

    running_total: total number of samples seen so far.
    """

    pbar = tqdm(dataloader, desc=f"{phase_name.capitalize()} batch progress", leave=False) # leave = False means that when the epoc finishes, the progress bar diasppears

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True) # Before computing new gradients, you must clear old ones setting them to none
        # Predicted class scores (logits) for one image. They are NOT the probabilities
        outputs = model(images) # forward pass: outpus = fx(images). If batch_size = 32 and num_classses = 12, then outputs.shape = [32,12]
        loss = criterion(outputs, labels) # Loss computation using Softmax + negative log likelihood

        loss.backward() # Backpropagation
        optimizer.step() # Weight update

        preds = outputs.argmax(dim=1) # Selects the index of the largest logit per image
        running_loss += loss.item() * images.size(0) # Accumulating metrics
        running_correct += (preds == labels).sum().item() # Count how many predictions where correct in the batch
        running_total += labels.size(0) # Adds number of samples in this batch

        # So for each batch a training step is 1. Reset gradients, 2. Forward pass, 3. Compute loss, 4. Backpropagation, 5. Update weights, 6. Update metrics
        # When all the dataset has been used, that is then one training epoch

        # Computes current epoch-level metrics (max(1, running_total) avoids division by zero)
        avg_loss = running_loss / max(1, running_total) 
        avg_acc = running_correct / max(1, running_total)
        pbar.set_postfix(loss=f"{avg_loss:.3f}", acc=f"{avg_acc*100:.1f}%")

    return running_loss / running_total, running_correct / running_total


@torch.no_grad() # Evaluation with validation dataset of the epoch's performance (no gradients computed)
def eval_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader, 
    criterion: torch.nn.Module, 
    device: str,
    phase_name: str = "val"
) -> Tuple[float, float]:
    """Evaluate for one epoch. Returns (loss, accuracy)."""
    model.eval() # Evaluation of the model
    running_loss, running_correct, running_total = 0.0, 0, 0
    pbar = tqdm(dataloader, desc=f"{phase_name.capitalize()} batch progress", leave=False)

    for images, labels in pbar: # image.shape = (batch_size = 32, dimensions = 3, high = 224, width = 224)
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)
        running_loss += loss.item() * images.size(0)
        running_correct += (preds == labels).sum().item()
        running_total += labels.size(0)

        avg_loss = running_loss / max(1, running_total)
        avg_acc = running_correct / max(1, running_total)
        pbar.set_postfix(loss=f"{avg_loss:.3f}", acc=f"{avg_acc*100:.1f}%")

    return running_loss / running_total, running_correct / running_total


# ----------------------------
# Plotting
# ----------------------------
def plot_curves(history: Dict[str, List[float]], out_dir: Path, prefix: str) -> None:
    """Save loss (error) and accuracy curves as PNGs.
        Each index correspond to one epoch in the history dictionary
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1)) # Epoch's index

    # Loss = "error per epoch"
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Loss (Error) per Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_dir / f"{prefix}_loss_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train acc")
    plt.plot(epochs, history["val_acc"], label="Val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_dir / f"{prefix}_acc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()


# def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], out_path: Path) -> None:
#     """Save a confusion matrix figure (PNG)."""
#     plt.figure()
#     plt.imshow(cm)
#     plt.title("Confusion Matrix (Test)")
#     plt.xlabel("Predicted")
#     plt.ylabel("True")
#     plt.xticks(range(len(class_names)), class_names, rotation=90)
#     plt.yticks(range(len(class_names)), class_names)
#     plt.colorbar()
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=300, bbox_inches="tight")
#     plt.close()

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    out_path: Path,
) -> None:
    """
    Saves a NORMALIZED confusion matrix:
    - Row-normalized (each true class sums to 1)
    - Blue colormap
    - Values printed in each cell
    """
    # Row-normalize: cm_norm[i, j] = cm[i, j] / sum_j cm[i, j]
    cm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    plt.figure(figsize=(12, 6))
    plt.imshow(cm_norm, vmin=0.0, vmax=1.0, cmap="Blues")
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()

    plt.xticks(range(len(class_names)), class_names, rotation=90)
    plt.yticks(range(len(class_names)), class_names)

    # Write numbers in cells
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# ----------------------------
# Splitting helpers
# ----------------------------
def stratified_train_val_split(
    targets: np.ndarray,
    val_ratio: float,
    seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split indices into train_idx and val_idx with stratification.
    """
    idx = np.arange(len(targets))
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(idx, targets))
    return train_idx, val_idx


# ----------------------------
# Balancing helpers
# ----------------------------
def get_class_count_summary(targets: np.ndarray, class_names: List[str]) -> Dict[str, int]:
    """Return a per-class sample count summary."""
    values, counts = np.unique(targets, return_counts=True)
    summary = {class_name: 0 for class_name in class_names}
    for value, count in zip(values, counts):
        summary[class_names[int(value)]] = int(count)
    return summary


def build_balanced_indices(
    targets: np.ndarray,
    seed: int,
    mode: str = "undersample",
) -> np.ndarray:
    """
    Build class-balanced indices from a target vector.

    Modes:
    - undersample: reduce each class to the minority class size
    - oversample: replicate minority-class samples up to the majority class size
    """
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(targets, return_counts=True)

    if mode not in {"undersample", "oversample"}:
        raise ValueError(f"Unsupported balance mode: {mode}")

    target_count = counts.min() if mode == "undersample" else counts.max()
    balanced_chunks = []

    for class_id in classes:
        class_indices = np.where(targets == class_id)[0]
        replace = mode == "oversample" and len(class_indices) < target_count
        sampled = rng.choice(class_indices, size=target_count, replace=replace)
        balanced_chunks.append(sampled)

    balanced_idx = np.concatenate(balanced_chunks)
    rng.shuffle(balanced_idx)
    return balanced_idx.astype(int)


def maybe_balance_dataset(
    targets: np.ndarray,
    class_names: List[str],
    seed: int,
    enable_balance: bool,
    balance_mode: str,
    dataset_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optionally balance a dataset and return:
    - selected indices in the original dataset
    - balanced targets aligned with those indices
    """
    original_idx = np.arange(len(targets))
    original_summary = get_class_count_summary(targets, class_names)
    print(f"[{dataset_name}] Class counts before balancing: {original_summary}")

    if not enable_balance:
        return original_idx, targets

    balanced_idx = build_balanced_indices(targets, seed=seed, mode=balance_mode)
    balanced_targets = targets[balanced_idx]
    balanced_summary = get_class_count_summary(balanced_targets, class_names)
    print(f"[{dataset_name}] Class counts after balancing ({balance_mode}): {balanced_summary}")
    return balanced_idx, balanced_targets


# ----------------------------
# Final training (one model)
# ----------------------------
def train_final_model_with_internal_val(
    dataset_dir_train: str,
    out_dir: Path,
    seed: int = 42,
    val_ratio: float = 0.15,
    batch_size: int = 32,
    num_workers: int = 0,
    warmup_epochs: int = 3,
    finetune_epochs: int = 5,
    lr_head: float = 1e-3,
    lr_finetune: float = 1e-4,
    device: str = "cpu",
    balance_train: bool = False,
    balance_mode: str = "undersample",
) -> Path:
    """
    Train ONE final model:
    - internal stratified split of TRAIN into train_sub / val_sub
    - warmup + finetune
    - saves BEST checkpoint by val accuracy

    Returns:
    - path to best final checkpoint
    """
    train_tf, eval_tf = get_transforms()

    # Two views of the same folder:
    ds_train_tf = datasets.ImageFolder(dataset_dir_train, transform=train_tf)
    ds_eval_tf  = datasets.ImageFolder(dataset_dir_train, transform=eval_tf)

    class_names = ds_train_tf.classes
    targets = np.array(ds_train_tf.targets)
    num_classes = len(class_names)

    selected_idx, selected_targets = maybe_balance_dataset(
        targets=targets,
        class_names=class_names,
        seed=seed,
        enable_balance=balance_train,
        balance_mode=balance_mode,
        dataset_name="FINAL TRAIN",
    )

    train_idx_local, val_idx_local = stratified_train_val_split(selected_targets, val_ratio=val_ratio, seed=seed)
    train_idx = selected_idx[train_idx_local]
    val_idx = selected_idx[val_idx_local]

    train_subset = Subset(ds_train_tf, train_idx.tolist())
    val_subset   = Subset(ds_eval_tf,  val_idx.tolist())

    # Batches definition (last batch can be smaller than batch_size)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # Best-checkpoint tracking
    best_val_acc = -1.0
    best_path = out_dir / "bird_species_resnet18_FINAL_best.pth"

    # Warmup
    print("[FINAL TRAIN] Warmup: head only...")
    # Head only
    freeze_backbone(model)
    opt_head = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_head)

    for epoch in range(1, warmup_epochs + 1):
        print(f"\n[FINAL WARMUP EPOCH {epoch}/{warmup_epochs}]")
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, opt_head, device, "train")
        va_loss, va_acc = eval_one_epoch(model, val_loader, criterion, device, "val")
        dt = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        print(f"TrainLoss={tr_loss:.4f} TrainAcc={tr_acc*100:.2f}% | "
              f"ValLoss={va_loss:.4f} ValAcc={va_acc*100:.2f}% | Time={dt:.1f}s")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "best_val_acc": best_val_acc,
                "history": history,
            }, best_path)

    ckpt = torch.load(best_path, map_location=device) # Checkpoint
    model.load_state_dict(ckpt["model_state_dict"]) # ckpt["model_state_dict"] is the best path saved before (the best model so far)

    # Fine-tune
    print("\n[FINAL TRAIN] Fine-tune: full network...")
    unfreeze_backbone(model)
    opt_full = optim.Adam(model.parameters(), lr=lr_finetune)

    for epoch in range(1, finetune_epochs + 1):
        global_epoch = warmup_epochs + epoch
        print(f"\n[FINAL FINETUNE EPOCH {epoch}/{finetune_epochs}] (Global epoch {global_epoch})")
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, opt_full, device, "train")
        va_loss, va_acc = eval_one_epoch(model, val_loader, criterion, device, "val")
        dt = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        print(f"TrainLoss={tr_loss:.4f} TrainAcc={tr_acc*100:.2f}% | "
              f"ValLoss={va_loss:.4f} ValAcc={va_acc*100:.2f}% | Time={dt:.1f}s")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "best_val_acc": best_val_acc,
                "history": history,
            }, best_path)

    print(f"\n[FINAL TRAIN] Best ValAcc: {best_val_acc*100:.2f}%")
    plot_curves(history, out_dir, prefix="FINAL")
    return best_path


# ----------------------------
# Optional K-Fold on TRAIN (for thesis reporting)
# ----------------------------
def run_kfold_on_train(
    dataset_dir_train: str,
    out_dir: Path,
    seed: int = 42,
    k: int = 5,
    batch_size: int = 32,
    num_workers: int = 0,
    warmup_epochs: int = 2,
    finetune_epochs: int = 3,
    lr_head: float = 1e-3,
    lr_finetune: float = 1e-4,
    device: str = "cpu",
    balance_train: bool = False,
    balance_mode: str = "undersample",
) -> None:
    """
    Runs K-Fold CV ONLY on the TRAIN folder.
    Saves per-fold best checkpoints + curves + summary CSV.
    """
    train_tf, eval_tf = get_transforms()
    ds_train_tf = datasets.ImageFolder(dataset_dir_train, transform=train_tf)
    ds_eval_tf  = datasets.ImageFolder(dataset_dir_train, transform=eval_tf)

    class_names = ds_train_tf.classes
    targets = np.array(ds_train_tf.targets)
    num_classes = len(class_names)

    selected_idx, selected_targets = maybe_balance_dataset(
        targets=targets,
        class_names=class_names,
        seed=seed,
        enable_balance=balance_train,
        balance_mode=balance_mode,
        dataset_name="K-FOLD TRAIN",
    )

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    fold_rows = []

    for fold_i, (tr_idx_local, va_idx_local) in enumerate(
        skf.split(np.zeros(len(selected_targets)), selected_targets),
        start=1
    ):
        tr_idx = selected_idx[tr_idx_local]
        va_idx = selected_idx[va_idx_local]
        print("\n" + "=" * 60)
        print(f"[K-FOLD] Fold {fold_i}/{k} | Train={len(tr_idx)} Val={len(va_idx)}")

        tr_subset = Subset(ds_train_tf, tr_idx.tolist())
        va_subset = Subset(ds_eval_tf,  va_idx.tolist())

        tr_loader = DataLoader(tr_subset, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
        va_loader = DataLoader(va_subset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True)

        model = build_model(num_classes).to(device)
        criterion = nn.CrossEntropyLoss()

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_acc = -1.0
        best_path = out_dir / f"kfold_best_fold_{fold_i}.pth"

        # Warmup
        freeze_backbone(model)
        opt_head = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_head)
        for epoch in range(1, warmup_epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, tr_loader, criterion, opt_head, device, "train")
            va_loss, va_acc = eval_one_epoch(model, va_loader, criterion, device, "val")
            history["train_loss"].append(tr_loss); history["val_loss"].append(va_loss)
            history["train_acc"].append(tr_acc);   history["val_acc"].append(va_acc)
            if va_acc > best_val_acc:
                best_val_acc = va_acc
                torch.save({"model_state_dict": model.state_dict(),
                            "class_names": class_names,
                            "best_val_acc": best_val_acc,
                            "fold": fold_i}, best_path)


        # After warmup: reload BEST warmup weights before fine-tuning
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        # Finetune
        unfreeze_backbone(model)
        opt_full = optim.Adam(model.parameters(), lr=lr_finetune)
        for epoch in range(1, finetune_epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, tr_loader, criterion, opt_full, device, "train")
            va_loss, va_acc = eval_one_epoch(model, va_loader, criterion, device, "val")
            history["train_loss"].append(tr_loss); history["val_loss"].append(va_loss)
            history["train_acc"].append(tr_acc);   history["val_acc"].append(va_acc)
            if va_acc > best_val_acc:
                best_val_acc = va_acc
                torch.save({"model_state_dict": model.state_dict(),
                            "class_names": class_names,
                            "best_val_acc": best_val_acc,
                            "fold": fold_i}, best_path)

        plot_curves(history, out_dir, prefix=f"KFold_F{fold_i}")
        fold_rows.append({"fold": fold_i, "best_val_acc": best_val_acc, "best_model_path": str(best_path.resolve())})
        print(f"[K-FOLD] Fold {fold_i} best ValAcc: {best_val_acc*100:.2f}%")

    df = pd.DataFrame(fold_rows)
    df.to_csv(out_dir / "kfold_summary.csv", index=False)
    accs = df["best_val_acc"].to_numpy()
    mean_acc = accs.mean()
    std_acc = accs.std(ddof=1) if len(accs) > 1 else 0.0
    print("\n" + "=" * 60)
    print(f"[K-FOLD SUMMARY] K={k} | Mean±Std best ValAcc = {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"[K-FOLD SUMMARY] Saved: {str((out_dir / 'kfold_summary.csv').resolve())}")


# ----------------------------
# Test evaluation
# ----------------------------
@torch.no_grad()
def evaluate_on_test(
    checkpoint_path: Path,
    test_dir: str,
    out_dir: Path,
    batch_size: int = 32,
    num_workers: int = 0,
    device: str = "cpu",
    balance_test: bool = False,
    balance_mode: str = "undersample",
    seed: int = 42,
) -> None:
    """
    Loads the final checkpoint and evaluates on TEST once.
    Saves confusion matrix PNG + classification report TXT.
    """
    _, eval_tf = get_transforms()
    test_ds = datasets.ImageFolder(test_dir, transform=eval_tf)

    ckpt = torch.load(checkpoint_path, map_location=device)
    class_names = ckpt["class_names"]

    # Safety: ensure class order matches
    if test_ds.classes != class_names:
        raise ValueError(
            "Class mismatch between TRAIN and TEST.\n"
            f"Train classes: {class_names}\n"
            f"Test classes : {test_ds.classes}\n"
            "Fix: ensure both folders have the same class subfolders and naming."
        )

    num_classes = len(class_names)
    test_targets = np.array(test_ds.targets)
    selected_idx, _ = maybe_balance_dataset(
        targets=test_targets,
        class_names=class_names,
        seed=seed,
        enable_balance=balance_test,
        balance_mode=balance_mode,
        dataset_name="TEST",
    )
    test_subset = Subset(test_ds, selected_idx.tolist())
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    model = build_model(num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    y_true, y_pred = [], []

    for images, labels in tqdm(test_loader, desc="Test batch progress", leave=False):
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()

        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.tolist())

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

    print("\n" + "=" * 60)
    print(f"[TEST] Accuracy: {acc*100:.2f}%")
    print(report)

    # Save artifacts
    out_dir.mkdir(parents=True, exist_ok=True)
    cm_path = out_dir / "TEST_confusion_matrix.png"
    rep_path = out_dir / "TEST_classification_report.txt"

    plot_confusion_matrix(cm, class_names, cm_path)
    rep_path.write_text(f"Test Accuracy: {acc:.6f}\n\n{report}", encoding="utf-8")

    print(f"[TEST] Saved confusion matrix: {cm_path.resolve()}")
    print(f"[TEST] Saved classification report: {rep_path.resolve()}")


# ----------------------------
# MAIN
# ----------------------------
def main():
    # ----------------------------
    # Update these two paths
    # ----------------------------
    TRAIN_DIR = r"F:/01_Univalle/01_TG/dataset_bbox/train"
    TEST_DIR  = r"F:/01_Univalle/01_TG/dataset_bbox/test"

    OUT_DIR = Path("outputs_resnet18_thesis")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Switches
    # ----------------------------
    RUN_KFOLD = True      # set False if you don't want K-Fold reporting
    K = 5

    # ----------------------------
    # Hyperparameters (adjust later)
    # ----------------------------
    SEED = 42
    set_seed(SEED)

    batch_size = 32
    num_workers = 0
    warmup_epochs = 3
    finetune_epochs = 5
    lr_head = 1e-3
    lr_finetune = 1e-4
    val_ratio = 0.15  # internal val split for the FINAL model
    balance_train = True
    balance_test = True
    balance_mode = "undersample"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # ----------------------------
    # (Optional) K-Fold on TRAIN for thesis robustness
    # ----------------------------
    if RUN_KFOLD:
        kfold_dir = OUT_DIR / "kfold_train_only"
        kfold_dir.mkdir(parents=True, exist_ok=True)
        run_kfold_on_train(
            dataset_dir_train=TRAIN_DIR,
            out_dir=kfold_dir,
            seed=SEED,
            k=K,
            batch_size=batch_size,
            num_workers=num_workers,
            warmup_epochs=2,     # keep CV shorter
            finetune_epochs=3,   # keep CV shorter
            lr_head=lr_head,
            lr_finetune=lr_finetune,
            device=device,
            balance_train=balance_train,
            balance_mode=balance_mode,
        )

    # ----------------------------
    # Train ONE final model (deployable)
    # ----------------------------
    final_dir = OUT_DIR / "final_train"
    final_dir.mkdir(parents=True, exist_ok=True)

    best_final_ckpt = train_final_model_with_internal_val(
        dataset_dir_train=TRAIN_DIR,
        out_dir=final_dir,
        seed=SEED,
        val_ratio=val_ratio,
        batch_size=batch_size,
        num_workers=num_workers,
        warmup_epochs=warmup_epochs,
        finetune_epochs=finetune_epochs,
        lr_head=lr_head,
        lr_finetune=lr_finetune,
        device=device,
        balance_train=balance_train,
        balance_mode=balance_mode,
    )

    print(f"\n[INFO] Final deployable checkpoint: {best_final_ckpt.resolve()}")

    # ----------------------------
    # Test evaluation (one time)
    # ----------------------------
    test_dir = OUT_DIR / "test_eval"
    test_dir.mkdir(parents=True, exist_ok=True)
    evaluate_on_test(
        checkpoint_path=best_final_ckpt,
        test_dir=TEST_DIR,
        out_dir=test_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        balance_test=balance_test,
        balance_mode=balance_mode,
        seed=SEED,
    )


if __name__ == "__main__":
    main()
