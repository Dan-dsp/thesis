"""
Compare the two ResNet-18 DL checkpoints and four SL models on their test sets.

The script checks that:
- all requested model files and test folders exist;
- DL checkpoint class names match the ImageFolder test classes;
- SL model class names and feature columns match the saved run metadata;
- each model can be loaded and has the expected prediction interface.

It prints macro metrics and classification reports, and also saves a summary CSV
plus one text report per model under comparison_results/.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torchvision import datasets, models, transforms
from torch import nn
from torch.utils.data import DataLoader, Subset


DL_TEST_DIR = Path(r"F:\01_Univalle\01_TG\dataset_bbox\test")
SL_TEST_DIR = Path(r"F:\01_Univalle\01_TG\dataset_features\test")
SL_RUN_DIR = Path(r"F:\01_Univalle\01_TG\sl_outputs\runs\run_20260619_005547")

DL_MODELS = {
    "resnet18_bbox_full_finetune": Path(
        r"F:\01_Univalle\01_TG\01_Python\outputs_resnet18_thesis_dl_bbox"
        r"\final_train\bird_species_resnet18_FINAL_best.pth"
    ),
    "resnet18_bbox_partial_finetune": Path(
        r"F:\01_Univalle\01_TG\01_Python\outputs_resnet18_thesis_partial_finetune"
        r"\final_train\bird_species_resnet18_FINAL_best.pth"
    ),
}

SL_MODELS = {
    "knn": SL_RUN_DIR / "models" / "knn_final.pkl",
    "random_forest": SL_RUN_DIR / "models" / "random_forest_final.pkl",
    "svm_rbf": SL_RUN_DIR / "models" / "svm_rbf_final.pkl",
    "xgboost": SL_RUN_DIR / "models" / "xgboost_final.pkl",
}


@dataclass
class EvalResult:
    family: str
    model: str
    samples: int
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    report_text: str
    confusion_matrix: np.ndarray

    def as_row(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "model": self.model,
            "samples": self.samples,
            "accuracy": self.accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
        }


def torch_load_checkpoint(path: Path, device: str) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise TypeError(f"{path} did not load as a checkpoint dictionary.")
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"{path} is missing 'model_state_dict'.")
    if "class_names" not in checkpoint:
        raise KeyError(f"{path} is missing 'class_names'.")

    return checkpoint


def assert_existing_path(path: Path, description: str, is_dir: bool = False) -> None:
    if is_dir:
        if not path.is_dir():
            raise FileNotFoundError(f"{description} directory not found: {path}")
    elif not path.is_file():
        raise FileNotFoundError(f"{description} file not found: {path}")


def build_resnet18(num_classes: int) -> torch.nn.Module:
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def get_dl_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def balanced_indices(labels: np.ndarray, seed: int) -> np.ndarray:
    classes, counts = np.unique(labels, return_counts=True)
    min_count = int(counts.min())
    rng = np.random.default_rng(seed)

    chosen_indices = []
    for cls in classes:
        cls_indices = np.flatnonzero(labels == cls)
        chosen_indices.extend(rng.choice(cls_indices, size=min_count, replace=False))

    return np.array(sorted(chosen_indices), dtype=np.int64)


def compute_metrics(
    family: str,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> EvalResult:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=class_names,
        average="macro",
        zero_division=0,
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=class_names,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=class_names)

    return EvalResult(
        family=family,
        model=model_name,
        samples=int(len(y_true)),
        accuracy=float(accuracy),
        precision_macro=float(precision),
        recall_macro=float(recall),
        f1_macro=float(f1),
        report_text=report,
        confusion_matrix=cm,
    )


def evaluate_dl_model(
    model_name: str,
    model_path: Path,
    test_dir: Path,
    device: str,
    batch_size: int,
    num_workers: int,
    balance_test: bool,
    seed: int,
) -> EvalResult:
    checkpoint = torch_load_checkpoint(model_path, device)
    class_names = list(checkpoint["class_names"])

    test_dataset = datasets.ImageFolder(str(test_dir), transform=get_dl_eval_transform())
    if test_dataset.classes != class_names:
        raise ValueError(
            f"DL class mismatch for {model_name}.\n"
            f"Checkpoint classes: {class_names}\n"
            f"Test folder classes: {test_dataset.classes}"
        )

    dataset_for_loader: Any = test_dataset
    if balance_test:
        selected = balanced_indices(np.asarray(test_dataset.targets), seed=seed)
        dataset_for_loader = Subset(test_dataset, selected.tolist())

    loader = DataLoader(
        dataset_for_loader,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )

    model = build_resnet18(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true_ids: list[int] = []
    y_pred_ids: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu().numpy()
            y_true_ids.extend(labels.numpy().tolist())
            y_pred_ids.extend(predictions.tolist())

    y_true = np.asarray([class_names[idx] for idx in y_true_ids])
    y_pred = np.asarray([class_names[idx] for idx in y_pred_ids])
    return compute_metrics("DL", model_name, y_true, y_pred, class_names)


def load_sl_metadata(run_dir: Path) -> tuple[list[str], list[str], Any]:
    metadata_dir = run_dir / "metadata"
    feature_columns_path = metadata_dir / "feature_columns.json"
    class_names_path = metadata_dir / "class_names.json"
    label_encoder_path = metadata_dir / "label_encoder.pkl"

    assert_existing_path(feature_columns_path, "SL feature columns")
    assert_existing_path(class_names_path, "SL class names")
    assert_existing_path(label_encoder_path, "SL label encoder")

    feature_columns = json.loads(feature_columns_path.read_text(encoding="utf-8"))
    class_names = json.loads(class_names_path.read_text(encoding="utf-8"))
    label_encoder = joblib.load(label_encoder_path)

    if list(label_encoder.classes_) != class_names:
        raise ValueError(
            "SL label encoder classes do not match metadata class_names.json.\n"
            f"Label encoder: {list(label_encoder.classes_)}\n"
            f"Metadata: {class_names}"
        )

    return feature_columns, class_names, label_encoder


def load_sl_test_data(
    test_dir: Path,
    feature_columns: list[str],
    class_names: list[str],
    balance_test: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    csv_paths = sorted(test_dir.glob("*/*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {test_dir}")

    frames = []
    parent_species_mismatches = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if df.empty:
            raise ValueError(f"Empty feature CSV: {csv_path}")
        expected_species = csv_path.parent.name
        if "species" in df.columns:
            mismatched = df["species"].astype(str) != expected_species
            if mismatched.any():
                parent_species_mismatches.append(str(csv_path))
        else:
            df["species"] = expected_species
        df["source_csv"] = str(csv_path)
        frames.append(df)

    if parent_species_mismatches:
        raise ValueError(
            "Some SL CSV files have a species column that does not match their "
            f"parent folder. First mismatches: {parent_species_mismatches[:5]}"
        )

    data = pd.concat(frames, ignore_index=True)
    folder_classes = sorted(path.name for path in test_dir.iterdir() if path.is_dir())
    if folder_classes != class_names:
        raise ValueError(
            "SL class mismatch between test folders and metadata.\n"
            f"Metadata classes: {class_names}\n"
            f"Test folder classes: {folder_classes}"
        )

    missing_columns = [column for column in feature_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(
            "SL test CSVs are missing feature columns required by the saved run. "
            f"First missing columns: {missing_columns[:10]}"
        )

    X = data[feature_columns].to_numpy(dtype=np.float32)
    y = data["species"].astype(str).to_numpy()

    if not np.isfinite(X).all():
        bad_rows, bad_cols = np.where(~np.isfinite(X))
        raise ValueError(
            "SL test features contain NaN or inf values. "
            f"First invalid row/column: {(int(bad_rows[0]), int(bad_cols[0]))}"
        )

    if balance_test:
        selected = balanced_indices(y, seed=seed)
        X = X[selected]
        y = y[selected]
        data = data.iloc[selected].reset_index(drop=True)

    return X, y, data


def decode_sl_predictions(raw_predictions: np.ndarray, label_encoder: Any) -> np.ndarray:
    predictions = np.asarray(raw_predictions)
    if np.issubdtype(predictions.dtype, np.number):
        return label_encoder.inverse_transform(predictions.astype(np.int64))
    return predictions.astype(str)


def evaluate_sl_model(
    model_name: str,
    model_path: Path,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    label_encoder: Any,
) -> EvalResult:
    model = joblib.load(model_path)
    if not hasattr(model, "predict"):
        raise TypeError(f"{model_path} loaded, but it has no predict() method.")

    n_features_in = getattr(model, "n_features_in_", None)
    if n_features_in is not None and int(n_features_in) != X_test.shape[1]:
        raise ValueError(
            f"Feature count mismatch for {model_name}: model expects "
            f"{n_features_in}, but test data has {X_test.shape[1]}."
        )

    raw_predictions = model.predict(X_test)
    y_pred = decode_sl_predictions(raw_predictions, label_encoder)
    return compute_metrics("SL", model_name, y_test, y_pred, class_names)


def print_result(result: EvalResult) -> None:
    print("\n" + "=" * 80)
    print(f"{result.family} | {result.model}")
    print("=" * 80)
    print(f"Samples:          {result.samples}")
    print(f"Accuracy:         {result.accuracy:.4f}")
    print(f"Precision macro:  {result.precision_macro:.4f}")
    print(f"Recall macro:     {result.recall_macro:.4f}")
    print(f"F1 macro:         {result.f1_macro:.4f}")
    print("\nClassification report:")
    print(result.report_text)


def save_outputs(results: list[EvalResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([result.as_row() for result in results])
    summary = summary.sort_values(["f1_macro", "accuracy"], ascending=False)
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)

    for result in results:
        safe_name = f"{result.family.lower()}_{result.model}"
        (output_dir / f"{safe_name}_classification_report.txt").write_text(
            result.report_text,
            encoding="utf-8",
        )
        pd.DataFrame(result.confusion_matrix).to_csv(
            output_dir / f"{safe_name}_confusion_matrix.csv",
            index=False,
        )

    print("\n" + "=" * 80)
    print("Summary, sorted by F1 macro then accuracy")
    print("=" * 80)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved metrics and reports to: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate and compare the saved DL and SL bird classifiers."
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--balance-test",
        action="store_true",
        help="Undersample each test set to the smallest class support before evaluation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("comparison_results"),
    )
    parser.add_argument("--skip-dl", action="store_true")
    parser.add_argument("--skip-sl", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    assert_existing_path(DL_TEST_DIR, "DL test data", is_dir=True)
    assert_existing_path(SL_TEST_DIR, "SL test data", is_dir=True)
    for name, path in DL_MODELS.items():
        assert_existing_path(path, f"DL model '{name}'")
    for name, path in SL_MODELS.items():
        assert_existing_path(path, f"SL model '{name}'")

    print(f"[INFO] Device: {args.device}")
    print(f"[INFO] DL test data: {DL_TEST_DIR}")
    print(f"[INFO] SL test data: {SL_TEST_DIR}")
    print(f"[INFO] Balance test: {args.balance_test}")

    results: list[EvalResult] = []

    if not args.skip_dl:
        for model_name, model_path in DL_MODELS.items():
            print(f"\n[INFO] Evaluating DL model: {model_name}")
            result = evaluate_dl_model(
                model_name=model_name,
                model_path=model_path,
                test_dir=DL_TEST_DIR,
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                balance_test=args.balance_test,
                seed=args.seed,
            )
            print_result(result)
            results.append(result)

    if not args.skip_sl:
        feature_columns, class_names, label_encoder = load_sl_metadata(SL_RUN_DIR)
        X_test, y_test, _ = load_sl_test_data(
            test_dir=SL_TEST_DIR,
            feature_columns=feature_columns,
            class_names=class_names,
            balance_test=args.balance_test,
            seed=args.seed,
        )
        print(f"\n[INFO] SL test matrix: {X_test.shape}")

        for model_name, model_path in SL_MODELS.items():
            print(f"\n[INFO] Evaluating SL model: {model_name}")
            result = evaluate_sl_model(
                model_name=model_name,
                model_path=model_path,
                X_test=X_test,
                y_test=y_test,
                class_names=class_names,
                label_encoder=label_encoder,
            )
            print_result(result)
            results.append(result)

    if not results:
        raise RuntimeError("No models were evaluated.")

    save_outputs(results, args.output_dir)


if __name__ == "__main__":
    main()
