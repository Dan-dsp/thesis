"""
Evaluate ResNet-18 checkpoints and SL models in independent family reports.

The script checks that:
- all requested model files and test folders exist;
- DL checkpoint class names match the ImageFolder test classes;
- SL model class names and feature columns match the saved run metadata;
- each model can be loaded and has the expected prediction interface.

It saves macro metrics and classification reports separately for SL and DL.
Cross-family performance statements are descriptive, without a pooled ranking.
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
from result_plots import plot_matrix, plot_model_metrics
from image_error_analysis import save_image_predictions, summarize_image_errors
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torchvision import datasets, models, transforms
from torch import nn
from torch.utils.data import DataLoader


DL_TEST_DIR = Path(r"F:\01_Univalle\01_TG\dataset_bbox\test")
SL_TEST_DIR = Path(r"F:\01_Univalle\01_TG\dataset_features\test")
SL_BATCH_ROOT = Path(r"F:\01_Univalle\01_TG\sl_outputs_batch")

DL_MODELS = {
    "resnet18_bbox_full_finetune": Path(
        r"F:\01_Univalle\01_TG\01_Python\outputs_resnet18_thesis"
        r"\final_train\bird_species_resnet18_FINAL_best.pth"
    ),
    "resnet18_bbox_partial_finetune": Path(
        r"F:\01_Univalle\01_TG\01_Python\outputs_resnet18_thesis_partial_finetune"
        r"\final_train\bird_species_resnet18_FINAL_best.pth"
    ),
}

SL_MODEL_NAMES = ("knn", "random_forest", "svm_rbf", "xgboost")


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
    class_names: list[str] | None = None

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
        class_names=class_names,
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
    audit_dir: Path | None = None,
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

    # Predict every image for error inspection; balance only the metric table.
    loader = DataLoader(
        test_dataset,
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
    if audit_dir is not None:
        metadata = pd.DataFrame([
            {"sample_key": f"{class_names[target]}/{Path(path).name}", "image_path": path}
            for path, target in test_dataset.samples
        ])
        save_image_predictions(audit_dir / f"{model_name}_test_predictions.csv",
                               "DL", model_name, metadata, y_true, y_pred)
    selected = (balanced_indices(np.asarray(test_dataset.targets), seed=seed) if balance_test
                else np.arange(len(y_true)))
    return compute_metrics("DL", model_name, y_true[selected], y_pred[selected], class_names)


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
    metadata: pd.DataFrame | None = None,
    audit_dir: Path | None = None,
    evaluation_indices: np.ndarray | None = None,
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
    if audit_dir is not None:
        if metadata is None or "orig_filename" not in metadata:
            raise ValueError("SL error analysis needs original image filenames in the test CSVs.")
        images = metadata.copy()
        images["sample_key"] = images.species.astype(str) + "/" + images.orig_filename.astype(str)
        images["image_path"] = [str(DL_TEST_DIR / str(species) / str(filename))
                                for species, filename in zip(images.species, images.orig_filename)]
        save_image_predictions(audit_dir / f"{model_name}_test_predictions.csv",
                               "SL", model_name, images[["sample_key", "image_path"]], y_test, y_pred)
    selected = evaluation_indices if evaluation_indices is not None else np.arange(len(y_test))
    return compute_metrics("SL", model_name, y_test[selected], y_pred[selected], class_names)


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
    # Families have different training protocols: report each independently.
    for family, table in summary.groupby("family"):
        family_dir = output_dir / family.lower()
        family_dir.mkdir(parents=True, exist_ok=True)
        table.sort_values("model").to_csv(family_dir / "summary_metrics.csv", index=False)
        plot_model_metrics(table.sort_values("model"),
                           {"accuracy": "Accuracy", "f1_macro": "Macro-F1",
                            "precision_macro": "Macro precision", "recall_macro": "Macro recall"},
                           family_dir / "summary_metrics.png", f"{family}: held-out test metrics")

    for result in results:
        safe_name = result.model
        family_dir = output_dir / result.family.lower()
        (family_dir / f"{safe_name}_classification_report.txt").write_text(
            result.report_text,
            encoding="utf-8",
        )
        pd.DataFrame(result.confusion_matrix).to_csv(
            family_dir / f"{safe_name}_confusion_matrix.csv",
            index=False,
        )
        labels = (result.class_names if result.class_names is not None else
                  [str(i) for i in range(len(result.confusion_matrix))])
        plot_matrix(result.confusion_matrix, labels,
                    family_dir / f"{safe_name}_confusion_matrix.png", f"{result.family}: {safe_name}",
                    xlabel="Predicted class", ylabel="True class")

    print("\n" + "=" * 80)
    print("Independent family reports; cross-family metrics are descriptive only")
    print("=" * 80)
    for family, table in summary.groupby("family"):
        print(f"\n{family} report")
        print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved metrics and reports to: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved classifiers into independent SL and DL reports."
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
        default=Path("family_evaluation_results"),
    )
    parser.add_argument("--skip-dl", action="store_true")
    parser.add_argument("--skip-sl", action="store_true")
    parser.add_argument("--sl-run-dir", type=Path, help="A reduced-feature SL run.")
    parser.add_argument("--dl-full-model", type=Path,
                        default=DL_MODELS["resnet18_bbox_full_finetune"],
                        help="Full-fine-tuning checkpoint (override for historical output folders).")
    parser.add_argument("--dl-partial-model", type=Path,
                        default=DL_MODELS["resnet18_bbox_partial_finetune"],
                        help="Partial-fine-tuning checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dl_models = {"resnet18_bbox_full_finetune": args.dl_full_model,
                 "resnet18_bbox_partial_finetune": args.dl_partial_model}
    sl_models = {}
    sl_run_dir = args.sl_run_dir
    if not args.skip_sl:
        if sl_run_dir is None:
            batch_dir = Path((SL_BATCH_ROOT / "latest_batch_run.txt").read_text().strip())
            selected = json.loads((batch_dir / "selected_model.json").read_text())
            sl_run_dir = Path(selected["model_path"]).parent.parent
        sl_models = {name: sl_run_dir / "models" / f"{name}_final.pkl"
                     for name in SL_MODEL_NAMES}
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    if not args.skip_dl:
        assert_existing_path(DL_TEST_DIR, "DL test data", is_dir=True)
        for name, path in dl_models.items():
            assert_existing_path(path, f"DL model '{name}'")
    if not args.skip_sl:
        assert_existing_path(SL_TEST_DIR, "SL test data", is_dir=True)
        for name, path in sl_models.items():
            assert_existing_path(path, f"SL model '{name}'")

    print(f"[INFO] Device: {args.device}")
    print(f"[INFO] DL test data: {DL_TEST_DIR}")
    print(f"[INFO] SL test data: {SL_TEST_DIR}")
    print(f"[INFO] Balance test: {args.balance_test}")

    results: list[EvalResult] = []
    prediction_paths: list[Path] = []

    if not args.skip_dl:
        for model_name, model_path in dl_models.items():
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
                audit_dir=args.output_dir / "dl" / "predictions",
            )
            print_result(result)
            results.append(result)
            prediction_paths.append(args.output_dir / "dl" / "predictions" /
                                    f"{model_name}_test_predictions.csv")

    if not args.skip_sl:
        feature_columns, class_names, label_encoder = load_sl_metadata(sl_run_dir)
        X_test, y_test, test_metadata = load_sl_test_data(
            test_dir=SL_TEST_DIR,
            feature_columns=feature_columns,
            class_names=class_names,
            balance_test=False,
            seed=args.seed,
        )
        print(f"\n[INFO] SL test matrix: {X_test.shape}")
        selected = (balanced_indices(y_test, seed=args.seed) if args.balance_test
                    else np.arange(len(y_test)))

        for model_name, model_path in sl_models.items():
            print(f"\n[INFO] Evaluating SL model: {model_name}")
            result = evaluate_sl_model(
                model_name=model_name,
                model_path=model_path,
                X_test=X_test,
                y_test=y_test,
                class_names=class_names,
                label_encoder=label_encoder,
                metadata=test_metadata,
                audit_dir=args.output_dir / "sl" / "predictions",
                evaluation_indices=selected,
            )
            print_result(result)
            results.append(result)
            prediction_paths.append(args.output_dir / "sl" / "predictions" /
                                    f"{model_name}_test_predictions.csv")

    if not results:
        raise RuntimeError("No models were evaluated.")

    save_outputs(results, args.output_dir)
    review_dir = args.output_dir / "image_error_analysis"
    overview = summarize_image_errors(prediction_paths, review_dir, image_root=DL_TEST_DIR)
    print(f"\n[IMAGE ERRORS] {len(overview)} unique test images inspected without rejection.")
    print(f"[IMAGE ERRORS] Open: {(review_dir / 'error_gallery.html').resolve()}")


if __name__ == "__main__":
    main()
