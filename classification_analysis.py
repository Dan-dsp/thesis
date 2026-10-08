"""Prediction audits and validation-selected rejection for SL and DL."""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def partition_id(keys):
    return hashlib.sha256("\n".join(sorted(keys)).encode("utf-8")).hexdigest()


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def choose_rejection_policy(y_true, y_pred, confidence, target_accuracy=0.90):
    """Maximize validation coverage subject to an empirical accuracy target."""
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(y_true) == np.asarray(y_pred)
    if confidence.shape != correct.shape or not len(correct):
        raise ValueError("Expected nonempty aligned labels, predictions and confidence.")
    if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("Confidence must be finite and in [0, 1].")
    if not 0 < target_accuracy <= 1:
        raise ValueError("target_accuracy must be in (0, 1].")
    min_accepted = min(len(correct), max(10, int(np.ceil(len(correct) * 0.1))))
    policy = {
        "threshold": 1.000001, "status": "target_not_met_reject_all",
        "target_validation_accuracy": target_accuracy,
        "validation_samples": len(correct), "validation_accepted": 0,
        "validation_accuracy_accepted": None, "minimum_validation_accepted": min_accepted,
        "confidence_kind": "uncalibrated_probability_of_predicted_class",
        "selection_data": "internal_validation_only",
    }
    for threshold in np.unique(np.r_[0.0, confidence]):
        accepted = confidence >= threshold
        count = int(accepted.sum())
        if count >= min_accepted and float(correct[accepted].mean()) >= target_accuracy:
            policy.update(threshold=float(threshold), status="validation_target_met",
                          validation_accepted=count,
                          validation_accuracy_accepted=float(correct[accepted].mean()))
            break
    return policy


def sl_predictions(model, X):
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    # SVC's probability argmax can disagree with predict; score the actual label.
    positions = {label: i for i, label in enumerate(model.classes_)}
    confidence = probabilities[np.arange(len(predictions)),
                               [positions[label] for label in predictions]]
    return predictions, confidence


def save_prediction_audit(out_dir, family, model_name, metadata, y_true, y_pred,
                          confidence, policy):
    """Keep raw mistakes and rejection outcomes, including every test image."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = metadata.reset_index(drop=True).copy()
    if len(rows) != len(y_true):
        raise ValueError("Prediction metadata must align with labels.")
    rows["family"] = family
    rows["model"] = model_name
    rows["true_label"] = y_true
    rows["predicted_label"] = y_pred
    rows["confidence"] = confidence
    rows["correct"] = np.asarray(y_true) == np.asarray(y_pred)
    rows["rejected"] = np.asarray(confidence) < policy["threshold"]
    rows["decision"] = np.where(rows["rejected"], "REJECT", rows["predicted_label"])
    if "sample_key" not in rows:
        raise ValueError("Stable sample_key required for image-level error analysis.")
    rows.to_csv(out_dir / f"{model_name}_test_predictions.csv", index=False)
    rows.loc[~rows["correct"]].to_csv(out_dir / f"{model_name}_test_errors.csv", index=False)
    accepted = rows.loc[~rows["rejected"]]
    save_json(out_dir / f"{model_name}_rejection.json", {
        "policy": policy, "samples": len(rows), "accepted": len(accepted),
        "coverage": len(accepted) / len(rows) if len(rows) else 0.0,
        "accuracy_without_rejection": float(rows["correct"].mean()) if len(rows) else None,
        "accuracy_accepted": float(accepted["correct"].mean()) if len(accepted) else None,
        "correct_and_accepted_fraction": float((rows["correct"] & ~rows["rejected"]).mean())
        if len(rows) else None,
    })
    return rows


def describe_sl_model(model, feature_names):
    """SL complexity has no single parameter count equivalent to a neural network."""
    clf = model.named_steps["clf"]
    result = {"family": "SL", "architecture": type(clf).__name__,
              "n_features": len(feature_names), "feature_columns": list(feature_names),
              "hyperparameters": model.get_params(deep=True),
              "parameter_count": None, "parameter_count_note": "Use model-specific complexity."}
    if hasattr(clf, "tree_"):
        result["tree_nodes"] = int(clf.tree_.node_count)
    elif hasattr(clf, "estimators_"):
        result["trees"] = len(clf.estimators_)
        result["tree_nodes"] = sum(int(t.tree_.node_count) for t in clf.estimators_)
        result["training_algorithm"] = "bootstrap decision trees with randomized feature splits"
    elif hasattr(clf, "support_vectors_"):
        result["support_vectors"] = len(clf.support_vectors_)
        result["training_algorithm"] = "libsvm margin optimization with RBF kernel"
    elif hasattr(clf, "get_booster"):
        result["boosted_rounds"] = clf.get_booster().num_boosted_rounds()
        result["trees"] = len(clf.get_booster().get_dump())
        result["training_algorithm"] = "gradient boosting with histogram tree construction"
    elif hasattr(clf, "_fit_X"):
        result["stored_training_samples"] = len(clf._fit_X)
        result["stored_feature_values"] = int(clf._fit_X.size)
        result["training_algorithm"] = "instance storage and distance-weighted neighbor voting"
    return result


def dl_predictions(model, loader, device):
    import torch
    labels, predictions, confidence = [], [], []
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            probabilities = torch.softmax(model(images.to(device)), dim=1)
            scores, predicted = probabilities.max(dim=1)
            labels.extend(targets.cpu().tolist())
            predictions.extend(predicted.cpu().tolist())
            confidence.extend(scores.cpu().tolist())
    return np.asarray(labels), np.asarray(predictions), np.asarray(confidence)


def finish_dl_training(model, checkpoint_path, val_loader, device, settings):
    import torch
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    truth, predictions, confidence = dl_predictions(model, val_loader, device)
    policy = choose_rejection_policy(truth, predictions, confidence)
    checkpoint["rejection_policy"] = policy
    checkpoint["training_analysis"] = {
        "family": "DL", "architecture": "ResNet18",
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters_final_stage": sum(p.numel() for p in model.parameters()
                                                if p.requires_grad),
        "training_algorithm": "Adam, cross-entropy, head warmup then fine-tuning",
        "hyperparameters": settings,
    }
    torch.save(checkpoint, checkpoint_path)
    analysis = dict(checkpoint["training_analysis"],
                    checkpoint_file_bytes=Path(checkpoint_path).stat().st_size)
    save_json(Path(checkpoint_path).parent / "model_analysis.json", analysis)
    save_json(Path(checkpoint_path).parent / "rejection_policy.json", policy)


def export_dl_test_audit(model, loader, dataset, indices, device, checkpoint, out_dir,
                         model_name="resnet18"):
    truth, predictions, confidence = dl_predictions(model, loader, device)
    names = checkpoint["class_names"]
    metadata = pd.DataFrame([
        {"sample_key": f"{names[dataset.targets[i]]}/{Path(dataset.samples[i][0]).name}",
         "image_path": dataset.samples[i][0]} for i in indices
    ])
    policy = checkpoint.get("rejection_policy")
    if policy is None:
        policy = {"threshold": 0.0, "status": "unavailable_legacy_checkpoint_no_rejection"}
    return save_prediction_audit(out_dir, "DL", model_name, metadata,
                                 [names[i] for i in truth], [names[i] for i in predictions],
                                 confidence, policy)


def summarize_errors(paths, out_dir, annotations=None):
    """Descriptive overlap only, with bird identity supplied explicitly by users."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = []
    for path in paths:
        table = pd.read_csv(path)
        required = {"sample_key", "family", "model", "true_label", "predicted_label", "correct"}
        if not required.issubset(table):
            raise ValueError(f"Missing prediction columns in {path}")
        if table.duplicated(["family", "model", "sample_key"]).any():
            raise ValueError(f"Duplicate image predictions in {path}")
        table["source"] = str(Path(path).resolve())
        tables.append(table)
    joined = pd.concat(tables, ignore_index=True)
    annotation = pd.DataFrame({"sample_key": sorted(joined.sample_key.unique())})
    annotation["bird_id"] = ""
    annotation["conditions"] = ""
    annotation["notes"] = ""
    annotation.to_csv(out_dir / "annotation_template.csv", index=False)
    if annotations:
        annotated = pd.read_csv(annotations)
        if annotated.sample_key.duplicated().any():
            raise ValueError("Annotation sample_key must be unique.")
        joined = joined.merge(annotated, on="sample_key", how="left", validate="many_to_one")
    joined.to_csv(out_dir / "all_predictions.csv", index=False)
    groups = joined.groupby("sample_key")
    overview = groups.agg(evaluations=("correct", "size"), correct_count=("correct", "sum"))
    overview["error_count"] = overview.evaluations - overview.correct_count
    overview.to_csv(out_dir / "image_error_frequency.csv")
    overlap = []
    sources = list(joined.source.unique())
    for left_idx, left in enumerate(sources):
        left_rows = joined.loc[joined.source.eq(left)].set_index("sample_key")
        for right in sources[left_idx + 1:]:
            right_rows = joined.loc[joined.source.eq(right)].set_index("sample_key")
            common = left_rows.index.intersection(right_rows.index)
            both_wrong = (~left_rows.loc[common, "correct"] &
                          ~right_rows.loc[common, "correct"])
            overlap.append({"left_source": left, "right_source": right,
                            "common_images": len(common),
                            "same_image_errors": int(both_wrong.sum())})
    pd.DataFrame(overlap).to_csv(out_dir / "descriptive_error_overlap.csv", index=False)
    errors = joined.loc[~joined["correct"]]
    errors.to_csv(out_dir / "error_review.csv", index=False)
    if "bird_id" in joined:
        identified = joined.dropna(subset=["bird_id"])
        identified = identified.loc[identified.bird_id.astype(str).str.strip().ne("")]
        identified.groupby("bird_id").agg(evaluations=("correct", "size"),
                                          accuracy=("correct", "mean")).to_csv(
                                              out_dir / "bird_error_summary.csv")
    if "conditions" in joined:
        tagged = joined.dropna(subset=["conditions"]).copy()
        tagged["condition"] = tagged.conditions.astype(str).str.split(";")
        tagged = tagged.explode("condition")
        tagged["condition"] = tagged.condition.str.strip()
        tagged = tagged.loc[tagged.condition.ne("")]
        tagged.groupby(["family", "model", "condition"]).agg(
            samples=("correct", "size"), accuracy=("correct", "mean")).to_csv(
                out_dir / "condition_summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect image and bird errors descriptively.")
    parser.add_argument("predictions", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    args = parser.parse_args()
    summarize_errors(args.predictions, args.out_dir, args.annotations)
