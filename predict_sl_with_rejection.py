"""Predict a feature CSV using the SL model selected by validation."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from classification_analysis import sl_predictions


def predict_selected(selection_path, csv_path, output_path):
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    model_path = Path(selection["model_path"])
    run_dir = model_path.parent.parent
    feature_columns = json.loads((run_dir / "metadata" / "feature_columns.json").read_text())
    encoder = joblib.load(run_dir / "metadata" / "label_encoder.pkl")
    model = joblib.load(model_path)
    data = pd.read_csv(csv_path)
    X = data[feature_columns].to_numpy(dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError("Features must be finite.")
    raw, confidence = sl_predictions(model, X)
    data["predicted_label"] = encoder.inverse_transform(raw)
    data["confidence"] = confidence
    threshold = selection["rejection_policy"]["threshold"]
    data["rejected"] = confidence < threshold
    data["decision"] = np.where(data.rejected, "REJECT", data.predicted_label)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection_json", type=Path)
    parser.add_argument("feature_csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predict_selected(args.selection_json, args.feature_csv, args.output)
