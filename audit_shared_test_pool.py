"""Prove SL/DL balanced-test identity equality without loading or training models."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from torchvision.datasets import ImageFolder

from shared_training_pool import balanced_indices
from shared_test_pool import ROOT, REFERENCE_CSV, imagefolder_metadata, matched_test_indices, sl_test_indices


def main():
    out = ROOT / "training_protocol" / "test_cohort_audit"
    reference = pd.read_csv(REFERENCE_CSV, usecols=["species", "orig_filename", "split"])
    reference = reference.loc[reference.split.eq("test")].reset_index(drop=True)
    dataset = ImageFolder(str(ROOT / "dataset_bbox" / "test"))
    images = imagefolder_metadata(dataset)
    sl_keys = reference.species + "/" + reference.orig_filename
    dl_keys = images.species + "/" + images.orig_filename
    original_sl = set(sl_keys.iloc[sl_test_indices(reference.species)])
    original_dl = set(dl_keys.iloc[balanced_indices(np.asarray(dataset.targets), 42)])
    sl = matched_test_indices(reference, out_dir=out / "sl")
    dl = matched_test_indices(images, out_dir=out / "dl")
    assert sl_keys.iloc[sl].tolist() == dl_keys.iloc[dl].tolist()
    exports = []
    manifest = ROOT / "sl_results" / "training_ready_datasets" / "training_ready_dataset_manifest.csv"
    if manifest.exists():
        for csv_path in pd.read_csv(manifest).csv_path:
            frame = pd.read_csv(csv_path, usecols=["species", "orig_filename", "split"])
            frame = frame.loc[frame.split.eq("test")].reset_index(drop=True)
            indices = matched_test_indices(frame)
            keys = frame.species + "/" + frame.orig_filename
            legacy_keys = set(keys.iloc[sl_test_indices(frame.species)])
            exports.append({"csv_path": str(csv_path),
                            "legacy_selected_ids_equal": legacy_keys == original_sl,
                            "matched_ordered_ids_equal": keys.iloc[indices].tolist() == sl_keys.iloc[sl].tolist()})
    report = {"seed": 42, "source_images": len(images),
              "legacy_sl_count": len(original_sl), "legacy_dl_count": len(original_dl),
              "legacy_intersection": len(original_sl & original_dl),
              "legacy_sl_only": sorted(original_sl - original_dl),
              "legacy_dl_only": sorted(original_dl - original_sl),
              "matched_image_count": len(sl), "matched_ordered_ids_equal": True,
              "exported_datasets_checked": len(exports), "exported_datasets": exports}
    (out / "identity_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Evidence: {out}")


if __name__ == "__main__":
    main()
