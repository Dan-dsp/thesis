"""Identity-based test cohort, anchored to the existing SL sampling order."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REFERENCE_CSV = ROOT / "dataset_features" / "shallow_learning_birds.csv"


def sl_test_indices(labels, seed=42):
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    if not len(classes):
        raise ValueError("Empty test cohort.")
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(labels == label),
                                              size=int(counts.min()), replace=False)
                                   for label in classes]))


def keyed_test_cohort(metadata, reference, seed=42):
    """Return caller indices in SL-reference order, independent of caller row order."""
    def keys(frame):
        if frame[["species", "orig_filename"]].isna().any().any():
            raise ValueError("Test image identity is missing.")
        values = frame.species.astype(str) + "/" + frame.orig_filename.astype(str)
        if values.duplicated().any():
            raise ValueError("Duplicate test image identity.")
        return values
    reference_keys, actual_keys = keys(reference), keys(metadata)
    if set(reference_keys) != set(actual_keys):
        raise ValueError("SL/DL test image identities do not match the reference test dataset.")
    selected = sl_test_indices(reference.species, seed)
    wanted = reference_keys.iloc[selected].tolist()
    lookup = {key: i for i, key in enumerate(actual_keys)}
    indices = np.array([lookup[key] for key in wanted], dtype=int)
    cohort = pd.DataFrame({"sample_key": wanted, "species": reference.iloc[selected].species.to_numpy()})
    return indices, cohort


def matched_test_indices(metadata, seed=42, out_dir=None, reference_csv=None):
    reference_csv = REFERENCE_CSV if reference_csv is None else reference_csv
    reference = pd.read_csv(reference_csv, usecols=["species", "orig_filename", "split"])
    reference = reference.loc[reference.split.eq("test")].reset_index(drop=True)
    indices, cohort = keyed_test_cohort(metadata.reset_index(drop=True), reference, seed)
    digest = hashlib.sha256("\n".join(cohort.sample_key).encode()).hexdigest()
    info = {"protocol": "sl_reference_order_test_v1", "seed": seed,
            "reference_csv": str(Path(reference_csv).resolve()), "source_images": len(reference),
            "selected_images": len(cohort), "test_pool_id": digest,
            "class_counts": {str(k): int(v) for k, v in cohort.species.value_counts().sort_index().items()}}
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        cohort.to_csv(out / "balanced_test_pool.csv", index=False)
        (out / "balanced_test_pool.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return indices


def imagefolder_metadata(dataset):
    return pd.DataFrame({"species": [dataset.classes[int(y)] for _, y in dataset.samples],
                         "orig_filename": [Path(p).name for p, _ in dataset.samples]})
