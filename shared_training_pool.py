"""One reproducible, DL-ordered undersampled image pool shared by SL and DL."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_ROOT = PROJECT_ROOT / "dataset_bbox" / "train"
DEFAULT_POOL_MANIFEST = PROJECT_ROOT / "training_protocol" / "balanced_training_pool_seed_42.csv"
POOL_PROTOCOL = "dl_ordered_undersample_v1"
SELECTION_PROTOCOL = "train_only_v4_shared_augmentation"


def balanced_indices(targets, seed, mode="undersample"):
    """Preserve the original DL RNG calls, per-class selection, and final shuffle."""
    targets = np.asarray(targets)
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(targets, return_counts=True)
    if mode not in {"undersample", "oversample"}:
        raise ValueError(f"Unsupported balance mode: {mode}")
    if not len(classes):
        raise ValueError("No training images to balance.")
    count = counts.min() if mode == "undersample" else counts.max()
    chunks = []
    for label in classes:
        positions = np.where(targets == label)[0]
        replace = mode == "oversample" and len(positions) < count
        chunks.append(rng.choice(positions, size=count, replace=replace))
    selected = np.concatenate(chunks)
    rng.shuffle(selected)
    return selected.astype(int)


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()


def get_training_pool(dataset=None, *, seed=42, manifest_path=DEFAULT_POOL_MANIFEST,
                      dataset_root=TRAIN_ROOT):
    """Create/reuse a manifest using ImageFolder order, never CSV row order."""
    if dataset is None:
        from torchvision.datasets import ImageFolder

        dataset = ImageFolder(str(dataset_root))
    targets = np.asarray(dataset.targets)
    names = list(dataset.classes)
    keys = [f"{names[int(target)]}/{Path(path).name}" for path, target in dataset.samples]
    if len(keys) != len(set(keys)):
        raise ValueError("Training images need unique species/filename keys.")
    selected = balanced_indices(targets, seed)
    rows = pd.DataFrame({"pool_order": np.arange(len(selected)), "source_index": selected,
                         "sample_key": [keys[i] for i in selected],
                         "species": [names[int(targets[i])] for i in selected]})
    info = {"protocol": POOL_PROTOCOL, "seed": int(seed),
            "dataset_root": str(Path(dataset_root).resolve()),
            "source_order_id": _digest(keys), "source_images": len(keys),
            "selected_images": len(rows),
            "class_counts": {str(k): int(v) for k, v in rows.species.value_counts().sort_index().items()},
            "training_pool_id": _digest({"protocol": POOL_PROTOCOL, "seed": seed,
                                         "sample_keys": rows.sample_key.tolist()})}
    path = Path(manifest_path)
    metadata_path = path.with_suffix(".json")
    info["manifest_path"] = str(path.resolve())
    if path.exists() or metadata_path.exists():
        if not path.is_file() or not metadata_path.is_file():
            raise ValueError(f"Incomplete shared training manifest: {path}. Use a new manifest path.")
        stored = pd.read_csv(path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        fields = ["protocol", "seed", "dataset_root", "source_order_id", "training_pool_id"]
        if any(metadata.get(field) != info[field] for field in fields):
            raise ValueError(f"Shared training manifest does not match DL image order/seed: {path}. "
                             "Do not reuse rankings or models from another training pool.")
        try:
            pd.testing.assert_frame_equal(stored[rows.columns], rows, check_dtype=False)
        except (KeyError, AssertionError) as exc:
            raise ValueError(f"Shared training image list was changed or reordered: {path}") from exc
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(path, index=False)
        metadata_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return selected, rows, info


def select_training_rows(df, *, seed=42, manifest_path=DEFAULT_POOL_MANIFEST,
                         dataset_root=TRAIN_ROOT, dataset=None):
    """Filter only train rows by image identity; preserve test/other rows as supplied."""
    if manifest_path is None:
        return df.copy(), None
    required = {"species", "orig_filename", "split"}
    if not required.issubset(df):
        raise ValueError("Shared SL training pool requires species, orig_filename and split columns.")
    _, pool, info = get_training_pool(dataset, seed=seed, manifest_path=manifest_path,
                                      dataset_root=dataset_root)
    train = df.loc[df["split"].eq("train")].copy()
    train.index = train.species.astype(str) + "/" + train.orig_filename.astype(str)
    if train.index.duplicated().any():
        raise ValueError("Duplicate image keys in SL training rows.")
    missing = set(pool.sample_key) - set(train.index)
    if missing:
        raise ValueError(f"SL descriptors missing {len(missing)} selected DL training images: "
                         f"{sorted(missing)[:5]}")
    # Explicit key lookup also establishes exactly the same shuffled order as DL.
    balanced = train.loc[pool.sample_key].reset_index(drop=True)
    augmented = df.loc[df["split"].eq("train_augmented")].copy()
    if len(augmented):
        if "augmentation_parent" not in augmented:
            raise ValueError("Augmented descriptors require their original image key.")
        augmented = augmented.loc[augmented.augmentation_parent.isin(pool.sample_key)]
        augmented = augmented.sort_values(["augmentation_parent", "orig_filename"])
        if augmented.duplicated(["species", "orig_filename"]).any():
            raise ValueError("Duplicate augmented image keys in SL training rows.")
    other = df.loc[~df["split"].isin(["train", "train_augmented"])]
    return pd.concat([balanced, augmented, other], ignore_index=True), info


def save_pool_record(info, out_dir):
    if info is None:
        return
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pool = pd.read_csv(info["manifest_path"])
    pool.to_csv(out / "training_pool.csv", index=False)
    (out / "training_pool.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def select_dl_training_pool(dataset, *, seed, enabled, mode, manifest_path, out_dir,
                             dataset_root):
    targets = np.asarray(dataset.targets)
    info = None
    if enabled and mode == "undersample":
        selected, _, info = get_training_pool(dataset, seed=seed, manifest_path=manifest_path,
                                              dataset_root=dataset_root)
        save_pool_record(info, out_dir)
    elif enabled:
        selected = balanced_indices(targets, seed, mode)
    else:
        selected = np.arange(len(targets))
    print(f"[TRAINING POOL] {len(selected)} images; "
          f"shared ID: {info['training_pool_id'] if info else 'not using shared undersampling'}")
    return selected, targets[selected], info


def save_dl_split_membership(dataset, train_indices, val_indices, path):
    rows = []
    for role, indices in [("train_internal", train_indices), ("val_internal", val_indices)]:
        for i in indices:
            image_path, target = dataset.samples[int(i)]
            rows.append({"sample_key": f"{dataset.classes[int(target)]}/{Path(image_path).name}",
                         "pipeline_split": role})
    pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_POOL_MANIFEST)
    args = parser.parse_args()
    _, _, info = get_training_pool(seed=args.seed, manifest_path=args.manifest)
    print(json.dumps(info, indent=2))
