"""Fixed image views shared by DL and SL, without preassigned validation splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, Subset

from shared_training_pool import DEFAULT_POOL_MANIFEST, PROJECT_ROOT, get_training_pool


ROOT = PROJECT_ROOT / "training_protocol" / "augmentation_seed_42"
MANIFEST = ROOT / "augmentation_manifest.csv"
PROTOCOL = "shared_fixed_flip_rotation_v1"
METADATA = ("augmentation_parent", "augmentation_id")
_VERIFIED_ASSETS = set()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_augmentation(*, seed=42, pool_manifest=DEFAULT_POOL_MANIFEST, out_dir=ROOT,
                       dataset_root=None):
    kwargs = {} if dataset_root is None else {"dataset_root": dataset_root}
    _, pool, info = get_training_pool(seed=seed, manifest_path=pool_manifest, **kwargs)
    source_root = Path(info["dataset_root"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in pool.sample_key:
        source = source_root / key
        source_digest = file_hash(source)
        with Image.open(source) as opened:
            image = opened.convert("RGB")
        for view in (1, 2):
            token = f"{PROTOCOL}|{seed}|{key}|{source_digest}|{view}"
            identity = hashlib.sha256(token.encode()).hexdigest()
            rng = np.random.default_rng(int(identity[:16], 16))
            angle = float(rng.uniform(-10, 10))
            flip = view == 1
            path = out / "images" / key.split("/")[0] / f"{identity}.png"
            transformed = ImageOps.mirror(image) if flip else image.copy()
            transformed = transformed.rotate(angle, resample=Image.Resampling.BILINEAR)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Regenerate deterministically, including files damaged since the previous run.
            transformed.save(path)
            rows.append({"augmentation_parent": key, "species": key.split("/")[0],
                         "orig_filename": path.name, "image_path": str(path.resolve()),
                         "view": view, "flip": flip, "angle": angle,
                         "source_sha256": source_digest, "image_sha256": file_hash(path)})
    table = pd.DataFrame(rows)
    augmentation_id = hashlib.sha256(
        table.drop(columns="image_path").to_json(orient="records").encode()).hexdigest()
    table["augmentation_id"] = augmentation_id
    table.to_csv(out / MANIFEST.name, index=False)
    (out / "augmentation_protocol.json").write_text(json.dumps({
        "protocol": PROTOCOL, "seed": seed, "training_pool_id": info["training_pool_id"],
        "dataset_root": str(source_root.resolve()),
        "augmentation_id": augmentation_id, "views_per_original": 2,
        "manifest_sha256": file_hash(out / MANIFEST.name),
        "split_assignment": "none; existing splitters assign originals at runtime",
        "transforms": "view 1: horizontal flip + rotation; view 2: rotation; angles [-10,10]",
        "source_images": len(pool), "augmented_images": len(table),
    }, indent=2), encoding="utf-8")
    return table


def load_augmentation(path=MANIFEST):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Shared augmentation missing. Run shared_augmentation.py first.")
    table = pd.read_csv(path)
    info = json.loads((path.parent / "augmentation_protocol.json").read_text(encoding="utf-8"))
    if (file_hash(path) != info["manifest_sha256"] or info["protocol"] != PROTOCOL or
            table.augmentation_id.nunique() != 1 or
            table.augmentation_id.iloc[0] != info["augmentation_id"] or
            table.duplicated(["augmentation_parent", "view"]).any() or
            not table.groupby("augmentation_parent").size().eq(2).all()):
        raise ValueError("Invalid shared augmentation manifest; rebuild it.")
    if info["augmentation_id"] not in _VERIFIED_ASSETS:
        for row in table.itertuples():
            image = Path(row.image_path)
            if not image.is_file() or file_hash(image) != row.image_sha256:
                raise ValueError(f"Missing/changed augmented pixels: {image}. Rebuild shared augmentation.")
        for row in table.drop_duplicates("augmentation_parent").itertuples():
            source = Path(info["dataset_root"]) / row.augmentation_parent
            if not source.is_file() or file_hash(source) != row.source_sha256:
                raise ValueError(f"Source pixels changed: {source}. Rebuild shared augmentation and descriptors.")
        _VERIFIED_ASSETS.add(info["augmentation_id"])
    return table, info


def validate_feature_augmentation(frame, pool_manifest):
    if pool_manifest is None or Path(pool_manifest).resolve() != DEFAULT_POOL_MANIFEST.resolve():
        return
    table, _ = load_augmentation()
    actual = frame.loc[frame.split.eq("train_augmented")]
    columns = ["augmentation_parent", "orig_filename", "augmentation_id"]
    if not set(columns).issubset(actual) or set(map(tuple, actual[columns].to_numpy())) != set(
            map(tuple, table[columns].to_numpy())) or len(actual) != len(table):
        raise ValueError("SL augmentation is missing/stale. Rerun sl_dataframe_main.py and feature selection.")


def expanded_training_indices(frame, original_indices):
    """Add only views of the supplied training originals, never validation relatives."""
    original_indices = np.asarray(original_indices, dtype=int)
    if "augmentation_parent" not in frame:
        return original_indices
    originals = frame.iloc[original_indices]
    keys = set(originals.species.astype(str) + "/" + originals.orig_filename.astype(str))
    augmented = np.flatnonzero(frame.split.eq("train_augmented") & frame.augmentation_parent.isin(keys))
    return np.concatenate([original_indices, augmented])


class TrainingOnlyAugmentationCV:
    """Stratify originals at runtime; fit their views and score original images only."""

    def __init__(self, frame, folds=5, seed=42):
        self.frame = frame.reset_index(drop=True)
        self.folds = folds
        self.seed = seed

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.folds

    def split(self, X, y, groups=None):
        if len(X) != len(self.frame):
            raise ValueError("Augmentation CV metadata and feature rows do not align.")
        originals = np.flatnonzero(self.frame.split.eq("train"))
        cv = StratifiedKFold(self.folds, shuffle=True, random_state=self.seed)
        for train, val in cv.split(np.zeros(len(originals)), np.asarray(y)[originals]):
            yield expanded_training_indices(self.frame, originals[train]), originals[val]


class SharedAugmentedDataset(Dataset):
    def __init__(self, dataset, indices, table):
        self.originals = Subset(dataset, list(map(int, indices)))
        keys = {f"{dataset.classes[dataset.samples[int(i)][1]]}/{Path(dataset.samples[int(i)][0]).name}"
                for i in indices}
        selected = table.loc[table.augmentation_parent.isin(keys)].sort_values(
            ["augmentation_parent", "orig_filename"])
        self.augmentation_id = str(table.augmentation_id.iloc[0])
        self.views = [(row.image_path, dataset.class_to_idx[row.species])
                      for row in selected.itertuples()]
        self.transform = dataset.transform

    def __len__(self):
        return len(self.originals) + len(self.views)

    def __getitem__(self, index):
        if index < len(self.originals):
            return self.originals[index]
        path, target = self.views[index - len(self.originals)]
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        return self.transform(image) if self.transform else image, target


def dl_training_subset(dataset, indices, pool_manifest, out_dir):
    # External/synthetic callers retain their explicitly supplied dataset contract.
    if pool_manifest is None or Path(pool_manifest).resolve() != DEFAULT_POOL_MANIFEST.resolve():
        return Subset(dataset, list(map(int, indices)))
    table, info = load_augmentation()
    _, _, pool = get_training_pool(dataset, manifest_path=pool_manifest,
                                   dataset_root=dataset.root, seed=info["seed"])
    if info["training_pool_id"] != pool["training_pool_id"]:
        raise ValueError("DL augmentation belongs to another training pool.")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "augmentation_protocol.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return SharedAugmentedDataset(dataset, indices, table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = build_augmentation(seed=args.seed)
    print(f"[AUGMENTATION] {len(result)} fixed views saved; no validation split assigned.")
