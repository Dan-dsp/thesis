from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from sl_methods import extract_all_features_torch, get_all_feature_names


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_dataset_images(dataset_root: str | Path) -> Iterable[Tuple[str, str, Path]]:
    """
    Iterate over a dataset organized like:

        dataset_root/
            train/
                species_a/
                    image_1.jpg
                species_b/
            test/
                species_a/
                species_b/

    The first folder level is treated as the split name (`train`, `test`, etc.).
    The second folder level is treated as the class/species name.
    """
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    split_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not split_dirs:
        raise ValueError(f"No split folders found in: {root}")

    for split_dir in split_dirs:
        split_name = split_dir.name
        species_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])

        for species_dir in species_dirs:
            species_name = species_dir.name
            for image_path in sorted(species_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    yield split_name, species_name, image_path


def extract_feature_row(
    image_path: str | Path,
    split: str,
    species: str,
    resize_to: Tuple[int, int] = (224, 224),
    sample_id: int | None = None,
    sift_centers=None,
    orb_centers=None,
    include_legacy_f_columns: bool = False,
) -> dict:
    """
    Extract one feature row from one image using the methods defined in sl_methods.py.
    """
    image_path = Path(image_path)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(resize_to, antialias=True),
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image)

    features_tensor = extract_all_features_torch(
        image_tensor,
        sift_centers=sift_centers,
        orb_centers=orb_centers,
    )
    features = features_tensor.tolist()
    feature_names = get_all_feature_names(
        sift_centers=sift_centers,
        orb_centers=orb_centers,
    )

    if len(features) != len(feature_names):
        raise ValueError(
            f"Feature count mismatch for {image_path}. "
            f"Got {len(features)} values but {len(feature_names)} names."
        )

    row = {
        "sample_id": sample_id,
        "sample_name": image_path.stem,
        "orig_filename": image_path.name,
        "species": species,
        "split": split,
    }
    for name, value in zip(feature_names, features):
        row[name] = float(value)
    if include_legacy_f_columns:
        for i, value in enumerate(features):
            row[f"f{i}"] = float(value)

    return row


def build_feature_dataframe(
    dataset_root: str | Path,
    resize_to: Tuple[int, int] = (224, 224),
    sift_centers=None,
    orb_centers=None,
    include_legacy_f_columns: bool = False,
) -> pd.DataFrame:
    """
    Build one flat DataFrame with the feature representation of all images.
    """
    rows: List[dict] = []
    entries = list(iter_dataset_images(dataset_root))

    for sample_id, (split, species, image_path) in enumerate(
        tqdm(entries, desc="Extracting image features", unit="image")
    ):
        row = extract_feature_row(
            image_path=image_path,
            split=split,
            species=species,
            resize_to=resize_to,
            sample_id=sample_id,
            sift_centers=sift_centers,
            orb_centers=orb_centers,
            include_legacy_f_columns=include_legacy_f_columns,
        )
        rows.append(row)

    return pd.DataFrame(rows)


def export_feature_dataset_with_structure(
    dataset_root: str | Path,
    output_root: str | Path,
    resize_to: Tuple[int, int] = (224, 224),
    sift_centers=None,
    orb_centers=None,
    output_suffix: str = ".csv",
    save_manifest: bool = True,
    include_legacy_f_columns: bool = False,
) -> pd.DataFrame:
    """
    Extract features from every image and save them mirroring the original dataset structure.

    Example:
        input:  dataset_root/train/species_a/img_001.jpg
        output: output_root/train/species_a/img_001.csv

    Each output CSV contains one row with metadata and descriptive feature
    columns. Legacy f0..fN aliases are included only if requested.
    """
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    entries = list(iter_dataset_images(dataset_root))

    for sample_id, (split, species, image_path) in enumerate(
        tqdm(entries, desc="Saving feature dataset", unit="image")
    ):
        row = extract_feature_row(
            image_path=image_path,
            split=split,
            species=species,
            resize_to=resize_to,
            sample_id=sample_id,
            sift_centers=sift_centers,
            orb_centers=orb_centers,
            include_legacy_f_columns=include_legacy_f_columns,
        )
        rows.append(row)

        relative_output_dir = output_root / split / species
        relative_output_dir.mkdir(parents=True, exist_ok=True)

        output_path = relative_output_dir / f"{image_path.stem}{output_suffix}"
        pd.DataFrame([row]).to_csv(output_path, index=False)

    manifest_df = pd.DataFrame(rows)
    if save_manifest:
        manifest_df.to_csv(output_root / "features_manifest.csv", index=False)

    return manifest_df
