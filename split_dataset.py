import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple


# Uncomment lines 155 and 156

def split_species_images(
    species_dir: Path,
    train_dir: Path,
    val_dir: Path,
    test_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    allowed_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"),
    seed: int = 42,
    copy_instead_of_move: bool = True,
):
    """
    Split one species folder into train/val/test subfolders.

    species_dir: path to one species (e.g. nuevo_dataset/chlorophanes_spiza)
    train_dir / val_dir / test_dir: destination dirs for that species
    ratios: must add up to 1.0
    seed: random seed for reproducibility
    copy_instead_of_move:
        True  -> shutil.copy2  (keeps original dataset intact, safer)
        False -> shutil.move   (saves disk but modifies your original dataset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    # list all valid image files for this species
    images = [p for p in species_dir.iterdir() if p.is_file() and p.suffix.lower() in allowed_exts]
    images = sorted(images)  # stable order before shuffling

    if len(images) == 0:
        print(f"[WARN] No images found in {species_dir}")
        return (0, 0, 0)

    # reproducible shuffle
    rng = random.Random(seed)
    rng.shuffle(images)

    n_total = len(images)
    n_train = int(train_ratio * n_total)
    n_val   = int(val_ratio * n_total)
    # whatever remains goes to test
    n_test  = n_total - n_train - n_val

    train_files = images[:n_train]
    val_files   = images[n_train:n_train + n_val]
    test_files  = images[n_train + n_val:]

    # make output dirs for this species
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # choose copy or move
    op = shutil.copy2 if copy_instead_of_move else shutil.move

    for src in train_files:
        dst = train_dir / src.name
        op(src, dst)
    for src in val_files:
        dst = val_dir / src.name
        op(src, dst)
    for src in test_files:
        dst = test_dir / src.name
        op(src, dst)

    return (len(train_files), len(val_files), len(test_files))


def split_full_dataset(
    source_root: str | Path,
    dest_root: str | Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    copy_instead_of_move: bool = True,
):
    """
    Walk through each species folder in source_root and split into train/val/test
    inside dest_root.
    """
    source_root = Path(source_root)
    dest_root   = Path(dest_root)

    train_root = dest_root / "train"
    val_root   = dest_root / "val"
    test_root  = dest_root / "test"

    summary = []

    if not source_root.exists():
        raise RuntimeError(f"Source dataset not found: {source_root}")

    # iterate all immediate subfolders = each is one species
    for species_dir in sorted([d for d in source_root.iterdir() if d.is_dir()]):
        species_name = species_dir.name

        train_dir = train_root / species_name
        val_dir   = val_root / species_name
        test_dir  = test_root / species_name

        n_train, n_val, n_test = split_species_images(
            species_dir=species_dir,
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            copy_instead_of_move=copy_instead_of_move,
        )

        summary.append((species_name, n_train, n_val, n_test))

    # print summary
    print("\n===== SPLIT SUMMARY =====")
    total_train = total_val = total_test = 0
    for species_name, n_train, n_val, n_test in summary:
        total_train += n_train
        total_val   += n_val
        total_test  += n_test
        print(f"{species_name}: train={n_train}, val={n_val}, test={n_test}")

    print("-------------------------")
    print(f"TOTAL: train={total_train}, val={total_val}, test={total_test}")
    print(f"Output root: {dest_root}")
    print("=========================\n")


if __name__ == "__main__":
    # EXAMPLE USAGE:
    #
    # Suppose you currently have:
    #   F:/Univalle/01_TG/nuevo_dataset/
    #       chlorophanes_spiza/
    #       turdus_ignobilis/
    #       anisognathus_somptuosus/
    #
    # We will create:
    #   F:/Univalle/01_TG/nuevo_dataset_split/
    #       train/...
    #       val/...
    #       test/...

    # source_root = r"F:/Univalle/01_TG/nuevo_dataset"
    # dest_root   = r"F:/Univalle/01_TG/nuevo_dataset_split"

    split_full_dataset(
        source_root=source_root,
        dest_root=dest_root,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
        copy_instead_of_move=True,  # True = copy (safe), False = move (saves space)
    )
