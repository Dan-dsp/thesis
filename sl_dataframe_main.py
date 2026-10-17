from pathlib import Path

from sl_dataframe_construction import export_feature_dataset_with_structure
from sl_dataframe_construction import extract_feature_row
from shared_augmentation import load_augmentation
import pandas as pd
from tqdm import tqdm


# Source image dataset. The construction function preserves its class/split
# directory structure when writing the extracted feature files.
DATASET_ROOT = Path(r"F:/01_Univalle/01_TG/dataset_bbox")
# Destination directory for feature datasets and the generated manifest files.
FEATURE_DATASET_ROOT = Path(r"F:/01_Univalle/01_TG/dataset_features")
# Consolidated CSV containing the features from all processed images.
FEATURES_CSV_PATH = Path(r"F:/01_Univalle/01_TG/dataset_features/shallow_learning_birds.csv")
# Set to True only when older code still requires the legacy f* feature columns.
INCLUDE_LEGACY_F_COLUMNS = False


def main() -> None:
    # Extract the handcrafted image features and return a manifest describing
    # the processed samples. Passing None omits SIFT and ORB BoVW features,
    # because no trained visual-word centers are supplied.
    manifest_df = export_feature_dataset_with_structure(
        dataset_root=DATASET_ROOT,
        output_root=FEATURE_DATASET_ROOT,
        # Resize every image to a common size before feature extraction.
        resize_to=(224, 224),
        sift_centers=None,
        orb_centers=None,
        output_suffix=".csv",
        save_manifest=True,
        include_legacy_f_columns=INCLUDE_LEGACY_F_COLUMNS,
    )

    views, _ = load_augmentation()
    augmented_rows = []
    for row in tqdm(list(views.itertuples()), desc="Shared augmented descriptors", unit="image"):
        features = extract_feature_row(row.image_path, "train_augmented", row.species,
                                       resize_to=(224, 224),
                                       sample_id=len(manifest_df) + len(augmented_rows),
                                       include_legacy_f_columns=INCLUDE_LEGACY_F_COLUMNS)
        features["augmentation_parent"] = row.augmentation_parent
        features["augmentation_id"] = row.augmentation_id
        augmented_rows.append(features)
    manifest_df = pd.concat([manifest_df, pd.DataFrame(augmented_rows)], ignore_index=True)
    manifest_df.to_csv(FEATURE_DATASET_ROOT / "features_manifest.csv", index=False)

    # Ensure the CSV destination exists even if the export function did not
    # need to create it, then save the combined manifest without row indices.
    FEATURES_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(FEATURES_CSV_PATH, index=False)

    # Report the output locations and a small preview for quick verification.
    print(f"[INFO] Feature dataset saved to: {FEATURE_DATASET_ROOT}")
    print(f"[INFO] Global feature CSV saved to: {FEATURES_CSV_PATH}")
    print(f"[INFO] Legacy f-columns enabled: {INCLUDE_LEGACY_F_COLUMNS}")
    print(manifest_df.head())


if __name__ == "__main__":
    # Allow this module to be run directly as the feature-extraction entrypoint.
    main()
