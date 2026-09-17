from pathlib import Path

from sl_dataframe_construction import export_feature_dataset_with_structure


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
