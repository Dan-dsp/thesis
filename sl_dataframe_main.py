from pathlib import Path

from sl_dataframe_construction import export_feature_dataset_with_structure


DATASET_ROOT = Path(r"F:/01_Univalle/01_TG/dataset_bbox")
FEATURE_DATASET_ROOT = Path(r"F:/01_Univalle/01_TG/dataset_features")
FEATURES_CSV_PATH = Path(r"F:/01_Univalle/01_TG/dataset_features/shallow_learning_birds.csv")
INCLUDE_LEGACY_F_COLUMNS = False


def main() -> None:
    manifest_df = export_feature_dataset_with_structure(
        dataset_root=DATASET_ROOT,
        output_root=FEATURE_DATASET_ROOT,
        resize_to=(224, 224),
        sift_centers=None,
        orb_centers=None,
        output_suffix=".csv",
        save_manifest=True,
        include_legacy_f_columns=INCLUDE_LEGACY_F_COLUMNS,
    )

    FEATURES_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(FEATURES_CSV_PATH, index=False)

    print(f"[INFO] Feature dataset saved to: {FEATURE_DATASET_ROOT}")
    print(f"[INFO] Global feature CSV saved to: {FEATURES_CSV_PATH}")
    print(f"[INFO] Legacy f-columns enabled: {INCLUDE_LEGACY_F_COLUMNS}")
    print(manifest_df.head())


if __name__ == "__main__":
    main()
