# `sl_dataframe_main.py`

Run shared_augmentation.py first (the full runner and sl_main do this). Extraction
now also reads the saved augmented PNGs and appends their descriptors as
train_augmented rows with parent IDs and augmentation provenance. Those rows
are pooled with originals before ordinary training/validation splitting. Originals and
test descriptors are still extracted from dataset_bbox. The combined feature
manifest and global CSV both include these rows.
See [Shared Augmentation](shared_augmentation.md).

Configuration entrypoint for the first SL stage. It defines the image dataset root, output feature directory, combined CSV path, and feature-column option, then calls `export_feature_dataset_with_structure` from `sl_dataframe_construction.py`.

Run this after the dataset has `train` and `test` species folders. Its generated feature CSV is the input to `sl_feature_comparison.py` and `sl_training_pipeline.py`.

See `sl_dataframe_main_guide.md` for the detailed guide.
