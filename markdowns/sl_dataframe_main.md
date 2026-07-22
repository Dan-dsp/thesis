# `sl_dataframe_main.py`

Configuration entrypoint for the first SL stage. It defines the image dataset root, output feature directory, combined CSV path, and feature-column option, then calls `export_feature_dataset_with_structure` from `sl_dataframe_construction.py`.

Run this after the dataset has `train` and `test` species folders. Its generated feature CSV is the input to `sl_feature_comparison.py` and `sl_training_pipeline.py`.

See `sl_dataframe_main_guide.md` for the detailed guide.
