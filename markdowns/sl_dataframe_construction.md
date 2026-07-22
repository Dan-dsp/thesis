# `sl_dataframe_construction.py`

Core SL feature-dataset builder. It walks an image dataset organized as split/species/image, loads and transforms each image, calls `extract_all_features_torch` from `sl_methods.py`, and attaches the `species`, `split`, and image-path metadata.

It can export one combined CSV and per-image feature CSV files that mirror the input structure. `sl_dataframe_main.py` supplies its configuration, while the generated CSV feeds feature selection and classical-model training.
