# `sl_dataframe_main.py` Guide

This file is the entrypoint for building the shallow-learning feature dataset.

It does not define the feature-extraction logic itself. Instead, it configures paths and options, then calls [sl_dataframe_construction.py](F:/01_Univalle/01_TG/01_Python/sl_dataframe_construction.py) to do the real work.

## Big Picture

The purpose of [sl_dataframe_main.py](F:/01_Univalle/01_TG/01_Python/sl_dataframe_main.py) is:

1. read the original image dataset organized by split and species
2. extract handcrafted shallow-learning features for each image
3. save one feature file per image, mirroring the original folder structure
4. save one global CSV containing all rows together

So this file is the dataset-construction stage for the shallow-learning pipeline.

It sits before:

- [sl_feature_comparison.py](F:/01_Univalle/01_TG/01_Python/sl_feature_comparison.py)
- [sl_training_pipeline.py](F:/01_Univalle/01_TG/01_Python/sl_training_pipeline.py)

---

## Current Configuration

The file currently defines:

- `DATASET_ROOT = F:/01_Univalle/01_TG/dataset_bbox`
- `FEATURE_DATASET_ROOT = F:/01_Univalle/01_TG/dataset_features`
- `FEATURES_CSV_PATH = F:/01_Univalle/01_TG/dataset_features/shallow_learning_birds.csv`
- `INCLUDE_LEGACY_F_COLUMNS = False`

What each one means:

- `DATASET_ROOT`
  - input image dataset
  - expected structure:
    - `train/species_name/image.jpg`
    - `test/species_name/image.jpg`

- `FEATURE_DATASET_ROOT`
  - output root where per-image feature CSV files will be saved
  - it mirrors the original `train/species/...` and `test/species/...` structure

- `FEATURES_CSV_PATH`
  - one global CSV with all samples together
  - this is the file later used by feature analysis and model training

- `INCLUDE_LEGACY_F_COLUMNS = False`
  - only descriptive feature names are written
  - example:
    - `hu_1`
    - `zernike_3`
    - `glcm_contrast`
  - legacy aliases like `f0`, `f1`, `f2`, ... are not added

---

## What `main()` Does

The function `main()` is short, but it is the execution wrapper for the whole feature-dataset build.

Workflow:

1. Call `export_feature_dataset_with_structure(...)` from [sl_dataframe_construction.py](F:/01_Univalle/01_TG/01_Python/sl_dataframe_construction.py).
2. Ask it to:
   - read every image from `DATASET_ROOT`
   - resize images to `(224, 224)`
   - extract all handcrafted features defined through the construction/methods layer
   - save one CSV per image under `FEATURE_DATASET_ROOT`
   - return a DataFrame containing all rows together
3. Ensure the parent folder for `FEATURES_CSV_PATH` exists.
4. Save the returned DataFrame again as:
   - `shallow_learning_birds.csv`
5. Print:
   - output folder path
   - global CSV path
   - whether legacy `f*` columns were enabled
   - the first rows of the resulting DataFrame

---

## Important Delegation

This file delegates the actual heavy work to:

- [sl_dataframe_construction.py](F:/01_Univalle/01_TG/01_Python/sl_dataframe_construction.py)

That construction file is the one that:

- iterates through images
- loads and resizes them
- calls feature extraction from [sl_methods.py](F:/01_Univalle/01_TG/01_Python/sl_methods.py)
- assigns metadata columns
- saves one CSV per image

So the design is:

- `sl_dataframe_main.py`: configuration + execution entrypoint
- `sl_dataframe_construction.py`: dataset traversal + row building + file export
- `sl_methods.py`: actual feature definitions

---

## Output Files

After running [sl_dataframe_main.py](F:/01_Univalle/01_TG/01_Python/sl_dataframe_main.py), you should get two output styles.

### 1. Per-image feature files

These are stored under:

- `F:/01_Univalle/01_TG/dataset_features/train/...`
- `F:/01_Univalle/01_TG/dataset_features/test/...`

Example:

- input:
  - `dataset_bbox/train/species_a/img_001.jpg`
- output:
  - `dataset_features/train/species_a/img_001.csv`

Each per-image CSV contains one row with:

- metadata columns
- feature columns

### 2. Global feature CSV

This file is:

- `F:/01_Univalle/01_TG/dataset_features/shallow_learning_birds.csv`

More precisely, at runtime it is saved to:

- `F:/01_Univalle/01_TG/dataset_features/shallow_learning_birds.csv`

This is the file that downstream scripts use.

### 3. `features_manifest.csv`

Because `save_manifest=True` is passed to `export_feature_dataset_with_structure(...)`, the construction layer also writes:

- `F:/01_Univalle/01_TG/dataset_features/features_manifest.csv`

In the current implementation:

- `features_manifest.csv`
- `shallow_learning_birds.csv`

contain the same rows and columns.

The difference is mostly naming and purpose:

- `features_manifest.csv` is the construction-layer manifest
- `shallow_learning_birds.csv` is the pipeline-facing dataset name used by later stages

---

## Feature Columns

With `INCLUDE_LEGACY_F_COLUMNS = False`, the output should contain:

- metadata columns:
  - `sample_id`
  - `sample_name`
  - `orig_filename`
  - `species`
  - `split`
- descriptive feature columns only

This is the recommended setting because the descriptive names tell you which extraction method each feature comes from.

If `INCLUDE_LEGACY_F_COLUMNS = True`, the output would also contain duplicate alias columns:

- `f0`
- `f1`
- `f2`
- ...

Those are not new features. They are just alternate names for the same feature values.

---

## What This File Does Not Do

This file does not:

- compare feature importance
- remove features
- train models
- evaluate models

It only builds the feature dataset.

That separation is intentional:

- dataset construction happens here
- feature analysis happens in [sl_feature_comparison.py](F:/01_Univalle/01_TG/01_Python/sl_feature_comparison.py)
- model training happens in [sl_training_pipeline.py](F:/01_Univalle/01_TG/01_Python/sl_training_pipeline.py)

---

## When To Run It

Run [sl_dataframe_main.py](F:/01_Univalle/01_TG/01_Python/sl_dataframe_main.py) when:

- the original image dataset changed
- feature extraction logic changed
- feature naming changed
- you need to regenerate `shallow_learning_birds.csv`

If the feature CSV already exists and has the correct schema, you do not need to rerun this file before every training run.

---

## Summary

[sl_dataframe_main.py](F:/01_Univalle/01_TG/01_Python/sl_dataframe_main.py) is the shallow-learning feature-dataset builder entrypoint.

It:

- reads the original bird image dataset
- calls the construction layer to extract handcrafted features
- saves per-image feature CSVs in mirrored folder structure
- saves the global dataset CSV used by the later shallow-learning stages

Its main role is orchestration of feature export, not analysis or training.
