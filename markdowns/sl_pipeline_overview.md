# Shallow Learning Pipeline Overview

This document describes the full shallow-learning process in your project from image folders to trained models and analysis outputs.

The goal is to make the workflow easy to understand now, and also make it easy to turn into future Sphinx / `github.io` documentation later.

## Purpose

Your shallow-learning pipeline is organized into three main stages:

1. **Feature extraction**
2. **Feature analysis / comparison**
3. **Model training and evaluation**

The high-level logic is:

- start from the bird image dataset
- convert each image into a handcrafted feature vector
- analyze which features look useful
- train classical ML models on the resulting tabular dataset

---

## Stage 0: Input Dataset

Current expected image dataset root:

- `F:/01_Univalle/01_TG/dataset_bbox`

Expected folder organization:

- `train/`
- `test/`

Inside each split:

- one folder per bird species

Example:

```text
dataset_bbox/
  train/
    species_a/
      img1.jpg
      img2.jpg
    species_b/
  test/
    species_a/
    species_b/
```

Important note:

- the training pipeline no longer requires a physical `val/` folder
- validation is now created internally from the `train` split

---

## Stage 1: Feature Extraction

Main entrypoint:

- [sl_dataframe_main.py](../sl_dataframe_main.py)

Main implementation:

- [sl_dataframe_construction.py](../sl_dataframe_construction.py)
- [sl_methods.py](../sl_methods.py)

### What happens here

1. Walk through every image in the dataset.
2. Convert each image into a PyTorch tensor.
3. Extract handcrafted features using [sl_methods.py](../sl_methods.py).
4. Save:
   - one per-image CSV mirroring the dataset structure
   - one global CSV with all samples together

### Current output folder

- `F:/01_Univalle/01_TG/dataset_features`

### Current global CSV

- `F:/01_Univalle/01_TG/dataset_features/shallow_learning_birds.csv`

### Current naming policy

The pipeline now defaults to:

- `INCLUDE_LEGACY_F_COLUMNS = False`

So the CSV uses descriptive feature names like:

- `hu_1`
- `zernike_4`
- `logpolarfft_r3_a7`
- `glcm_contrast`

instead of legacy aliases like:

- `f0`
- `f1`
- `f2`

### Why this is better

Descriptive names make it immediately clear:

- which method produced a column
- which family a feature belongs to
- how to interpret rankings and plots later

---

## Stage 2: Feature Analysis / Comparison

Main entrypoint:

- [sl_feature_comparison.py](../sl_feature_comparison.py)

Guide:

- [sl_feature_comparison_guide.md](../sl_feature_comparison_guide.md)

### What happens here

This stage reads the global feature CSV and runs:

- ANOVA
- Fisher score
- Mutual Information
- PCA
- violin plots
- correlation matrix
- RFECV

### Current input CSV

- `F:/01_Univalle/01_TG/dataset_features/shallow_learning_birds.csv`

### Current output folder

- `F:/01_Univalle/01_TG/sl_results`

### Important behavior

- `logpolarfft_*` is included again by default because the extractor was stabilized
- duplicated old affine columns are excluded if found:
  - `affine_6`
  - `f39`
- this analysis step may still use median imputation for numerical analysis convenience

That is different from the training stage, which is stricter.

---

## Stage 3: Model Training and Evaluation

Main entrypoint:

- [sl_training_pipeline.py](../sl_training_pipeline.py)

Guide:

- [sl_training_pipeline_guide.md](../sl_training_pipeline_guide.md)

### What happens here

1. Load the global feature CSV.
2. Detect the feature columns.
3. Use the original `train` rows only for model development.
4. Create an **internal validation split** from the `train` rows.
5. Tune models with cross-validation on the internal train subset.
6. Evaluate tuned models on the internal validation subset.
7. Refit the selected configuration on the full original `train` split.
8. Evaluate once on the untouched `test` split.

### Models currently supported

- SVM (RBF)
- Random Forest
- k-NN
- XGBoost, if installed

### Current input CSV

- `F:/01_Univalle/01_TG/dataset_features/shallow_learning_birds.csv`

### Current output root

- `F:/01_Univalle/01_TG/sl_outputs`

### Current output organization

Each run gets its own folder:

```text
sl_outputs/
  latest_run.txt
  runs/
    run_YYYYMMDD_HHMMSS/
      internal_validation_comparison.csv
      test_comparison.csv
      best_params.json
      models/
      reports/
      cm/
      metadata/
```

Inside `metadata/`:

- `feature_columns.json`
- `feature_columns.txt`
- `run_config.json`
- `split_summary.json`
- `split_membership.csv`

### Why this organization matters

This prevents different training runs from overwriting each other.

It also preserves:

- exact feature order
- run configuration
- internal split membership

So later you can trace:

- which data was used
- which schema the model expects
- which run produced which metrics

---

## Why There Are Different Output Folders

### `dataset_features`

Purpose:

- store extracted feature representations

Contains:

- per-image feature CSVs
- global feature CSV

### `sl_results`

Purpose:

- store feature-analysis outputs

Contains:

- rankings
- figures
- PCA results
- RFE results

### `sl_outputs`

Purpose:

- store model-training outputs

Contains:

- trained models
- confusion matrices
- reports
- run metadata

These folders are separate because they serve different stages of the workflow.

---

## Recommended Execution Order

If you want to rerun the shallow-learning pipeline from scratch, the current order is:

1. Run [sl_dataframe_main.py](../sl_dataframe_main.py)
   - regenerates the feature dataset
   - creates `dataset_features/shallow_learning_birds.csv`

2. Run [sl_feature_comparison.py](../sl_feature_comparison.py)
   - analyzes the regenerated feature CSV
   - writes outputs into `sl_results`

3. Run [sl_training_pipeline.py](../sl_training_pipeline.py)
   - trains and compares models
   - writes outputs into `sl_outputs/runs/...`

Optional:

4. Review older helper scripts such as [sl_models_evaluation.py](../sl_models_evaluation.py) only if you want custom extra plots beyond what the training pipeline already saves

---

## If `dataset_features` or `sl_results` Already Exist but Are Empty

That is not a problem.

The code uses `mkdir(..., exist_ok=True)` style behavior, so:

- existing empty folders are fine
- you do **not** need to delete them just to rerun

You may want to delete old contents only when:

- you want a perfectly clean result folder
- you want to avoid confusing old outputs with new outputs

But empty folders by themselves are harmless.

### Practical recommendation

- if the folders are empty: leave them
- if the folders contain old outputs you no longer want: clear the contents before rerunning

For `sl_outputs`, old runs are less dangerous now because each run gets its own timestamped directory.

---

## Important Behavioral Differences Between Analysis and Training

### Feature analysis

[sl_feature_comparison.py](../sl_feature_comparison.py) may use median imputation when needed for analysis methods.

Why:

- many statistical analysis methods are easier to run on a fully numeric matrix

### Model training

[sl_training_pipeline.py](../sl_training_pipeline.py) does **not** silently impute.

Instead it:

- checks for `NaN` / `inf`
- fails fast if invalid values are present

Why:

- model-training experiments should not quietly hide feature-extraction errors

---

## Future Documentation Direction

This project is already in a good shape for future Sphinx or `github.io` documentation because:

- the code is being documented file by file
- the workflow is now separated into clear stages
- the outputs are organized by purpose
- training runs are now isolated in per-run folders

For a future docs website, the natural documentation structure would be:

1. Dataset format
2. Feature extraction
3. Feature families
4. Feature analysis
5. Training pipeline
6. Output folders and artifacts
7. Reproducibility notes

---

## Short Summary

The current shallow-learning process is:

1. images in `dataset_bbox`
2. handcrafted features generated into `dataset_features`
3. feature analysis saved into `sl_results`
4. model training runs saved into `sl_outputs/runs/...`

And the current recommended execution order is:

1. [sl_dataframe_main.py](../sl_dataframe_main.py)
2. [sl_feature_comparison.py](../sl_feature_comparison.py)
3. [sl_training_pipeline.py](../sl_training_pipeline.py)

You can also run the same three stages from one orchestrator:

- [sl_main.py](../sl_main.py)

This gives you two valid run modes:

### Run each stage individually

Use this when you only want one part:

1. [sl_dataframe_main.py](../sl_dataframe_main.py)
2. [sl_feature_comparison.py](../sl_feature_comparison.py)
3. [sl_training_pipeline.py](../sl_training_pipeline.py)

### Run the whole pipeline in one Python process

Use this when you want to avoid repeating import startup across separate script runs:

1. [sl_main.py](../sl_main.py)

Important design detail:

- the three stage files still remain runnable on their own
- the orchestrator is an extra convenience layer, not a dependency
