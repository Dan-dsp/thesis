# `sl_training_pipeline.py` Guide

This guide explains the current behavior of [sl_training_pipeline.py](f:/01_Univalle/01_TG/01_Python/sl_training_pipeline.py), including the recent changes:

- it reads the CSV from `dataset_features`
- it does not require a pre-made `val` split
- it creates validation internally from the `train` rows
- it balances internal validation and test evaluation by downsampling each species to the smallest class support
- it checks for `NaN` / `inf` before training
- it saves the exact ordered feature-column list used for the models
- it writes each training run into its own timestamped run folder
- it reduces nested parallelism by keeping tree-based inner models single-threaded during grid search
- it uses the `fast` XGBoost grid by default and keeps slower alternatives documented in comments
- it measures runtime per stage and per model, then saves timing files in each run folder
- it now shows a live progress bar for completed cross-validation fits when `tqdm` is available
- it remains runnable on its own and can also be called by [sl_main.py](f:/01_Univalle/01_TG/01_Python/sl_main.py)

## Big Picture

This file is the main shallow-learning experiment pipeline for:

- SVM with RBF kernel
- Random Forest
- k-NN
- XGBoost, if available

The pipeline assumes the feature CSV already exists and contains at least:

- `species`
- `split`
- numerical feature columns

It expects `split` to contain:

- `train`
- `test`

If older CSVs also contain `val`, those rows are ignored because validation is now built internally from the `train` split.

## Workflow

1. Load the CSV.
2. Detect feature columns.
3. Keep only the rows marked as `train` and `test`.
4. Split the `train` rows into:
   - internal train
   - internal validation
5. Build balanced evaluation subsets for:
   - internal validation
   - test
6. Tune models with cross-validation on the internal train subset.
7. Evaluate the tuned model on the balanced internal validation subset.
8. Refit the chosen configuration on the full original `train` split.
9. Evaluate once on the balanced test subset.
10. Save models, metrics, reports, confusion matrices, feature-column order, and run configuration.

---

## Main Helpers

### `get_feature_columns(df)`

Purpose:

- detect which numeric columns are real features
- prefer descriptive names over legacy `f*` columns
- exclude deprecated duplicates:
  - `affine_6`
  - `f39`

This matters because the model must see the same feature order every time.

### `assert_all_finite(name, X)`

Purpose:

- fast verification that an array contains only finite values

It checks for:

- `NaN`
- `+inf`
- `-inf`

If any are found, it raises a clear error and shows the first bad positions.

This is a strict validation step, not an imputation step.

### `make_balanced_subset(X, y, seedd= int = 42)`

Purpose:

* create a balanced subset of the dataset
* downsample every class to match the smallest class count
* make evaluation fair by giving each class the same number of samples

Behavior:

1. Find all unique classes in `y`.
2. Count how many samples each class has.
3. Find the smallest class count.
4. Create a deterministic random generator using `seed`.
5. For each class:

   * find all sample indices belonging to that class
   * randomly select `min_count` samples
   * store the selected indices
6. Combine all selected indices.
7. Sort the selected indices to preserve the original dataset order.
8. Use those indices to create:

   * balanced `X`
   * balanced `y`
9. Create a support summary showing the final number of samples per class.

Returned data includes:

* `X[selected_indices]`
* `y[selected_indices]`
* `support_summary`

Important detail:

* the function uses deterministic random downsampling
* using the same `seed` gives the same balanced subset every time
* `replace=False` means no sample is selected more than once

Additional metadata returned:

* `support_summary`

Important balancing detail:

* every class is reduced to the size of the smallest class
* larger classes lose samples
* smaller classes are kept unchanged
* this is useful for fair validation or test evaluation
* it is usually better to use this for evaluation, not for final model training

Example:

* original class counts:

  * species A: 100
  * species B: 60
  * species C: 25

After balancing:

* species A: 25
* species B: 25
* species C: 25

Final dataset size:

* 75 samples

---


### `prepare_run_output_dirs(output_root)`

Purpose:

* create the folder structure for a new experiment run
* organize model outputs, reports, confusion matrices, and metadata in separate directories
* keep track of the most recent run directory

Behavior:

1. Create the main `output_root` directory if it does not already exist.
2. Create a `runs` directory inside `output_root`.
3. Generate a unique run name using the current date and time.
4. Create a new run directory inside `runs`.
5. Create the following subdirectories inside the run directory:

   * `models`
   * `reports`
   * `cm`
   * `metadata`
6. Save the path of the current run directory into:

   * `latest_run.txt`
7. Return a dictionary containing all important output paths.

Returned data includes:

* `output_root`
* `runs_dir`
* `run_dir`
* `models_dir`
* `reports_dir`
* `cm_dir`
* `metadata_dir`

Important detail:

* each run gets a timestamped folder name
* this prevents results from different executions from being overwritten
* `latest_run.txt` stores the path to the most recent run
* this makes it easier to find the latest experiment output later

Additional metadata returned:

* no model metadata is created directly by this function
* however, the function creates a dedicated `metadata_dir` where later metadata files can be saved

Important output organization detail:

* trained models should be saved in `models_dir`
* evaluation reports should be saved in `reports_dir`
* confusion matrices should be saved in `cm_dir`
* experiment metadata should be saved in `metadata_dir`
* all these folders belong to one specific timestamped run

Example output structure:

```text
output_root/
├── latest_run.txt
└── runs/
    └── run_20260626_143015/
        ├── models/
        ├── reports/
        ├── cm/
        └── metadata/
```

---

### `load_split_dataset(csv_path, val_fraction=0.2, seed=42)`

Purpose:

* load the shallow-learning feature CSV
* detect and extract feature columns
* encode species labels into numeric classes
* create an internal validation split from the original `train` rows only
* prepare balanced validation and test subsets for fair evaluation

Behavior:

1. Read the CSV file from `csv_path`.
2. Detect the feature columns using `get_feature_columns(df)`.
3. Check that feature columns exist.
4. Check that the required columns are present:

   * `species`
   * `split`
5. Build the feature matrix `X` using the detected feature columns.
6. Build the raw label array from the `species` column.
7. Encode species names into numeric labels using `LabelEncoder`.
8. Store the class names from the label encoder.
9. Read the original split assignment from the `split` column.
10. Create masks for:

* `train` rows
* `test` rows

11. Check that the CSV contains at least one `train` row.
12. Check that the CSV contains at least one `test` row.
13. If existing `val` rows are found:

* print an information message
* ignore those rows

14. Extract:

* `X_train_full`, `y_train_full` from the original `train` rows
* `X_test`, `y_test` from the original `test` rows

15. Check that `val_fraction` is between `0` and `1`.
16. Use `StratifiedShuffleSplit` to split the original `train` rows into:

* internal train
* internal validation

17. If the stratified split fails, raise an error explaining that some classes may not have enough training samples.
18. Create:

* `X_train`, `y_train`
* `X_val`, `y_val`

19. Downsample the internal validation subset using `make_balanced_subset`.
20. Downsample the test subset using `make_balanced_subset`.
21. Run finite-value checks on:

* `X_train`
* `X_val`
* `X_val_balanced`
* `X_test`
* `X_test_balanced`

22. Detect the feature schema as either:

* `descriptive`
* `legacy_f`

23. Collect available metadata columns from the CSV.
24. Reconstruct the absolute row indices for:

* internal train
* internal validation
* test

25. Create a `split_membership_df` showing how each selected CSV row is used in the pipeline.
26. Create a `split_summary` dictionary with dataset and split statistics.
27. Return all feature arrays, label arrays, class information, split metadata, and balancing metadata.

Returned data includes:

* `X_train`, `y_train`
* `X_val`, `y_val`
* `X_train_full`, `y_train_full`
* `X_test`, `y_test`
* `X_val_balanced`, `y_val_balanced`
* `X_test_balanced`, `y_test_balanced`
* `classes`
* `label_encoder`
* `n_features`
* `feature_cols`
* `val_fraction`

Important detail:

* `X_train_full` contains all original rows marked as `train`
* `X_train` is smaller than `X_train_full` because part of the original training data is carved out for internal validation
* existing `val` rows in the CSV are ignored
* validation is created only from the original `train` rows
* the final model can later be trained on `X_train_full` before final testing

Additional metadata returned:

* `feature_schema`
* `split_membership_df`
* `split_summary`
* `val_balanced_support`
* `test_balanced_support`

Important validation detail:

* the validation split is stratified
* this means the internal train and validation subsets try to preserve the class distribution of the original training split
* the split may fail if some species have too few samples in the original `train` split

Important balancing detail:

* model fitting still uses the natural training distribution
* balancing is applied only to evaluation subsets
* this means every species contributes the same number of samples when computing validation and test metrics
* the balancing method is deterministic downsampling to the smallest class count in each evaluation split
* the same `seed` produces the same balanced validation and test subsets

Important feature detail:

* the function supports two feature column formats:

  * descriptive feature names
  * legacy `f` columns
* if no valid feature columns are found, the function raises an error
* all extracted features are converted to `np.float32`

Important label detail:

* species names are converted into numeric class labels using `LabelEncoder`
* `classes` stores the original species names
* `label_encoder` can later be used to convert between numeric predictions and species names

Important split tracking detail:

* `split_membership_df` records which original CSV rows were used as:

  * `train_internal`
  * `val_internal`
  * `test`
* rows not used by the pipeline are marked as ignored internally
* this is useful for reproducibility and for checking exactly which samples were used in each stage

Example split behavior:

```text
Original CSV split column:

train rows → internally split into train_internal and val_internal
test rows  → kept as final test set
val rows   → ignored
```

Example returned split summary:

```text
split_summary = {
    "train_internal_samples": number of internal training samples,
    "val_internal_samples": number of internal validation samples,
    "val_balanced_samples": number of balanced validation samples,
    "train_full_samples": number of original train samples,
    "test_samples": number of original test samples,
    "test_balanced_samples": number of balanced test samples,
    "n_features": number of extracted features,
    "feature_schema": "descriptive" or "legacy_f",
    "internal_val_fraction": validation fraction used,
    "n_classes": number of species
}
```

---

### `make_model_spaces(seed=42, tree_model_n_jobs=1, xgb_n_jobs=1)`

Purpose:

* define the shallow-learning models used in the experiment
* define the hyperparameter search space for each model
* return each model as a `Pipeline` together with its parameter grid
* optionally include XGBoost if the `xgboost` package is installed

Behavior:

1. Create an empty dictionary called `spaces`.
2. Define an RBF Support Vector Machine pipeline.
3. Add `StandardScaler` before the SVM classifier.
4. Define the SVM hyperparameter grid for:

   * `C`
   * `gamma`
5. Store the SVM pipeline and grid under:

   * `svm_rbf`
6. Define a Random Forest pipeline.
7. Define the Random Forest hyperparameter grid for:

   * number of trees
   * maximum tree depth
   * minimum samples required to split a node
   * minimum samples required at a leaf node
8. Store the Random Forest pipeline and grid under:

   * `random_forest`
9. Define a k-Nearest Neighbors pipeline.
10. Add `StandardScaler` before the k-NN classifier.
11. Define the k-NN hyperparameter grid for:

* number of neighbors
* neighbor weighting strategy
* distance metric
* Minkowski distance parameter

12. Store the k-NN pipeline and grid under:

* `knn`

13. Try to import `XGBClassifier` from `xgboost`.
14. If XGBoost is installed:

* define an XGBoost pipeline
* configure it for multiclass classification
* define the active XGBoost hyperparameter grid
* store the XGBoost pipeline and grid under `xgboost`

15. If XGBoost is not installed:

* print a warning message
* skip the XGBoost model

16. Return the dictionary containing all available model pipelines and grids.

Returned data includes:

* `spaces`

Where `spaces` is a dictionary with model names as keys and tuples as values:

```python
{
    "model_name": (pipeline, parameter_grid)
}
```

Possible model entries include:

* `svm_rbf`
* `random_forest`
* `knn`
* `xgboost`

Important detail:

* not all models require feature scaling
* SVM and k-NN use `StandardScaler`
* Random Forest and XGBoost do not use `StandardScaler`
* this is because distance-based and margin-based models are sensitive to feature scale, while tree-based models are generally not

Additional metadata returned:

* no separate metadata dictionary is returned
* however, each model entry contains:

  * the complete model pipeline
  * the hyperparameter grid used for tuning

Important model-selection detail:

* this function does not train the models
* it only prepares the model definitions and hyperparameter spaces
* training and model selection are expected to happen later using these pipelines and grids
* typically, these returned grids are used with cross-validation methods such as `GridSearchCV`

Important reproducibility detail:

* `seed` is passed to models that support random-state control
* this helps make model training and hyperparameter search more reproducible
* the same seed should produce more consistent results across runs, especially for models such as Random Forest and XGBoost

Important parallelization detail:

* `tree_model_n_jobs` controls the number of CPU jobs used by the Random Forest model
* `xgb_n_jobs` controls the number of CPU jobs used by XGBoost
* using more jobs can speed up training
* using too many jobs can increase memory usage or overload the computer

Important XGBoost detail:

* XGBoost is optional
* if the `xgboost` library is not installed, the function does not fail
* instead, it prints a warning and continues with the remaining models
* this makes the pipeline more robust across different environments

Active models:

#### `svm_rbf`

Purpose:

* train a Support Vector Machine with an RBF kernel
* useful for nonlinear classification boundaries
* appropriate for shallow-learning feature vectors

Pipeline:

```text
StandardScaler → SVC
```

Hyperparameters searched:

* `clf__C`
* `clf__gamma`

Grid:

```python
{
    "clf__C": [0.1, 1.0, 10.0, 30.0],
    "clf__gamma": ["scale", 0.1, 0.01, 0.001],
}
```

#### `random_forest`

Purpose:

* train an ensemble of decision trees
* useful for nonlinear feature interactions
* provides a strong tree-based baseline

Pipeline:

```text
RandomForestClassifier
```

Hyperparameters searched:

* `clf__n_estimators`
* `clf__max_depth`
* `clf__min_samples_split`
* `clf__min_samples_leaf`

Grid:

```python
{
    "clf__n_estimators": [200, 400],
    "clf__max_depth": [None, 15, 30],
    "clf__min_samples_split": [2, 5],
    "clf__min_samples_leaf": [1, 2],
}
```

#### `knn`

Purpose:

* train a k-Nearest Neighbors classifier
* classify samples based on the labels of nearby samples in feature space
* useful as a simple distance-based baseline

Pipeline:

```text
StandardScaler → KNeighborsClassifier
```

Hyperparameters searched:

* `clf__n_neighbors`
* `clf__weights`
* `clf__metric`
* `clf__p`

Grid:

```python
{
    "clf__n_neighbors": [3, 5, 7, 11],
    "clf__weights": ["uniform", "distance"],
    "clf__metric": ["minkowski"],
    "clf__p": [1, 2],
}
```

#### `xgboost`

Purpose:

* train a gradient-boosted tree classifier
* useful for strong performance on tabular feature datasets
* included only when the `xgboost` library is installed

Pipeline:

```text
XGBClassifier
```

Configuration:

* multiclass objective: `multi:softprob`
* evaluation metric: `mlogloss`
* tree method: `hist`
* verbosity disabled
* controlled by `seed`
* CPU usage controlled by `xgb_n_jobs`

Active hyperparameters searched:

* `clf__n_estimators`
* `clf__max_depth`
* `clf__learning_rate`
* `clf__subsample`
* `clf__colsample_bytree`

Active grid:

```python
{
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [3, 4],
    "clf__learning_rate": [0.1],
    "clf__subsample": [0.8],
    "clf__colsample_bytree": [0.8, 1.0],
}
```

Important XGBoost grid detail:

* the active XGBoost grid is the `FAST` grid
* it is designed for quick benchmarking and debugging
* it has 8 parameter combinations
* with 5-fold cross-validation, this means 40 XGBoost fits

Alternative XGBoost grids:

* `BALANCED`

  * intended for regular experiments
  * stronger search than `FAST`
  * 48 combinations
  * 240 fits with 5-fold cross-validation

* `THOROUGH`

  * intended for final tuning when XGBoost already looks promising
  * much slower
  * 144 combinations
  * 720 fits with 5-fold cross-validation

Important pipeline detail:

* parameter names use the `clf__` prefix
* this prefix is required because the classifier step inside each pipeline is named `clf`
* for example:

  * `clf__C`
  * `clf__max_depth`
  * `clf__n_neighbors`

Example returned structure:

```python
spaces = {
    "svm_rbf": (svm_pipe, svm_grid),
    "random_forest": (rf_pipe, rf_grid),
    "knn": (knn_pipe, knn_grid),
    "xgboost": (xgb_pipe, xgb_grid),
}
```

If XGBoost is not installed:

```python
spaces = {
    "svm_rbf": (svm_pipe, svm_grid),
    "random_forest": (rf_pipe, rf_grid),
    "knn": (knn_pipe, knn_grid),
}
```

---


## Model Search Space

### SVM

Pipeline:

- `StandardScaler`
- `SVC(kernel="rbf", probability=True)`

Grid:

- `C`: `[0.1, 1.0, 10.0, 30.0]`
- `gamma`: `["scale", 0.1, 0.01, 0.001]`

### Random Forest

Pipeline:

- `RandomForestClassifier`

Parallelism detail:

- the classifier now uses `n_jobs=1` by default inside the grid-search loop
- this helps reduce nested parallelism because the outer `GridSearchCV` is already parallel

Grid:

- `n_estimators`: `[200, 400]`
- `max_depth`: `[None, 15, 30]`
- `min_samples_split`: `[2, 5]`
- `min_samples_leaf`: `[1, 2]`

### k-NN

Pipeline:

- `StandardScaler`
- `KNeighborsClassifier`

Grid:

- `n_neighbors`: `[3, 5, 7, 11]`
- `weights`: `["uniform", "distance"]`
- `metric`: `["minkowski"]`
- `p`: `[1, 2]`

### XGBoost

Only used if import succeeds.

Parallelism detail:

- it now uses `n_jobs=1` by default inside the grid-search loop for the same reason

Active grid:

- `n_estimators`: `[100, 200]`
- `max_depth`: `[3, 4]`
- `learning_rate`: `[0.1]`
- `subsample`: `[0.8]`
- `colsample_bytree`: `[0.8, 1.0]`

Why this is the active default:

- the previous XGBoost search was valid but slow
- it used 48 parameter combinations
- with 5-fold CV, that meant 240 XGBoost fits
- because `GridSearchCV(verbose=1)` prints the start of the search and then stays quiet during fitting, the program could look frozen even while it was still running

The current `fast` grid reduces the search to:

- `2 * 2 * 1 * 1 * 2 = 8` combinations
- with 5 folds = `40` fits

Commented alternatives left in the code:

- `balanced`
  - `48` combinations
  - `240` fits with 5-fold CV
- `thorough`
  - `144` combinations
  - `720` fits with 5-fold CV

Suggested use:

- `fast`: pipeline checks and routine reruns
- `balanced`: normal experiments
- `thorough`: heavier final tuning when XGBoost already looks useful

---

## Evaluation Logic

### `evaluate(model, X, y, labels)`

Returns:

- `accuracy`
- `precision_macro`
- `recall_macro`
- `f1_macro`
- confusion matrix
- full classification report

Macro metrics are used because they treat each bird class equally.

### `save_confusion_matrix(...)`

Saves a `.png` confusion matrix with:

- axis labels
- class names
- cell counts

---

## `main()`

Current defaults:

- `DATA_CSV_PATH = F:/01_Univalle/01_TG/dataset_features/shallow_learning_birds.csv`
- `OUT_DIR = F:/01_Univalle/01_TG/sl_outputs`
- `SEED = 42`
- `CV_FOLDS = 5`
- `SCORING = "f1_macro"`
- `VAL_FRACTION = 0.2`
- `GRIDSEARCH_N_JOBS = -1`
- `GRIDSEARCH_PRE_DISPATCH = "2*n_jobs"`
- `TREE_MODEL_N_JOBS = 1`
- `XGB_N_JOBS = 1`

Execution flow:

1. Load the dataset with internal validation creation.
2. Create a timestamped run directory under `sl_outputs/runs/`.
3. Save metadata:
   - `feature_columns.json`
   - `feature_columns.txt`
   - `run_config.json`
   - `split_summary.json`
   - `split_membership.csv`
4. Tune each model with `GridSearchCV` on the internal train subset.
5. Evaluate each tuned model on the internal validation subset.
6. Save:
   - `*_internal_val_report.txt`
   - `*_internal_val_cm.png`
   - `*_train_only.pkl`
7. Refit each selected model on the full original `train` split.
8. Evaluate on `test`.
9. Save:
   - `*_test_report.txt`
   - `*_test_cm.png`
   - `*_final.pkl`
10. Save summary tables:
   - `internal_validation_comparison.csv`
   - `test_comparison.csv`
   - `best_params.json`
11. Save timing artifacts:
   - `stage_timings.csv`
   - `model_timings.csv`
   - `runtime_summary.json`

Runtime behavior:

- each major pipeline stage is timed
- each model is timed during:
  - internal train + validation
  - final refit + test
- terminal messages now include `[TIME] ...`
- each model search now shows a live progress bar for completed CV fits
- the same timing information is saved to disk for later review

Important detail:

- previous versions only printed elapsed time after a stage had finished
- that means XGBoost could appear stuck during `GridSearchCV`
- the current version uses a `tqdm` + `joblib` hook so the CV search now updates as fits finish
- if `tqdm` is not installed, the pipeline still runs, but the live progress bar will not appear

---

## Output Organization

The folder `sl_outputs` is created only when this training pipeline runs.

So if you do not currently see:

- `F:/01_Univalle/01_TG/sl_outputs`

that usually just means you have not run [sl_training_pipeline.py](f:/01_Univalle/01_TG/01_Python/sl_training_pipeline.py) yet with the current setup.

Inside `sl_outputs`, the code now writes:

- `latest_run.txt`
- `runs/`

Inside each `runs/run_YYYYMMDD_HHMMSS/` folder, the code writes:

- `models/`
- `reports/`
- `cm/`
- `metadata/`
- `internal_validation_comparison.csv`
- `test_comparison.csv`
- `best_params.json`
- `stage_timings.csv`
- `model_timings.csv`
- `runtime_summary.json`

Inside `metadata/`, the code writes:

- `feature_columns.json`
- `feature_columns.txt`
- `class_names.json`
- `label_encoder.pkl`
- `run_config.json`
- `split_summary.json`
- `balanced_eval_support.json`
- `split_membership.csv`

Why `feature_columns.json` matters:

- it records the exact ordered feature list the models were trained on
- this protects you if the CSV schema changes later

Why `run_config.json` matters:

- it records the training configuration used for that run
- this helps reproducibility

Why `split_membership.csv` matters:

- it tells you exactly which samples were used as:
  - internal train
  - internal validation
  - test

This is useful when you want to reproduce or audit a specific run later.

Why `balanced_eval_support.json` matters:

- it records the exact per-species support used for balanced validation and balanced test evaluation
- this lets you verify that every class contributed equally to the reported metrics

---

## Suggestions

### 1. The current internal validation design is reasonable

Using a stratified split from the `train` rows is a good default when your dataset folders only provide:

- `train`
- `test`

It keeps the final test set untouched.

The only added adjustment is that validation and test metrics are now computed on balanced subsets, not on the raw class-count distribution.

### 2. Finite-value checks are worth keeping

This is a fast safeguard against bad feature extraction.

It is better to fail early than to let `GridSearchCV` crash deep inside a long run.

### 3. Parallelism is now safer by default

The code now uses:

- `GridSearchCV(n_jobs=-1)`
- `RandomForestClassifier(n_jobs=1)`
- `XGBClassifier(n_jobs=1)` if available

This keeps the outer hyperparameter search parallel while avoiding the most obvious nested parallelism problem inside tree-based models.

The grid search also sets:

- `pre_dispatch="2*n_jobs"`

which is aligned with scikit-learn's documented control over job dispatching.

This is a practical runtime improvement, not a change in ML logic.

### 4. The default XGBoost search is intentionally smaller

This is a runtime choice, not a theoretical ML restriction.

The main reason is usability:

- a long XGBoost search can make the terminal look stuck
- that is especially misleading when the other models finish quickly
- the `fast` grid gives you a much cheaper first pass

If later you want stronger XGBoost tuning, the file already contains commented `balanced` and `thorough` grid options, so you do not need to redesign the search from scratch.

### 5. Runtime tracking is now part of the experiment outputs

This makes performance review easier.

You can now inspect:

- which stage was slow
- which model took the longest to tune
- how expensive the final full-train refit was

That is useful both for debugging and for planning future grid sizes.

### 6. Time counting and progress bars are different things

This distinction matters:

- time counting means measuring how long a stage took after it finishes
- a progress bar means seeing movement while the stage is still running

The pipeline now does both:

- `[TIME] ...` messages for measured duration
- `tqdm` bars for live cross-validation progress

---

## Short Summary

The current pipeline is now organized as:

- feature CSV in `dataset_features`
- internal validation created from `train`
- balanced validation and balanced test evaluation
- final test kept separate
- strict finite checks before model training
- per-run output folders under `sl_outputs/runs`
- runtime tracking for stages and models
- reproducibility files saved alongside the model outputs

That is a cleaner and safer setup than the previous version if your dataset no longer has a real `val` folder.
