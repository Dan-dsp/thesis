# `sl_feature_comparison_tools.py` Guide

This file contains the **implementation methods** used by:

- [sl_feature_comparison.py](f:/01_Univalle/01_TG/01_Python/sl_feature_comparison.py)

So the split is now:

- [sl_feature_comparison.py](f:/01_Univalle/01_TG/01_Python/sl_feature_comparison.py): workflow and comparison logic
- [sl_feature_comparison_tools.py](f:/01_Univalle/01_TG/01_Python/sl_feature_comparison_tools.py): reusable statistical, model, and plotting tools

## Big Picture

The tools file contains:

- dataset loading and preprocessing helpers
- exploratory analysis helpers
- filter methods
- wrapper-model helpers
- embedded-model helpers
- plotting utilities

It is meant to keep the main comparison file shorter and easier to read.

## Data and Utility Helpers

## `ensure_dir(...)`

Creates an output directory if it does not exist.

## `save_json(...)`

Writes small summary dictionaries to disk in JSON format.

## `get_feature_columns(...)`

Detects which columns are real numeric features by:

- excluding metadata
- preferring descriptive names over legacy `f*` columns
- excluding deprecated duplicates such as `affine_6` and `f39`

## `load_features_and_labels(...)`

Loads the CSV and returns:

- original DataFrame
- cleaned numeric matrix `X`
- labels `y`
- detected feature names

It also:

- converts features to numeric
- replaces `inf` with `NaN`
- imputes missing values with the median when needed

## `encode_labels(...)`

Encodes class labels into integers for models such as XGBoost and for wrapper / embedded routines that work more cleanly with numeric targets.

## Exploratory Tools

## `run_shapiro_tests(...)`

Runs a Shapiro-Wilk normality test for each feature.

Design notes:

- this is the chosen Gaussianity test
- it is one of the most common normality tests in practice
- for very large feature vectors, the function can subsample to keep Shapiro in a safer range

Saved outputs include:

- full CSV with `W`, `p_value`, and Gaussian/non-Gaussian flag
- summary JSON

## `plot_correlation_matrix(...)`

Computes a Pearson correlation matrix and saves:

- the full CSV
- a heatmap for up to `max_features`

This is used both at the exploratory stage and again on final selected lists.

## Filter Methods

## `compute_anova_scores(...)`

Computes ANOVA F-scores and p-values for each feature.

The ranking is descending by `F_score`.

## `compute_fisher_scores(...)`

Computes a multi-class Fisher score for each feature.

The ranking is descending by `fisher_score`.

## `compute_mutual_information(...)`

Computes mutual information between each feature and the class label.

The ranking is descending by `mutual_info`.

## `run_pca_analysis(...)`

Fits PCA and saves:

- explained variance ratios
- component loadings
- explained variance plot

## `rank_pca_features(...)`

Converts PCA loadings into a feature-level ranking using:

- **weighted absolute loading**

The weights come from the explained variance ratio of the retained components, which was the agreed rule for turning PCA into a feature-ranking method.

## Ranked-List Helper

## `save_top_ranked_feature_slices(...)`

This helper applies the agreed top-list rule:

1. take the ordered top-100 list
2. derive top-80 from its first 80 rows
3. derive top-50 from its first 50 rows

So 50 and 80 are not recomputed independently.

## Wrapper Methods

## `build_wrapper_model_spaces(...)`

Defines the wrapper models and their search spaces:

- `svm_rbf`
- `knn`

## `PermutationImportanceWrapper`

This class is the key technical bridge that lets RFECV work with:

- SVM with RBF kernel
- k-NN

These models do not expose native coefficients or feature importances, so the wrapper:

1. fits the tuned estimator
2. computes permutation importance
3. exposes that importance vector to RFECV

That makes recursive elimination possible with non-linear models.

## `run_wrapper_rfecv(...)`

Runs the wrapper stage for one model.

Behavior:

- starts from the full feature space
- performs RFECV
- uses GridSearch inside the evaluation
- refits using `f1_macro`
- also records `accuracy` and `MCC`

Saved outputs include:

- RFECV ranking
- selected feature list
- RFECV curve
- full grid-search results
- best-summary JSON
- derived top-100 / top-80 / top-50 ranked lists

## Embedded Methods

## `build_embedded_model_spaces(...)`

Defines the embedded models and their grids:

- `random_forest`
- `xgboost`, when available

## `run_embedded_feature_importance(...)`

Fits the tuned embedded model and extracts:

- `feature_importances_`

Saved outputs include:

- full importance ranking
- full grid-search results
- best-summary JSON
- top-100 / top-80 / top-50 ranked lists

## Plotting for Final Inspection

## `plot_violin_for_features(...)`

Creates class-wise violin plots for selected features.

This is mainly used at the end of the workflow to inspect the final candidate lists visually.

## Short Summary

[sl_feature_comparison_tools.py](f:/01_Univalle/01_TG/01_Python/sl_feature_comparison_tools.py) is the file that **does the actual work**:

- it loads and cleans the data
- computes the statistical rankings
- runs wrapper and embedded selectors
- saves reusable outputs

[sl_feature_comparison.py](f:/01_Univalle/01_TG/01_Python/sl_feature_comparison.py) then uses these tools to compare stages and organize the final feature lists.
