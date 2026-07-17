# `sl_feature_comparison.py` Guide

This file is now the **high-level orchestrator** for the shallow-learning feature-selection workflow.

It no longer stores all statistical methods directly. Instead:

- [sl_feature_comparison.py](f:/01_Univalle/01_TG/01_Python/sl_feature_comparison.py) coordinates the workflow
- [sl_feature_comparison_tools.py](f:/01_Univalle/01_TG/01_Python/sl_feature_comparison_tools.py) contains the reusable methods

If you want the implementation details of ANOVA, Fisher, MI, PCA, RFECV, Random Forest, or XGBoost importance, the companion guide is:

- [sl_feature_comparison_tools_guide.md](f:/01_Univalle/01_TG/01_Python/sl_feature_comparison_tools_guide.md)

## Big Picture

The file runs the full feature-comparison procedure in stages:

1. exploratory analysis
2. filter methods
3. wrapper methods
4. embedded methods
5. final overlap comparison
6. diagnostic plots for the final lists

The goal is not just to score features once, but to compare several selection strategies and save separate final lists for each one.

## Workflow

### 1. Exploratory stage

The script first loads the dataset and runs:

- Shapiro-Wilk normality test
- correlation matrix

This gives a quick statistical and redundancy check before feature selection starts.

### 2. Filter stage

The orchestrator calls the tools module to compute:

- ANOVA
- Fisher
- mutual information
- PCA with **weighted absolute loading**

Then it compares the four rankings using:

- a shared top-100 reference list for each method
- majority voting with threshold `>= 3 of 4`
- Borda-style tie breaking

After the top-100 consensus list is ordered, the script simply derives:

- top-80 = first 80 rows of the top-100 list
- top-50 = first 50 rows of the top-100 list

That matches the agreed rule of not reprocessing 50, 80, and 100 independently.

### 3. Wrapper stage

The orchestrator then runs RFECV-based wrapper selection for:

- SVM with RBF kernel
- k-NN

These wrapper runs use:

- the **full feature space**
- GridSearch inside the evaluation
- `f1_macro` as the main refit metric
- `accuracy` and `MCC` also recorded in the grid-search outputs

Each wrapper model saves:

- the RFECV ranking
- the selected feature list
- grid-search results
- best-parameter summary

### 4. Embedded stage

The orchestrator runs embedded importance-based selection for:

- Random Forest
- XGBoost, when available

For each embedded model, it saves:

- the full importance ranking
- top-100
- top-80
- top-50
- grid-search summary

### 5. Stage comparison

After the separate stages finish, the file compares the resulting feature lists across stages.

Examples:

- filter consensus top-100
- SVM selected subset
- k-NN selected subset
- Random Forest top-100
- XGBoost top-100

It saves overlap counts and Jaccard similarities so you can see how much agreement exists between stages.

### 6. Final diagnostics

At the end, the script checks selected lists again using:

- correlation matrices
- violin plots

This helps you inspect whether final selected features still contain strong redundancy or visually weak class separation.

## Main Functions

## `compare_filter_method_rankings(...)`

Purpose:

- compare ANOVA, Fisher, MI, and PCA rankings
- build a consensus top-100 list
- derive top-80 and top-50 from that same ordered list

Key logic:

- vote count = in how many method top-100 lists a feature appears
- majority vote = `vote_count >= 3`
- tie breaking = Borda score and average present rank

## `compare_stage_feature_lists(...)`

Purpose:

- compare the final lists produced by filters, wrappers, and embedded methods

Outputs:

- pairwise overlap counts
- Jaccard similarity
- feature-count summary per stage

## `run_final_selection_diagnostics(...)`

Purpose:

- save correlation plots and violin plots for the chosen final feature lists

This is the final interpretability check after the numeric ranking stages.

## `run_feature_comparison_workflow(...)`

Purpose:

- run the complete multi-stage procedure
- create the output folder structure
- call the tools module
- save summaries and diagnostics

This is the main workflow entry point used by both:

- direct execution of [sl_feature_comparison.py](f:/01_Univalle/01_TG/01_Python/sl_feature_comparison.py)
- orchestration from [sl_main.py](f:/01_Univalle/01_TG/01_Python/sl_main.py)

## Output Structure

The workflow now writes stage-oriented outputs under `sl_results`:

- `exploratory/`
- `filters/`
- `wrappers/`
- `embedded/`
- `stage_comparison/`
- `diagnostics/`

This is easier to navigate than placing every method directly at the root level.

## Short Summary

[sl_feature_comparison.py](f:/01_Univalle/01_TG/01_Python/sl_feature_comparison.py) is now the **controller** of the feature-selection experiment.

It:

- loads the data through the tools module
- runs exploratory, filter, wrapper, and embedded stages
- compares their outputs
- saves the separate final lists
- finishes with diagnostic correlation and violin plots
