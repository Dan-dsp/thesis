# `result_plots.py`

Shared PNG rendering helpers used by selection, training, independent CV and
family evaluation. This is a library, not another script to run. It uses an
Agg canvas directly: no desktop windows, Tkinter objects or extra model fits.
Existing CSV/JSON outputs remain the numerical source of truth.

## Feature Selection Images

Under `F:/01_Univalle/01_TG/sl_results`:

| Folder | Images |
| --- | --- |
| `exploratory/shapiro` | Shapiro p-value histogram; normality is exploratory, not a Gaussian requirement for the classifiers |
| `filters/anova`, `filters/fisher`, `filters/mutual_information` | Existing top-20 score plots |
| `filters/pca` | Existing explained-variance plot, plus original-feature loading ranking |
| `filters/comparison` | Consensus votes and pairwise top-100 Jaccard overlap |
| `wrappers/knn`, `wrappers/svm_rbf` | RFECV score versus retained feature count, selected-count marker, fold-SD band; initial/final grid-search scores; selected-subset permutation importance |
| `embedded/random_forest`, `embedded/xgboost` | Top-20 native feature importance and grid-search scores |
| `stage_comparison` | Pairwise Jaccard overlap and selected-feature counts |
| `diagnostics` | Existing correlation heatmaps and species-wise violin plots |

Use the output folders printed by the pipeline; the exact exploratory/filter
subfolders are configured by `sl_feature_comparison.py`. SD bands/error bars show
fold variability, not confidence intervals or independent statistical evidence.
Search candidate numbers refer to CSV row order, not epochs. SVM/k-NN permutation
importance is measured on the selected training subset, not held-out test data.
Negative permutation importances are retained, not clipped to zero.

## Training and Evaluation Images

- Each SL run: `cm/` contains existing validation/test confusion matrices,
  plus per-model grid-search diagnostics and grouped validation/test metric plots.
- Both DL trainers: existing epoch loss/accuracy curves and final test confusion
  matrices remain; `kfold_train_only` also gets independent outer-fold confusion
  matrices and an outer-CV macro-F1 curve. Epoch validation is the inner split;
  the new outer plots describe held-out outer folds, not checkpoint selection.
- `thesis_validation.py sl-cv`: per-model outer-fold confusion matrices and an
  `outer_cv_scores.png` summary. Its `compare` command also saves the score plot,
  separately within SL or DL. It is not itself a significance visualization.
- `compare_dl_sl_models.py`: labeled confusion-matrix PNGs and `summary_metrics.png`
  in each of `family_evaluation_results/sl` and `family_evaluation_results/dl`.
  No pooled SL/DL ranking or significance test is added.

No epoch loss/accuracy history is invented for SL: the current SVM, k-NN, forest
and XGBoost procedures do not record such a history. Collecting new learning
curves would require extra fits and is intentionally not part of this change.

## Rerun Order

For the Fourier correction and new SL figures, from `01_Python` run:

```powershell
python sl_feature_comparison.py
python sl_training_pipeline.py --run-exported-batch
python compare_dl_sl_models.py
```

Existing extracted Fourier features and DL checkpoints can be reused. Do not
repeat cropping, dataset splitting or extraction solely for these figures.
New DL outer-fold figures are generated on future DL runs; existing DL epoch
plots do not need regeneration. Recreate SL nested-CV/statistical outputs if used
as thesis evidence after the Fourier correction. No full experiment is run just
by importing these plotting helpers.
