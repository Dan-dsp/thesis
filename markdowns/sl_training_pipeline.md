# `sl_training_pipeline.py`

The `cm/` folder also contains per-model grid-search diagnostics and grouped
validation/test metrics, alongside the existing confusion matrices.
See [Result Plots](result_plots.md).

Training now defaults to the reduced consensus top-50 CSV; batch mode requires a new train-only selection manifest. It writes per-image prediction/error audits, rejection coverage and accuracy, model-specific complexity, and a validation-selected model descriptor. See [Thesis Evaluation Protocol](thesis_evaluation_protocol.md).

Legacy column names are only `f` followed by digits (for example, `f0`).
Descriptive `fourier_1` through `fourier_20` are eligible predictors, not legacy
columns. Batch training requires `train_only_v5_pooled_augmented_validation` exports with a matching
training_pool_id; regenerate selection with `python sl_feature_comparison.py` first.

All training inputs are matched to DL's shared balanced image list (1,140 images,
95/species), plus 2,280 fixed views. All 3,420 rows enter the ordinary runtime
split: 2,736 fitting rows and 684 validation rows. Versions are not grouped;
related source images can cross those subsets and inflate validation scores.
Ordinary tuning CV also splits mixed rows. Separate nested independent CV retains
source isolation. The final SL refit uses all 3,420 rows. Test stays unaugmented.
See [Shared Augmentation](shared_augmentation.md).
See [Shared Training Pool](shared_training_pool.md).

Main SL model-training pipeline. It loads the handcrafted feature CSV or reduced feature-set CSVs, uses only the declared training rows to build an internal validation split, tunes classical models, and evaluates the selected models on the held-out test rows.

It supports SVM, Random Forest, k-NN, and XGBoost when installed. Each run records models, feature columns, metrics, confusion matrices, comparisons, and timing in a timestamped output directory. Batch mode consumes the feature-set manifest from `sl_feature_comparison.py`.

See `sl_training_pipeline_guide.md` for the detailed guide.

Plots use the non-interactive Matplotlib `Agg` backend and are saved as PNG files.
This avoids Tkinter cleanup errors during threaded grid search on Windows
(`main thread is not in main loop` and `Tcl_AsyncDelete`). It does not change
feature selection, model settings, or scoring.

If feature selection already saved the training-ready datasets, restart Python
and run `python sl_training_pipeline.py --run-exported-batch` from `01_Python`.
This starts a new training batch without repeating extraction or RFECV; it does
not resume a partially completed batch. Joblib temporary-folder cleanup warnings
after an abrupt exit should be investigated separately if they persist.
