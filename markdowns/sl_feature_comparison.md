# `sl_feature_comparison.py`

Filter consensus votes, filter overlap, stage overlap and stage subset sizes are
also saved as PNGs beside their CSVs. See [Result Plots](result_plots.md).

The export manifest records selection_split=train, selection_protocol=train_only_v5_pooled_augmented_validation,
and training_pool_id. All selection methods now see the same 1,140-image balanced pool
as DL (95 per species), in the same order. Existing test rows are preserved in exports.
Augmented descriptors are retained in reduced CSVs for fitting, but excluded from
rankings and diagnostics. PCA and MI receive the workflow seed explicitly.
See [Shared Augmentation](shared_augmentation.md).
Batch training refuses older manifests, including v2 exports that used unbalanced train rows.
See [Shared Training Pool](shared_training_pool.md) and [Thesis Evaluation Protocol](thesis_evaluation_protocol.md).

High-level SL feature-selection workflow. It loads the handcrafted feature dataset, uses only rows marked train for every exploratory and feature-selection calculation, then runs filter, wrapper, and embedded ranking methods, compares their selected feature sets, and saves diagnostic outputs.

It exports reduced, training-ready CSV datasets containing both train and untouched test rows. The feature columns were selected from training rows only, and the manifest can be processed in batch mode by sl_training_pipeline.py. The implementation methods live in sl_feature_comparison_tools.py.

See `sl_feature_comparison_guide.md` for the detailed workflow.
