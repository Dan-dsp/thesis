# `sl_feature_comparison.py`

Filter consensus votes, filter overlap, stage overlap and stage subset sizes are
also saved as PNGs beside their CSVs. See [Result Plots](result_plots.md).

The export manifest now records selection_split=train and selection_protocol=train_only_v2_fourier. Batch training refuses older manifests, including v1 exports that excluded Fourier features. See [Thesis Evaluation Protocol](thesis_evaluation_protocol.md).

High-level SL feature-selection workflow. It loads the handcrafted feature dataset, uses only rows marked train for every exploratory and feature-selection calculation, then runs filter, wrapper, and embedded ranking methods, compares their selected feature sets, and saves diagnostic outputs.

It exports reduced, training-ready CSV datasets containing both train and untouched test rows. The feature columns were selected from training rows only, and the manifest can be processed in batch mode by sl_training_pipeline.py. The implementation methods live in sl_feature_comparison_tools.py.

See `sl_feature_comparison_guide.md` for the detailed workflow.
