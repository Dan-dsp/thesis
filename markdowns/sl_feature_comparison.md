# `sl_feature_comparison.py`

High-level SL feature-selection workflow. It loads the handcrafted feature dataset, runs exploratory checks and several filter, wrapper, and embedded ranking methods, compares their selected feature sets, and saves diagnostic outputs.

It also exports reduced, training-ready CSV datasets plus a manifest that `sl_training_pipeline.py` can process in batch mode. The implementation methods live in `sl_feature_comparison_tools.py`.

See `sl_feature_comparison_guide.md` for the detailed workflow.
