# `sl_main.py`

When extraction is enabled, the first stage now creates/verifies the shared pool
and generates fixed image views before extracting their descriptors. This is an
SL-only launcher; run_pipeline.py rebuilds both SL and DL with the same policy.
See [Shared Augmentation](shared_augmentation.md).

The training stage now invokes manifest-driven batch training on selected feature datasets. It no longer invokes the full-feature single-dataset training default.

Top-level SL orchestrator. It can run feature extraction, feature comparison, and training in sequence, while allowing each stage to be turned on or off through its `main()` arguments.

The stages are `sl_dataframe_main.py`, `sl_feature_comparison.py`, and `sl_training_pipeline.py`. Use this script for an end-to-end SL run; use the individual scripts when you want to repeat just one stage.

See `sl_main_guide.md` for the detailed guide.
